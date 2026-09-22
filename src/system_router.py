"""
system_router.py — Unified System Router
=========================================
Merges: neural_router.py + Garlic-Components/ + Ada Step Entropy

Architecture: 13-stage forward pass
  Stage 0:  Context encoding
  Stage 1:  GlobalGraphMemory — prepend history
  Stage 2:  HSGM compression
  Stage 3-4: Neural slot prediction
  Stage 5-6: Safety validation + template selection
  Stage 7:  Model slot selection
  Stage 8:  Step Entropy (Ada native or Python fallback)
  Stage 9:  Semantic signals (ReasoningScaffolder)
  Stage 10: Expert routing (EntropyRegularizedRouter)
  Stage 11: Garlic Bridge memory injection
  Stage 12: Template assembly
  Stage 13: GlobalGraphMemory update

Papers:
  Step Entropy — arXiv:2508.03346
  HSGM — Hierarchical Segment-Graph Memory
  Reasoning Scaffolding — arXiv:2509.23619
  Entropy Router — Pangu 5.5
  DARE — ICLR 2026 (difficulty-aware allocation)
  MoDE — AAAI-22 (GNN expert collaboration)
"""

import os
import sys
import time as _time
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ============================================================================
# Ada Bridge — optional, falls back to Python
# ============================================================================

ADA_AVAILABLE = False
_GarlicAdaStepEntropy = None
_AdaptiveThresholdManager = None

try:
    _ada_bridge_dir = str(Path(__file__).resolve().parent / "ada_bridge")
    if _ada_bridge_dir not in sys.path:
        sys.path.insert(0, _ada_bridge_dir)
    from ada_bridge import GarlicAdaStepEntropy as _GarlicAdaStepEntropy
    from ada_bridge import AdaptiveThresholdManager as _AdaptiveThresholdManager
    ADA_AVAILABLE = True
    logger.info("Ada Step Entropy bridge loaded (GNAT 13 native)")
except Exception as _e:
    logger.info(f"Ada bridge unavailable ({_e}) — Python fallback active")

# ============================================================================
# Import Garlic-Components (working standalone modules)
# ============================================================================

_components_dir = str(Path(__file__).resolve().parent / "Garlic-Components")
if _components_dir not in sys.path:
    sys.path.insert(0, _components_dir)

from hsgm_local_graph_builder import HSGMLocalGraphBuilder, LocalGraphConfig
from gnn_summary_extractor import GNNSummaryExtractor, SummaryExtractionConfig
from hierarchical_query_processor import HierarchicalQueryProcessor, HierarchicalQueryConfig
from reasoning_scaffolder_prod import ReasoningScaffolder, SemanticSignal
from entropy_regularized_router import EntropyRegularizedRouter, RoutingDecision
from garlic_injection_layer import GarlicInjectionLayer

# Import updated neural router components (post Phase 1 renames)
_router_dir = str(Path(__file__).resolve().parent)
if _router_dir not in sys.path:
    sys.path.insert(0, _router_dir)

from neural_router import (
    RouterConfig,
    ModelSlot,
    ModelSlotConfig,
    SlotPredictions,
    SystemOutput,
    ModelLoader,
    ContextFeatures,
    ContextEncoder,
    HashTextEncoder,
    ProfileEncoder,
    MetadataEncoder,
    InputPreparer,
    SlotPredictorNetwork,
    TemplateSelectorNetwork,
    SafetyValidator,
    TemplateLibrary,
    NeuralPromptRouter,
)

# ============================================================================
# Step Entropy Interface + Fallback
# ============================================================================


@runtime_checkable
class StepEntropyInterface(Protocol):
    """Type contract satisfied by Ada bridge and Python fallback."""

    def calculate_batch_step_entropy(
        self,
        token_logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor: ...


class PythonStepEntropyFallback:
    """Pure-Python step entropy — slower, no Ada compile-time guarantees.
    Interface-compatible with GarlicAdaStepEntropy."""

    def __init__(self, threshold_low: float = 2.0, threshold_high: float = 4.0):
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def calculate_batch_step_entropy(
        self,
        token_logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """[B, S, V] → [B] mean token entropy (bits) per sample."""
        probs = torch.softmax(token_logits.float(), dim=-1)
        log_probs = torch.log2(probs + 1e-10)
        token_entropy = -(probs * log_probs).sum(dim=-1)  # [B, S]
        return token_entropy.mean(dim=-1)  # [B]

    def route_step(self, entropy_value: float) -> str:
        if entropy_value < self.threshold_low:
            return "fast"
        elif entropy_value > self.threshold_high:
            return "slow"
        return "normal"


class _AdaBridgeAdapter:
    """
    Adapts GarlicAdaStepEntropy to the StepEntropyInterface protocol.

    GarlicAdaStepEntropy.calculate_step_entropy() → StepEntropyData with .avg_entropy
    StepEntropyInterface.calculate_batch_step_entropy() → torch.Tensor [B]

    This thin wrapper bridges the method-name gap without modifying the Ada bridge.
    """

    def __init__(self, bridge: Any, threshold_low: float, threshold_high: float):
        self._bridge = bridge
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def calculate_batch_step_entropy(
        self,
        token_logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """[B, S, V], [B, S] → [B] mean entropy per sample (bits)."""
        batch_size = token_logits.shape[0]
        entropies = []
        for b in range(batch_size):
            logits_b = token_logits[b]   # [S, V]
            ids_b = token_ids[b]         # [S]
            texts_b = [""] * ids_b.shape[0]
            try:
                step_data = self._bridge.calculate_step_entropy(
                    logits_b, ids_b, texts_b, step_id=b,
                )
                entropies.append(float(step_data.avg_entropy))
            except Exception:
                entropies.append(0.0)
        return torch.tensor(entropies, dtype=torch.float32)

    def route_step(self, entropy_value: float) -> str:
        if entropy_value < self.threshold_low:
            return "fast"
        elif entropy_value > self.threshold_high:
            return "slow"
        return "normal"


def build_step_entropy(
    threshold_low: float = 2.0,
    threshold_high: float = 4.0,
) -> StepEntropyInterface:
    """Factory: Ada native when available, Python fallback otherwise."""
    if ADA_AVAILABLE and _GarlicAdaStepEntropy is not None:
        try:
            bridge = _GarlicAdaStepEntropy(
                threshold_low=threshold_low,
                threshold_high=threshold_high,
            )
            adapter = _AdaBridgeAdapter(bridge, threshold_low, threshold_high)
            logger.info("Step entropy: Ada native (via adapter)")
            return adapter
        except Exception as e:
            logger.warning(f"Ada bridge init failed: {e} — using Python fallback")
    logger.info("Step entropy: Python fallback")
    return PythonStepEntropyFallback(threshold_low, threshold_high)


# ============================================================================
# GlobalGraphMemory
# ============================================================================


@dataclass
class TurnRecord:
    turn_id: int
    input_graph: torch.Tensor
    entropy_value: float
    model_slot: ModelSlot
    timestamp: float = field(default_factory=_time.time)


class GlobalGraphMemory:
    """Turn-to-turn persistent HSGM state. Never flushes until session end."""

    def __init__(self, hidden_dim: int = 768, max_turns: int = 1000):
        self.hidden_dim = hidden_dim
        self.max_turns = max_turns
        self.turn_history: List[TurnRecord] = []
        self._global_graph: Optional[torch.Tensor] = None

    @property
    def is_empty(self) -> bool:
        return self._global_graph is None

    @property
    def turn_count(self) -> int:
        return len(self.turn_history)

    def add_turn(
        self,
        input_graph: torch.Tensor,
        entropy_value: float,
        model_slot: ModelSlot,
    ) -> None:
        record = TurnRecord(
            turn_id=len(self.turn_history),
            input_graph=input_graph.detach(),
            entropy_value=entropy_value,
            model_slot=model_slot,
        )
        self.turn_history.append(record)
        if len(self.turn_history) > self.max_turns:
            self.turn_history.pop(0)
        pooled = input_graph.mean(dim=1, keepdim=True).detach()  # [B, 1, D]
        if self._global_graph is None:
            self._global_graph = pooled
        else:
            n = float(len(self.turn_history))
            self._global_graph = (self._global_graph * (n - 1) + pooled) / n

    def get_context(
        self, device: Optional[torch.device] = None
    ) -> Optional[torch.Tensor]:
        if self._global_graph is None:
            return None
        return self._global_graph.to(device) if device else self._global_graph

    def flush(self) -> None:
        self.turn_history.clear()
        self._global_graph = None


# ============================================================================
# SOTA++ Innovations
# ============================================================================


class DifficultyAwareExpertAllocator(nn.Module):
    """
    DARE-inspired (ICLR 2026): Step Entropy as difficulty proxy → dynamic K.
    Learnable thresholds trained end-to-end.
    """

    def __init__(
        self,
        num_experts: int = 4,
        min_experts: int = 1,
        max_experts: int = 3,
        init_threshold_low: float = 2.0,
        init_threshold_high: float = 4.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.min_experts = min_experts
        self.max_experts = max_experts
        self.threshold_low = nn.Parameter(torch.tensor(init_threshold_low))
        self.threshold_high = nn.Parameter(torch.tensor(init_threshold_high))
        self.register_buffer("optimal_pruning_ratio", torch.tensor(0.80))

    def forward(self, entropy_values: torch.Tensor) -> torch.Tensor:
        """[B, S] entropy → [B, S] int expert counts."""
        counts = torch.full_like(entropy_values, 2, dtype=torch.long)
        counts[entropy_values < self.threshold_low] = self.min_experts
        counts[entropy_values >= self.threshold_high] = self.max_experts
        return counts

    def get_routing_loss(self) -> torch.Tensor:
        ordering = F.relu(self.threshold_low - self.threshold_high + 0.5)
        bound = F.relu(1.0 - self.threshold_low) + F.relu(self.threshold_high - 6.0)
        return ordering + 0.1 * bound


class MultiScaleGraphBuilder(nn.Module):
    """
    Build HSGM graphs at 3 scales for hierarchical injection.
    Fine (seg=16) → medium (seg=32) → coarse (seg=64).
    """

    SCALES = {"fine": 16, "medium": 32, "coarse": 64}

    def __init__(self, base_config: LocalGraphConfig, hidden_dim: int = 768):
        super().__init__()
        import copy

        self.builders = nn.ModuleDict()
        self.extractors = nn.ModuleDict()
        for scale_name, seg_size in self.SCALES.items():
            cfg = copy.copy(base_config)
            cfg.segment_size = seg_size
            self.builders[scale_name] = HSGMLocalGraphBuilder(cfg)
            self.extractors[scale_name] = GNNSummaryExtractor(
                SummaryExtractionConfig(
                    hidden_dim=hidden_dim,
                    summary_ratio=0.25,
                )
            )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Returns {'fine': [B,N1,D], 'medium': [B,N2,D], 'coarse': [B,N3,D]}"""
        results: Dict[str, torch.Tensor] = {}
        for scale_name in self.SCALES:
            local_graphs = self.builders[scale_name](hidden_states)
            summary = self.extractors[scale_name](
                local_graphs["node_embeddings"],
                local_graphs["edge_index"],
                local_graphs.get("edge_weights"),
            )
            results[scale_name] = summary["summary_embeddings"]
        return results


class GNNExpertCollaborationLayer(nn.Module):
    """
    MoDE-inspired (AAAI-22): experts form a graph, collaborate via message passing.
    Gracefully skips if torch_geometric absent.
    """

    def __init__(
        self,
        num_experts: int = 4,
        expert_dim: int = 768,
        num_gnn_layers: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_adjacency = nn.Parameter(torch.eye(num_experts))
        self._gnn_available = False
        try:
            from torch_geometric.nn import GATConv

            self.gnn_layers = nn.ModuleList(
                [
                    GATConv(expert_dim, expert_dim // 4, heads=4, concat=True)
                    for _ in range(num_gnn_layers)
                ]
            )
            self._gnn_available = True
        except ImportError:
            logger.info("torch_geometric absent — GNN expert collab uses weighted-sum fallback")

    def forward(
        self,
        expert_outputs: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        expert_outputs: [B, num_experts, D]
        expert_weights: [B, num_experts]
        → collaborated_output: [B, D]
        """
        if not self._gnn_available:
            return torch.einsum("be,bed->bd", expert_weights, expert_outputs)

        batch_size = expert_outputs.shape[0]
        edge_weights = torch.sigmoid(self.expert_adjacency)
        edge_index = (edge_weights > 0.1).nonzero(as_tuple=False).t()
        x = expert_outputs
        for gnn_layer in self.gnn_layers:
            processed = []
            for b in range(batch_size):
                x_b = gnn_layer(x[b], edge_index)
                processed.append(x_b)
            x = torch.stack(processed, dim=0)
        return torch.einsum("be,bed->bd", expert_weights, x)

    def get_graph_sparsity_loss(self) -> torch.Tensor:
        return torch.sigmoid(self.expert_adjacency).sum() * 0.01


class EntropyAwareInjectionController(nn.Module):
    """
    Dynamic injection: entropy → layer mask + strength.
    Low entropy → inject late only. High → inject all layers. (Novel — not in papers.)
    """

    def __init__(self, num_layers: int = 32, hidden_dim: int = 768):
        super().__init__()
        self.num_layers = num_layers
        self.entropy_to_layer_probs = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, num_layers), nn.Sigmoid()
        )
        self.entropy_to_scale = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, entropy_value: float) -> Dict[str, Any]:
        e = torch.tensor([[entropy_value]], dtype=torch.float32)
        layer_probs = self.entropy_to_layer_probs(e).squeeze(0)
        scale = self.entropy_to_scale(e).item()

        if entropy_value < 2.0:
            mask = torch.zeros(self.num_layers)
            mask[-1] = 1.0
        elif entropy_value < 4.0:
            k = max(1, self.num_layers // 3)
            topk_idx = torch.topk(layer_probs, k).indices
            mask = torch.zeros(self.num_layers)
            mask[topk_idx] = 1.0
        else:
            mask = (layer_probs > 0.3).float()

        return {
            "layer_mask": mask,
            "injection_scale": scale,
            "layer_scales": mask * layer_probs * scale,
        }


# ============================================================================
# SystemRouterConfig
# ============================================================================


@dataclass
class SystemRouterConfig:
    """Merged config: RouterConfig + GarlicOrchestratorConfig + SOTA++ fields."""

    # Neural routing (from RouterConfig)
    context_dim: int = 768
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    num_templates: int = 16
    num_tools: int = 32
    learning_rate: float = 1e-4
    dropout: float = 0.1
    weight_decay: float = 0.01
    num_model_slots: int = 3
    builtin_tools: Optional[List[str]] = None

    # HSGM
    enable_hsgm: bool = True
    segment_size: int = 32
    overlap_size: int = 8
    similarity_threshold: float = 0.7
    summary_ratio: float = 0.25

    # Step Entropy
    enable_entropy_routing: bool = True
    entropy_threshold_low: float = 2.0
    entropy_threshold_high: float = 4.0

    # Entropy Router (from GarlicOrchestratorConfig)
    num_experts: int = 4
    top_k: int = 2
    temperature: float = 1.0
    lambda_entropy: float = 0.01
    router_entropy_threshold_low: float = 0.2
    router_entropy_threshold_high: float = 0.8

    # Semantic signals
    enable_semantic_signals: bool = True

    # GNN routing
    enable_gnn_routing: bool = False

    # Garlic injection
    enable_garlic_injection: bool = True
    injection_scale_init: float = 0.1

    # Global memory
    global_memory_max_turns: int = 1000

    # SOTA++
    use_difficulty_aware_allocation: bool = True
    use_multi_scale_graphs: bool = False  # Expensive — opt-in
    use_gnn_expert_collab: bool = False   # Requires torch_geometric
    use_entropy_aware_injection: bool = True

    def __post_init__(self) -> None:
        if self.builtin_tools is None:
            self.builtin_tools = ["browser", "python", "web_search"]

    def to_router_config(self) -> RouterConfig:
        """Downcast to RouterConfig for neural_router components."""
        return RouterConfig(
            context_dim=self.context_dim,
            num_transformer_layers=self.num_transformer_layers,
            num_attention_heads=self.num_attention_heads,
            num_templates=self.num_templates,
            num_tools=self.num_tools,
            learning_rate=self.learning_rate,
            dropout=self.dropout,
            weight_decay=self.weight_decay,
            num_model_slots=self.num_model_slots,
            builtin_tools=list(self.builtin_tools),
        )


# ============================================================================
# SystemRouter — 13-Stage Forward Pass
# ============================================================================


class SystemRouter(nn.Module):
    """
    Unified system router — orchestrator internalized.

    All 13 stages of the forward pass in one module.
    Ada Step Entropy or Python fallback, with automatic selection.
    """

    def __init__(self, config: SystemRouterConfig):
        super().__init__()
        self.config = config
        router_cfg = config.to_router_config()

        # ── Neural routing components (from neural_router.py) ──────────────
        self.template_library = TemplateLibrary(router_cfg)
        router_cfg.num_templates = len(self.template_library.templates)
        self.context_encoder = ContextEncoder(router_cfg)
        self.slot_predictor = SlotPredictorNetwork(router_cfg)
        self.template_selector = TemplateSelectorNetwork(router_cfg)
        self.safety_validator = SafetyValidator(router_cfg)
        self.tool_embeddings = nn.Parameter(
            torch.randn(config.num_tools, config.context_dim)
        )

        # ── Model loader ───────────────────────────────────────────────────
        self.model_loader = ModelLoader(num_slots=config.num_model_slots)

        # ── HSGM pipeline ─────────────────────────────────────────────────
        if config.enable_hsgm:
            _hsgm_cfg = LocalGraphConfig(
                segment_size=config.segment_size,
                overlap_size=config.overlap_size,
                similarity_threshold=config.similarity_threshold,
                hidden_dim=config.context_dim,
            )
            if config.use_multi_scale_graphs:
                self.multi_scale_builder = MultiScaleGraphBuilder(
                    _hsgm_cfg, hidden_dim=config.context_dim
                )
            else:
                self.local_graph_builder = HSGMLocalGraphBuilder(_hsgm_cfg)
                self.summary_extractor = GNNSummaryExtractor(
                    SummaryExtractionConfig(
                        hidden_dim=config.context_dim,
                        summary_ratio=config.summary_ratio,
                    )
                )

        # ── Reasoning scaffolder ───────────────────────────────────────────
        if config.enable_semantic_signals:
            self.reasoning_scaffolder = ReasoningScaffolder(
                hidden_dim=config.context_dim
            )

        # ── Entropy regularized router ─────────────────────────────────────
        self.entropy_router = EntropyRegularizedRouter(
            hidden_dim=config.context_dim,
            graph_context_dim=config.context_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            temperature=config.temperature,
            entropy_threshold_high=config.router_entropy_threshold_high,
            entropy_threshold_low=config.router_entropy_threshold_low,
            lambda_entropy=config.lambda_entropy,
            use_gnn=config.enable_gnn_routing,
        )

        # ── Garlic bridge layers ───────────────────────────────────────────
        if config.enable_garlic_injection:
            self.garlic_bridge = nn.Linear(
                config.context_dim * 2, config.context_dim
            )
            self.signal_projection = nn.Linear(
                config.context_dim, config.context_dim
            )

        # ── Step Entropy (Ada or Python) ───────────────────────────────────
        if config.enable_entropy_routing:
            self.step_entropy: StepEntropyInterface = build_step_entropy(
                threshold_low=config.entropy_threshold_low,
                threshold_high=config.entropy_threshold_high,
            )

        # ── Global graph memory ────────────────────────────────────────────
        self.global_memory = GlobalGraphMemory(
            hidden_dim=config.context_dim,
            max_turns=config.global_memory_max_turns,
        )

        # ── SOTA++ components ──────────────────────────────────────────────
        if config.use_difficulty_aware_allocation:
            self.difficulty_allocator = DifficultyAwareExpertAllocator(
                num_experts=config.num_experts,
            )

        if config.use_gnn_expert_collab:
            self.expert_collab = GNNExpertCollaborationLayer(
                num_experts=config.num_experts,
                expert_dim=config.context_dim,
            )

        if config.use_entropy_aware_injection:
            self.injection_controller = EntropyAwareInjectionController(
                hidden_dim=config.context_dim,
            )

        # Internal state for injection controller
        self._last_entropy: float = 0.0

        logger.info(
            f"SystemRouter initialized — ADA={ADA_AVAILABLE}, "
            f"HSGM={config.enable_hsgm}, "
            f"SOTA++=[DARE={config.use_difficulty_aware_allocation}, "
            f"MultiScale={config.use_multi_scale_graphs}, "
            f"GNN={config.use_gnn_expert_collab}]"
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _compress_hsgm(
        self, hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """HSGM compression → [B, N_summary, D]."""
        if not self.config.enable_hsgm:
            return None
        try:
            if self.config.use_multi_scale_graphs and hasattr(self, "multi_scale_builder"):
                return self.multi_scale_builder(hidden_states)["medium"]
            if hasattr(self, "local_graph_builder"):
                local_graphs = self.local_graph_builder(hidden_states)
                summary = self.summary_extractor(
                    local_graphs["node_embeddings"],
                    local_graphs["edge_index"],
                    local_graphs.get("edge_weights"),
                )
                return summary["summary_embeddings"]
        except Exception as e:
            logger.warning(f"HSGM compression failed: {e}")
        return None

    def inject_memory_garlic_bridge(
        self,
        hidden_states: torch.Tensor,
        graph_memory: torch.Tensor,
        semantic_signals: torch.Tensor,
    ) -> torch.Tensor:
        """
        h' = h + W_bridge([pool(graph); proj(signals)])

        Inlined from garlic_orchestrator.py:467-511.
        """
        batch_size, seq_len, _ = hidden_states.shape
        graph_pooled = graph_memory.mean(dim=1, keepdim=True).expand(
            -1, seq_len, -1
        )
        signals_proj = self.signal_projection(semantic_signals)
        memory_context = torch.cat([graph_pooled, signals_proj], dim=-1)
        memory_injection = self.garlic_bridge(memory_context)

        # Apply entropy-aware scaling
        if hasattr(self, "injection_controller"):
            ctrl = self.injection_controller(self._last_entropy)
            memory_injection = memory_injection * ctrl["injection_scale"]

        return hidden_states + memory_injection

    # ── Main forward pass ──────────────────────────────────────────────────

    def forward(
        self,
        message_embs: torch.Tensor,
        user_profile: torch.Tensor,
        metadata: torch.Tensor,
        context_metadata: Dict[str, Any],
        hidden_states: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        message_mask: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> SystemOutput:
        """13-stage forward pass."""
        trace: Optional[Dict[str, Any]] = {} if return_trace else None
        batch_size = message_embs.shape[0]
        device = message_embs.device

        # ── STAGE 0: Context encoding ──────────────────────────────────────
        context_emb = self.context_encoder(
            message_embs, user_profile, metadata, message_mask
        )
        if trace is not None:
            trace["stage0_context_norm"] = context_emb.norm(dim=-1).mean().item()

        # ── STAGE 1: Global Memory — prepend history ───────────────────────
        history_context = self.global_memory.get_context(device=device)
        if history_context is not None and hidden_states is not None:
            hidden_states = torch.cat(
                [history_context.expand(batch_size, -1, -1), hidden_states], dim=1
            )

        # ── STAGE 2: HSGM compression ──────────────────────────────────────
        original_tokens = hidden_states.shape[1] if hidden_states is not None else 0
        compressed_context: Optional[torch.Tensor] = None
        if hidden_states is not None:
            compressed_context = self._compress_hsgm(hidden_states)
        compressed_tokens = (
            compressed_context.shape[1]
            if compressed_context is not None
            else original_tokens
        )
        compression_ratio = float(original_tokens) / max(compressed_tokens, 1)
        if trace is not None:
            trace["stage2_compression_ratio"] = compression_ratio

        # ── STAGE 3-4: Neural slot prediction ──────────────────────────────
        slot_preds = self.slot_predictor(context_emb, self.tool_embeddings)

        # ── STAGE 5-6: Safety + template selection ─────────────────────────
        slot_preds, violations = self.safety_validator.validate_slots(
            slot_preds, context_metadata
        )
        template_weights = self.template_selector(slot_preds)
        if trace is not None:
            trace["stage56_violations"] = violations
            trace["stage56_selected_template"] = template_weights.argmax(dim=-1).item()

        # ── STAGE 7: Model slot selection ──────────────────────────────────
        active_slot = self.model_loader.predict_slot(slot_preds.model_slot)
        if trace is not None:
            trace["stage7_model_slot"] = active_slot.name

        # ── STAGE 8: Step Entropy ──────────────────────────────────────────
        entropy_value = 0.0
        entropy_level = "normal"
        if (
            self.config.enable_entropy_routing
            and logits is not None
            and hidden_states is not None
        ):
            token_ids = context_metadata.get(
                "token_ids",
                torch.zeros(
                    batch_size, logits.shape[1], dtype=torch.long, device=device
                ),
            )
            entropy_tensor = self.step_entropy.calculate_batch_step_entropy(
                logits, token_ids
            )
            entropy_value = entropy_tensor.mean().item()
            if entropy_value < self.config.entropy_threshold_low:
                entropy_level = "fast"
            elif entropy_value > self.config.entropy_threshold_high:
                entropy_level = "slow"
        self._last_entropy = entropy_value
        if trace is not None:
            trace["stage8_entropy"] = {"value": entropy_value, "level": entropy_level}

        # ── STAGE 9: Semantic signals ──────────────────────────────────────
        semantic_signals: Optional[torch.Tensor] = None
        if (
            self.config.enable_semantic_signals
            and hidden_states is not None
            and hasattr(self, "reasoning_scaffolder")
        ):
            try:
                pooled = hidden_states.mean(dim=1)
                signal_pred, signal_emb = self.reasoning_scaffolder.predict_and_embed(
                    pooled
                )
                semantic_signals = signal_emb.unsqueeze(1).expand(
                    -1, hidden_states.shape[1], -1
                )
                if trace is not None:
                    trace["stage9_signal"] = signal_pred.signal.value if hasattr(signal_pred, "signal") else str(signal_pred)
            except Exception as e:
                logger.debug(f"Semantic signal failed: {e}")

        # ── STAGE 10: Expert routing ───────────────────────────────────────
        routing_decision: Optional[RoutingDecision] = None
        if hidden_states is not None and compressed_context is not None:
            try:
                # Align compressed_context to seq_len for router
                seq_len = hidden_states.shape[1]
                graph_aligned = compressed_context.mean(dim=1, keepdim=True).expand(
                    -1, seq_len, -1
                )
                _, expert_indices, router_meta = self.entropy_router(
                    hidden_states, graph_aligned, semantic_signals=semantic_signals
                )
                routing_decision = RoutingDecision(
                    path=entropy_level,
                    expert_indices=expert_indices,
                    gating_weights=router_meta.get("entropy", torch.zeros(1)),
                    entropy=router_meta.get("entropy", torch.zeros(batch_size, seq_len)),
                    is_ood=router_meta.get("is_ood", torch.zeros(batch_size, seq_len, dtype=torch.bool)),
                    retrieve_memory=router_meta.get("retrieve_memory", torch.zeros(batch_size, seq_len, dtype=torch.bool)),
                )
                if trace is not None:
                    trace["stage10_entropy_mean"] = router_meta["entropy"].mean().item()
            except Exception as e:
                logger.warning(f"Expert routing failed: {e}")

        # ── STAGE 11: Garlic Bridge memory injection ───────────────────────
        augmented_states: Optional[torch.Tensor] = None
        if (
            self.config.enable_garlic_injection
            and hidden_states is not None
            and compressed_context is not None
            and semantic_signals is not None
            and hasattr(self, "garlic_bridge")
        ):
            try:
                augmented_states = self.inject_memory_garlic_bridge(
                    hidden_states, compressed_context, semantic_signals
                )
            except Exception as e:
                logger.warning(f"Garlic bridge injection failed: {e}")

        # ── STAGE 12: Template assembly ────────────────────────────────────
        prompt = self.template_library.assemble(
            template_weights, slot_preds, context_metadata
        )
        prompt, _ = self.safety_validator.validate_output(prompt)

        # ── STAGE 13: Global memory update ────────────────────────────────
        if compressed_context is not None:
            self.global_memory.add_turn(compressed_context, entropy_value, active_slot)

        return SystemOutput(
            prompt=prompt,
            model_slot=active_slot,
            routing_decision=routing_decision,
            entropy_level=entropy_level,
            entropy_value=entropy_value,
            compression_ratio=compression_ratio,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            augmented_states=augmented_states,
            hsgm_summary=compressed_context,
            trace=trace,
        )

    def shutdown_session(self) -> None:
        """Flush global memory at session end."""
        self.global_memory.flush()
        logger.info("Session ended — GlobalGraphMemory flushed")


# ============================================================================
# SystemRouterWrapper — production wrapper with fallback
# ============================================================================


class SystemRouterWrapper:
    """
    Production wrapper for SystemRouter.
    Returns SystemOutput; falls back to Jinja2-only on neural failure.

    Replaces SafeRouterWrapper from neural_router.py (extended output type).
    """

    def __init__(self, config: Optional[SystemRouterConfig] = None):
        self.config = config or SystemRouterConfig()
        try:
            self.router = SystemRouter(self.config)
            self._neural_available = True
        except Exception as e:
            logger.error(f"SystemRouter init failed: {e}")
            self._neural_available = False
            self.router = None

    def route(
        self,
        message_embs: torch.Tensor,
        user_profile: torch.Tensor,
        metadata: torch.Tensor,
        context_metadata: Dict[str, Any],
        hidden_states: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> SystemOutput:
        if self._neural_available and self.router is not None:
            try:
                return self.router(
                    message_embs=message_embs,
                    user_profile=user_profile,
                    metadata=metadata,
                    context_metadata=context_metadata,
                    hidden_states=hidden_states,
                    logits=logits,
                    return_trace=return_trace,
                )
            except Exception as e:
                logger.warning(f"Neural routing failed: {e} — Jinja2 fallback")

        # Jinja2 fallback — template only, no neural
        return SystemOutput(
            prompt="<|start|>system<|message|>Fallback routing active.<|end|>",
            model_slot=ModelSlot.SLOT_B,
            entropy_level="normal",
            trace={"method": "jinja2_fallback"} if return_trace else None,
        )
