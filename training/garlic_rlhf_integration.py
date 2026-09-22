"""
Garlic Architecture + Production RLHF Integration

Source: /home/daeron/Projects/Garlic/garlic_gpt2_adapter.py (Garlic old)
        /home/daeron/Projects/ADA-Step-Entropy/ada_bridge.py (Ada step entropy bridge for runtime import of compiled step_entropy)
Integrated: 2026-05-05
Purpose: Wires GarlicGPT2 into the Full-RLHF-Pipeline production surface
         (rlhf.py, inference_optimizations.py, inference_protocols.py,
          telemetry.py, benchmark_harness.py) with Ada 2022 step entropy
          as the routing/compression backbone.

Architecture philosophy:
  Garlic lays the nail — structural routing/compression learning.
  Cherry hammers it right — quality alignment via PRM (separate stage).
  GarlicRewardModel stays lean; Cherry's process reward model handles quality.

Ada step entropy (arXiv:2508.03346):
  - Token entropy H(t) = -Σ p(w) log₂ p(w), compiled Ada 2022 kernel
  - Adaptive Welford thresholds replacing static 2.0/4.0 bit values
  - GRPO reward components R_skip_ratio, R_skip_num, R_response_len (Table 3)
  - Max_Vocab_Size = 262_144 (model-agnostic ceiling, rebuilt 2026-05-05)
  - ADA_STEP_ENTROPY_AVAILABLE reflects live bridge status at import time

Override env vars:
  GARLIC_ROOT            path to garlic_gpt2_adapter.py directory
  ADA_STEP_ENTROPY_ROOT  path to ada_bridge.py directory
"""

from __future__ import annotations

import logging
import os as _os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

# ---------------------------------------------------------------------------
# Portable path resolution — override via env vars for non-standard layouts
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent

_GARLIC_ROOT = _os.environ.get(
    "GARLIC_ROOT",
    str(_REPO_ROOT.parent / "Garlic"),
)
if _GARLIC_ROOT not in sys.path:
    sys.path.insert(0, _GARLIC_ROOT)

# Ada bridge is loaded via importlib by absolute path — NOT via sys.path —
# to avoid shadowing Garlic's neural_router.py with the homonymous file in
# ADA-Step-Entropy/System-Router/. sys.path manipulation for two projects
# that share file names is a silent namespace collision waiting to happen.
_ADA_STEP_ENTROPY_ROOT = _os.environ.get(
    "ADA_STEP_ENTROPY_ROOT",
    str(_REPO_ROOT.parent / "ADA-Step-Entropy"),
)

# ---------------------------------------------------------------------------
# rlhf.py — the only canon RLHF runtime spine
# ---------------------------------------------------------------------------
from rlhf import (
    # Orchestrator
    RLHFOrchestrator,
    # Infrastructure
    DeviceManager,
    CheckpointManager,
    TrainingLogger,
    EarlyStopping,
    # Configs
    BaseConfig,
    SFTConfig,
    RewardModelConfig,
    DPOConfig,
    GRPOConfig,
    TreeGRPOConfig,
    SimPOConfig,
    KTOConfig,
    PPOConfig,
    # Models
    PolicyModel,
    RewardModel,
    ProcessRewardModel,
    ValueModel,
    ContextCompressor,
    # In-memory datasets
    PreferenceDataset,
    SFTDataset,
    KTODataset,
    GRPODataset,
    # Streaming datasets (RAM-efficient, for JSONL)
    StreamingPreferenceDataset,
    StreamingSFTDataset,
    StreamingKTODataset,
    StreamingGRPODataset,
    # Trainers
    SFTTrainer,
    DPOTrainer,
    GRPOTrainer,
    TreeGRPOTrainer,
    SimPOTrainer,
    KTOTrainer,
    PPOTrainer,
    # Self-improvement
    AdversarialValidator,
    CapabilityTester,
    IterativeRefiner,
    # Evaluation
    RLHFEvaluator,
    # Reward utilities
    RewardFunctionFactory,
    ConstitutionalRewardWrapper,
)

# ---------------------------------------------------------------------------
# Inference / search-time capabilities
# ---------------------------------------------------------------------------
from inference_protocols import PolicyAdapter, ProcessRewardModelAdapter
from inference_optimizations import (
    BestOfNConfig,
    BestOfNSampler,
    MCTSConfig,
    MCTSGenerator,
    ChainOfThoughtConfig,
    ChainOfThoughtGenerator,
    AStarConfig,
    AStarGenerator,
    TreeRolloutCollector,
    RolloutSample,
    compile_model,
)

# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
from telemetry import TelemetryRecorder
from benchmark_harness import BenchmarkHarness

# ---------------------------------------------------------------------------
# Garlic architecture
# ---------------------------------------------------------------------------
from garlic_gpt2_adapter import GarlicGPT2

# ---------------------------------------------------------------------------
# Ada 2022 step entropy bridge (arXiv:2508.03346)
#
# Loaded via importlib by absolute path — NOT via sys.path — to prevent
# ADA-Step-Entropy's neural_router.py from shadowing Garlic's copy.
# Falls back gracefully: routing still functions via garlic_output keys.
# ---------------------------------------------------------------------------
import importlib.util as _importlib_util


def _load_ada_bridge_module() -> Optional[Any]:
    """Load ada_bridge.py by absolute path to avoid sys.path conflicts.

    Returns:
        Loaded module object, or None if not found or failed to load.
    """
    candidates = [
        Path(_ADA_STEP_ENTROPY_ROOT) / "ada_bridge.py",
        Path(_REPO_ROOT.parent / "ADA-Step-Entropy" / "ada_bridge.py"),
    ]
    for candidate in candidates:
        if candidate.exists():
            spec = _importlib_util.spec_from_file_location("ada_bridge", str(candidate))
            if spec is None or spec.loader is None:
                continue
            mod = _importlib_util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                return mod
            except Exception as _exc:
                continue  # try next candidate
    return None


_ada_bridge_mod = _load_ada_bridge_module()

if _ada_bridge_mod is not None:
    AdaStepEntropy = _ada_bridge_mod.AdaStepEntropy
    GarlicAdaStepEntropy = _ada_bridge_mod.GarlicAdaStepEntropy
    AdaptiveThresholdManager = _ada_bridge_mod.AdaptiveThresholdManager
    StepEntropyData = _ada_bridge_mod.StepEntropyData
    GRPORewardsData = _ada_bridge_mod.GRPORewardsData
    ADA_STEP_ENTROPY_AVAILABLE: bool = True
else:
    AdaStepEntropy = None  # type: ignore[assignment,misc]
    GarlicAdaStepEntropy = None  # type: ignore[assignment,misc]
    AdaptiveThresholdManager = None  # type: ignore[assignment,misc]
    StepEntropyData = None  # type: ignore[assignment,misc]
    GRPORewardsData = None  # type: ignore[assignment,misc]
    ADA_STEP_ENTROPY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if ADA_STEP_ENTROPY_AVAILABLE:
    logger.info("Ada step entropy bridge: LIVE (Max_Vocab_Size=262144)")
else:
    logger.warning(
        "Ada step entropy bridge: UNAVAILABLE — routing via garlic_output only"
    )


# ---------------------------------------------------------------------------
# TELEMETRY HELPER
# ---------------------------------------------------------------------------


def _sync_default_telemetry_recorder(recorder: TelemetryRecorder) -> None:
    """Sync a TelemetryRecorder into both telemetry.py and rlhf.py singletons.

    Mirrors run_pipeline.py lines 262-269 so TreeGRPOTrainer and all
    downstream rlhf.py trainers share the same recorder instance.

    Args:
        recorder: Initialized TelemetryRecorder to propagate.
    """
    import telemetry as _telemetry_module

    _telemetry_module.default_recorder = recorder
    try:
        import rlhf as _rlhf_module

        _rlhf_module.default_recorder = recorder
    except Exception as exc:
        logger.warning("Could not sync telemetry recorder into rlhf.py: %s", exc)


# ============================================================================
# GARLIC POLICY MODEL
# ============================================================================


class GarlicPolicyModel(PolicyModel):
    """PolicyModel wrapper over GarlicGPT2 for production RLHF training.

    Source: garlic_gpt2_adapter.GarlicGPT2 (Garlic project)
    Integrated: 2026-05-05
    Purpose: Presents the Garlic orchestration layer as a PolicyModel so
             all rlhf.py trainers (SFT, DPO, GRPO, TreeGRPO, PPO, …) can
             drive it without knowing about Garlic internals.

    Design notes:
      - Calls nn.Module.__init__ directly instead of PolicyModel.__init__
        because PolicyModel.__init__ downloads and instantiates a fresh base
        model from HuggingFace. GarlicGPT2 is already instantiated; re-
        loading would waste memory and break the orchestrator wiring.
      - Ada step entropy (if available) runs on the last-token logits of each
        forward pass and stores routing metadata in _last_routing for
        GarlicRewardModel to consume without a redundant forward call.
    """

    def __init__(
        self,
        garlic_model: GarlicGPT2,
        tokenizer: PreTrainedTokenizer,
        freeze_base_model: bool = False,
    ) -> None:
        """Initialize the Garlic policy model wrapper.

        Args:
            garlic_model: Fully initialized GarlicGPT2 instance.
            tokenizer: Tokenizer matching the base model.
            freeze_base_model: If True, freeze base model weights so only
                the Garlic orchestrator trains. Default False (both train).
        """
        # Intentional bypass — see class docstring for rationale.
        nn.Module.__init__(self)

        self.garlic_model: GarlicGPT2 = garlic_model
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.freeze_base_model: bool = freeze_base_model

        # Expose config and base model at the PolicyModel expected surface.
        self.model = garlic_model.base_model
        self.config = garlic_model.config

        # Freeze / unfreeze base model weights.
        base_params = 0
        for param in garlic_model.base_model.parameters():
            param.requires_grad = not freeze_base_model
            base_params += param.numel()

        # Garlic orchestrator is always trainable.
        garlic_params = sum(
            p.numel()
            for p in garlic_model.orchestrator.parameters()
            if hasattr(garlic_model, "orchestrator")
        )
        for param in garlic_model.orchestrator.parameters():
            param.requires_grad = True

        trainable_total = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Ada entropy bridge (per-instance, not shared across workers).
        self._ada_entropy: Optional[Any] = (
            GarlicAdaStepEntropy() if ADA_STEP_ENTROPY_AVAILABLE else None
        )

        # Last forward routing metadata — read by GarlicRewardModel.
        self._last_routing: Dict[str, Any] = {}

        logger.info("GarlicPolicyModel initialized")
        status = "FROZEN" if freeze_base_model else "TRAINABLE"
        logger.info("  base_model params : %d (%s)", base_params, status)
        logger.info("  orchestrator params: %d (TRAINABLE)", garlic_params)
        logger.info("  total trainable   : %d", trainable_total)
        logger.info(
            "  Ada step entropy  : %s", "LIVE" if self._ada_entropy else "FALLBACK"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_garlic: bool = True,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Forward pass through GarlicGPT2 with optional Ada entropy routing.

        Args:
            input_ids: Token IDs [batch, seq_len].
            attention_mask: Padding mask [batch, seq_len].
            labels: Target IDs for loss computation [batch, seq_len].
            use_garlic: Route through Garlic orchestration when True.
            output_hidden_states: If True, also return the final hidden
                states from the base model (costs an extra forward call).

        Returns:
            Dict with keys:
              loss          – CrossEntropy loss if labels provided else None
              logits        – [batch, seq_len, vocab]
              hidden_states – [batch, seq_len, hidden] or None
              step_entropy  – float (Ada avg entropy in bits) or None
              routing_path  – str ('fast'|'normal'|'slow') or garlic path
              compression_ratio – float or None
        """
        if attention_mask is not None and attention_mask.dtype == torch.int64:
            attention_mask = attention_mask.float()

        garlic_out = self.garlic_model.forward(
            input_ids,
            attention_mask=attention_mask,
            use_garlic=use_garlic,
            return_trace=False,
        )

        logits: torch.Tensor = garlic_out["logits"]
        routing_path: str = garlic_out.get("routing_path", "unknown")
        compression_ratio: Optional[float] = garlic_out.get("compression_ratio")

        # --- Ada step entropy on last-token logits ---
        step_entropy: Optional[float] = None
        if self._ada_entropy is not None and logits is not None:
            try:
                import numpy as _np

                last_logits_np = (
                    logits[0, -1, :].detach().cpu().numpy().astype(_np.float32)
                )
                step_data = self._ada_entropy.calculate_step_entropy(
                    token_logits=[last_logits_np],
                    token_ids=[0],
                    token_texts=[""],
                )
                step_entropy = step_data.avg_entropy
                routing_path = step_data.level.name.lower()

                # Update Welford thresholds if a manager is available.
                atm: Optional[Any] = getattr(self, "_threshold_manager", None)
                if atm is not None:
                    atm.update(step_entropy)
            except Exception as _exc:
                logger.debug("Ada entropy failed on forward: %s", _exc)

        # Persist routing metadata for GarlicRewardModel (no extra forward needed).
        self._last_routing = {
            "routing_path": routing_path,
            "compression_ratio": compression_ratio
            if compression_ratio is not None
            else 0.0,
            "step_entropy": step_entropy,
            "seq_len": input_ids.shape[1],
        }

        # --- Optional hidden states (extra base-model forward, output_hs=True) ---
        hidden_states: Optional[torch.Tensor] = None
        if output_hidden_states:
            with torch.no_grad():
                base_out = self.garlic_model.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            hidden_states = base_out.hidden_states[-1]

        # --- Loss ---
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
            "step_entropy": step_entropy,
            "routing_path": routing_path,
            "compression_ratio": compression_ratio,
        }

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        use_garlic: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate with Garlic orchestration, returning a token ID tensor.

        GarlicGPT2.generate() takes a text string, not token IDs, so we
        decode first then re-encode. This maintains the PolicyModel interface.

        Args:
            input_ids: Prompt token IDs [1, seq_len] (batch size 1 required).
            max_length: Maximum total generation length.
            temperature: Sampling temperature.
            top_k: Top-K sampling parameter.
            top_p: Nucleus sampling threshold.
            use_garlic: Route through Garlic orchestration when True.

        Returns:
            Generated token ID tensor [1, total_len].
        """
        prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        results: List[Dict[str, Any]] = self.garlic_model.generate(
            input_text=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            use_garlic=use_garlic,
        )

        generated_text: str = results[0].get("text", prompt) if results else prompt
        generated_ids = self.tokenizer.encode(generated_text, return_tensors="pt")
        return generated_ids.to(input_ids.device)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, save_path: str) -> None:
        """Save base model weights, tokenizer, and Garlic orchestrator state.

        Args:
            save_path: Directory path. Created if it does not exist.
        """
        import os

        os.makedirs(save_path, exist_ok=True)

        self.garlic_model.base_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        garlic_state: Dict[str, Any] = {
            "orchestrator": self.garlic_model.orchestrator.state_dict(),
            "injection_layer": self.garlic_model.injection_layer,
        }
        torch.save(garlic_state, os.path.join(save_path, "garlic_components.pt"))
        logger.info("GarlicPolicyModel saved to %s", save_path)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_garlic_statistics(self) -> Dict[str, Any]:
        """Return Garlic orchestration statistics for the current session.

        Returns:
            Dict with routing and memory stats from the Garlic orchestrator.
        """
        stats: Dict[str, Any] = dict(self._last_routing)

        orchestrator = self.garlic_model.orchestrator
        if hasattr(orchestrator, "get_stats"):
            stats.update(orchestrator.get_stats())
        elif hasattr(orchestrator, "get_statistics"):
            stats.update(orchestrator.get_statistics())

        # Adaptive thresholds if Welford manager is present.
        atm: Optional[Any] = getattr(self, "_threshold_manager", None)
        if atm is not None:
            try:
                lo, hi = atm.get_thresholds()
                stats["adaptive_threshold_low"] = lo
                stats["adaptive_threshold_high"] = hi
            except Exception:
                pass

        return stats


# ============================================================================
# GARLIC REWARD MODEL
# ============================================================================


class GarlicRewardModel(RewardModel):
    """Lean reward model for Garlic's structural training stage.

    Source: garlic_gpt2_adapter.GarlicGPT2, ada_bridge.AdaStepEntropy
    Integrated: 2026-05-05
    Purpose: Combines base sequence quality reward with Ada GRPO compression
             signal (arXiv:2508.03346 Eq.13-15). Kept deliberately lean —
             quality alignment is Cherry's PRM responsibility, not Garlic's.

    Reward decomposition:
      R = α_base * R_base + α_compression * R_compression
      R_base        — base RewardModel scalar (sequence quality proxy)
      R_compression — Ada GRPO reward: R_skip_ratio + R_skip_num + R_response_len
                      range [-2.0, 1.0], 0.0 when Ada unavailable
    """

    def __init__(
        self,
        base_model: nn.Module,
        garlic_policy: GarlicPolicyModel,
        alpha_base: float = 0.85,
        alpha_compression: float = 0.15,
    ) -> None:
        """Initialize the Garlic reward model.

        Args:
            base_model: Pre-trained model for the base reward head.
            garlic_policy: GarlicPolicyModel whose _last_routing carries
                routing metadata from the most recent forward pass.
            alpha_base: Weight for the base quality reward. Default 0.85.
            alpha_compression: Weight for the Ada GRPO compression reward.
                Default 0.15. Must satisfy alpha_base + alpha_compression == 1.
        """
        super().__init__(base_model)

        self.garlic_policy: GarlicPolicyModel = garlic_policy
        self.alpha_base: float = alpha_base
        self.alpha_compression: float = alpha_compression

        # Reuse policy Ada bridge instance if live; otherwise None.
        self._ada: Optional[Any] = (
            AdaStepEntropy if ADA_STEP_ENTROPY_AVAILABLE else None
        )

        logger.info("GarlicRewardModel initialized")
        logger.info("  alpha_base       : %.2f", alpha_base)
        logger.info("  alpha_compression: %.2f", alpha_compression)
        logger.info("  Ada GRPO rewards : %s", "LIVE" if self._ada else "SKIPPED")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute composite Garlic reward.

        Reads routing metadata from garlic_policy._last_routing — no second
        Garlic forward needed.

        Args:
            input_ids: Token IDs [batch, seq_len].
            attention_mask: Padding mask [batch, seq_len].

        Returns:
            Scalar reward tensor [batch].
        """
        base_reward: torch.Tensor = super().forward(input_ids, attention_mask)

        compression_signal = torch.zeros(base_reward.shape, device=input_ids.device)

        if self._ada is not None:
            routing = self.garlic_policy._last_routing
            try:
                grpo: Any = self._ada.compute_grpo_rewards(
                    skip_ratio=float(routing.get("compression_ratio", 0.0)),
                    skip_count=0,
                    response_len=int(routing.get("seq_len", input_ids.shape[1])),
                )
                compression_signal = torch.full(
                    base_reward.shape,
                    float(grpo.r_total),
                    device=input_ids.device,
                )
            except Exception as exc:
                logger.debug("Ada GRPO reward failed: %s", exc)

        return (
            self.alpha_base * base_reward + self.alpha_compression * compression_signal
        )


# ============================================================================
# GARLIC RLHF SYSTEM
# ============================================================================


class GarlicRLHFSystem(RLHFOrchestrator):
    """Production RLHF orchestrator for the Garlic architecture.

    Source: rlhf.RLHFOrchestrator, garlic_gpt2_adapter.GarlicGPT2
    Integrated: 2026-05-05
    Purpose: Thin wrapper that wires GarlicGPT2 + Ada step entropy into
             RLHFOrchestrator without duplicating its training/evaluation
             machinery. All trainers, datasets, and config objects come
             from rlhf.py.

    Supported training methods (all from rlhf.py):
      SFT, DPO, GRPO, TreeGRPO, SimPO, KTO, PPO

    Stage architecture:
      Garlic (this system)  → structural routing/compression learning
      Cherry (future stage) → quality alignment via PRM + TreeGRPO
    """

    def __init__(
        self,
        base_model: str = "gpt2-large",
        output_dir: str = "./garlic_rlhf_output",
        use_garlic_rewards: bool = True,
        use_self_improvement: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Garlic RLHF system.

        Args:
            base_model: HuggingFace model identifier or local path.
            output_dir: Root output directory for checkpoints and logs.
            use_garlic_rewards: If True, wrap the reward model with Ada GRPO
                compression signal after reward model training.
            use_self_improvement: Pass-through to RLHFOrchestrator.
            **kwargs: Additional kwargs forwarded to RLHFOrchestrator.
        """
        # Build GarlicGPT2 before parent init so self.garlic_gpt2 is ready
        # when we override self.policy_model below.
        logger.info("Building GarlicGPT2: %s", base_model)
        self.garlic_gpt2: GarlicGPT2 = GarlicGPT2(base_model)

        super().__init__(
            base_model=base_model,
            output_dir=output_dir,
            use_self_improvement=use_self_improvement,
            **kwargs,
        )

        # Replace parent's policy model with our Garlic wrapper.
        self.garlic_policy: GarlicPolicyModel = GarlicPolicyModel(
            self.garlic_gpt2, self.tokenizer
        )
        self.policy_model = self.garlic_policy

        self.use_garlic_rewards: bool = use_garlic_rewards
        self._garlic_base_model: str = base_model

        # --- Telemetry ---
        self._telemetry_recorder: TelemetryRecorder = TelemetryRecorder()
        _sync_default_telemetry_recorder(self._telemetry_recorder)

        # --- Welford adaptive threshold manager ---
        self._threshold_manager: Optional[Any] = (
            AdaptiveThresholdManager() if ADA_STEP_ENTROPY_AVAILABLE else None
        )
        # Share manager with the policy model's forward so thresholds adapt
        # during training based on live entropy observations.
        if self._threshold_manager is not None:
            self.garlic_policy._threshold_manager = self._threshold_manager

        logger.info("GarlicRLHFSystem initialized")
        logger.info("  base_model     : %s", base_model)
        logger.info("  garlic_rewards : %s", use_garlic_rewards)
        logger.info(
            "  Ada thresholds : %s", "WELFORD" if self._threshold_manager else "STATIC"
        )

    # ------------------------------------------------------------------
    # SFT override — preserve GarlicPolicyModel
    # ------------------------------------------------------------------

    def run_sft(
        self,
        sft_data: List[Dict[str, str]],
        config: Optional[SFTConfig] = None,
        batch_size: int = 8,
        num_epochs: int = 3,
    ) -> GarlicPolicyModel:
        """Run supervised fine-tuning without replacing GarlicPolicyModel.

        Parent RLHFOrchestrator.run_sft() creates a fresh PolicyModel from
        scratch. This override skips that step and drives SFTTrainer directly
        on the existing GarlicPolicyModel so Garlic orchestrator state is
        preserved across SFT.

        Args:
            sft_data: List of {'prompt': str, 'response': str} dicts.
            config: SFTConfig. Built from defaults if None.
            batch_size: Batch size when config is auto-built.
            num_epochs: Epoch count when config is auto-built.

        Returns:
            The trained GarlicPolicyModel (same object, updated weights).
        """
        from rlhf import SFTDataset, SFTTrainer, logger as rlhf_logger
        from torch.utils.data import DataLoader

        rlhf_logger.info("=" * 72)
        rlhf_logger.info("GARLIC SFT — preserving GarlicPolicyModel")
        rlhf_logger.info("=" * 72)

        if config is None:
            config = SFTConfig(
                output_dir=str(self.output_dir / "sft"),
                batch_size=batch_size,
                num_epochs=num_epochs,
            )

        trainable = sum(
            p.numel() for p in self.policy_model.parameters() if p.requires_grad
        )
        rlhf_logger.info("Trainable params: %d", trainable)

        trainer = SFTTrainer(
            self.policy_model,
            self.tokenizer,
            config,
            self.device_manager,
        )

        max_length = getattr(config, "max_seq_length", 512)
        dataset = SFTDataset(sft_data, self.tokenizer, max_length=max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        rlhf_logger.info("SFT: %d examples, %d epochs", len(sft_data), num_epochs)
        results = trainer.train(dataloader)
        self.training_history["sft"] = results

        rlhf_logger.info("SFT complete")
        return self.policy_model  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Long-context loading (HSGM)
    # ------------------------------------------------------------------

    def load_long_context(self, document: str) -> Dict[str, Any]:
        """Load a long document into the Garlic orchestrator's HSGM memory.

        Delegates to garlic_gpt2.orchestrator.load_long_context() if the
        orchestrator supports it, otherwise returns a stub summary.

        Args:
            document: Raw text to compress and store.

        Returns:
            Dict with memory stats (segments, compression_ratio, entities).
        """
        logger.info("Loading long context (%d chars) into HSGM", len(document))
        orchestrator = self.garlic_gpt2.orchestrator

        if hasattr(orchestrator, "load_long_context"):
            stats: Dict[str, Any] = orchestrator.load_long_context(document)
        else:
            stats = {
                "segments": 0,
                "entities": 0,
                "compression_ratio": 1.0,
                "note": "Orchestrator does not expose load_long_context",
            }

        logger.info(
            "Context loaded: segments=%s compression=%.2f%%",
            stats.get("segments", "?"),
            float(stats.get("compression_ratio", 1.0)) * 100,
        )
        return stats

    # ------------------------------------------------------------------
    # Reward model training override
    # ------------------------------------------------------------------

    def run_reward_model_training(
        self,
        preference_data: List[Dict[str, str]],
        config: Optional[RewardModelConfig] = None,
    ) -> Dict[str, Any]:
        """Train reward model, then optionally wrap with Ada compression signal.

        Args:
            preference_data: List of {'prompt', 'chosen', 'rejected'} dicts.
            config: RewardModelConfig. Uses parent defaults if None.

        Returns:
            Training result dict from the parent implementation.
        """
        result = super().run_reward_model_training(preference_data, config)

        if self.use_garlic_rewards and self.reward_models:
            logger.info("Wrapping reward model with Garlic Ada compression signal")
            base_rm_model = self.reward_models[0].model
            garlic_rm = GarlicRewardModel(base_rm_model, self.garlic_policy)
            self.reward_models[0] = garlic_rm
            logger.info("GarlicRewardModel ready")

        return result

    # ------------------------------------------------------------------
    # PolicyAdapter for BenchmarkHarness / search-time evaluation
    # ------------------------------------------------------------------

    def get_policy_adapter(self) -> PolicyAdapter:
        """Wrap GarlicPolicyModel as a PolicyAdapter for BenchmarkHarness.

        Returns:
            PolicyAdapter compatible with BestOfNSampler, MCTSGenerator,
            AStarGenerator, and the run_pipeline.py benchmark surface.
        """
        return PolicyAdapter(
            model=self.garlic_policy,
            tokenizer=self.tokenizer,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_garlic_statistics(self) -> Dict[str, Any]:
        """Return comprehensive Garlic + Ada routing statistics.

        Returns:
            Dict with orchestrator stats, last routing metadata, and
            Welford adaptive threshold values if available.
        """
        return self.garlic_policy.get_garlic_statistics()


# ============================================================================
# CONVENIENCE FACTORY
# ============================================================================


def create_garlic_rlhf_system(
    base_model: str = "gpt2-large",
    output_dir: str = "./garlic_rlhf_output",
    use_garlic_rewards: bool = True,
    **kwargs: Any,
) -> GarlicRLHFSystem:
    """Create and return a ready-to-use GarlicRLHFSystem.

    Args:
        base_model: HuggingFace model identifier or local checkpoint path.
        output_dir: Root directory for training outputs.
        use_garlic_rewards: Enable Ada GRPO compression reward signal.
        **kwargs: Additional kwargs forwarded to GarlicRLHFSystem.

    Returns:
        Initialized GarlicRLHFSystem with telemetry and Welford thresholds
        active (or graceful fallback if Ada bridge unavailable).
    """
    return GarlicRLHFSystem(
        base_model=base_model,
        output_dir=output_dir,
        use_garlic_rewards=use_garlic_rewards,
        **kwargs,
    )
