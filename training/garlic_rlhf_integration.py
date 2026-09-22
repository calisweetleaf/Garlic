"""
Garlic Architecture + Production RLHF Integration

Source: src/system_router.py (SystemRouter — the real Garlic architecture)
        ada_bridge.py (Ada step entropy bridge, Garlic project root)
        rlhf-pipeline-cherry-revision/ (external production RLHF framework)
Integrated: 2026-05-05
Rewired:    2026-09-22 — GarlicPolicyModel now wraps a real
            AutoModelForCausalLM + SystemRouterWrapper (src/system_router.py)
            instead of the removed GarlicGPT2 dependency, which does not
            exist in this repo's architecture.
Purpose: Wires the Garlic SystemRouter into the Full-RLHF-Pipeline
         production surface (rlhf.py, inference_optimizations.py,
         inference_protocols.py, telemetry.py, benchmark_harness.py) with
         Ada 2022 step entropy as the routing/compression backbone.

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
  ADA_STEP_ENTROPY_ROOT  path to ada_bridge.py's directory (Garlic project root)
  GARLIC_SRC_ROOT        path to system_router.py's directory (Garlic/src)
  RLHF_PIPELINE_ROOT     path to the rlhf-pipeline-cherry-revision checkout
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util as _importlib_util
import logging
import os as _os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Portable path resolution — override via env vars for non-standard layouts
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent  # .../Garlic/training

# ada_bridge.py lives at the Garlic project root itself (one level above
# this file), not in a separate ADA-Step-Entropy checkout — that path does
# not exist on this machine; the prior default was stale.
_ADA_STEP_ENTROPY_ROOT = _os.environ.get(
    "ADA_STEP_ENTROPY_ROOT",
    str(_REPO_ROOT.parent),
)

# system_router.py — the real Garlic architecture — lives in Garlic/src.
_GARLIC_SRC_ROOT = _os.environ.get(
    "GARLIC_SRC_ROOT",
    str(_REPO_ROOT.parent / "src"),
)

# External production RLHF framework (rlhf.py + siblings) — a separate
# project, not vendored into this repo.
_RLHF_PIPELINE_ROOT = _os.environ.get(
    "RLHF_PIPELINE_ROOT",
    "/home/daeron/LAB/Experiments/projects/full-rlhf-pipeline/rlhf-pipeline-cherry-revision",
)

# ---------------------------------------------------------------------------
# External RLHF pipeline (rlhf.py + 5 siblings) — a separate project, not
# vendored into this repo. Loaded via a *scoped* sys.path insertion (not a
# permanent one) because rlhf.py and benchmark_harness.py use bare absolute
# sibling imports (`from telemetry import ...`, `from model_merging import
# ...`) that only resolve when the pipeline directory itself is importable
# as a path root — per-file importlib.util.spec_from_file_location loading
# (used below for system_router.py / ada_bridge.py, which are self-
# contained) does not make those internal sibling imports resolve. Fails
# loud — no fallback — if the directory or any module is missing.
# ---------------------------------------------------------------------------

_PIPELINE_MODULE_NAMES = (
    "telemetry",                # leaf dependency of rlhf.py, benchmark_harness.py
    "model_merging",            # leaf dependency of benchmark_harness.py
    "inference_optimizations",  # leaf dependency of benchmark_harness.py
    "inference_protocols",      # standalone (torch/stdlib only)
    "rlhf",                     # depends on telemetry
    "benchmark_harness",        # depends on model_merging, inference_optimizations, telemetry
)


@contextlib.contextmanager
def _temporary_sys_path(path: str):
    """Insert `path` at sys.path[0] for the duration of the block only."""
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def _load_rlhf_pipeline_modules() -> Dict[str, Any]:
    """Import rlhf.py and its sibling modules from RLHF_PIPELINE_ROOT.

    Returns:
        Dict mapping module name -> loaded module object, for every name in
        _PIPELINE_MODULE_NAMES.

    Raises:
        RuntimeError: if RLHF_PIPELINE_ROOT does not exist, or if any
            required module fails to import.
    """
    root = Path(_RLHF_PIPELINE_ROOT)
    if not root.is_dir():
        raise RuntimeError(
            f"RLHF_PIPELINE_ROOT does not exist: {root}. Set the "
            "RLHF_PIPELINE_ROOT env var to the rlhf-pipeline-cherry-revision "
            "checkout."
        )
    if not (root / "rlhf.py").exists():
        raise RuntimeError(f"rlhf.py not found under RLHF_PIPELINE_ROOT: {root}")

    loaded: Dict[str, Any] = {}
    with _temporary_sys_path(str(root)):
        for name in _PIPELINE_MODULE_NAMES:
            try:
                loaded[name] = importlib.import_module(name)
            except ImportError as exc:
                raise RuntimeError(
                    f"Failed to import pipeline module '{name}' from {root}: {exc}"
                ) from exc
    return loaded


_pipeline = _load_rlhf_pipeline_modules()
rlhf = _pipeline["rlhf"]
inference_protocols = _pipeline["inference_protocols"]
inference_optimizations = _pipeline["inference_optimizations"]
telemetry = _pipeline["telemetry"]
benchmark_harness = _pipeline["benchmark_harness"]

# Pulled off the module objects (not a second `from rlhf import (...)`
# statement) so there is exactly one import path, governed by the scoped
# sys.path block above.
RLHFOrchestrator = rlhf.RLHFOrchestrator
DeviceManager = rlhf.DeviceManager
CheckpointManager = rlhf.CheckpointManager
TrainingLogger = rlhf.TrainingLogger
EarlyStopping = rlhf.EarlyStopping
BaseConfig = rlhf.BaseConfig
SFTConfig = rlhf.SFTConfig
RewardModelConfig = rlhf.RewardModelConfig
DPOConfig = rlhf.DPOConfig
GRPOConfig = rlhf.GRPOConfig
TreeGRPOConfig = rlhf.TreeGRPOConfig
SimPOConfig = rlhf.SimPOConfig
KTOConfig = rlhf.KTOConfig
PPOConfig = rlhf.PPOConfig
PolicyModel = rlhf.PolicyModel
RewardModel = rlhf.RewardModel
ProcessRewardModel = rlhf.ProcessRewardModel
ValueModel = rlhf.ValueModel
ContextCompressor = rlhf.ContextCompressor
PreferenceDataset = rlhf.PreferenceDataset
SFTDataset = rlhf.SFTDataset
KTODataset = rlhf.KTODataset
GRPODataset = rlhf.GRPODataset
StreamingPreferenceDataset = rlhf.StreamingPreferenceDataset
StreamingSFTDataset = rlhf.StreamingSFTDataset
StreamingKTODataset = rlhf.StreamingKTODataset
StreamingGRPODataset = rlhf.StreamingGRPODataset
SFTTrainer = rlhf.SFTTrainer
DPOTrainer = rlhf.DPOTrainer
GRPOTrainer = rlhf.GRPOTrainer
TreeGRPOTrainer = rlhf.TreeGRPOTrainer
SimPOTrainer = rlhf.SimPOTrainer
KTOTrainer = rlhf.KTOTrainer
PPOTrainer = rlhf.PPOTrainer
AdversarialValidator = rlhf.AdversarialValidator
CapabilityTester = rlhf.CapabilityTester
IterativeRefiner = rlhf.IterativeRefiner
RLHFEvaluator = rlhf.RLHFEvaluator
RewardFunctionFactory = rlhf.RewardFunctionFactory
ConstitutionalRewardWrapper = rlhf.ConstitutionalRewardWrapper

PolicyAdapter = inference_protocols.PolicyAdapter
ProcessRewardModelAdapter = inference_protocols.ProcessRewardModelAdapter
BestOfNConfig = inference_optimizations.BestOfNConfig
BestOfNSampler = inference_optimizations.BestOfNSampler
MCTSConfig = inference_optimizations.MCTSConfig
MCTSGenerator = inference_optimizations.MCTSGenerator
ChainOfThoughtConfig = inference_optimizations.ChainOfThoughtConfig
ChainOfThoughtGenerator = inference_optimizations.ChainOfThoughtGenerator
AStarConfig = inference_optimizations.AStarConfig
AStarGenerator = inference_optimizations.AStarGenerator
TreeRolloutCollector = inference_optimizations.TreeRolloutCollector
RolloutSample = inference_optimizations.RolloutSample
compile_model = inference_optimizations.compile_model
TelemetryRecorder = telemetry.TelemetryRecorder
BenchmarkHarness = benchmark_harness.BenchmarkHarness

# ---------------------------------------------------------------------------
# Garlic architecture — SystemRouter (src/system_router.py). Loaded by
# absolute path; the module resolves its own internal imports (neural_router,
# garlic-components/) relative to its own __file__, so no sys.path insertion
# is needed here. Unconditional / fails loud — SystemRouter is the real
# Garlic architecture, not an optional component.
# ---------------------------------------------------------------------------


def _load_system_router_module() -> Any:
    candidate = Path(_GARLIC_SRC_ROOT) / "system_router.py"
    if not candidate.exists():
        raise RuntimeError(f"system_router.py not found at {candidate}")
    spec = _importlib_util.spec_from_file_location(
        "garlic_system_router", str(candidate)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build import spec for {candidate}")
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_system_router_mod = _load_system_router_module()
SystemRouter = _system_router_mod.SystemRouter
SystemRouterConfig = _system_router_mod.SystemRouterConfig
SystemRouterWrapper = _system_router_mod.SystemRouterWrapper
ModelSlot = _system_router_mod.ModelSlot
SystemOutput = _system_router_mod.SystemOutput
SYSTEM_ROUTER_ADA_AVAILABLE: bool = _system_router_mod.ADA_AVAILABLE

# ---------------------------------------------------------------------------
# Ada 2022 step entropy bridge (arXiv:2508.03346) — repo-root ada_bridge.py.
# Loaded via importlib by absolute path to avoid any sys.path collision.
# Used directly by GarlicRewardModel for GRPO reward computation; SystemRouter
# above loads its own separate instance internally for routing decisions.
# ---------------------------------------------------------------------------


def _load_ada_bridge_module() -> Optional[Any]:
    """Load ada_bridge.py by absolute path.

    Returns:
        Loaded module object, or None if not found or failed to load.
    """
    candidate = Path(_ADA_STEP_ENTROPY_ROOT) / "ada_bridge.py"
    if not candidate.exists():
        logger.warning("ada_bridge.py not found at %s", candidate)
        return None
    spec = _importlib_util.spec_from_file_location("ada_bridge", str(candidate))
    if spec is None or spec.loader is None:
        return None
    mod = _importlib_util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    except Exception as exc:
        logger.warning("Failed to load ada_bridge.py: %s", exc)
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

if ADA_STEP_ENTROPY_AVAILABLE:
    logger.info("Ada step entropy bridge: LIVE (Max_Vocab_Size=262144)")
else:
    logger.warning("Ada step entropy bridge: UNAVAILABLE — GRPO compression reward disabled")

if SYSTEM_ROUTER_ADA_AVAILABLE:
    logger.info("SystemRouter Ada step entropy: LIVE")
else:
    logger.warning("SystemRouter Ada step entropy: Python fallback active")


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


def _resolve_hidden_size(cfg: Any) -> int:
    """Resolve a HF config's hidden size across architecture families.

    RewardModel's own _resolve_hidden_size (rlhf.py) only checks
    hidden_size/text_config/language_config, which misses GPT-2-family
    configs that expose n_embd instead. This checks the wider set actually
    seen across causal-LM configs.
    """
    for attr in ("hidden_size", "n_embd", "d_model", "hidden_dim"):
        val = getattr(cfg, attr, None)
        if val is not None:
            return int(val)
    for sub in ("text_config", "language_config"):
        subcfg = getattr(cfg, sub, None)
        if subcfg is not None:
            for attr in ("hidden_size", "n_embd", "d_model"):
                val = getattr(subcfg, attr, None)
                if val is not None:
                    return int(val)
    raise ValueError(f"Cannot resolve hidden size from config {cfg!r}")


class GarlicPolicyModel(PolicyModel):
    """PolicyModel wrapper over a real causal LM + SystemRouter.

    Source: rlhf.PolicyModel (base), src/system_router.py (SystemRouter)
    Integrated: 2026-05-05
    Rewired:    2026-09-22 — wraps a genuine AutoModelForCausalLM (via the
                real PolicyModel.__init__) plus SystemRouterWrapper, in
                place of the removed GarlicGPT2 dependency (that class does
                not exist in this repo's architecture).
    Purpose: Presents Garlic's SystemRouter as a PolicyModel so all rlhf.py
             trainers (SFT, DPO, GRPO, TreeGRPO, PPO, …) can drive it
             without knowing about SystemRouter internals.

    Design notes:
      - SystemRouter does not itself run a language model — it CONSUMES
        hidden_states/logits (computed here by the real base LM) and
        produces a routing decision, HSGM compression, and (optionally)
        memory-augmented hidden states. This wrapper is the seam between
        the two: real LM forward -> SystemRouterWrapper.route() -> logit
        correction from augmented_states (if any) -> loss.
      - SystemRouter owns its own step-entropy engine internally
        (build_step_entropy(), Ada or Python fallback); this wrapper does
        not duplicate a second entropy engine.
    """

    def __init__(
        self,
        base_model_name: str,
        tokenizer: PreTrainedTokenizer,
        router_config: Optional[Any] = None,
        freeze_base_model: bool = False,
        use_gradient_checkpointing: bool = False,
        **policy_model_kwargs: Any,
    ) -> None:
        """Initialize the Garlic policy model wrapper.

        Args:
            base_model_name: HuggingFace model identifier or local path.
            tokenizer: Tokenizer matching the base model.
            router_config: SystemRouterConfig to use. Built from the base
                model's real hidden size if None.
            freeze_base_model: If True, freeze base model weights so only
                the SystemRouter trains. Default False (both train).
            use_gradient_checkpointing: Forwarded to PolicyModel.__init__.
            **policy_model_kwargs: Forwarded to PolicyModel.__init__
                (load_in_4bit, attn_implementation, etc.).
        """
        super().__init__(
            base_model_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            **policy_model_kwargs,
        )

        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.freeze_base_model: bool = freeze_base_model

        base_params = 0
        for param in self.model.parameters():
            param.requires_grad = not freeze_base_model
            base_params += param.numel()

        hidden_size = _resolve_hidden_size(self.model.config)
        self._router_config = router_config or SystemRouterConfig(
            context_dim=hidden_size
        )
        if self._router_config.context_dim != hidden_size:
            raise ValueError(
                f"router_config.context_dim ({self._router_config.context_dim}) "
                f"must equal the base model's hidden size ({hidden_size})"
            )

        self.system_router: Any = SystemRouterWrapper(self._router_config)

        router_params = 0
        if self.system_router.router is not None:
            # SystemRouterWrapper is a plain Python class, not an nn.Module,
            # so assigning it above does NOT auto-register its wrapped
            # SystemRouter (which IS an nn.Module) with PyTorch's parameter
            # tracking — self.parameters() would silently skip all of its
            # weights (context_encoder, slot_predictor, entropy_router,
            # etc.), breaking optimizer construction, .to(device), and
            # state_dict() for anything walking self.parameters()/.modules().
            # add_module() registers the *same* object under self._modules
            # so it's tracked, while self.system_router.route(...) still
            # operates on that identical instance.
            self.add_module("_system_router_module", self.system_router.router)
            router_params = sum(
                p.numel() for p in self.system_router.router.parameters()
            )
            for param in self.system_router.router.parameters():
                param.requires_grad = True

        # Last forward routing metadata — read by GarlicRewardModel.
        self._last_routing: Dict[str, Any] = {}
        # Set externally by GarlicRLHFSystem when a Welford threshold
        # manager is active.
        self._threshold_manager: Optional[Any] = None

        trainable_total = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info("GarlicPolicyModel initialized")
        status = "FROZEN" if freeze_base_model else "TRAINABLE"
        logger.info("  base_model params  : %d (%s)", base_params, status)
        logger.info("  system_router params: %d (TRAINABLE)", router_params)
        logger.info("  total trainable    : %d", trainable_total)
        logger.info(
            "  SystemRouter       : %s",
            "LIVE" if self.system_router.router is not None else "JINJA2_FALLBACK",
        )
        logger.info(
            "  Ada step entropy   : %s",
            "LIVE" if SYSTEM_ROUTER_ADA_AVAILABLE else "PYTHON_FALLBACK",
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        user_profile: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        context_metadata: Optional[Dict[str, Any]] = None,
        use_garlic: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Forward pass: real LM -> SystemRouter routing -> loss.

        Args:
            input_ids: Token IDs [batch, seq_len].
            attention_mask: Padding mask [batch, seq_len].
            labels: Target IDs for loss computation [batch, seq_len].
            user_profile: [batch, 128] profile features, or None for a
                documented zero-signal default (no per-example profile
                data flows through the standard RLHF trainers today).
            metadata: [batch, 64] metadata features, or None for the same
                zero-signal default.
            context_metadata: Passed through to SystemRouter.route().
            use_garlic: If False, skip SystemRouter entirely and return
                the raw base-LM forward pass (a real, meaningful toggle —
                e.g. for baseline comparisons against the routed path).

        Returns:
            Dict with keys: loss, logits, hidden_states, step_entropy,
            routing_path, compression_ratio, system_output (the full
            SystemOutput, or None when use_garlic=False).
        """
        if attention_mask is not None and attention_mask.dtype == torch.int64:
            attention_mask = attention_mask.float()

        base_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        base_logits: torch.Tensor = base_out.logits
        base_hidden: torch.Tensor = base_out.hidden_states[-1]

        if not use_garlic:
            loss = self._compute_loss(base_logits, labels)
            self._last_routing = {}
            return {
                "loss": loss,
                "logits": base_logits,
                "hidden_states": base_hidden,
                "step_entropy": None,
                "routing_path": None,
                "compression_ratio": None,
                "system_output": None,
            }

        batch_size = input_ids.shape[0]
        device = input_ids.device
        if user_profile is None:
            user_profile = torch.zeros(batch_size, 128, device=device)
        if metadata is None:
            metadata = torch.zeros(batch_size, 64, device=device)
        context_metadata = context_metadata or {}

        system_output: Any = self.system_router.route(
            message_embs=base_hidden,
            user_profile=user_profile,
            metadata=metadata,
            context_metadata=context_metadata,
            hidden_states=base_hidden,
            logits=base_logits,
            return_trace=False,
        )

        final_logits = base_logits
        if system_output.augmented_states is not None:
            aug = system_output.augmented_states
            prefix_len = aug.shape[1] - base_hidden.shape[1]
            if prefix_len > 0:
                aug = aug[:, prefix_len:, :]
            final_logits = self.model.lm_head(aug)

        loss = self._compute_loss(final_logits, labels)

        original_tokens = system_output.original_tokens or input_ids.shape[1]
        compressed_tokens = system_output.compressed_tokens or original_tokens
        # Bounded [0,1) skip fraction — NOT the raw HSGM N:1 compression_ratio,
        # which would miscalibrate compute_grpo_rewards' skip_ratio in [0,1] contract.
        skip_ratio = 1.0 - (compressed_tokens / max(original_tokens, 1))

        self._last_routing = {
            "routing_path": system_output.entropy_level or "unknown",
            "model_slot": system_output.model_slot.name,
            "compression_ratio": skip_ratio,
            "hsgm_compression_ratio": system_output.compression_ratio,
            "step_entropy": system_output.entropy_value,
            "seq_len": original_tokens,
            "augmented": system_output.augmented_states is not None,
        }

        if self._threshold_manager is not None and system_output.entropy_value:
            self._threshold_manager.update(system_output.entropy_value)

        return {
            "loss": loss,
            "logits": final_logits,
            "hidden_states": base_hidden,
            "step_entropy": system_output.entropy_value,
            "routing_path": system_output.entropy_level,
            "compression_ratio": skip_ratio,
            "system_output": system_output,
        }

    @staticmethod
    def _compute_loss(
        logits: torch.Tensor, labels: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        use_garlic: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate token IDs, with a real routing pass for telemetry.

        Runs one no-grad forward() pass first (when use_garlic=True) purely
        to populate _last_routing / emit a routing decision — real work,
        not a stub. Actual decoding is delegated entirely to the base
        model's own complete HF generate() implementation; per-step Garlic
        injection during autoregressive decoding is out of scope (would
        need a custom LogitsProcessor re-running SystemRouter every N
        tokens — a real design with its own tradeoffs, not decided here).

        Args:
            input_ids: Prompt token IDs [batch, seq_len].
            attention_mask: Padding mask [batch, seq_len].
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K sampling parameter.
            top_p: Nucleus sampling threshold.
            do_sample: Whether to sample (vs. greedy decoding).
            use_garlic: Route through SystemRouter for telemetry when True.

        Returns:
            Generated token ID tensor [batch, seq_len + new_tokens].
        """
        if use_garlic:
            with torch.no_grad():
                self.forward(input_ids, attention_mask=attention_mask, use_garlic=True)

        pad_token_id = self.model.config.eos_token_id
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, save_path: str) -> None:
        """Save base model weights, tokenizer, and full SystemRouter state.

        Args:
            save_path: Directory path. Created if it does not exist.
        """
        _os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        if self.system_router.router is not None:
            # Full state_dict, not just the two named bridge layers —
            # SystemRouter has many more trainable submodules (context_encoder,
            # slot_predictor, entropy_router, difficulty_allocator, ...) that
            # accumulate gradient during GRPO/reward training.
            torch.save(
                self.system_router.router.state_dict(),
                _os.path.join(save_path, "system_router.pt"),
            )
        else:
            logger.warning(
                "SystemRouter unavailable (Jinja2 fallback active) — "
                "system_router.pt not saved; only base LM weights persisted."
            )
        logger.info("GarlicPolicyModel saved to %s", save_path)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_garlic_statistics(self) -> Dict[str, Any]:
        """Return Garlic routing statistics for the current session.

        Returns:
            Dict with routing metadata from the last forward() call, plus
            global-memory turn count and adaptive thresholds when available.
        """
        stats: Dict[str, Any] = dict(self._last_routing)

        router = self.system_router.router
        if router is not None:
            stats["global_memory_turns"] = router.global_memory.turn_count
            stats["global_memory_empty"] = router.global_memory.is_empty

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


class GarlicRewardModel(nn.Module):
    """Lean reward model for Garlic's structural training stage.

    Source: rlhf.RewardModel (wrapped, not subclassed), ada_bridge.AdaStepEntropy
    Integrated: 2026-05-05
    Rewired:    2026-09-22 — changed from subclassing RewardModel to
                *wrapping* an already-built RewardModel instance. The real
                RewardModel.__init__ takes a base_model_name: str (it builds
                its own AutoModel backbone internally) — the original call
                site passed an already-built nn.Module into that slot, which
                raises. Composition over an existing RewardModel sidesteps
                the constructor-arg mismatch entirely while keeping the same
                forward() delegation.
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
        base_reward_model: Any,
        garlic_policy: GarlicPolicyModel,
        alpha_base: float = 0.85,
        alpha_compression: float = 0.15,
    ) -> None:
        """Initialize the Garlic reward model.

        Args:
            base_reward_model: An already-built RewardModel instance (e.g.
                self.reward_models[0] from RLHFOrchestrator.run_reward_model_training).
            garlic_policy: GarlicPolicyModel whose _last_routing carries
                routing metadata from the most recent forward pass.
            alpha_base: Weight for the base quality reward. Default 0.85.
            alpha_compression: Weight for the Ada GRPO compression reward.
                Default 0.15. Must satisfy alpha_base + alpha_compression == 1.
        """
        super().__init__()

        self.base_reward_model = base_reward_model
        self.garlic_policy: GarlicPolicyModel = garlic_policy
        self.alpha_base: float = alpha_base
        self.alpha_compression: float = alpha_compression

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
        base_reward: torch.Tensor = self.base_reward_model(input_ids, attention_mask)

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
        super().__init__(
            base_model=base_model,
            output_dir=output_dir,
            use_self_improvement=use_self_improvement,
            **kwargs,
        )

        # Replace parent's policy model with our Garlic wrapper. self.tokenizer
        # is set by RLHFOrchestrator.__init__ just above (AutoTokenizer.from_pretrained).
        self.garlic_policy: GarlicPolicyModel = GarlicPolicyModel(
            base_model, self.tokenizer, freeze_base_model=False,
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
        """Load a long document into SystemRouter's HSGM memory.

        Thin delegation to SystemRouter.load_long_context() (src/system_router.py),
        which owns the real HSGM ingestion pipeline (global_memory,
        local_graph_builder, summary_extractor). Raises loud if the router
        is in its documented Jinja2-fallback state rather than returning
        fabricated stats.

        Args:
            document: Raw text to compress and store.

        Returns:
            Dict with real memory stats (segments, summary_nodes, edges,
            original_tokens, compressed_tokens, compression_ratio).
        """
        logger.info("Loading long context (%d chars) into HSGM", len(document))
        router = self.garlic_policy.system_router.router
        if router is None:
            raise RuntimeError(
                "SystemRouter unavailable (Jinja2 fallback active) — cannot "
                "ingest long context without the neural router."
            )
        stats = router.load_long_context(document)
        logger.info(
            "Context loaded: segments=%s compression=%.2fx",
            stats["segments"], stats["compression_ratio"],
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
            # Pass the built RewardModel instance itself (not .model) — the
            # real RewardModel.__init__ takes a base_model_name: str, not a
            # pre-built module, so GarlicRewardModel wraps the instance via
            # composition rather than subclassing (see GarlicRewardModel).
            garlic_rm = GarlicRewardModel(self.reward_models[0], self.garlic_policy)
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
        return PolicyAdapter.from_rlhf_model(
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
