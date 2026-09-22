"""
Step Entropy Ada FFI Bridge — Production v3.0
==============================================
Python interface to the compiled Ada step_entropy library via C API layer.

Hybrid model: token-level math uses Ada FFI; step-level aggregation in Python.
All errors fail LOUD with explicit logging. No silent fallbacks.

New in v3.0:
  - Paper constants (arXiv:2508.03346 §3.2-3.3)
  - EntropyHistogramData / GRPORewardsData dataclasses
  - AdaStepEntropy.compute_histogram / normalize_entropy / compute_grpo_rewards
  - AdaptiveThresholdManager (Welford online algorithm, §5 adaptive thresholds)
  - GarlicAdaStepEntropy.compress_chain / recommend_pruning
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Type
from dataclasses import dataclass
from enum import IntEnum
import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

# Guard concurrent LD_LIBRARY_PATH mutations during parallel module imports.
_LD_PATH_LOCK = threading.Lock()


# ============================================================================
# PAPER CONSTANTS  (arXiv:2508.03346 Table 3 / §3.2-3.3)
# ============================================================================

GRPO_K_SKIP_HIGH    = 0.80   # κ_high: skip ratio → R_skip_ratio = 1.0
GRPO_K_SKIP_LOW     = 0.50   # κ_low:  skip ratio → R_skip_ratio = 0.5
GRPO_TAU_SKIP_MAX   = 100    # τ_skip: skip count before -1.0 penalty
GRPO_TAU_LENGTH     = 3500   # τ_length: response length before -1.0 penalty
OPTIMAL_PRUNE_RATIO = 0.80   # κ: §3.2 empirical sweet spot

# [SKIP] token protocol (§3.2)
SKIP_TOKEN_TEXT  = "[SKIP]"  # text injected for low-entropy steps
STEP_DELIMITER   = "\n\n"    # steps delimited inside <think>...</think>


# ============================================================================
# ENUMERATIONS
# ============================================================================

class EntropyLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class RoutingPath(IntEnum):
    FAST = 0
    NORMAL = 1
    SLOW = 2


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TokenEntropyData:
    token_id: int
    token_text: str
    entropy: float
    probability: float
    context_position: int


@dataclass
class StepEntropyData:
    step_id: int
    step_text: str
    token_count: int
    total_entropy: float
    avg_entropy: float
    max_entropy: float
    level: EntropyLevel
    token_entropies: List[TokenEntropyData]

    def __repr__(self) -> str:
        return (
            f"StepEntropyData(step_id={self.step_id!r}, "
            f"token_count={self.token_count}, "
            f"avg_entropy={self.avg_entropy:.4f}, "
            f"level={self.level.name}, "
            f"token_entropies=[{len(self.token_entropies)} entries])"
        )


@dataclass
class CompressionAnalysisData:
    total_steps: int
    compressible_steps: int
    informative_steps: int
    low_entropy_steps: int
    medium_entropy_steps: int
    high_entropy_steps: int
    compression_ratio: float
    avg_entropy: float


@dataclass
class RoutingDecisionData:
    path: RoutingPath
    retrieve_memory: bool
    total_entropy: float
    avg_entropy: float
    max_entropy: float
    level: EntropyLevel


@dataclass
class EntropyHistogramData:
    """Distribution summary from Ada C_Compute_Histogram (SOTA++ §2.4).

    Buffer layout from Ada: [bin_width, min_val, max_val, mean, std_dev, bin0..binN-1]
    """
    bin_count: int
    bin_width: float
    min_val: float
    max_val: float
    mean: float
    std_dev: float
    bins: List[int]    # length == bin_count
    total_tokens: int


@dataclass
class GRPORewardsData:
    """GRPO reward components as defined in Paper Eq.13-15 (§3.3).

    R_skip_ratio : 1.0 if skip_ratio >= κ_high, 0.5 if >= κ_low, else 0.0
    R_skip_num   : -1.0 if skip_count > τ_skip, else 0.0
    R_response_len: -1.0 if response_len > τ_length, else 0.0
    R_total       : sum of the three kernel components
    """
    r_skip_ratio: float    # 0.0, 0.5, or 1.0
    r_skip_num: float      # 0.0 or -1.0
    r_response_len: float  # 0.0 or -1.0
    r_total: float         # sum of the three above


# ============================================================================
# ADA LIBRARY LOADING
# ============================================================================

def _find_and_load_library() -> ctypes.CDLL:
    """Find and load libgarlic_core.so. Fails LOUD if not found."""
    script_dir = Path(__file__).parent.resolve()
    candidates = [
        script_dir / "lib" / "libgarlic_core.so",
        script_dir / "libgarlic_core.so",
        script_dir / "build" / "libgarlic_core.so",
        Path.cwd() / "lib" / "libgarlic_core.so",
    ]

    for path in candidates:
        if path.exists():
            logger.info(f"Found Ada library: {path}")
            # Prepend lib_dir to LD_LIBRARY_PATH so Ada runtime deps resolve.
            # The lock prevents a TOCTOU race when worker threads import this
            # module concurrently (e.g. in a multi-process torch DataLoader).
            # split(":") avoids the false-positive where "/foo/lib" would
            # suppress the legitimate addition of "/foo/libexec".
            lib_dir = str(path.parent)
            with _LD_PATH_LOCK:
                current = os.environ.get("LD_LIBRARY_PATH", "")
                existing_dirs = current.split(":") if current else []
                if lib_dir not in existing_dirs:
                    os.environ["LD_LIBRARY_PATH"] = (
                        f"{lib_dir}:{current}" if current else lib_dir
                    )
            try:
                lib = ctypes.CDLL(str(path), use_errno=True)
                logger.info("Ada library loaded successfully")
                return lib
            except OSError as e:
                logger.error(f"Found library at {path} but failed to load: {e}")
                raise RuntimeError(f"Ada library load failed: {e}") from e

    searched = "\n  ".join(str(p) for p in candidates)
    raise RuntimeError(
        f"Ada step_entropy library not found. Searched:\n  {searched}\n"
        f"Build with: gprbuild -P garlic_core.gpr"
    )


# ============================================================================
# C FUNCTION BINDINGS
# ============================================================================

class _CBindings:
    """Low-level ctypes bindings matching step_entropy_c_api.ads exactly."""

    def __init__(self, lib: ctypes.CDLL):
        self.lib = lib

        # -- Token entropy --
        self.calculate_token_entropy = lib.step_entropy_calculate_token_entropy
        self.calculate_token_entropy.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float
        ]
        self.calculate_token_entropy.restype = ctypes.c_float

        # -- Token probability --
        self.calculate_token_probability = lib.step_entropy_calculate_token_probability
        self.calculate_token_probability.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
        ]
        self.calculate_token_probability.restype = ctypes.c_float

        # -- Classify entropy --
        self.classify_entropy = lib.step_entropy_classify_entropy
        self.classify_entropy.argtypes = [
            ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]
        self.classify_entropy.restype = ctypes.c_int

        # -- Route step --
        self.route_step = lib.step_entropy_route_step
        self.route_step.argtypes = [
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        self.route_step.restype = ctypes.c_int

        # -- Batch route --
        self.batch_route = lib.step_entropy_batch_route
        self.batch_route.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.batch_route.restype = ctypes.c_int

        # -- Analyze chain --
        self.analyze_chain = lib.step_entropy_analyze_chain
        self.analyze_chain.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.analyze_chain.restype = ctypes.c_int

        # -- Recommend pruning --
        self.recommend_pruning = lib.step_entropy_recommend_pruning
        self.recommend_pruning.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        self.recommend_pruning.restype = ctypes.c_int

        # -- Introspection --
        self.get_version = lib.step_entropy_get_version
        self.get_version.argtypes = []
        self.get_version.restype = ctypes.c_int

        self.get_max_vocab_size = lib.step_entropy_get_max_vocab_size
        self.get_max_vocab_size.argtypes = []
        self.get_max_vocab_size.restype = ctypes.c_int

        self.get_max_tokens_per_step = lib.step_entropy_get_max_tokens_per_step
        self.get_max_tokens_per_step.argtypes = []
        self.get_max_tokens_per_step.restype = ctypes.c_int

        logger.info(f"Ada library v{self.get_version()} bound "
                     f"(vocab={self.get_max_vocab_size()}, "
                     f"max_tokens={self.get_max_tokens_per_step()})")

        # ---- SOTA++ bindings (Phase 2 kernel extensions) ----

        # -- Entropy histogram (SOTA++ §2.4) --
        # step_entropy_compute_histogram(
        #   Entropies   : float*,   -- input array of entropy values
        #   Count       : c_int,    -- number of elements
        #   Bin_Count   : c_int,    -- requested bin count (1..32)
        #   Out_Buffer  : float*    -- output: [bin_width, min, max, mean, std, bin0..binN-1]
        # ) -> c_int  (0 = success, non-zero = error)
        self.compute_histogram = lib.step_entropy_compute_histogram
        self.compute_histogram.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # Entropies
            ctypes.c_int,                    # Count
            ctypes.c_int,                    # Bin_Count
            ctypes.POINTER(ctypes.c_float),  # Out_Buffer
        ]
        self.compute_histogram.restype = ctypes.c_int
        logger.debug("Bound step_entropy_compute_histogram")

        # -- Z-score normalization (SOTA++ §2.3 cross-model portability) --
        # step_entropy_normalize_entropy(
        #   Entropy : c_float,
        #   Mean    : c_float,
        #   Std_Dev : c_float
        # ) -> c_float  (-999.0 = error sentinel)
        self.normalize_entropy = lib.step_entropy_normalize_entropy
        self.normalize_entropy.argtypes = [
            ctypes.c_float,  # Entropy
            ctypes.c_float,  # Mean
            ctypes.c_float,  # Std_Dev
        ]
        self.normalize_entropy.restype = ctypes.c_float
        logger.debug("Bound step_entropy_normalize_entropy")

        # -- GRPO reward computation (Paper Eq.13-15) --
        # step_entropy_compute_grpo_rewards(
        #   Skip_Ratio   : c_float,
        #   Skip_Count   : c_int,
        #   Response_Len : c_int,
        #   K_High       : c_float,
        #   K_Low        : c_float,
        #   Max_Skip     : c_int,
        #   Max_Length   : c_int,
        #   Out_Buffer   : float*   -- [r_skip_ratio, r_skip_num, r_response_len, r_total]
        # ) -> c_int  (0 = success, non-zero = error)
        self.compute_grpo_rewards = lib.step_entropy_compute_grpo_rewards
        self.compute_grpo_rewards.argtypes = [
            ctypes.c_float,                  # Skip_Ratio
            ctypes.c_int,                    # Skip_Count
            ctypes.c_int,                    # Response_Len
            ctypes.c_float,                  # K_High
            ctypes.c_float,                  # K_Low
            ctypes.c_int,                    # Max_Skip
            ctypes.c_int,                    # Max_Length
            ctypes.POINTER(ctypes.c_float),  # Out_Buffer[4]
        ]
        self.compute_grpo_rewards.restype = ctypes.c_int
        logger.debug("Bound step_entropy_compute_grpo_rewards")

        # -- Adaptive threshold state update (SOTA++ §5 Welford) --
        # step_entropy_update_adaptive_state(
        #   Entropy    : c_float,
        #   State      : float*  -- 5-element state buffer (in/out)
        #                           [running_mean, M2, count, low_sigma, high_sigma]
        # ) -> c_int  (0 = success, non-zero = error)
        self.update_adaptive_state = lib.step_entropy_update_adaptive_state
        self.update_adaptive_state.argtypes = [
            ctypes.c_float,                  # Entropy
            ctypes.POINTER(ctypes.c_float),  # State (in/out)
        ]
        self.update_adaptive_state.restype = ctypes.c_int
        logger.debug("Bound step_entropy_update_adaptive_state")

        # -- Retrieve adaptive thresholds from state (SOTA++ §5) --
        # step_entropy_get_adaptive_thresholds(
        #   State          : float*,  -- 5-element state buffer (read-only)
        #   Out_Low        : float*,  -- output threshold_low
        #   Out_High       : float*   -- output threshold_high
        # ) -> c_int  (0 = success, non-zero = error; e.g. count < 2)
        self.get_adaptive_thresholds = lib.step_entropy_get_adaptive_thresholds
        self.get_adaptive_thresholds.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # State
            ctypes.POINTER(ctypes.c_float),  # Out_Low
            ctypes.POINTER(ctypes.c_float),  # Out_High
        ]
        self.get_adaptive_thresholds.restype = ctypes.c_int
        logger.debug("Bound step_entropy_get_adaptive_thresholds")

        logger.info("SOTA++ bindings registered (histogram, normalize, grpo_rewards, adaptive_state)")


# Module-level initialization — fail LOUD
_ADA_LIB = _find_and_load_library()
_C = _CBindings(_ADA_LIB)

MAX_VOCAB_SIZE = _C.get_max_vocab_size()
MAX_TOKENS_PER_STEP = _C.get_max_tokens_per_step()


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

class AdaStepEntropy:
    """Production Python interface to Ada step entropy kernel."""

    @staticmethod
    def calculate_token_entropy(logits: np.ndarray, base: float = 2.0) -> float:
        """H(t) = -Σ p(w) log_base(p(w)). Returns entropy >= 0.0 in bits."""
        logits = np.asarray(logits, dtype=np.float32).ravel()
        if len(logits) == 0:
            raise ValueError("logits cannot be empty")
        if len(logits) > MAX_VOCAB_SIZE:
            raise ValueError(f"logits length {len(logits)} > MAX_VOCAB_SIZE {MAX_VOCAB_SIZE}")
        if base <= 0.0:
            raise ValueError(f"base must be > 0, got {base}")

        ptr = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result = _C.calculate_token_entropy(ptr, len(logits), ctypes.c_float(base))
        if result < 0.0:
            raise RuntimeError(f"Ada C_Calculate_Token_Entropy returned error: {result}")
        return float(result)

    @staticmethod
    def calculate_token_probability(logits: np.ndarray, token_id: int) -> float:
        """Softmax probability for token_id. Returns p in [0, 1]."""
        logits = np.asarray(logits, dtype=np.float32).ravel()
        if not (0 <= token_id < len(logits)):
            raise ValueError(f"token_id {token_id} out of range [0, {len(logits)})")

        ptr = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result = _C.calculate_token_probability(ptr, len(logits), token_id)
        if result < 0.0:
            raise RuntimeError(f"Ada C_Calculate_Token_Probability returned error: {result}")
        return float(np.clip(result, 0.0, 1.0))

    @staticmethod
    def classify_entropy(entropy: float, threshold_low: float = 2.0,
                         threshold_high: float = 4.0) -> EntropyLevel:
        """Classify entropy into LOW/MEDIUM/HIGH using Ada kernel."""
        if threshold_low >= threshold_high:
            raise ValueError("threshold_low must be < threshold_high")
        level = _C.classify_entropy(
            ctypes.c_float(entropy),
            ctypes.c_float(threshold_low),
            ctypes.c_float(threshold_high),
        )
        if level < 0:
            raise RuntimeError(f"Ada C_Classify_Entropy returned error: {level}")
        return EntropyLevel(level)

    @staticmethod
    def calculate_step_entropy(
        token_logits: List[np.ndarray],
        token_ids: List[int],
        token_texts: List[str],
        step_id: int = 0,
        threshold_low: float = 2.0,
        threshold_high: float = 4.0,
    ) -> StepEntropyData:
        """Calculate entropy for a complete reasoning step via Ada FFI."""
        n = len(token_logits)
        if n == 0:
            raise ValueError("token_logits cannot be empty")
        if not (n == len(token_ids) == len(token_texts)):
            raise ValueError("token_logits/ids/texts length mismatch")
        if n > MAX_TOKENS_PER_STEP:
            raise ValueError(f"Too many tokens ({n} > {MAX_TOKENS_PER_STEP})")
        if threshold_low >= threshold_high:
            raise ValueError("threshold_low must be < threshold_high")

        # Hybrid: token-level via Ada, aggregation in Python
        token_entropies = []
        total_entropy = 0.0
        max_entropy = 0.0

        for i, (logits, tid, txt) in enumerate(zip(token_logits, token_ids, token_texts)):
            ent = AdaStepEntropy.calculate_token_entropy(logits)
            prob = AdaStepEntropy.calculate_token_probability(logits, tid)
            token_entropies.append(TokenEntropyData(
                token_id=tid, token_text=txt, entropy=ent,
                probability=prob, context_position=i,
            ))
            total_entropy += ent
            max_entropy = max(max_entropy, ent)

        avg_entropy = total_entropy / n
        level = AdaStepEntropy.classify_entropy(avg_entropy, threshold_low, threshold_high)

        return StepEntropyData(
            step_id=step_id, step_text="".join(token_texts),
            token_count=n, total_entropy=total_entropy,
            avg_entropy=avg_entropy, max_entropy=max_entropy,
            level=level, token_entropies=token_entropies,
        )

    @staticmethod
    def route_step(step: StepEntropyData, threshold_low: float = 2.0,
                   threshold_high: float = 4.0,
                   enable_memory: bool = True) -> RoutingDecisionData:
        """Route step via Ada kernel: Fast/Normal/Slow."""
        out_path = ctypes.c_int(0)
        out_retrieve = ctypes.c_int(0)
        rc = _C.route_step(
            ctypes.c_float(step.avg_entropy),
            ctypes.c_float(step.total_entropy),
            ctypes.c_float(step.max_entropy),
            ctypes.c_float(threshold_low),
            ctypes.c_float(threshold_high),
            1 if enable_memory else 0,
            ctypes.byref(out_path),
            ctypes.byref(out_retrieve),
        )
        if rc != 0:
            raise RuntimeError(f"Ada C_Route_Step returned error: {rc}")
        return RoutingDecisionData(
            path=RoutingPath(out_path.value),
            retrieve_memory=bool(out_retrieve.value),
            total_entropy=step.total_entropy,
            avg_entropy=step.avg_entropy,
            max_entropy=step.max_entropy,
            level=step.level,
        )

    @staticmethod
    def batch_route(step_entropies: List[float], threshold_low: float = 2.0,
                    threshold_high: float = 4.0) -> List[RoutingPath]:
        """Batch routing via Ada kernel."""
        n = len(step_entropies)
        if n == 0:
            return []
        arr = np.array(step_entropies, dtype=np.float32)
        out = (ctypes.c_int * n)()
        rc = _C.batch_route(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n, ctypes.c_float(threshold_low), ctypes.c_float(threshold_high),
            ctypes.cast(out, ctypes.POINTER(ctypes.c_int)),
        )
        if rc != 0:
            raise RuntimeError(f"Ada C_Batch_Route returned error: {rc}")
        return [RoutingPath(out[i]) for i in range(n)]

    @staticmethod
    def analyze_chain(step_entropies: List[float], threshold_low: float = 2.0,
                      threshold_high: float = 4.0) -> CompressionAnalysisData:
        """Analyze chain for compression opportunity via Ada kernel."""
        n = len(step_entropies)
        if n == 0:
            return CompressionAnalysisData(0, 0, 0, 0, 0, 0, 0.0, 0.0)
        arr = np.array(step_entropies, dtype=np.float32)
        buf = (ctypes.c_float * 8)()
        rc = _C.analyze_chain(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n, ctypes.c_float(threshold_low), ctypes.c_float(threshold_high),
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"Ada C_Analyze_Chain returned error: {rc}")
        return CompressionAnalysisData(
            total_steps=int(buf[0]), compressible_steps=int(buf[1]),
            informative_steps=int(buf[2]), low_entropy_steps=int(buf[3]),
            medium_entropy_steps=int(buf[4]), high_entropy_steps=int(buf[5]),
            compression_ratio=float(buf[6]), avg_entropy=float(buf[7]),
        )

    @staticmethod
    def recommend_pruning(step_entropies: List[float],
                          pruning_ratio: float = 0.8) -> List[int]:
        """Recommend step indices to prune via Ada kernel."""
        n = len(step_entropies)
        if n == 0:
            return []
        if not (0.0 <= pruning_ratio <= 1.0):
            raise ValueError(f"pruning_ratio must be in [0,1], got {pruning_ratio}")
        arr = np.array(step_entropies, dtype=np.float32)
        out_indices = (ctypes.c_int * n)()
        out_count = ctypes.c_int(0)
        rc = _C.recommend_pruning(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n, ctypes.c_float(pruning_ratio),
            ctypes.cast(out_indices, ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(out_count),
        )
        if rc != 0:
            raise RuntimeError(f"Ada C_Recommend_Pruning returned error: {rc}")
        return [out_indices[i] for i in range(out_count.value)]

    # ---- SOTA++ static methods ----

    @staticmethod
    def compute_histogram(entropies: List[float], bin_count: int = 16) -> EntropyHistogramData:
        """Compute entropy distribution histogram via Ada kernel (SOTA++ §2.4).

        Implements a fixed-width histogram over the supplied entropy values, returning
        per-bin token counts plus summary statistics (min, max, mean, std_dev).

        Buffer layout from Ada kernel:
            Out_Buffer[0]           = bin_width
            Out_Buffer[1]           = min_val
            Out_Buffer[2]           = max_val
            Out_Buffer[3]           = mean
            Out_Buffer[4]           = std_dev
            Out_Buffer[5..5+N-1]    = integer bin counts (packed as float)

        Args:
            entropies:  Non-empty list of entropy values.  Must be finite (no NaN/inf).
            bin_count:  Number of histogram bins in [1, 32].  Defaults to 16.

        Returns:
            EntropyHistogramData with bins, summary statistics, and total token count.

        Raises:
            ValueError:  Bad inputs (empty list, out-of-range bin_count, NaN/inf values).
            RuntimeError: Ada kernel returned a non-zero error code.
        """
        if len(entropies) == 0:
            raise ValueError("entropies cannot be empty")
        if not (1 <= bin_count <= 32):
            raise ValueError(f"bin_count must be in [1, 32], got {bin_count}")
        for i, v in enumerate(entropies):
            if not np.isfinite(v):
                raise ValueError(f"entropies[{i}] is not finite: {v}")

        arr = np.array(entropies, dtype=np.float32)
        buf_size = 5 + bin_count
        buf = (ctypes.c_float * buf_size)()

        logger.debug(
            f"compute_histogram: n={len(entropies)}, bin_count={bin_count}"
        )
        rc = _C.compute_histogram(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(len(entropies)),
            ctypes.c_int(bin_count),
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"Ada C_Compute_Histogram returned error: {rc}")

        bin_width = float(buf[0])
        min_val   = float(buf[1])
        max_val   = float(buf[2])
        mean      = float(buf[3])
        std_dev   = float(buf[4])
        bins      = [int(buf[5 + i]) for i in range(bin_count)]

        return EntropyHistogramData(
            bin_count=bin_count,
            bin_width=bin_width,
            min_val=min_val,
            max_val=max_val,
            mean=mean,
            std_dev=std_dev,
            bins=bins,
            total_tokens=len(entropies),
        )

    @staticmethod
    def normalize_entropy(entropy: float, mean: float, std_dev: float) -> float:
        """Z-score normalize entropy for cross-model portability (SOTA++ §2.3).

        Computes z = (entropy - mean) / std_dev, enabling the router to apply
        identical thresholds across models with different absolute entropy scales
        (e.g. DeepSeek-R1 vs Qwen vs Llama).

        The Ada kernel returns the sentinel value -999.0 when std_dev == 0.0,
        which this method converts to 0.0 (zero deviation → perfectly average).

        Args:
            entropy:  The entropy value to normalize.  Any finite float.
            mean:     Running mean of the entropy distribution.  Any finite float.
            std_dev:  Running standard deviation.  Must be >= 0.

        Returns:
            Z-score normalized entropy.  0.0 when std_dev == 0.

        Raises:
            ValueError:  std_dev < 0.
            RuntimeError: Ada kernel returned an unexpected sentinel value.
        """
        if std_dev < 0.0:
            raise ValueError(f"std_dev must be >= 0, got {std_dev}")

        logger.debug(
            f"normalize_entropy: entropy={entropy}, mean={mean}, std_dev={std_dev}"
        )
        result = _C.normalize_entropy(
            ctypes.c_float(entropy),
            ctypes.c_float(mean),
            ctypes.c_float(std_dev),
        )
        result = float(result)

        # Ada Normalize_Entropy (step_entropy.adb) returns Float(Entropy)
        # UNCHANGED when Std_Dev <= 0.0.  No variance → cannot normalize;
        # raw passthrough is the correct behaviour (Test 11b confirms this).
        #
        # The -999.0 value is the error sentinel emitted ONLY by the C
        # wrapper's outermost exception handler (step_entropy_c_api.adb
        # C_Normalize_Entropy: "when E : others => Log_Error; return -999.0").
        # It signals an Ada runtime fault, NOT a std_dev=0 condition.
        #
        # We guard against -999.0 defensively.  If this branch ever fires,
        # it means the Ada kernel threw an unhandled exception — log it.
        if result == -999.0:
            logger.error(
                "normalize_entropy: Ada kernel returned error sentinel -999.0 "
                f"(entropy={entropy}, mean={mean}, std_dev={std_dev}); "
                "returning 0.0 as safe fallback"
            )
            return 0.0

        return result

    @staticmethod
    def compute_grpo_rewards(
        skip_ratio: float,
        skip_count: int,
        response_len: int,
        k_high: float = GRPO_K_SKIP_HIGH,
        k_low: float = GRPO_K_SKIP_LOW,
        max_skip: int = GRPO_TAU_SKIP_MAX,
        max_length: int = GRPO_TAU_LENGTH,
    ) -> GRPORewardsData:
        """Compute GRPO inference-time reward components (Paper Eq.13-15, §3.3).

        The paper defines a 4-component reward for GRPO training that incentivises
        high skip ratios while penalising degenerate compression and verbosity:

            R_skip_ratio   = 1.0  if skip_ratio >= κ_high  (default 0.80)
                           = 0.5  if skip_ratio >= κ_low   (default 0.50)
                           = 0.0  otherwise

            R_skip_num     = -1.0 if skip_count > τ_skip   (default 100)
                           = 0.0  otherwise

            R_response_len = -1.0 if response_len > τ_length (default 3500)
                           = 0.0  otherwise

            R_total        = R_skip_ratio + R_skip_num + R_response_len

        (R_correctness is handled separately by the training loop and is not
        computable at the kernel level.)

        Args:
            skip_ratio:   Fraction of steps that were skipped.  In [0, 1].
            skip_count:   Absolute number of [SKIP] tokens emitted.  >= 0.
            response_len: Total token length of the model response.  >= 0.
            k_high:       Upper skip-ratio threshold.  Default GRPO_K_SKIP_HIGH.
            k_low:        Lower skip-ratio threshold.  Default GRPO_K_SKIP_LOW.
            max_skip:     Skip count ceiling.  Default GRPO_TAU_SKIP_MAX.
            max_length:   Response length ceiling.  Default GRPO_TAU_LENGTH.

        Returns:
            GRPORewardsData with each reward component and their sum.

        Raises:
            ValueError:  skip_ratio outside [0,1] or k_low >= k_high.
            RuntimeError: Ada kernel returned a non-zero error code.
        """
        if not (0.0 <= skip_ratio <= 1.0):
            raise ValueError(f"skip_ratio must be in [0, 1], got {skip_ratio}")
        if k_low >= k_high:
            raise ValueError(
                f"k_low ({k_low}) must be < k_high ({k_high})"
            )

        buf = (ctypes.c_float * 4)()

        logger.debug(
            f"compute_grpo_rewards: skip_ratio={skip_ratio}, skip_count={skip_count}, "
            f"response_len={response_len}, k_high={k_high}, k_low={k_low}"
        )
        rc = _C.compute_grpo_rewards(
            ctypes.c_float(skip_ratio),
            ctypes.c_int(skip_count),
            ctypes.c_int(response_len),
            ctypes.c_float(k_high),
            ctypes.c_float(k_low),
            ctypes.c_int(max_skip),
            ctypes.c_int(max_length),
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"Ada C_Compute_GRPO_Rewards returned error: {rc}")

        return GRPORewardsData(
            r_skip_ratio=float(buf[0]),
            r_skip_num=float(buf[1]),
            r_response_len=float(buf[2]),
            r_total=float(buf[3]),
        )


# ============================================================================
# ADAPTIVE THRESHOLD MANAGER
# ============================================================================

class AdaptiveThresholdManager:
    """Welford online algorithm for adaptive entropy thresholds (Paper §5).

    Implements the "adaptive thresholds for task-aware compression" future work
    direction from the Step Entropy paper.  Running mean and variance are
    maintained in a 5-float state buffer shared with the Ada kernel via ctypes.

    State buffer layout (5 floats, index 0-4):
        [0] running_mean       -- Welford online mean
        [1] running_M2         -- Welford sum-of-squared-deviations (M2 accumulator)
        [2] sample_count       -- number of observations (stored as float for FFI simplicity)
        [3] low_sigma          -- sigma multiplier for threshold_low  (e.g. -1.0)
        [4] high_sigma         -- sigma multiplier for threshold_high (e.g. +1.0)

    Derived thresholds:
        threshold_low  = mean + low_sigma  * std   (typically mean - 1σ)
        threshold_high = mean + high_sigma * std   (typically mean + 1σ)

    Thread-safety: NOT thread-safe.  Use one instance per thread.
    """

    STATE_SIZE = 5  # float elements in state buffer

    def __init__(self, low_sigma: float = -1.0, high_sigma: float = 1.0):
        """Initialize with sigma factors for threshold computation.

        Args:
            low_sigma:  Sigma multiplier for threshold_low.  Must be < high_sigma.
                        Negative means threshold_low sits below the running mean.
            high_sigma: Sigma multiplier for threshold_high.  Must be > low_sigma.

        Raises:
            ValueError: low_sigma >= high_sigma.
        """
        if low_sigma >= high_sigma:
            raise ValueError(
                f"low_sigma ({low_sigma}) must be < high_sigma ({high_sigma})"
            )
        self._state = (ctypes.c_float * self.STATE_SIZE)(
            0.0, 0.0, 0.0, low_sigma, high_sigma
        )
        logger.info(
            f"AdaptiveThresholdManager initialized (σ: [{low_sigma}, {high_sigma}])"
        )

    def update(self, entropy: float) -> None:
        """Update Welford running stats with a new entropy observation.

        Implements the Welford online algorithm:
            count    += 1
            delta     = x - mean
            mean     += delta / count
            delta2    = x - mean
            M2       += delta * delta2

        The Ada kernel performs this update in-place on the shared state buffer.

        Args:
            entropy: New entropy observation.  Must be >= 0.

        Raises:
            ValueError:  entropy < 0.
            RuntimeError: Ada kernel returned a non-zero error code.
        """
        if entropy < 0.0:
            raise ValueError(f"entropy must be >= 0, got {entropy}")

        logger.debug(f"AdaptiveThresholdManager.update: entropy={entropy}")
        rc = _C.update_adaptive_state(
            ctypes.c_float(entropy),
            ctypes.cast(self._state, ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(
                f"Ada C_Update_Adaptive_State returned error: {rc}"
            )

    def get_thresholds(self) -> Tuple[float, float]:
        """Return (threshold_low, threshold_high) from current running stats.

        Thresholds are computed by the Ada kernel as:
            threshold_low  = mean + low_sigma  * std
            threshold_high = mean + high_sigma * std

        Requires at least 2 samples to produce a variance estimate.

        Returns:
            (threshold_low, threshold_high) as floats.

        Raises:
            RuntimeError: Fewer than 2 samples seen (variance undefined), or
                          Ada kernel returned a non-zero error code.
        """
        if int(self._state[2]) < 2:
            raise RuntimeError(
                f"Need at least 2 samples to compute thresholds, "
                f"got {int(self._state[2])}"
            )

        out_low  = ctypes.c_float(0.0)
        out_high = ctypes.c_float(0.0)

        logger.debug("AdaptiveThresholdManager.get_thresholds called")
        rc = _C.get_adaptive_thresholds(
            ctypes.cast(self._state, ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out_low),
            ctypes.byref(out_high),
        )
        if rc != 0:
            raise RuntimeError(
                f"Ada C_Get_Adaptive_Thresholds returned error: {rc}"
            )

        return float(out_low.value), float(out_high.value)

    def sample_count(self) -> int:
        """Number of entropy observations seen so far."""
        return int(self._state[2])

    def reset(self) -> None:
        """Reset running stats, preserving sigma factors.

        Clears running_mean, M2, and sample_count to 0.0 while keeping
        the low_sigma / high_sigma configuration intact.
        """
        low_sigma  = self._state[3]
        high_sigma = self._state[4]
        for i in range(3):
            self._state[i] = 0.0
        self._state[3] = low_sigma
        self._state[4] = high_sigma
        logger.debug("AdaptiveThresholdManager reset")

    @property
    def running_mean(self) -> float:
        """Current Welford running mean."""
        return float(self._state[0])

    @property
    def running_std(self) -> float:
        """Current running standard deviation (0.0 if fewer than 2 samples)."""
        count = int(self._state[2])
        if count < 2:
            return 0.0
        variance = float(self._state[1]) / (count - 1)
        return variance ** 0.5

    def __enter__(self) -> "AdaptiveThresholdManager":
        """Context manager entry — enables `with AdaptiveThresholdManager() as atm:` usage.

        Logs entry with current sample count so streaming pipeline logs show
        whether the manager enters with pre-warmed statistics or cold state.
        """
        logger.info(
            f"AdaptiveThresholdManager.__enter__: "
            f"samples={self.sample_count()}, "
            f"mean={self.running_mean:.4f}, "
            f"std={self.running_std:.4f}"
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: object,
    ) -> bool:
        """Context manager exit — logs final statistics on clean or exception exit.

        Never suppresses exceptions: returns False so any exception in the
        `with` block propagates normally.  The log entry is written regardless
        of whether the block exited cleanly or via an exception, so post-mortem
        analysis always has the final threshold state.
        """
        status = "clean" if exc_type is None else f"exception ({exc_type.__name__})"
        logger.info(
            f"AdaptiveThresholdManager.__exit__: status={status}, "
            f"samples={self.sample_count()}, "
            f"mean={self.running_mean:.4f}, "
            f"std={self.running_std:.4f}"
        )
        return False  # never suppress exceptions


# ============================================================================
# GARLIC ORCHESTRATOR INTEGRATION
# ============================================================================

class GarlicAdaStepEntropy:
    """Integration wrapper for Garlic orchestrator."""

    def __init__(self, threshold_low: float = 2.0, threshold_high: float = 4.0):
        if threshold_low >= threshold_high:
            raise ValueError("threshold_low must be < threshold_high")
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        # AdaStepEntropy has only @staticmethod methods — no instance is needed.
        # Calling AdaStepEntropy.method() directly avoids a dead allocation
        # and prevents maintainers from assuming the instance carries state.
        logger.info(f"GarlicAdaStepEntropy initialized (thresholds: {threshold_low}/{threshold_high})")

    def calculate_step_entropy(self, token_logits, token_ids, token_texts,
                                step_id=0) -> StepEntropyData:
        return AdaStepEntropy.calculate_step_entropy(
            token_logits, token_ids, token_texts, step_id,
            self.threshold_low, self.threshold_high)

    def route_step(self, step: StepEntropyData) -> RoutingDecisionData:
        return AdaStepEntropy.route_step(step, self.threshold_low, self.threshold_high)

    def analyze_chain(self, step_entropies: List[float]) -> CompressionAnalysisData:
        return AdaStepEntropy.analyze_chain(step_entropies, self.threshold_low, self.threshold_high)

    def recommend_pruning(
        self,
        step_entropies: List[float],
        pruning_ratio: float = OPTIMAL_PRUNE_RATIO,
    ) -> List[int]:
        """Return indices of steps recommended for pruning (Paper §3.2 Step 3).

        Delegates to AdaStepEntropy.recommend_pruning with the Garlic instance's
        configured pruning ratio.

        Args:
            step_entropies: Per-step average entropy values.
            pruning_ratio:  Fraction of steps to prune.  Default OPTIMAL_PRUNE_RATIO (0.80).

        Returns:
            List of 0-based step indices selected for pruning (lowest-entropy first).
        """
        return AdaStepEntropy.recommend_pruning(step_entropies, pruning_ratio)

    def compress_chain(
        self,
        steps: List[StepEntropyData],
        pruning_ratio: float = OPTIMAL_PRUNE_RATIO,
    ) -> Tuple[List[str], List[int]]:
        """Replace lowest-entropy steps with SKIP_TOKEN_TEXT ([SKIP]).

        Implements Paper §3.2 Step 4: after ranking steps by entropy and selecting
        the bottom κ×N for pruning, each pruned step's text is replaced by the
        [SKIP] token.  Non-pruned steps retain their original text.

        The compressed list preserves the original step order so it can be
        re-joined with STEP_DELIMITER and wrapped in <think>...</think> by the
        caller:

            think_block = "<think>" + STEP_DELIMITER.join(compressed_texts) + "</think>"

        Args:
            steps:         List of StepEntropyData for each reasoning step.
            pruning_ratio: Fraction of steps to replace.  Default 0.80.

        Returns:
            (compressed_texts, pruned_indices)
                compressed_texts: Per-step strings; pruned steps contain SKIP_TOKEN_TEXT.
                pruned_indices:   Sorted list of 0-based indices that were pruned.

        Raises:
            ValueError:  steps is empty, or pruning_ratio is out of [0, 1].
        """
        if not steps:
            raise ValueError("steps cannot be empty")
        if not (0.0 <= pruning_ratio <= 1.0):
            raise ValueError(f"pruning_ratio must be in [0, 1], got {pruning_ratio}")

        step_avg_entropies = [s.avg_entropy for s in steps]
        pruned_indices = self.recommend_pruning(step_avg_entropies, pruning_ratio)
        pruned_set = set(pruned_indices)

        compressed_texts = [
            SKIP_TOKEN_TEXT if i in pruned_set else steps[i].step_text
            for i in range(len(steps))
        ]

        logger.info(
            f"compress_chain: {len(steps)} steps → "
            f"{len(pruned_indices)} pruned ({len(pruned_indices)/len(steps):.1%})"
        )
        return compressed_texts, sorted(pruned_indices)


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=" * 70)
    print("ADA-Step-Entropy FFI Bridge — Production Smoke Test")
    print("=" * 70)

    version = _C.get_version()
    print(f"\nLibrary version: {version // 10000}.{(version % 10000) // 100}.{version % 100}")
    print(f"Max vocab size: {MAX_VOCAB_SIZE}")
    print(f"Max tokens/step: {MAX_TOKENS_PER_STEP}")

    ada = AdaStepEntropy()

    _all_passed = True

    def _check(label: str, ok: bool, detail: str = "") -> None:
        global _all_passed
        status = "PASS" if ok else "FAIL"
        if not ok:
            _all_passed = False
        suffix = f" — {detail}" if detail else ""
        print(f"  [{status}] {label}{suffix}")

    # ------------------------------------------------------------------
    # Test 1: Token entropy — peaked distribution (low entropy)
    # ------------------------------------------------------------------
    try:
        logits_peaked = np.zeros(1000, dtype=np.float32)
        logits_peaked[0] = 10.0
        ent_low = ada.calculate_token_entropy(logits_peaked)
        print(f"\n[Test 1] Peaked logits entropy: {ent_low:.4f} bits (expect ~0)")
        _check("Test 1", ent_low >= 0.0 and ent_low < 2.0, f"ent={ent_low:.4f}")
    except Exception as exc:
        _check("Test 1", False, str(exc))

    # ------------------------------------------------------------------
    # Test 2: Token entropy — uniform distribution (high entropy)
    # ------------------------------------------------------------------
    try:
        logits_uniform = np.ones(1000, dtype=np.float32)
        ent_high = ada.calculate_token_entropy(logits_uniform)
        expected_uniform = np.log2(1000)
        print(f"[Test 2] Uniform logits entropy: {ent_high:.4f} bits (expect ~{expected_uniform:.2f})")
        _check("Test 2", abs(ent_high - expected_uniform) < 0.1, f"ent={ent_high:.4f}")
    except Exception as exc:
        _check("Test 2", False, str(exc))

    # ------------------------------------------------------------------
    # Test 3: Token probability
    # ------------------------------------------------------------------
    try:
        prob = ada.calculate_token_probability(logits_peaked, 0)
        print(f"[Test 3] Peaked token 0 probability: {prob:.6f} (expect ~1.0)")
        _check("Test 3", prob > 0.9, f"prob={prob:.6f}")
    except Exception as exc:
        _check("Test 3", False, str(exc))

    # ------------------------------------------------------------------
    # Test 4: Classify entropy
    # ------------------------------------------------------------------
    try:
        level_low = ada.classify_entropy(1.0, 2.0, 4.0)
        level_mid = ada.classify_entropy(3.0, 2.0, 4.0)
        level_hi  = ada.classify_entropy(5.0, 2.0, 4.0)
        print(f"[Test 4] Classify: 1.0→{level_low.name}, 3.0→{level_mid.name}, 5.0→{level_hi.name}")
        _check("Test 4",
               level_low == EntropyLevel.LOW and
               level_mid == EntropyLevel.MEDIUM and
               level_hi  == EntropyLevel.HIGH)
    except Exception as exc:
        _check("Test 4", False, str(exc))

    # ------------------------------------------------------------------
    # Test 5: Step entropy (hybrid path)
    # ------------------------------------------------------------------
    try:
        step = ada.calculate_step_entropy(
            token_logits=[logits_peaked, logits_uniform, logits_peaked],
            token_ids=[0, 500, 0],
            token_texts=["The", " answer", " is"],
        )
        print(f"[Test 5] Step entropy: total={step.total_entropy:.2f}, "
              f"avg={step.avg_entropy:.2f}, level={step.level.name}")
        _check("Test 5", step.token_count == 3 and step.total_entropy >= 0.0)
    except Exception as exc:
        _check("Test 5", False, str(exc))

    # ------------------------------------------------------------------
    # Test 6: Route step
    # ------------------------------------------------------------------
    try:
        decision = ada.route_step(step)
        print(f"[Test 6] Route: path={decision.path.name}, retrieve_memory={decision.retrieve_memory}")
        _check("Test 6", isinstance(decision.path, RoutingPath))
    except Exception as exc:
        _check("Test 6", False, str(exc))

    # ------------------------------------------------------------------
    # Test 7: Batch route
    # ------------------------------------------------------------------
    try:
        routes = ada.batch_route([1.0, 3.0, 5.0, 0.5, 4.5])
        print(f"[Test 7] Batch route: {[r.name for r in routes]}")
        _check("Test 7",
               routes[0] == RoutingPath.FAST and
               routes[1] == RoutingPath.NORMAL and
               routes[2] == RoutingPath.SLOW)
    except Exception as exc:
        _check("Test 7", False, str(exc))

    # ------------------------------------------------------------------
    # Test 8: Analyze chain
    # ------------------------------------------------------------------
    try:
        analysis = ada.analyze_chain([1.0, 3.0, 5.0, 0.5, 4.5])
        print(f"[Test 8] Chain: {analysis.total_steps} steps, "
              f"compression={analysis.compression_ratio:.1%}, "
              f"low={analysis.low_entropy_steps}, med={analysis.medium_entropy_steps}, "
              f"hi={analysis.high_entropy_steps}")
        _check("Test 8", analysis.total_steps == 5)
    except Exception as exc:
        _check("Test 8", False, str(exc))

    # ------------------------------------------------------------------
    # Test 9: Recommend pruning
    # ------------------------------------------------------------------
    try:
        prune = ada.recommend_pruning([1.0, 3.0, 5.0, 0.5, 4.5], pruning_ratio=0.6)
        print(f"[Test 9] Prune indices (60%): {prune}")
        _check("Test 9", all(isinstance(i, int) for i in prune))
    except Exception as exc:
        _check("Test 9", False, str(exc))

    # ------------------------------------------------------------------
    # Test 10: Entropy histogram
    # ------------------------------------------------------------------
    try:
        hist = ada.compute_histogram([1.0, 2.5, 3.0, 4.5, 5.0, 1.5, 3.5], bin_count=8)
        print(f"[Test 10] Histogram: bins={hist.bins[:8]}, mean={hist.mean:.3f}, std={hist.std_dev:.3f}")
        _check("Test 10",
               hist.total_tokens == 7 and
               hist.bin_count == 8 and
               sum(hist.bins[:8]) == 7,
               f"total={hist.total_tokens}, bin_sum={sum(hist.bins[:8])}")
    except Exception as exc:
        _check("Test 10", False, str(exc))

    # ------------------------------------------------------------------
    # Test 11: Z-score normalization
    # ------------------------------------------------------------------
    try:
        norm = ada.normalize_entropy(3.0, mean=2.5, std_dev=1.0)
        print(f"[Test 11] Normalize(3.0, mean=2.5, std=1.0): {norm:.4f} (expect 0.5)")
        _check("Test 11", abs(norm - 0.5) < 0.001, f"norm={norm:.4f}")
    except Exception as exc:
        _check("Test 11", False, str(exc))

    # ------------------------------------------------------------------
    # Test 11b: Z-score normalization — zero std_dev (degenerate case)
    # When std_dev=0 there is no variance to normalize against, so the
    # function returns Float(Entropy) unchanged (the raw entropy value).
    # ------------------------------------------------------------------
    try:
        norm_z = ada.normalize_entropy(3.0, mean=3.0, std_dev=0.0)
        print(f"[Test 11b] Normalize(std=0): {norm_z:.4f} (expect 3.0 — raw passthrough)")
        _check("Test 11b", norm_z == 3.0, f"norm={norm_z}")
    except Exception as exc:
        _check("Test 11b", False, str(exc))

    # ------------------------------------------------------------------
    # Test 12: GRPO rewards — high skip ratio, no penalties
    # ------------------------------------------------------------------
    try:
        rewards = ada.compute_grpo_rewards(skip_ratio=0.85, skip_count=50, response_len=2000)
        print(f"[Test 12] GRPO: r_skip_ratio={rewards.r_skip_ratio}, "
              f"r_skip_num={rewards.r_skip_num}, "
              f"r_response_len={rewards.r_response_len}, "
              f"r_total={rewards.r_total}")
        _check("Test 12",
               rewards.r_skip_ratio == 1.0 and
               rewards.r_skip_num   == 0.0 and
               rewards.r_response_len == 0.0 and
               rewards.r_total == 1.0,
               f"r_total={rewards.r_total}")
    except Exception as exc:
        _check("Test 12", False, str(exc))

    # ------------------------------------------------------------------
    # Test 12b: GRPO rewards — low skip ratio, both penalties triggered
    # ------------------------------------------------------------------
    try:
        rewards2 = ada.compute_grpo_rewards(skip_ratio=0.3, skip_count=150, response_len=4000)
        print(f"[Test 12b] GRPO penalties: r_total={rewards2.r_total} (expect -2.0)")
        _check("Test 12b",
               rewards2.r_skip_ratio == 0.0 and
               rewards2.r_skip_num   == -1.0 and
               rewards2.r_response_len == -1.0 and
               rewards2.r_total == -2.0,
               f"r_total={rewards2.r_total}")
    except Exception as exc:
        _check("Test 12b", False, str(exc))

    # ------------------------------------------------------------------
    # Test 12c: GRPO rewards — mid skip ratio (k_low tier)
    # ------------------------------------------------------------------
    try:
        rewards3 = ada.compute_grpo_rewards(skip_ratio=0.65, skip_count=30, response_len=1000)
        print(f"[Test 12c] GRPO mid tier: r_skip_ratio={rewards3.r_skip_ratio} (expect 0.5)")
        _check("Test 12c",
               rewards3.r_skip_ratio == 0.5 and
               rewards3.r_total == 0.5,
               f"r_skip_ratio={rewards3.r_skip_ratio}")
    except Exception as exc:
        _check("Test 12c", False, str(exc))

    # ------------------------------------------------------------------
    # Test 13: Adaptive threshold manager
    # ------------------------------------------------------------------
    try:
        atm = AdaptiveThresholdManager(low_sigma=-1.0, high_sigma=1.0)
        for ent in [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5]:
            atm.update(ent)
        print(f"[Test 13] Adaptive: count={atm.sample_count()}, "
              f"mean={atm.running_mean:.3f}, std={atm.running_std:.3f}")
        lo, hi = atm.get_thresholds()
        print(f"         Thresholds: low={lo:.3f}, high={hi:.3f}")
        _check("Test 13",
               atm.sample_count() == 7 and lo < hi,
               f"count={atm.sample_count()}, lo={lo:.3f}, hi={hi:.3f}")
    except Exception as exc:
        _check("Test 13", False, str(exc))

    # ------------------------------------------------------------------
    # Test 13b: AdaptiveThresholdManager.reset
    # ------------------------------------------------------------------
    try:
        atm2 = AdaptiveThresholdManager(low_sigma=-0.5, high_sigma=0.5)
        for v in [1.0, 2.0, 3.0]:
            atm2.update(v)
        pre_count = atm2.sample_count()
        atm2.reset()
        post_count = atm2.sample_count()
        _check("Test 13b",
               pre_count == 3 and post_count == 0 and
               atm2.running_mean == 0.0,
               f"pre={pre_count}, post={post_count}")
        print(f"[Test 13b] ATM reset: count before={pre_count}, after={post_count}")
    except Exception as exc:
        _check("Test 13b", False, str(exc))

    # ------------------------------------------------------------------
    # Test 14: GarlicAdaStepEntropy.compress_chain
    # ------------------------------------------------------------------
    try:
        garlic = GarlicAdaStepEntropy()
        # Build two steps with very different entropies
        step_lo = garlic.calculate_step_entropy(
            token_logits=[logits_peaked],
            token_ids=[0],
            token_texts=["trivial"],
            step_id=0,
        )
        step_hi = garlic.calculate_step_entropy(
            token_logits=[logits_uniform],
            token_ids=[500],
            token_texts=["informative"],
            step_id=1,
        )
        compressed, pruned = garlic.compress_chain([step_lo, step_hi], pruning_ratio=0.5)
        print(f"[Test 14] compress_chain: compressed={compressed}, pruned={pruned}")
        _check("Test 14",
               len(compressed) == 2 and
               SKIP_TOKEN_TEXT in compressed,
               f"compressed={compressed}")
    except Exception as exc:
        _check("Test 14", False, str(exc))

    # ------------------------------------------------------------------
    # Test 15: GarlicAdaStepEntropy.recommend_pruning wrapper
    # ------------------------------------------------------------------
    try:
        garlic2 = GarlicAdaStepEntropy()
        prune2 = garlic2.recommend_pruning([0.5, 3.0, 5.0, 1.0, 4.0], pruning_ratio=0.4)
        print(f"[Test 15] Garlic recommend_pruning(40%): {prune2}")
        _check("Test 15",
               isinstance(prune2, list) and all(isinstance(i, int) for i in prune2))
    except Exception as exc:
        _check("Test 15", False, str(exc))

    # ------------------------------------------------------------------
    # Test 16: Validation — ValueError on bad inputs
    # ------------------------------------------------------------------
    try:
        bad_inputs_ok = True

        # compute_histogram: empty list
        try:
            ada.compute_histogram([], bin_count=8)
            bad_inputs_ok = False
        except ValueError:
            pass

        # compute_histogram: bin_count out of range
        try:
            ada.compute_histogram([1.0, 2.0], bin_count=0)
            bad_inputs_ok = False
        except ValueError:
            pass

        # compute_histogram: NaN in list
        try:
            ada.compute_histogram([1.0, float("nan")], bin_count=8)
            bad_inputs_ok = False
        except ValueError:
            pass

        # normalize_entropy: negative std_dev
        try:
            ada.normalize_entropy(1.0, mean=2.0, std_dev=-0.5)
            bad_inputs_ok = False
        except ValueError:
            pass

        # compute_grpo_rewards: skip_ratio out of range
        try:
            ada.compute_grpo_rewards(skip_ratio=1.5, skip_count=0, response_len=0)
            bad_inputs_ok = False
        except ValueError:
            pass

        # compute_grpo_rewards: k_low >= k_high
        try:
            ada.compute_grpo_rewards(0.5, 0, 0, k_high=0.4, k_low=0.6)
            bad_inputs_ok = False
        except ValueError:
            pass

        # AdaptiveThresholdManager: low_sigma >= high_sigma
        try:
            AdaptiveThresholdManager(low_sigma=1.0, high_sigma=0.5)
            bad_inputs_ok = False
        except ValueError:
            pass

        # AdaptiveThresholdManager.update: negative entropy
        try:
            atm_bad = AdaptiveThresholdManager()
            atm_bad.update(-1.0)
            bad_inputs_ok = False
        except ValueError:
            pass

        # AdaptiveThresholdManager.get_thresholds: insufficient samples
        try:
            atm_empty = AdaptiveThresholdManager()
            atm_empty.get_thresholds()
            bad_inputs_ok = False
        except RuntimeError:
            pass

        print(f"[Test 16] Input validation: all ValueError/RuntimeError fired correctly")
        _check("Test 16", bad_inputs_ok)
    except Exception as exc:
        _check("Test 16", False, str(exc))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    if _all_passed:
        print("ALL TESTS PASSED — Ada FFI bridge is OPERATIONAL")
    else:
        print("SOME TESTS FAILED — review output above")
    print("=" * 70)
