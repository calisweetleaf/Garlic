# HIGH-LEVEL PLAN: ADA-Step-Entropy Transition

This document outlines the strategic roadmap for finalizing the Ada 2022 migration of the Step Entropy kernel and advancing it to a SOTA++ state.

---

## Phase 1: Unblock the FFI Bridge (Immediate Priority)

**Objective:** Establish a rock-solid, C-compatible ABI between the Ada math kernel and the Python orchestration layer.

- [x] **Task 1.1:** Create `step_entropy_c_api.ads` — thin `Convention => C` translation layer.
- [ ] **Task 1.2:** Create `step_entropy_c_api.adb` — marshaling body (System.Address → Ada arrays).
  - First attempt used `access all C_Float` → **segfault** (fat pointer vs thin pointer mismatch).
  - Fix: use `System.Address` for all pointer params (matches C `void*` exactly).
  - Need to use address overlay pattern from Ada manual (B.3) for array access.
- [ ] **Task 1.3:** Recompile `libgarlic_core.so` with fixed C API and verify symbols (`nm -D`).
- [x] **Task 1.4:** Rewrite `ada_bridge.py` ctypes bindings to consume C-ABI cleanly.
  - Fixed library search path (checks `lib/` subdir first).
  - All FFI calls check error codes, fail loud.
  - Includes self-test (`python ada_bridge.py`).
- [ ] **Task 1.5:** Execute smoke tests — verify lossless float/integer marshaling across boundary.

**Current blocker:** Task 1.2 — the .adb body needs rewrite using System.Address overlay pattern.

**Previous session progress (2026-04-25 earlier):**

- GNAT 13.3.0 + gprbuild confirmed working
- `.so` rebuilt with GNAT 13, links against libgnat-13.so ✅
- C-exported symbols verified with `nm -D` (10 single-underscore symbols) ✅
- BUG-001 (symbol mismatch) resolved at the spec level
- BUG-002 (libgnat-12) resolved

## Phase 2: SOTA++ Kernel Extensions

**Objective:** Move beyond the arXiv:2508.03346 paper by implementing the 6 identified SOTA++ features directly in the Ada kernel.

- **Task 2.1 (Adaptive Thresholds):** Implement a running mean/variance state machine in Ada to dynamically adjust `Threshold_Low` and `Threshold_High` per model/task, eliminating hardcoded heuristics.
- **Task 2.2 (Kernel-Side GRPO Rewards):** Port the 4-component reward function (Eq. 15) into Ada to calculate `R_skip_ratio` and penalties at bare-metal speeds during training rollouts.
- **Task 2.3 (Cross-Model Normalization):** Add z-score normalization routines to allow the entropy router to work flawlessly across DeepSeek-R1, Qwen, and Llama architectures.
- **Task 2.4 (Entropy Distributions):** Implement an `Entropy_Histogram` record type to return rich, bin-based distribution data back to Python for logging/telemetry.

## Phase 3: Garlic / Orchestration Integration

**Objective:** Plug the Ada kernel into the broader ML orchestration stack (Garlic/Lisan al-Gaib).

- **Task 3.1:** Wire the Ada kernel's `Routing_Decision` into the Thompson Contextual Bandit and Spectral Intent Decomposer.
- **Task 3.2:** Implement the `<SKIP>` token injection protocol at the Python orchestration layer based on the fast-path routing signals emitted by Ada.
- **Task 3.3:** Connect the high-entropy "Slow Path" signal to trigger the actual Hierarchical Spectral Graph Memory (HSGM) retrieval.

## Phase 4: SPARK 2014 Formal Verification (The Crown Jewel)

**Objective:** Mechanically prove the correctness and safety of the Step Entropy kernel.

- **Task 4.1:** Annotate the core math routines (`Calculate_Token_Entropy`, `Calculate_Step_Entropy`) with SPARK `Global`, `Depends`, and `Pre`/`Post` contracts.
- **Task 4.2:** Prove the absence of runtime errors (AoRE) — guaranteeing no array out-of-bounds, no divide-by-zero, and no floating-point overflows.
- **Task 4.3:** Prove functional correctness — mathematically guaranteeing that `Entropy_Value` is strictly non-negative and that probabilities normalize correctly.
- **Outcome:** Produce the world's first formally verified, mathematically guaranteed LLM reasoning compressor.
