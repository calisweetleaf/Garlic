# FAILURE_GRAMMAR.md — What Wrong Looks Like Before It's Provable

> Load this when: something smells off, the bridge is misbehaving, Ada won't compile,
> entropy values look wrong, or confidence in results drops.
> This is NOT an error list. This is a taxonomy of pre-failure signatures.

---

## PRE-FAILURE SIGNATURES

### SIG-01: "Ada library not loaded" at bridge import

**Smell**: `RuntimeError: Ada step_entropy library not found` during `import ada_bridge`  
**What it actually is**: One of:

- `libgarlic_core.so` not in search paths
- `libgnat-12.so` (or libgnat-13.so) not found by dynamic linker
- `.so` built for wrong architecture (x86_64 vs ARM)
**Distinguish by**: `ldd lib/libgarlic_core.so` — check for `not found` dependencies  
**False signal**: The library IS there but path resolution fails → check `script_dir` in `AdaLibraryLoader.find_library`

### SIG-02: `AttributeError: step_entropy_calculate_token_entropy`

**Smell**: AttributeError when bridge tries to resolve function from library  
**What it actually is**: Symbol name mismatch — Ada uses double-underscore mangling  
**Distinguish by**: `nm -D lib/libgarlic_core.so | grep token_entropy`  
**If you see**: `step_entropy__calculate_token_entropy` (double underscore) → BUG-001, missing C-export pragmas  
**False signal**: Thinking the bridge has a typo — the bridge is correct, the Ada exports are wrong

### SIG-03: Entropy values consistently = 0.0

**Smell**: All entropy outputs are exactly 0.0  
**What it actually is**: One of:

- Logits are all the same value (uniform) → entropy is log2(vocab_size) not 0 (this is opposite)
- Logits are extreme (one token probability → 1.0) → entropy legitimately ~0 bits
- Softmax underflow: `Sum_Exp <= Epsilon` triggers uniform fallback, then entropy is computed correctly
- Bug in `Safe_Log` not being called (entropy sum stays 0.0)
**Distinguish by**: Test with `logits = [10.0] + [0.0] * (vocab_size-1)` → should give ~0 bits

### SIG-04: Entropy values are suspiciously high for confident tokens

**Smell**: A clearly peaked logit distribution gives entropy > log2(vocab_size)  
**What it actually is**: Log base not applied correctly — using natural log but reporting as bits  
**Distinguish by**: `entropy_bits = entropy_nats / log(2)` — check if Ada divides by `Log_2_Base`  
**Ada location**: `Safe_Log` base=2.0 path uses `Log(X) / Log_2_Base` ✓

### SIG-05: Routing always returns NORMAL (1) regardless of entropy

**Smell**: Batch routing returns all-1s, single routing always says Normal  
**What it actually is**: Threshold comparison using wrong value (total vs avg entropy)  
**Distinguish by**: Check if `Avg_Entropy` or `Total_Entropy` is being compared to thresholds  
**Critical**: With 10-token steps and per-token entropy ~3.0 bits, total=30 bits >> threshold_high=4.0 → everything would route to HIGH. Mean=3.0 → correctly routes to MEDIUM.

### SIG-06: `Constraint_Error` on Token_Index in `Calculate_Token_Probability`

**Smell**: Crash on `Token_Index out of vocabulary range`  
**What it actually is**: 0-indexed Python token_id passed where 1-indexed Ada index expected  
**Ada code**: `Index := Integer(Token_Index) + 1` — Ada converts 0-indexed token_id to 1-indexed array index  
**False signal**: Thinking the vocab size is wrong — it's an off-by-one on indexing convention  
**Watch for**: Any token_id >= Max_Vocab_Size (50257) will raise constraint error by design

### SIG-07: Token text truncation without warning

**Smell**: Token text is cut off at 64 characters in step analysis  
**What it actually is**: `Token_Text` is `String(1..64)` — fixed-width, silently truncates  
**This is NOT a bug**: It's baked-in Ada design. 64 chars covers all real tokenizer tokens.  
**Would be a bug if**: Passing multi-token spans as a single token_text string

### SIG-08: Compilation error: "Threshold_Low must be less than Threshold_High"

**Smell**: Ada raises `Constraint_Error` with this message  
**What it actually is**: Pre-condition violation — caller passed equal or inverted thresholds  
**Most common cause**: Default parameter shadowing — accidentally calling with `(2.0, 2.0)` or `(4.0, 2.0)`

### SIG-09: `gprbuild` fails with "project not found" or "no sources"

**Smell**: Build fails immediately after GNAT install  
**What it actually is**: One of:

- `garlic_core.gpr` not found (run from wrong directory)
- `Source_Dirs = (".")` doesn't include `.ads`/`.adb` files (they're in root, which is correct)
- GNAT 13 and GPR have different project file syntax expectations
**Fix**: Always run `gprbuild -P garlic_core.gpr` from the project root

### SIG-10: Bridge `calculate_step_entropy` returns wrong avg_entropy

**Smell**: avg_entropy doesn't match sum/count of individual token entropies  
**What it actually is**: The bridge's `calculate_step_entropy` is Python-side (hybrid model)  

- It calls Ada for each token's entropy individually
- Then aggregates in Python
- The Ada implementation aggregates differently (in-loop)
**This is by design** — but cross-validate if you suspect a bug: sum all token entropies / count == avg_entropy

---

## FALSE SUCCESS PATTERNS (The Worst Failure Mode)

### FS-01: Bridge loads without error but all FFI calls silently use fallback

**Pattern**: `_ADA_LIB = None` (import error caught and suppressed at module level)  
**Why dangerous**: Code appears to work (Python fallback paths exist for classify_entropy, analyze_chain) but Ada FFI paths are actually dead. The HIGH-VALUE Ada path (entropy calculation) has no Python fallback — those will crash on actual use.  
**Where in code**: Lines 173-177 of `ada_bridge.py` — `_ADA_LIB = None` set silently  
**Detection**: `from ada_bridge import _ADA_LIB; assert _ADA_LIB is not None`

### FS-02: `Analyze_Chain` and `Classify_Entropy` appear to work with Ada loaded

**Pattern**: These functions have Python fallback paths even when Ada is loaded (they fall through to Python logic)  
**Why dangerous**: Masks FFI failures. Even if Ada is broken, classify/analyze appear functional.  
**Detection**: Explicitly verify Ada path: check `_ADA_LIB is not None` before trusting results

### FS-03: Entropy looks correct but is computed with natural log

**Pattern**: Results are plausible but systematically ~1.44x higher than expected  
**Why dangerous**: log_e / log_2 = 1.4427... — values look sane but are in nats not bits  
**Detection**: Test with uniform distribution over 2 classes → should give exactly 1.0 bit  
  `logits = [1.0, 1.0] + [float('-inf')] * (vocab_size-2)` → H = 1.0 bit

### FS-04: Pruning indices look sorted but are shuffled after sort

**Pattern**: `Recommend_Pruning` returns indices that look reasonable but aren't the true lowest-entropy steps  
**Why dangerous**: The bubble sort in the function is O(n²) and correct, but the `Sort_By_Entropy` generic sort happens on the `Indexed_Entropy` array. If the indexed array was built with wrong base indices, pruning targets wrong steps.  
**Detection**: Verify that returned indices correspond to minimum avg_entropy steps

---

## RECOVERY PROTOCOLS

### When Ada won't compile (after GNAT install)

1. `gnat --version` — confirm GNAT 13 is active
2. Check `garlic_core.gpr` syntax — GNAT 13 may have different project format requirements
3. Remove `obj/` directory and rebuild clean: `rm -rf obj/* && gprbuild -P garlic_core.gpr`
4. Check `step_entropy.adb.stderr` in `obj/` for detailed error messages
5. Verify `-gnat2022` flag is supported by GNAT 13 (it is — Ada 2022 is supported)

### When bridge fails to load

1. `ldd lib/libgarlic_core.so` — find missing deps
2. `nm -D lib/libgarlic_core.so | grep " T "` — list all exported symbols
3. Compare against what bridge expects (`grep "ada_lib\." ada_bridge.py`)
4. Recompile after adding C-export pragmas

### When entropy values are wrong

1. Test with known inputs: uniform distribution → H = log2(vocab_size)
2. Test with peaked distribution: one logit=10.0, rest=0.0 → H ≈ 0.0 bits
3. Test with 2-class uniform → H = exactly 1.0 bit
4. Compare Ada result with Python reference: `original-code/step_entropy.py calculate_token_entropy()`

---

## REJECTION CRITERIA (Three Strikes)

The system fails hard if any of these are true:

1. **STRIKE 1**: `_ADA_LIB is None` after bridge import — Ada is not operational
2. **STRIKE 2**: Any token entropy is < 0.0 — mathematical invariant violated
3. **STRIKE 3**: `avg_entropy` for a routing decision is out of range [0, log2(50257)] ≈ [0, 15.6] — invalid entropy value in use


---

## 2026-04-26 ADDENDUM — System-Router Failure Signatures

### SIG-SR-01: Router tests pass while Ada native path is unavailable

**Smell:** `System-Router/tests/test_system_router.py` passes but `System-Router/system_router.py::ADA_AVAILABLE` is false.  
**Likely cause:** Python entropy fallback was used.  
**Recovery:** Run `.venv/bin/python ada_bridge.py` at root, then verify nested `System-Router/ada_bridge` import path and assert `ADA_AVAILABLE=True`.

### SIG-SR-02: `SystemOutput.trace` lacks high-value stage evidence

**Smell:** Output returns a prompt but trace only shows template/model-slot fields.  
**Likely cause:** No `hidden_states`, no `logits`, or disabled flags.  
**Recovery:** Provide `hidden_states`, `logits`, and `context_metadata['token_ids']`; verify stage 2/8/9/10/11/13 behavior.

### SIG-SR-03: Constitution/runtime mismatch

**Smell:** An agent starts copying all Garlic components into `system_router.py` because `SYSTEM_ROUTER_CONSTITUTION.md` describes a monolithic target.  
**Likely cause:** Treating target architecture as current runtime truth.  
**Recovery:** Read `SYSTEM_ROUTER_TOPOLOGY.md`; inspect live imports and tests before editing.

### SIG-SR-04: Root/nested bridge drift

**Smell:** Root `ada_bridge.py` passes but SystemRouter entropy behavior differs or falls back.  
**Likely cause:** `System-Router/ada_bridge/` copy drifted from root.  
**Recovery:** Diff root bridge/Ada files against nested bridge; sync deliberately; rerun root smoke and router tests.
