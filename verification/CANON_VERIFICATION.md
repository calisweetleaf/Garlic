# Canon verification snapshot

## Selected source hashes
```
7a86b65e56a79aa595728847d95586953567cb03dfaf1d8f9e8eb79caeb3110c  step_entropy.ads
8184dfe930d098a37e4f424f3dfa9de37862be5445699b7bab90fade089edd58  step_entropy.adb
f67c1b18ed4a626ace34bb687315a00c7ae7071dab3968af92d3e2dd05ae6445  step_entropy_c_api.ads
232b34cc6c284bc870a400b85c047c41d013d55324c2e1af0fe339ed6b18ee29  step_entropy_c_api.adb
54e1d32c295cff7b32d5ddfa2a97656cf0cafe71aa6b4d4bfb713f7d53ccba61  garlic_core.gpr
4351aedc7597571cdf79a2b61b1a9d7f248b65fef82ee7961720878620b44083  ada_bridge.py
05190f2a10d8940a9c297e344ee0865e9d3158b10903d5a73898d52398b36ed2  lib/libgarlic_core.so
```

## Native artifact identity
```
/mnt/data/ADA-Step-Entropy-CANON-v2.1/lib/libgarlic_core.so: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, BuildID[sha1]=2b51f4534433f6bf60437bdcdef43bf4b6c43b9d, not stripped
```

## Exported C ABI symbols
```
step_entropy_analyze_chain
step_entropy_batch_route
step_entropy_calculate_token_entropy
step_entropy_calculate_token_probability
step_entropy_classify_entropy
step_entropy_compute_grpo_rewards
step_entropy_compute_histogram
step_entropy_get_adaptive_thresholds
step_entropy_get_max_tokens_per_step
step_entropy_get_max_vocab_size
step_entropy_get_version
step_entropy_normalize_entropy
step_entropy_recommend_pruning
step_entropy_route_step
step_entropy_update_adaptive_state
```

## Extraction decision
- Root `step_entropy.ads` SHA256: 7a86b65e56a79aa595728847d95586953567cb03dfaf1d8f9e8eb79caeb3110c
- Nested `System-Router/ada_bridge/step_entropy.ads` SHA256: 2357d885646e9dfe0ecad04d6d0c3e3286622fe2eb7b1869f0ede7badcdab318
- Root `step_entropy.adb` SHA256: 8184dfe930d098a37e4f424f3dfa9de37862be5445699b7bab90fade089edd58
- Nested `System-Router/ada_bridge/step_entropy.adb` SHA256: 59460926b902e1d6b75b6eb3df81f7cb67b729e5d431642799fc4fe9e0bf9283
- Root and nested `ada_bridge.py`, C API sources, and `garlic_core.gpr` were hash-identical in the supplied archive.
- The selected root shared library contains the `step_entropy__compression_ratio_valuePredicate` symbol introduced by the v2.1 core spec.
