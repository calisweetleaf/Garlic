daeron@daeron-hpenvyx3602in1laptop15ey0xxx:~/Projects/ADA-Step-Entropy$ source .venv/bin/activate
.venv/bin/python ada_bridge.py
.venv/bin/python -m pytest System-Router/tests/test_system_router.py -q
======================================================================
ADA-Step-Entropy FFI Bridge — Production Smoke Test
======================================================================

Library version: 2.0.0
Max vocab size: 262144
Max tokens/step: 512

[Test 1] Peaked logits entropy: 0.6899 bits (expect ~0)
  [PASS] Test 1 — ent=0.6899
[Test 2] Uniform logits entropy: 9.9659 bits (expect ~9.97)
  [PASS] Test 2 — ent=9.9659
[Test 3] Peaked token 0 probability: 0.956596 (expect ~1.0)
  [PASS] Test 3 — prob=0.956596
[Test 4] Classify: 1.0→LOW, 3.0→MEDIUM, 5.0→HIGH
  [PASS] Test 4
[Test 5] Step entropy: total=11.35, avg=3.78, level=MEDIUM
  [PASS] Test 5
[Test 6] Route: path=NORMAL, retrieve_memory=False
  [PASS] Test 6
[Test 7] Batch route: ['FAST', 'NORMAL', 'SLOW', 'FAST', 'SLOW']
  [PASS] Test 7
[Test 8] Chain: 5 steps, compression=40.0%, low=2, med=1, hi=2
  [PASS] Test 8
[Test 9] Prune indices (60%): [0, 1, 3]
  [PASS] Test 9
[Test 10] Histogram: bins=[1, 1, 0, 1, 1, 1, 0, 2], mean=3.000, std=1.472
  [PASS] Test 10 — total=7, bin_sum=7
[Test 11] Normalize(3.0, mean=2.5, std=1.0): 0.5000 (expect 0.5)
  [PASS] Test 11 — norm=0.5000
[Test 11b] Normalize(std=0): 3.0000 (expect 3.0 — raw passthrough)
  [PASS] Test 11b — norm=3.0
[Test 12] GRPO: r_skip_ratio=1.0, r_skip_num=0.0, r_response_len=0.0, r_total=1.0
  [PASS] Test 12 — r_total=1.0
[Test 12b] GRPO penalties: r_total=-2.0 (expect -2.0)
  [PASS] Test 12b — r_total=-2.0
[Test 12c] GRPO mid tier: r_skip_ratio=0.5 (expect 0.5)
  [PASS] Test 12c — r_skip_ratio=0.5
INFO: AdaptiveThresholdManager initialized (σ: [-1.0, 1.0])
[Test 13] Adaptive: count=7, mean=3.000, std=1.323
         Thresholds: low=1.677, high=4.323
  [PASS] Test 13 — count=7, lo=1.677, hi=4.323
INFO: AdaptiveThresholdManager initialized (σ: [-0.5, 0.5])
  [PASS] Test 13b — pre=3, post=0
[Test 13b] ATM reset: count before=3, after=0
INFO: GarlicAdaStepEntropy initialized (thresholds: 2.0/4.0)
INFO: compress_chain: 2 steps → 1 pruned (50.0%)
[Test 14] compress_chain: compressed=['[SKIP]', 'informative'], pruned=[0]
  [PASS] Test 14 — compressed=['[SKIP]', 'informative']
INFO: GarlicAdaStepEntropy initialized (thresholds: 2.0/4.0)
[Test 15] Garlic recommend_pruning(40%): [0, 3]
  [PASS] Test 15
INFO: AdaptiveThresholdManager initialized (σ: [-1.0, 1.0])
INFO: AdaptiveThresholdManager initialized (σ: [-1.0, 1.0])
[Test 16] Input validation: all ValueError/RuntimeError fired correctly
  [PASS] Test 16

======================================================================
ALL TESTS PASSED — Ada FFI bridge is OPERATIONAL
======================================================================
................................                  [100%]
32 passed in 2.97s
(.venv) daeron@daeron-hpenvyx3602in1laptop15ey0xxx:~/Projects/ADA-Step-Entropy$ ^C
(.venv) daeron@daeron-hpenvyx3602in1laptop15ey0xxx:~/Projects/ADA-Step-Entropy$  source /home/daeron/Projects/ADA-