# Contributing to ADA-Step-Entropy

First off, thank you for considering contributing to ADA-Step-Entropy! 

This repository operates under a strict cognitive topology and engineering doctrine to ensure that the mathematical formalism of the **Step Entropy** paper (`arXiv:2508.03346`) remains flawlessly represented in our **Ada 2022 Kernel**.

## Engineering Doctrine

1. **Ada 2022 is the Kernel:** All core logic, memory manipulations, and numerical stabilization must happen in Ada. Do not implement these in the Python FFI wrapper.
2. **Contracts First:** Any new functionality added to the Ada Kernel (`step_entropy.ads` / `step_entropy.adb`) must be defended by `Pre`, `Post`, and `Dynamic_Predicate` contracts. Failures should be loud, hard, and happen before bad data crosses the ABI boundary.
3. **Paper as Ground Truth:** `original-code/step_entropy.py` and the `Step-Entropy.pdf` are the absolute truth. The Ada code must match these perfectly or explicitly document justified deltas inside the Ada spec files.
4. **Run in the `.venv`:** Ensure all Python bridge and System-Router testing occurs within the `/home/daeron/Projects/ADA-Step-Entropy/.venv/` environment.

## Submitting Pull Requests

1. **Check the Trackers:** Ensure an issue isn't already open. If one is, comment your intent to help.
2. **Read the Docs:** Familiarize yourself with `AGENTS.md` and `SYSTEM_ROUTER_CONSTITUTION.md` before architecture-level changes.
3. **Pass the Tests:** 
   - Root Ada bridge smoke tests (`.venv/bin/python ada_bridge.py`) must pass (16/16).
   - System-Router tests must pass (`.venv/bin/python -m pytest System-Router/tests/test_system_router.py -q`).
4. **Draft the PR:** Clearly describe what was changed, referencing specific theorems or sections of the Step-Entropy paper if applicable.

## Reporting Bugs

Please use the provided issue templates. When submitting a bug:
- Provide the exact Git commit hash you are running.
- Detail the exception traceback or Ada assertion failure.
- If it is a routing logic failure, explain why the Mean Step Entropy calculation led to an incorrect `[Fast/Normal/Slow]` classification.
- Consult `FAILURE_GRAMMAR.md` to see if your bug matches known failure topologies.

## Autonomous Agents

If you are an autonomous agent operating in this repository, **you must read `AGENTS.md`** first. Never use `cat`, always utilize your execution tools in the correct `.venv`, and make sure you understand the `C ABI` before touching `.ads` files.
