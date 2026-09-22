# CONTEXT.md — Current State

> Ephemeral repo state. Update when runtime truth changes.
> Updated: 2026-04-26 by Codex CTM learning pass.

---

## CURRENT STATUS

**Phase:** ADA-Step-Entropy SOTA+++ kernel operational + System-Router composed integration present.  
**Blocking Issues:** No native Ada bridge blocker found in current verification.  
**Primary current task lane:** documentation/topology lock for ADA-Step-Entropy + System-Router integration.

### Verified commands from 2026-04-26 pass

All commands were run from `/home/daeron/Projects/ADA-Step-Entropy` using MCP execution tools. Python commands used `.venv`.

| Check | Command | Result |
|---|---|---|
| Host | `uname -a` | Linux Ubuntu 24.04 x86_64 |
| Python | `.venv/bin/python --version` | Python 3.12.3 |
| Python deps | `.venv/bin/python -c "import sys,numpy,torch; ..."` | numpy 2.4.4, torch 2.11.0+cu130 |
| GNAT | `gnat --version` | GNAT 13.3.0 |
| gprbuild | `gprbuild --version` | GPRBUILD Pro 18.0w |
| Native library deps | `ldd lib/libgarlic_core.so` | links to `/lib/x86_64-linux-gnu/libgnat-13.so` |
| C symbols | `nm -D lib/libgarlic_core.so` | single-underscore C API symbols present |
| Ada rebuild | `gprbuild -P garlic_core.gpr` | success |
| Root Ada smoke | `.venv/bin/python ada_bridge.py` | 16/16 PASS |
| Router tests | `.venv/bin/python -m pytest System-Router/tests/test_system_router.py -q` | 32 passed |

### Git status before doc edits

Before this CTM pass edited docs, git showed:

```text
M 4-26-filetree-current-with-router.md
?? 4-26-2026-garlic-router.zip
```

Those were pre-existing in the working tree and were not created by this pass.

---

## CURRENT NAVIGATION SURFACES

Load order for future agents:

1. `AGENTS.md` — canonical entry point.
2. `4-26-filetree-current-with-router.md` — current file tree and navigation anchor.
3. `CONTEXT.md` — this current-state file.
4. `SYSTEM_ROUTER_TOPOLOGY.md` — new CTM for System-Router + Ada integration.
5. `TOPOLOGY.md` — Ada Step Entropy kernel CTM.
6. `FAILURE_GRAMMAR.md` — failure signatures.
7. `MEMORY.md` — durable decisions and known deltas.
8. `SYSTEM_ROUTER_CONSTITUTION.md` — architecture constitution / target design record.

---

## WHAT CHANGED IN THIS CTM PASS

- Redeveloped `README.md` using Defense-Grade Documentation Engine (`/doc-engine`) standards, incorporating Mermaid diagrams, executive summaries, and formal architecture models.
- Integrated **Google Jules** autonomous coding agent protocols directly into `AGENTS.md` and repository standards. Jules uses standard `Makefile` protocols (`make test`) and requires no standalone config files.
- Added `CONTRIBUTING.md`, `LICENSE`, and GitHub PR/Issue templates for professional engineering governance.
- Created `SYSTEM_ROUTER_TOPOLOGY.md` as the current System-Router + Ada integration topology map.
- Updated `AGENTS.md` to remove stale “bridge broken” framing and include System-Router navigation.
- Replaced root `CLAUDE.md` and `.claude/CLAUDE.md` with small mirrors pointing back to canonical `AGENTS.md`, reducing doctrine drift.
- Appended topology/failure-memory corrections to local docs.
- Wrote validation report and JSON manifest under `reports/`.

---

## CURRENT ARCHITECTURE TRUTH

### Ada side

- Core math is in `step_entropy.ads` / `step_entropy.adb`.
- C ABI is in `step_entropy_c_api.ads` / `step_entropy_c_api.adb`.
- Build config is `garlic_core.gpr`; `Library_Interface` includes both `Step_Entropy` and `Step_Entropy_C_API`.
- Native library is `lib/libgarlic_core.so`.
- Root Python bridge is `ada_bridge.py` and is operational.

### System-Router side

- Current navigation tree is in `4-26-filetree-current-with-router.md`.
- `System-Router/system_router.py` is a composed integration file, not a fully monolithic internalization.
- It imports:
  - `System-Router/neural_router.py` for ModelSlot, template, and neural routing primitives.
  - `System-Router/Garlic-Components/*` for HSGM/scaffolding/router/injection modules.
  - `System-Router/ada_bridge` for the nested Ada bridge.
- Tests currently cover ModelSlot regression, GlobalGraphMemory, StepEntropyInterface, SOTA++ components, and template-only SystemRouter integration.

---

## KNOWN RISKS / NEXT CHECKS

1. **Root/nested bridge parity:** root `ada_bridge.py` and `System-Router/ada_bridge/ada_bridge.py` appear same line count but should be checksum/diff verified before release claims.
2. **Fallback false success:** SystemRouter can use Python entropy fallback if nested Ada import fails. Native readiness claims must assert `ADA_AVAILABLE=True` and run the root/nested bridge path.
3. **Hidden-state path coverage:** Current test suite passes 32 tests, but many are small-dim/template/fallback-aware. Add a full-stage test with `hidden_states`, `logits`, `token_ids`, semantic signals, HSGM, expert routing, injection, and GlobalGraphMemory update.
4. **Constitution/runtime distinction:** `SYSTEM_ROUTER_CONSTITUTION.md` is a target architecture + plan; live code currently composes modules rather than fully internalizing everything in one giant file.
5. **Stale old text:** if any old doc still says BUG-001 is blocking, treat this as stale unless a fresh native smoke test fails.

---

## IMMEDIATE NEXT TASKS

### TASK-1: Diff root vs nested Ada bridge

Use MCP-only execution:

```bash
diff -ru \
  --exclude='__pycache__' \
  ada_bridge.py step_entropy.ads step_entropy.adb step_entropy_c_api.ads step_entropy_c_api.adb garlic_core.gpr \
  System-Router/ada_bridge/
```

If this command shape is awkward, compare file-by-file.

### TASK-2: Add a full-stage SystemRouter integration test

Goal: validate that with `hidden_states`, `logits`, and `token_ids`, the high-value path engages:

- HSGM compression creates `hsgm_summary`.
- Step entropy produces nonzero `entropy_value`.
- Routing decision attempts stage 10.
- Garlic injection produces `augmented_states` when semantic signal path succeeds.
- GlobalGraphMemory `turn_count` increments.

### TASK-3: Split or annotate `SYSTEM_ROUTER_CONSTITUTION.md`

Preserve it, but add a short “runtime truth vs target architecture” header so agents stop treating planned 3500-4500 line monolith as current file reality.

### TASK-4: Optional Codex skill package

Once CTM stabilizes, create a local skill such as `ada-step-entropy-system-router` that loads `AGENTS.md`, `SYSTEM_ROUTER_TOPOLOGY.md`, and task-relevant code only. Do not auto-load all docs.

---

## HISTORICAL NOTE

The 2026-04-25 state files correctly captured the initial bridge blockers and C ABI plan at the time. The 2026-04-26 verification supersedes those blocker claims for this snapshot: native Ada FFI is now operational and tested.


## ADDITIONAL CTM ASSET CREATED

A dedicated local Codex skill was created outside the repo at:

`/home/daeron/.codex/skills/custom/ada-step-entropy-system-router/SKILL.md`

It is intentionally thin and conditionally loads repo docs instead of auto-loading the whole codebase.
