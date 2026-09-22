# ADA-Step-Entropy Canonical Extraction

This package is a clean extraction of the latest Ada Step Entropy implementation
present in `garlic-ADA-step-entropy.zip`. It intentionally removes the
System-Router integration tree, historical ZIPs, duplicate source copies,
compiler intermediates, reports, prompt templates, caches, and unrelated Garlic
components.

## Canon selected

The source authority in the supplied archive is the project-root implementation:

- `step_entropy.ads` — **v2.1**, last updated 2026-05-24
- `step_entropy.adb` — status **v2.1**, last updated 2026-05-24
- `step_entropy_c_api.ads` / `.adb` — production C ABI layer v2.0, unchanged from
  the earlier System-Router copy
- `garlic_core.gpr` — GNAT/GPR dynamic-library build
- `ada_bridge.py` — Python ctypes bridge, Production v3.0
- `lib/libgarlic_core.so` — newest prebuilt native library in the supplied archive,
  dated 2026-05-24

The nested `System-Router/ada_bridge/step_entropy.ads` and `.adb` are older than
these root sources. The C API, GPR project, and Python bridge are hash-identical
between root and nested copies, but the core Ada v2.1 source is newer at root.
The root native library is also newer and differs from the nested May-05 build.

## Version caveat

The core Ada package was advanced to v2.1 on 2026-05-24, while the stable C ABI
introspection function still returns `2.0.0` (`20000`). That is how the supplied
working tree is encoded; this extraction does not rewrite provenance or bump the
ABI version.

## Build

Requires GNAT / gprbuild with Ada 2022 support.

```bash
gprbuild -P garlic_core.gpr
```

The project writes objects to `obj/` and the shared library to `lib/`.

## Smoke test

The Python bridge requires NumPy:

```bash
python3 -m pip install -r requirements.txt
python3 ada_bridge.py
```

The supplied repository's recorded verification log is preserved under
`verification/recorded-success-2026-05-24.md`. It records a clean native build
and all 16 bridge smoke tests passing. The original System-Router regression
suite is intentionally not included because this package is the Ada subsystem,
not the router.

## What was deliberately excluded

- `System-Router/`
- `obj/` and `.ali` / `.o` compiler intermediates
- historical ZIP archives
- prompt templates and router documentation
- the original Python-only reference implementation
- unrelated Garlic components

This leaves one source tree, one ABI layer, one bridge, and one native artifact.
