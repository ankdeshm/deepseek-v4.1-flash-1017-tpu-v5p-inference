#!/usr/bin/env python3
"""Optional v5p host-RAM patch: skip TpuCompilerAdaptor serialize() when the vLLM compile cache is disabled.

With VLLM_DISABLE_COMPILE_CACHE=1 the serialized bundled AOTAutograd entry (~1-2 GiB per bucket per
rank) is never written, but `_run_backend` still builds it, inflating each worker's RSS by ~4.5 GiB per
compiled bucket on v5p (glibc does not return the freed arena memory). Idempotent.
"""

from __future__ import annotations

import pathlib
import sys

P = pathlib.Path("/workspace/vllm-torchtpu/src/vllm_torchtpu/compilation/tpu_compiler.py")
OLD = """        entry = None
        serialize = getattr(compiled_fn, "serialize", None)
        if serialize is not None:"""
NEW = """        entry = None
        serialize = getattr(compiled_fn, "serialize", None)
        if getattr(self, "_disable_cache", False):  # V5P_SKIP_SERIALIZE_WHEN_CACHE_DISABLED
            serialize = None
        if serialize is not None:"""


def main() -> int:
    txt = P.read_text()
    if "V5P_SKIP_SERIALIZE_WHEN_CACHE_DISABLED" in txt:
        print("already patched")
        return 0
    if OLD not in txt:
        print("anchor not found; not patching", file=sys.stderr)
        return 1
    P.write_text(txt.replace(OLD, NEW, 1))
    print("patched", P)
    return 0


if __name__ == "__main__":
    sys.exit(main())
