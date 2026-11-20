#!/usr/bin/env python3
"""SASS static verifier for ApexPRL v0.2 cubin modules (strict 0 spills on Direct-L2, IMMA.16832, LDSM, TMA)."""
import re
import subprocess
import sys
from pathlib import Path

CUOBJDUMP = "/usr/local/cuda-12.8/bin/cuobjdump"


def check_cubin(path: Path, expect_tma: bool, max_spills: int) -> None:
    sass = subprocess.check_output([CUOBJDUMP, "--dump-sass", str(path)], text=True)
    imma = len(re.findall(r"\bIMMA\.16832\.U8\.U8\b", sass))
    ldsm = len(re.findall(r"\bLDSM\b", sass))
    spills = len(re.findall(r"\b(?:STL|LDL)\b", sass))
    hmma = len(re.findall(r"\bHMMA\b", sass))
    tma = len(re.findall(r"\b(?:UBLKCP|UTMACMDFLUSH|UTMALDG|LDGSTS)\b", sass))

    print(
        f"[SASS] {path.name}: IMMA.16832.U8.U8={imma}, LDSM={ldsm}, "
        f"Async/TMA={tma}, STL/LDL_spills={spills} (limit={max_spills}), HMMA={hmma}"
    )
    if imma == 0:
        raise SystemExit(f"FAIL {path.name}: missing IMMA.16832.U8.U8 instructions!")
    if ldsm == 0:
        raise SystemExit(f"FAIL {path.name}: missing LDSM instructions!")
    if spills > max_spills:
        raise SystemExit(
            f"FAIL {path.name}: detected register spills ({spills} STL/LDL > limit {max_spills})!"
        )
    if hmma != 0:
        raise SystemExit(f"FAIL {path.name}: unexpected floating-point HMMA in V3 INT8 kernel!")
    if expect_tma and tma == 0:
        raise SystemExit(f"FAIL {path.name}: expected Blackwell TMA (UBLKCP/UTMALDG) in SASS!")


def main() -> None:
    cubin_dir = Path(sys.argv[1])
    # v0.2 Hyper-L2 Direct engines: STRICT 0 register spills enforced!
    check_cubin(cubin_dir / "apexprl_direct_sm89.cubin", expect_tma=False, max_spills=0)
    check_cubin(cubin_dir / "apexprl_direct_sm120a.cubin", expect_tma=True, max_spills=0)
    # v0.1 legacy Basis-160x256 engines
    check_cubin(cubin_dir / "apexprl_sm89.cubin", expect_tma=False, max_spills=32)
    check_cubin(cubin_dir / "apexprl_sm120a.cubin", expect_tma=False, max_spills=32)
    print("[SASS] ALL 4 CUBIN INVARIANTS PASSED (0 SPILLS ON ALL DIRECT-L2 KERNELS)!")


if __name__ == "__main__":
    main()
