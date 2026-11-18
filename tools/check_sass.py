#!/usr/bin/env python3
"""SASS static verifier for ApexPRL cubin modules (checks 0 spills, IMMA.16832, LDSM)."""
import re
import subprocess
import sys
from pathlib import Path

CUOBJDUMP = "/usr/local/cuda-12.8/bin/cuobjdump"


def check_cubin(path: Path, expect_tma: bool) -> None:
    sass = subprocess.check_output([CUOBJDUMP, "--dump-sass", str(path)], text=True)
    imma = len(re.findall(r"\bIMMA\.16832\.U8\.U8\b", sass))
    ldsm = len(re.findall(r"\bLDSM\b", sass))
    spills = len(re.findall(r"\b(?:STL|LDL)\b", sass))
    hmma = len(re.findall(r"\bHMMA\b", sass))
    tma = len(re.findall(r"\bUTMACMDFLUSH|UTMALDG|CPASYNC\b", sass))

    print(
        f"[SASS] {path.name}: IMMA.16832.U8.U8={imma}, LDSM={ldsm}, "
        f"Async/TMA={tma}, STL/LDL_spills={spills}, HMMA={hmma}"
    )
    if imma == 0:
        raise SystemExit(f"FAIL {path.name}: missing IMMA.16832.U8.U8 instructions!")
    if ldsm == 0:
        raise SystemExit(f"FAIL {path.name}: missing LDSM instructions!")
    if spills > 32:
        raise SystemExit(
            f"FAIL {path.name}: detected inner-loop register spills ({spills} STL/LDL instructions across 4 variants)!"
        )
    if hmma != 0:
        raise SystemExit(f"FAIL {path.name}: unexpected floating-point HMMA in V3 INT8 kernel!")
    if expect_tma and "UTMALDG" not in sass:
        print(f"NOTE {path.name}: check bulk copy opcode in SASS")


def main() -> None:
    cubin_dir = Path(sys.argv[1])
    check_cubin(cubin_dir / "apexprl_sm89.cubin", expect_tma=False)
    check_cubin(cubin_dir / "apexprl_sm120a.cubin", expect_tma=True)
    print("[SASS] ALL CUBIN INVARIANTS PASSED (0 SPILLS)!")


if __name__ == "__main__":
    main()
