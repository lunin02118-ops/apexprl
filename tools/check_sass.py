#!/usr/bin/env python3
"""SASS static verifier for ApexPRL v0.4 cubin modules (strict 0 spills + 0 BRX on Basis-128, Direct-L2, & EDCP-120)."""
import re
import subprocess
import sys
from pathlib import Path

CUOBJDUMP = "/usr/local/cuda-12.8/bin/cuobjdump"


def check_cubin(
    path: Path,
    expect_tma: bool,
    max_spills: int,
    max_warpsync: int | None = None,
    max_brx: int | None = None,
) -> None:
    sass = subprocess.check_output([CUOBJDUMP, "--dump-sass", str(path)], text=True)
    imma = len(re.findall(r"\bIMMA\.(?:16832|16816)\.(?:U8\.U8|S8\.S8)\b", sass))
    ldsm = len(re.findall(r"\bLDSM\b", sass))
    spills = len(re.findall(r"\b(?:STL|LDL)\b", sass))
    brx = len(re.findall(r"\bBRX\b", sass))
    hmma = len(re.findall(r"\bHMMA\b", sass))
    tma = len(re.findall(r"\b(?:UBLKCP|UTMACMDFLUSH|UTMALDG|LDGSTS)\b", sass))
    mine_bodies = [m.group(1) for m in re.finditer(r"Function : apx_b128_mine_v\d+\b(.*?)(?=Function : |\Z)", sass, re.S)]
    ws = sum(len(re.findall(r"\bWARPSYNC\b", b)) for b in mine_bodies) if mine_bodies else len(re.findall(r"\bWARPSYNC\b", sass))

    print(
        f"[SASS] {path.name:28s}: IMMA={imma:4d}, LDSM={ldsm:3d}, "
        f"Async/TMA={tma:3d}, WARPSYNC={ws:3d}, BRX={brx:2d}, STL/LDL={spills:2d} (lim={max_spills}), HMMA={hmma}"
    )
    if imma == 0:
        raise SystemExit(f"FAIL {path.name}: missing IMMA.16832/16816 instructions!")
    if ldsm == 0:
        raise SystemExit(f"FAIL {path.name}: missing LDSM instructions!")
    if spills > max_spills:
        raise SystemExit(
            f"FAIL {path.name}: detected register spills ({spills} STL/LDL > limit {max_spills})!"
        )
    if max_brx is not None and brx > max_brx:
        raise SystemExit(
            f"FAIL {path.name}: detected indirect branch jump table ({brx} BRX > limit {max_brx})!"
        )
    if max_warpsync is not None and ws > max_warpsync:
        raise SystemExit(
            f"FAIL {path.name}: unexpected WARPSYNC count ({ws} > limit {max_warpsync})!"
        )
    if hmma != 0:
        raise SystemExit(f"FAIL {path.name}: unexpected floating-point HMMA in V3 INT8 kernel!")
    if expect_tma and tma == 0:
        raise SystemExit(f"FAIL {path.name}: expected Blackwell TMA (UBLKCP/UTMALDG) in SASS!")


def main() -> None:
    cubin_dir = Path(sys.argv[1])
    # v0.5 Persistent-A Unpredicated Basis-128x256 (DEFAULT): STRICT 0 spills, STRICT 0 BRX, 0 inner-loop WARPSYNC!
    check_cubin(cubin_dir / "apexprl_b128_sm89.cubin", expect_tma=False, max_spills=0, max_warpsync=12, max_brx=0)
    check_cubin(cubin_dir / "apexprl_b128_sm120a.cubin", expect_tma=True, max_spills=0, max_warpsync=24, max_brx=0)
    # v0.5 Zero-Outlier EDCP-120 Cycle-Packing: STRICT 0 spills, STRICT 0 BRX!
    check_cubin(cubin_dir / "apexprl_edcp_sm89.cubin", expect_tma=False, max_spills=0, max_brx=0)
    check_cubin(cubin_dir / "apexprl_edcp_sm120a.cubin", expect_tma=True, max_spills=0, max_brx=0)
    # v0.5 Zero-Eviction Hyper-L2 Direct engines: STRICT 0 register spills, STRICT 0 BRX!
    check_cubin(cubin_dir / "apexprl_direct_sm89.cubin", expect_tma=False, max_spills=0, max_brx=0)
    check_cubin(cubin_dir / "apexprl_direct_sm120a.cubin", expect_tma=True, max_spills=0, max_brx=0)
    # v0.1 legacy Basis-160x256 engines
    check_cubin(cubin_dir / "apexprl_sm89.cubin", expect_tma=False, max_spills=32)
    check_cubin(cubin_dir / "apexprl_sm120a.cubin", expect_tma=False, max_spills=32)
    print("[SASS] ALL 8 CUBIN INVARIANTS PASSED (0 SPILLS & 0 BRX ON ALL v0.5 BASIS-128, EDCP-120 & DIRECT-L2 KERNELS)!")


if __name__ == "__main__":
    main()
