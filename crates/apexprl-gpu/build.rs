use std::{env, path::PathBuf, process::Command};

fn compile_cubin(nvcc: &PathBuf, src: &PathBuf, arch: &str, opt: &str, out_file: &PathBuf) {
    let status = Command::new(nvcc)
        .args([
            "-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            &format!("-arch={arch}"),
            "-Xptxas",
            &format!("{opt},-v"),
            "-diag-suppress",
            "177,549",
            "--cubin",
        ])
        .arg(src)
        .arg("-o")
        .arg(out_file)
        .status()
        .unwrap_or_else(|e| panic!("Failed to invoke nvcc ({}): {e}", nvcc.display()));
    assert!(status.success(), "nvcc failed compiling {} for {arch} ({opt})", src.display());
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let kernels_dir = workspace_root.join("kernels");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed={}", kernels_dir.display());
    println!("cargo:rerun-if-changed={}", workspace_root.join("tools/opt_strassen_sm120.py").display());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUBIT_ROOT");
    println!("cargo:rerun-if-env-changed=APX_STRASSEN_SM120_CUBIN");

    let cuda_home = env::var("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let p128 = PathBuf::from("/usr/local/cuda-12.8");
            if p128.exists() { p128 } else { PathBuf::from("/usr/local/cuda") }
        });
    let nvcc = env::var("NVCC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| cuda_home.join("bin/nvcc"));

    // Uniform-3 On-Chip L2-Pinned Basis 128x256 cubins (v0.3 default engine: 0 spills, 0 GB/s DRAM at 810 MHz!)
    compile_cubin(
        &nvcc,
        &kernels_dir.join("apexprl_b128_sm89.cu"),
        "sm_89",
        "-O1",
        &out_dir.join("apexprl_b128_sm89.cubin"),
    );
    compile_cubin(
        &nvcc,
        &kernels_dir.join("apexprl_b128_sm120a.cu"),
        "sm_120a",
        "-O1",
        &out_dir.join("apexprl_b128_sm120a.cubin"),
    );

    // Hyper-L2 Direct 2-CTA/SM cubins (`--engine direct`)
    compile_cubin(
        &nvcc,
        &kernels_dir.join("apexprl_direct_sm89.cu"),
        "sm_89",
        "-O3",
        &out_dir.join("apexprl_direct_sm89.cubin"),
    );
    compile_cubin(
        &nvcc,
        &kernels_dir.join("apexprl_direct_sm120a.cu"),
        "sm_120a",
        "-O1",
        &out_dir.join("apexprl_direct_sm120a.cubin"),
    );

    // Classic On-Chip Basis 160x256 cubins (`--engine basis160`)
    compile_cubin(
        &nvcc,
        &kernels_dir.join("apexprl_sm89.cu"),
        "sm_89",
        "-O1",
        &out_dir.join("apexprl_sm89.cubin"),
    );
    compile_cubin(
        &nvcc,
        &kernels_dir.join("apexprl_sm120a.cu"),
        "sm_120a",
        "-O1",
        &out_dir.join("apexprl_sm120a.cubin"),
    );

    // Zero-Outlier Graph-Theoretic EDCP-120 2-CTA/SM cubins (`--engine edcp`)
    compile_cubin(
        &nvcc,
        &kernels_dir.join("apexprl_edcp_sm89.cu"),
        "sm_89",
        "-O3",
        &out_dir.join("apexprl_edcp_sm89.cubin"),
    );
    compile_cubin(
        &nvcc,
        &kernels_dir.join("apexprl_edcp_sm120a.cu"),
        "sm_120a",
        "-O1",
        &out_dir.join("apexprl_edcp_sm120a.cubin"),
    );

    // v0.7 On-Chip RAW-24K Basis-Strassen <2,2,2:7> cubins (`--engine strassen`, default on SM120 Blackwell!)
    compile_cubin(
        &nvcc,
        &kernels_dir.join("apexprl_strassen_sm89.cu"),
        "sm_89",
        "-O3",
        &out_dir.join("apexprl_strassen_sm89.cubin"),
    );
    let strassen_sm120_out = out_dir.join("apexprl_strassen_sm120a.cubin");
    if let Ok(prebuilt) = env::var("APX_STRASSEN_SM120_CUBIN") {
        std::fs::copy(&prebuilt, &strassen_sm120_out)
            .unwrap_or_else(|e| panic!("Failed to copy APX_STRASSEN_SM120_CUBIN ({prebuilt}): {e}"));
    } else {
        let strassen_sm120_raw = out_dir.join("apexprl_strassen_sm120a_raw.cubin");
        compile_cubin(
            &nvcc,
            &kernels_dir.join("apexprl_strassen_sm120a.cu"),
            "sm_120a",
            "-O3",
            &strassen_sm120_raw,
        );
        let opt_script = workspace_root.join("tools/opt_strassen_sm120.py");
        let status = Command::new("python3")
            .arg(&opt_script)
            .arg(&strassen_sm120_raw)
            .arg(&strassen_sm120_out)
            .arg("apx_strassen_mine_v0")
            .status()
            .unwrap_or_else(|e| panic!("Failed to invoke python3 {}: {e}", opt_script.display()));
        assert!(
            status.success(),
            "SM120 Strassen SASS optimizer/verifier failed on {}",
            strassen_sm120_raw.display()
        );
    }
}
