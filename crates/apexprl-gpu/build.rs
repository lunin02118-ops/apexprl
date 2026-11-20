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
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");

    let cuda_home = env::var("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let p128 = PathBuf::from("/usr/local/cuda-12.8");
            if p128.exists() { p128 } else { PathBuf::from("/usr/local/cuda") }
        });
    let nvcc = env::var("NVCC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| cuda_home.join("bin/nvcc"));

    // Hyper-L2 Direct 2-CTA/SM cubins (v0.2 default engine!)
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

    // Classic On-Chip Basis 160x256 cubins (`--engine basis`)
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
}
