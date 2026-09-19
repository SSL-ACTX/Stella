use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=shaders/stella_gemm.comp");
    println!("cargo:rerun-if-changed=native/mali_vk_runner.cpp");
    println!("cargo:rerun-if-changed=native/vk_probe.c");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let _target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // Build native Vulkan runner and shader binaries if native-vulkan or android-radb is active
    if env::var("CARGO_FEATURE_NATIVE_VULKAN").is_ok()
        || env::var("CARGO_FEATURE_ANDROID_RADB").is_ok()
    {
        build_artifacts(&out_dir);
    }
}

fn build_artifacts(out_dir: &Path) {
    // 1. Compile GLSL compute shader into SPIR-V using glslangValidator
    let comp_shader = Path::new("shaders/stella_gemm.comp");
    let spv_out = out_dir.join("stella_gemm.spv");

    let glsl_status = Command::new("glslangValidator")
        .args([
            "-V",
            comp_shader.to_str().unwrap(),
            "-o",
            spv_out.to_str().unwrap(),
        ])
        .status();

    match glsl_status {
        Ok(status) if status.success() => {
            println!(
                "cargo:warning=Compiled {} to {}",
                comp_shader.display(),
                spv_out.display()
            );
        }
        _ => {
            println!(
                "cargo:warning=glslangValidator not found or failed, checking precompiled fallback"
            );
        }
    }

    // 2. Compile native/mali_vk_runner.cpp
    let runner_src = Path::new("native/mali_vk_runner.cpp");
    let runner_out = out_dir.join("stella_mali_runner");

    let runner_status = Command::new("clang++")
        .args([
            "-O2",
            "-std=c++17",
            runner_src.to_str().unwrap(),
            "-o",
            runner_out.to_str().unwrap(),
        ])
        .status();

    if let Ok(status) = runner_status {
        if status.success() {
            println!("cargo:warning=Compiled {}", runner_out.display());
        }
    }

    // 3. Compile native/vk_probe.c
    let probe_src = Path::new("native/vk_probe.c");
    let probe_out = out_dir.join("stella_vk_probe");

    let probe_status = Command::new("clang")
        .args([
            "-O2",
            probe_src.to_str().unwrap(),
            "-o",
            probe_out.to_str().unwrap(),
        ])
        .status();

    if let Ok(status) = probe_status {
        if status.success() {
            println!("cargo:warning=Compiled {}", probe_out.display());
        }
    }
}
