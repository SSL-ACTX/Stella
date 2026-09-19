// crates/stella_gpu/src/backends/native_vulkan.rs
//! Desktop / Server Native Vulkan Backend (Linux, Windows, macOS via MoltenVK).
//!
//! Executes compiled Vulkan compute shader kernels directly on the local host GPU.

use crate::traits::{ComputeDimensions, ComputeResult, GpuComputeBackend};
use std::process::Command;

const RUNNER_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stella_mali_runner"));
const PROBE_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stella_vk_probe"));
const SHADER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stella_gemm.spv"));

pub struct NativeVulkanBackend;

impl NativeVulkanBackend {
    pub fn new() -> Self {
        Self
    }

    fn ensure_runner_extracted(&self) -> Result<std::path::PathBuf, String> {
        let temp_dir = std::env::temp_dir().join(".stella_vk");
        let _ = std::fs::create_dir_all(&temp_dir);

        let runner_path = temp_dir.join("stella_mali_runner");
        let probe_path = temp_dir.join("stella_vk_probe");
        let shader_path = temp_dir.join("stella_gemm.spv");

        if !runner_path.exists() {
            let _ = std::fs::write(&runner_path, RUNNER_BINARY);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ =
                    std::fs::set_permissions(&runner_path, std::fs::Permissions::from_mode(0o755));
            }
        }

        if !probe_path.exists() {
            let _ = std::fs::write(&probe_path, PROBE_BINARY);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ =
                    std::fs::set_permissions(&probe_path, std::fs::Permissions::from_mode(0o755));
            }
        }

        if !shader_path.exists() {
            let _ = std::fs::write(&shader_path, SHADER_SPV);
        }

        Ok(runner_path)
    }

    fn parse_runner_output(&self, raw: &str) -> Result<ComputeResult, String> {
        let mut device_name = "Desktop/Server Native Vulkan Compute GPU".to_string();
        let mut exec_time_ms = 0.0;
        let mut giga_macs = 0.0;
        let mut agent_vel = 0.0;
        let mut sample_out = 0.0;

        for line in raw.lines() {
            let line = line.trim();
            if line.starts_with("Device") {
                if let Some(pos) = line.find(':') {
                    device_name = line[pos + 1..].trim().to_string();
                }
            } else if line.starts_with("Hardware Execution Time:") {
                if let Some(val) = line.split_whitespace().nth(3) {
                    exec_time_ms = val.parse::<f64>().unwrap_or(0.0);
                }
            } else if line.starts_with("Physical GPU Compute") {
                if let Some(val) = line.split_whitespace().nth(4) {
                    giga_macs = val.parse::<f64>().unwrap_or(0.0);
                }
            } else if line.starts_with("Swarm Agent Velocity") {
                if let Some(val) = line.split_whitespace().nth(4) {
                    agent_vel = val.parse::<f64>().unwrap_or(0.0);
                }
            } else if line.starts_with("Sample State") {
                if let Some(pos) = line.find('=') {
                    sample_out = line[pos + 1..].trim().parse::<f32>().unwrap_or(0.0);
                }
            }
        }

        Ok(ComputeResult {
            device_name,
            execution_time_ms: exec_time_ms,
            giga_macs_per_sec: giga_macs,
            agent_cycles_per_sec: agent_vel,
            sample_output: sample_out,
        })
    }
}

impl Default for NativeVulkanBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuComputeBackend for NativeVulkanBackend {
    fn name(&self) -> &'static str {
        "Host Native Vulkan Compute (Desktop/Server)"
    }

    fn is_available(&self) -> bool {
        let temp_dir = std::env::temp_dir().join(".stella_vk");
        let probe_path = temp_dir.join("stella_vk_probe");
        if !probe_path.exists() {
            if self.ensure_runner_extracted().is_err() {
                return false;
            }
        }

        let output = Command::new(&probe_path).output();
        match output {
            Ok(out) => {
                out.status.success() && String::from_utf8_lossy(&out.stdout).contains("Device")
            }
            Err(_) => false,
        }
    }

    fn query_device(&self) -> Result<String, String> {
        let temp_dir = std::env::temp_dir().join(".stella_vk");
        let probe_path = temp_dir.join("stella_vk_probe");
        if !probe_path.exists() {
            self.ensure_runner_extracted()?;
        }

        let output = Command::new(&probe_path)
            .output()
            .map_err(|e| format!("Failed to execute probe: {}", e))?;

        if !output.status.success() {
            return Err("Vulkan hardware probe failed".into());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            let line = line.trim();
            if line.starts_with("Device") {
                if let Some(pos) = line.find(':') {
                    return Ok(line[pos + 1..].trim().to_string());
                }
            }
        }

        Ok("Vulkan Hardware Device".into())
    }

    fn execute(&self, dims: ComputeDimensions) -> Result<ComputeResult, String> {
        let runner_path = self.ensure_runner_extracted()?;
        let temp_dir = runner_path.parent().unwrap();

        let output = Command::new(&runner_path)
            .current_dir(temp_dir)
            .args([
                dims.state_size.to_string(),
                dims.batch_size.to_string(),
                dims.cycles.to_string(),
            ])
            .output()
            .map_err(|e| format!("Native Vulkan execution failed: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        if !output.status.success() {
            return Err(format!(
                "Vulkan runner failed with exit code {:?}: {}",
                output.status.code(),
                stdout
            ));
        }

        self.parse_runner_output(&stdout)
    }
}
