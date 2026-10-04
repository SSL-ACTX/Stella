// crates/stella_gpu/src/backends/android_radb.rs
//! Android Hardware GPU Backend using pure-Rust Wireless ADB (`radb`) Bridge.
//!
//! Enables direct dispatch onto physical ARM Mali / Adreno GPUs by executing inside
//! Android's UID 2000 (`/data/local/tmp/`) shell context, bypassing Android's
//! bionic linker restrictions.

use crate::traits::{ComputeDimensions, ComputeResult, GpuComputeBackend};

/// Embedded compiled binaries generated at build-time by build.rs
const PROBE_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stella_vk_probe"));
const RUNNER_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stella_mali_runner"));
const SHADER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stella_gemm.spv"));

const STELLA_DIR: &str = "/data/local/tmp/.stella";
const REMOTE_PROBE: &str = "/data/local/tmp/.stella/vk_probe";
const REMOTE_RUNNER: &str = "/data/local/tmp/.stella/mali_runner";
const REMOTE_SHADER: &str = "/data/local/tmp/.stella/gemm.spv";

pub struct AndroidRadbBackend {
    client: radb::RuriClient,
}

impl AndroidRadbBackend {
    /// Attempts to connect to localhost wireless ADB.
    pub fn try_connect() -> Result<Self, String> {
        let client = radb::RuriClient::auto_connect()
            .map_err(|e| format!("radb auto-connect failed: {}", e))?;
        Ok(Self { client })
    }

    /// Synchronizes required native compute binaries into an isolated /data/local/tmp/.stella/ directory.
    /// Only uploads if the binaries are missing or corrupted.
    fn ensure_deployed(&self) -> Result<(), String> {
        // Fast-path: check if runner already exists
        let check_cmd = format!(
            "if [ -x {} ] && [ -f {} ]; then echo 1; else echo 0; fi",
            REMOTE_RUNNER, REMOTE_SHADER
        );
        if let Ok(resp) = self.client.exec_str(&check_cmd) {
            if resp.trim() == "1" {
                return Ok(());
            }
        }

        // Ensure isolated directory exists
        let _ = self.client.exec_str(&format!("mkdir -p {}", STELLA_DIR));

        // Deploy probe
        self.client
            .push_bytes(REMOTE_PROBE, PROBE_BINARY, 0o755)
            .map_err(|e| format!("Failed to push probe: {}", e))?;
        let _ = self.client.exec_str(&format!("chmod 755 {}", REMOTE_PROBE));

        // Deploy runner
        self.client
            .push_bytes(REMOTE_RUNNER, RUNNER_BINARY, 0o755)
            .map_err(|e| format!("Failed to push runner: {}", e))?;
        let _ = self
            .client
            .exec_str(&format!("chmod 755 {}", REMOTE_RUNNER));

        // Deploy shader SPIR-V
        self.client
            .push_bytes(REMOTE_SHADER, SHADER_SPV, 0o644)
            .map_err(|e| format!("Failed to push SPIR-V shader: {}", e))?;

        Ok(())
    }

    /// Cleans up the runtime artifacts from /data/local/tmp/.stella/
    pub fn clean_remote_artifacts(&self) -> Result<(), String> {
        self.client
            .exec_str(&format!("rm -rf {}", STELLA_DIR))
            .map_err(|e| format!("Failed to clean remote artifacts: {}", e))?;
        Ok(())
    }

    fn parse_runner_output(&self, raw: &str) -> Result<ComputeResult, String> {
        let mut device_name = "ARM Mali GPU (Direct Driver Pipeline)".to_string();
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
            } else if line.starts_with("Attractor Convergence") {
                if let Some(pos) = line.find('=') {
                    sample_out = line[pos + 1..].trim().parse::<f32>().unwrap_or(0.0);
                }
            }
        }

        if exec_time_ms == 0.0 {
            return Err(format!("Invalid runner response:\n{}", raw));
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

impl GpuComputeBackend for AndroidRadbBackend {
    fn name(&self) -> &'static str {
        "Android Wireless ADB Vulkan Bridge (radb)"
    }

    fn is_available(&self) -> bool {
        radb::scan_local_adbd(30000, 45000).is_some()
    }

    fn query_device(&self) -> Result<String, String> {
        self.ensure_deployed()?;
        let cmd = format!("chmod 755 {} && {}", REMOTE_PROBE, REMOTE_PROBE);
        self.client
            .exec_str(&cmd)
            .map_err(|e| format!("Device query error: {}", e))
    }

    fn execute(&self, dims: ComputeDimensions) -> Result<ComputeResult, String> {
        self.ensure_deployed()?;
        let cmd = format!(
            "{} {} {} {}",
            REMOTE_RUNNER, dims.state_size, dims.batch_size, dims.cycles
        );
        let out = self
            .client
            .exec_str(&cmd)
            .map_err(|e| format!("Remote GPU execution failed: {}", e))?;

        self.parse_runner_output(&out)
    }
}
