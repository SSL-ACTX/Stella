// crates/stella_gpu/src/traits.rs
//! Extensible Abstractions for GPU & Accelerated Compute Backends.

/// Dimensional parameters for a neural swarm compute pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeDimensions {
    /// Number of neurons / state vector size (N)
    pub state_size: usize,
    /// Number of parallel agents executing simultaneously (B)
    pub batch_size: usize,
    /// Number of clock cycles to propagate
    pub cycles: usize,
}

impl ComputeDimensions {
    pub fn new(state_size: usize, batch_size: usize, cycles: usize) -> Self {
        Self {
            state_size,
            batch_size,
            cycles,
        }
    }

    /// Total Multiply-Accumulate operations in this compute workload.
    pub fn total_macs(&self) -> u64 {
        (self.state_size as u64)
            * (self.state_size as u64)
            * (self.batch_size as u64)
            * (self.cycles as u64)
    }
}

/// Standardized telemetry returned by any GPU or accelerator backend.
#[derive(Debug, Clone)]
pub struct ComputeResult {
    /// Name and architecture of the executing device
    pub device_name: String,
    /// Pure hardware execution duration in milliseconds
    pub execution_time_ms: f64,
    /// Computational throughput in GigaMACs/sec
    pub giga_macs_per_sec: f64,
    /// Velocity in thousand agent-cycles/sec
    pub agent_cycles_per_sec: f64,
    /// Verification sample of attractor state S[0, 0]
    pub sample_output: f32,
}

/// Unified Trait implemented by all compute backends (CPU, Vulkan, Metal, CUDA).
pub trait GpuComputeBackend: Send + Sync {
    /// Friendly human-readable identifier of this backend.
    fn name(&self) -> &'static str;

    /// Checks if this backend is supported and available on the host system.
    fn is_available(&self) -> bool;

    /// Queries the physical hardware device name and vendor details.
    fn query_device(&self) -> Result<String, String>;

    /// Dispatches the continuous neural swarm GEMM across compute units.
    fn execute(&self, dims: ComputeDimensions) -> Result<ComputeResult, String>;
}
