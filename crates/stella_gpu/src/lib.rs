// crates/stella_gpu/src/lib.rs
//! Clean, Extensible GPU Compute Backend Architecture for Project Stella.
//!
//! Provides high-throughput continuous-state swarm simulations through:
//! - Multi-threaded CPU SIMD Tiled GEMM engine (Universal fallback)
//! - Android Wireless ADB Vulkan Hardware Bridge (for ARM Mali / Adreno on Android/Termux)
//! - Native Desktop/Server Vulkan (extensible future target)

pub mod backends;
pub mod dispatcher;
pub mod traits;

pub use backends::*;
pub use dispatcher::*;
pub use traits::*;

use stella_core::layer::Dense;
use stella_core::math::fix::Q32;

/// A Batched Neural VM Swarm executing across multiple parallel instances.
/// Computes S_{t+1}[N, B] = Clamp01(W[N, N] * S_t[N, B] + B[N, 1])
pub struct SwarmGpuBackend {
    pub state_size: usize,
    pub batch_size: usize,
    pub weights_f32: Vec<f32>,
    pub biases_f32: Vec<f32>,
    pub states_f32: Vec<f32>,
    scratch_f32: Vec<f32>,
}

impl SwarmGpuBackend {
    /// Creates a new GPU swarm backend from a compiled Dense core and batch size.
    pub fn new(core: &Dense, batch_size: usize) -> Self {
        let n = core.weights.rows;
        let weights_f32: Vec<f32> = core
            .weights
            .data
            .iter()
            .map(|q| q.to_f64() as f32)
            .collect();
        let biases_f32: Vec<f32> = core.biases.data.iter().map(|q| q.to_f64() as f32).collect();
        let total_states = n * batch_size;

        Self {
            state_size: n,
            batch_size,
            weights_f32,
            biases_f32,
            states_f32: vec![0.0; total_states],
            scratch_f32: vec![0.0; total_states],
        }
    }

    /// Injects initial states for a specific instance in the swarm batch.
    pub fn set_instance_state(&mut self, instance_idx: usize, initial: &[Q32]) {
        assert!(instance_idx < self.batch_size);
        assert!(initial.len() <= self.state_size);
        for (row, &val) in initial.iter().enumerate() {
            self.states_f32[row * self.batch_size + instance_idx] = val.to_f64() as f32;
        }
    }

    /// Reads back states for a specific instance in the swarm batch as Q32.
    pub fn get_instance_state(&self, instance_idx: usize) -> Vec<Q32> {
        assert!(instance_idx < self.batch_size);
        let mut out = Vec::with_capacity(self.state_size);
        for row in 0..self.state_size {
            let val = self.states_f32[row * self.batch_size + instance_idx] as f64;
            out.push(Q32::from_f64(val));
        }
        out
    }

    /// Tiled parallel Batch GEMM Execution Kernel.
    pub fn step_batch(&mut self) {
        use rayon::prelude::*;

        let n = self.state_size;
        let b = self.batch_size;
        let w = &self.weights_f32;
        let bias = &self.biases_f32;
        let cur_s = &self.states_f32;

        self.scratch_f32
            .par_chunks_mut(b)
            .enumerate()
            .for_each(|(r, out_row)| {
                let b_val = bias[r];
                let row_offset = r * n;

                for col in 0..b {
                    let mut sum = b_val;
                    let mut k = 0;
                    while k + 4 <= n {
                        let w0 = w[row_offset + k];
                        let w1 = w[row_offset + k + 1];
                        let w2 = w[row_offset + k + 2];
                        let w3 = w[row_offset + k + 3];

                        let s0 = cur_s[k * b + col];
                        let s1 = cur_s[(k + 1) * b + col];
                        let s2 = cur_s[(k + 2) * b + col];
                        let s3 = cur_s[(k + 3) * b + col];

                        sum += w0 * s0 + w1 * s1 + w2 * s2 + w3 * s3;
                        k += 4;
                    }
                    while k < n {
                        sum += w[row_offset + k] * cur_s[k * b + col];
                        k += 1;
                    }

                    out_row[col] = sum.max(0.0).min(1.0);
                }
            });

        core::mem::swap(&mut self.states_f32, &mut self.scratch_f32);
    }

    /// Propagates the swarm forward for a designated number of clock cycles.
    pub fn run_swarm(&mut self, cycles: usize) {
        for _ in 0..cycles {
            self.step_batch();
        }
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;
    use stella_core::math::Matrix;

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_swarm_gpu_batch_execution() {
        let w = Matrix::from_vec(
            2,
            2,
            vec![
                Q32::from_f64(-1.0),
                Q32::ZERO,
                Q32::from_f64(1.0),
                Q32::ZERO,
            ],
        );
        let b = Matrix::from_vec(2, 1, vec![Q32::from_f64(1.0), Q32::ZERO]);
        let core = Dense::new(w, b);

        let mut swarm = SwarmGpuBackend::new(&core, 4);
        swarm.set_instance_state(0, &[Q32::ZERO, Q32::ZERO]);
        swarm.set_instance_state(1, &[Q32::ONE, Q32::ZERO]);

        swarm.step_batch();

        let inst0_c1 = swarm.get_instance_state(0);
        let inst1_c1 = swarm.get_instance_state(1);

        assert_eq!(inst0_c1[0], Q32::ONE);
        assert_eq!(inst0_c1[1], Q32::ZERO);

        assert_eq!(inst1_c1[0], Q32::ZERO);
        assert_eq!(inst1_c1[1], Q32::ONE);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_backend_dispatcher() {
        let backend = auto_detect_backend();
        assert!(backend.is_available());
        assert!(!backend.name().is_empty());
    }
}
