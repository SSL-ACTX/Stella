// crates/stella_gpu/src/backends/cpu_simd.rs
//! Universal Fallback Backend: Multi-threaded CPU Tiled Batch GEMM Engine.

use crate::traits::{ComputeDimensions, ComputeResult, GpuComputeBackend};
use rayon::prelude::*;
use std::time::Instant;

pub struct CpuSimdBackend;

impl CpuSimdBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuSimdBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuComputeBackend for CpuSimdBackend {
    fn name(&self) -> &'static str {
        "Multi-threaded CPU SIMD (Rayon + ARM NEON/AVX)"
    }

    fn is_available(&self) -> bool {
        true // CPU SIMD is always available on all operating systems
    }

    fn query_device(&self) -> Result<String, String> {
        let threads = rayon::current_num_threads();
        let arch = std::env::consts::ARCH;
        Ok(format!("Host CPU ({}, {} worker threads)", arch, threads))
    }

    fn execute(&self, dims: ComputeDimensions) -> Result<ComputeResult, String> {
        let n = dims.state_size;
        let b = dims.batch_size;
        let cycles = dims.cycles;

        // Initialize synthetic benchmark matrix matching GPU shader test patterns
        let mut weights = vec![0.01f32; n * n];
        let biases = vec![0.05f32; n];
        let mut states = vec![0.5f32; n * b];
        let mut scratch = vec![0.0f32; n * b];

        for i in 0..n {
            weights[i * n + i] = 0.85f32;
        }

        let start = Instant::now();

        for _ in 0..cycles {
            let cur_s = &states;
            scratch
                .par_chunks_mut(b)
                .enumerate()
                .for_each(|(r, out_row)| {
                    let b_val = biases[r];
                    let row_offset = r * n;

                    for col in 0..b {
                        let mut sum = b_val;
                        let mut k = 0;
                        while k + 4 <= n {
                            let w0 = weights[row_offset + k];
                            let w1 = weights[row_offset + k + 1];
                            let w2 = weights[row_offset + k + 2];
                            let w3 = weights[row_offset + k + 3];

                            let s0 = cur_s[k * b + col];
                            let s1 = cur_s[(k + 1) * b + col];
                            let s2 = cur_s[(k + 2) * b + col];
                            let s3 = cur_s[(k + 3) * b + col];

                            sum += w0 * s0 + w1 * s1 + w2 * s2 + w3 * s3;
                            k += 4;
                        }
                        while k < n {
                            sum += weights[row_offset + k] * cur_s[k * b + col];
                            k += 1;
                        }
                        out_row[col] = sum.max(0.0).min(1.0);
                    }
                });

            std::mem::swap(&mut states, &mut scratch);
        }

        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let total_macs = dims.total_macs() as f64;
        let giga_macs_sec = (total_macs / (ms / 1000.0)) / 1e9;
        let agent_cycles_sec = ((b * cycles) as f64) / (ms / 1000.0);

        Ok(ComputeResult {
            device_name: self.query_device()?,
            execution_time_ms: ms,
            giga_macs_per_sec: giga_macs_sec,
            agent_cycles_per_sec: agent_cycles_sec / 1000.0,
            sample_output: states[0],
        })
    }
}
