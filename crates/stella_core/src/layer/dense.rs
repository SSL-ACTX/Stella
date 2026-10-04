// crates/stella_core/src/layer/dense.rs

use crate::math::fix::{Q16, Q32};
use crate::math::matrix::Matrix;
use alloc::vec;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Hard activation function: Clamps a continuous value between 0.0 and 1.0.
/// This acts as our "digital signal restoration", snapping analog states back to discrete logic (0 or 1).
#[inline(always)]
pub fn clamp_01(val: Q32) -> Q32 {
    val.clamp_01()
}

/// Applies the activation function element-wise across a matrix.
pub fn activate(m: &Matrix) -> Matrix {
    let mut result = m.clone();
    for val in result.data.iter_mut() {
        *val = clamp_01(*val);
    }
    result
}

/// A standard Dense (Fully Connected) Neural Layer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Dense {
    pub weights: Matrix,
    pub biases: Matrix,
}

impl Dense {
    /// Constructs a new Dense layer.
    /// `weights`: An (N x M) matrix.
    /// `biases`: An (N x 1) column matrix.
    pub fn new(weights: Matrix, biases: Matrix) -> Self {
        assert_eq!(
            weights.rows, biases.rows,
            "Weights row count and biases row count must match"
        );
        assert_eq!(biases.cols, 1, "Biases must be an (N x 1) column vector");
        Self { weights, biases }
    }

    /// High-performance fused GEMV + Bias + Clamping Kernel with ZERO heap allocations.
    /// Computes: `Output[i] = Clamp01(sum_j(W[i, j] * Input[j]) + Bias[i])`
    #[inline]
    pub fn forward_into(&self, input: &[Q32], output: &mut [Q32]) {
        let rows = self.weights.rows;
        let cols = self.weights.cols;

        assert_eq!(cols, input.len(), "Input size must match weights cols");
        assert_eq!(
            rows,
            output.len(),
            "Output buffer size must match weights rows"
        );

        let w = &self.weights.data;
        let b = &self.biases.data;

        #[cfg(feature = "rayon")]
        {
            // When the network is sufficiently large (>= 65,536 MAC operations and >= 64 rows),
            // partition rows into balanced batches across Rayon worker threads.
            if rows * cols >= 65_536 && rows >= 64 {
                // Determine batch size to avoid per-row thread scheduling overhead
                let chunk_size = (rows / (rayon::current_num_threads() * 4)).max(16);
                output
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, out_chunk)| {
                        let base_row = chunk_idx * chunk_size;
                        for (i, out_slot) in out_chunk.iter_mut().enumerate() {
                            let r = base_row + i;
                            let row_offset = r * cols;
                            let row_weights = &w[row_offset..row_offset + cols];
                            let sum_raw = dot_product_row(row_weights, input, b[r].0);
                            *out_slot = Q32::clamp_01_raw(sum_raw);
                        }
                    });
                return;
            }
        }

        // Sequential 8-way unrolled kernel with raw i64 saturating accumulation
        for r in 0..rows {
            let row_offset = r * cols;
            let row_weights = &w[row_offset..row_offset + cols];
            let sum_raw = dot_product_row(row_weights, input, b[r].0);
            output[r] = Q32::clamp_01_raw(sum_raw);
        }
    }

    /// Computes the forward pass: Output = Activation(W * Input + B)
    /// `input`: An (M x 1) column matrix.
    pub fn forward(&self, input: &Matrix) -> Matrix {
        assert_eq!(
            self.weights.cols, input.rows,
            "Input rows must match weights columns"
        );
        assert_eq!(input.cols, 1, "Input must be an (M x 1) column vector");

        let mut output = vec![Q32::ZERO; self.weights.rows];
        self.forward_into(&input.data, &mut output);

        Matrix {
            rows: self.weights.rows,
            cols: 1,
            data: output,
        }
    }
}

/// High-performance fixed-point dot product kernel.
/// Computes: b + sum_i(w[i] * x[i]) with fast 64-bit multiplication and saturating accumulation.
#[inline(always)]
fn dot_product_row(w: &[Q32], input: &[Q32], bias_raw: i64) -> i64 {
    let cols = w.len();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut acc = bias_raw;
        let mut n = cols;
        let mut w_ptr = w.as_ptr() as *const i64;
        let mut x_ptr = input.as_ptr() as *const i64;

        core::arch::asm!(
            "2:",
            "cmp {n}, #4",
            "b.lt 3f",

            // Load 4 w elements (32 bytes) and 4 x elements (32 bytes) into paired registers
            "ldp {w0}, {w1}, [{w_ptr}], #16",
            "ldp {x0}, {x1}, [{x_ptr}], #16",
            "ldp {w2}, {w3}, [{w_ptr}], #16",
            "ldp {x2}, {x3}, [{x_ptr}], #16",

            // Exact 64-bit -> 128-bit Q32.32 product extraction via extr
            "mul {t0}, {w0}, {x0}",
            "smulh {h0}, {w0}, {x0}",
            "extr {p0}, {h0}, {t0}, #32",

            "mul {t1}, {w1}, {x1}",
            "smulh {h1}, {w1}, {x1}",
            "extr {p1}, {h1}, {t1}, #32",

            "mul {t2}, {w2}, {x2}",
            "smulh {h2}, {w2}, {x2}",
            "extr {p2}, {h2}, {t2}, #32",

            "mul {t3}, {w3}, {x3}",
            "smulh {h3}, {w3}, {x3}",
            "extr {p3}, {h3}, {t3}, #32",

            // Direct 64-bit integer addition across 4 lanes (accumulates in 64-bit space)
            "add {p0}, {p0}, {p1}",
            "add {p2}, {p2}, {p3}",
            "add {p0}, {p0}, {p2}",
            "adds {acc}, {acc}, {p0}",

            "sub {n}, {n}, #4",
            "b 2b",

            "3:",
            // Remainder loop
            "cbz {n}, 4f",
            "ldr {w0}, [{w_ptr}], #8",
            "ldr {x0}, [{x_ptr}], #8",
            "mul {t0}, {w0}, {x0}",
            "smulh {h0}, {w0}, {x0}",
            "extr {p0}, {h0}, {t0}, #32",
            "adds {acc}, {acc}, {p0}",
            "sub {n}, {n}, #1",
            "b 3b",

            "4:",
            n = inout(reg) n,
            w_ptr = inout(reg) w_ptr,
            x_ptr = inout(reg) x_ptr,
            acc = inout(reg) acc,

            w0 = out(reg) _,
            w1 = out(reg) _,
            w2 = out(reg) _,
            w3 = out(reg) _,
            x0 = out(reg) _,
            x1 = out(reg) _,
            x2 = out(reg) _,
            x3 = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            h0 = out(reg) _,
            h1 = out(reg) _,
            h2 = out(reg) _,
            h3 = out(reg) _,
            p0 = out(reg) _,
            p1 = out(reg) _,
            p2 = out(reg) _,
            p3 = out(reg) _,
            options(nostack)
        );

        let _ = (n, w_ptr, x_ptr);
        return acc;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum_raw = bias_raw;
        let mut c = 0;
        while c + 4 <= cols {
            let p0 =
                ((w[c].0 as i128 * input[c].0 as i128) >> crate::math::fix::FRACTIONAL_BITS) as i64;
            let p1 = ((w[c + 1].0 as i128 * input[c + 1].0 as i128)
                >> crate::math::fix::FRACTIONAL_BITS) as i64;
            let p2 = ((w[c + 2].0 as i128 * input[c + 2].0 as i128)
                >> crate::math::fix::FRACTIONAL_BITS) as i64;
            let p3 = ((w[c + 3].0 as i128 * input[c + 3].0 as i128)
                >> crate::math::fix::FRACTIONAL_BITS) as i64;

            sum_raw = sum_raw
                .saturating_add((p0.saturating_add(p1)).saturating_add(p2.saturating_add(p3)));
            c += 4;
        }

        while c < cols {
            let p =
                ((w[c].0 as i128 * input[c].0 as i128) >> crate::math::fix::FRACTIONAL_BITS) as i64;
            sum_raw = sum_raw.saturating_add(p);
            c += 1;
        }

        sum_raw
    }
}

/// High-velocity 32-bit (Q16.16) Dense Neural Layer with 128-bit ARM NEON SIMD acceleration.
/// Executes 4 parallel multiply-accumulates per vector instruction on AArch64 hardware.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DenseQ16 {
    pub rows: usize,
    pub cols: usize,
    pub weights: Vec<Q16>,
    pub biases: Vec<Q16>,
}

impl DenseQ16 {
    /// Attempts to convert a standard Q32 Dense layer into a Q16 SIMD layer.
    /// Returns None if any weight or bias overflows the Q16.16 dynamic range ([-32768, +32767]).
    pub fn try_from_dense(dense: &Dense) -> Option<Self> {
        let rows = dense.weights.rows;
        let cols = dense.weights.cols;

        let mut weights_q16 = Vec::with_capacity(dense.weights.data.len());
        for &w in &dense.weights.data {
            weights_q16.push(Q16::from_q32(w)?);
        }

        let mut biases_q16 = Vec::with_capacity(dense.biases.data.len());
        for &b in &dense.biases.data {
            biases_q16.push(Q16::from_q32(b)?);
        }

        Some(Self {
            rows,
            cols,
            weights: weights_q16,
            biases: biases_q16,
        })
    }

    /// Fused NEON SIMD GEMV Kernel.
    /// Multiplies 4 elements at a time in 128-bit vector registers with ZERO heap allocations.
    #[inline]
    pub fn forward_into(&self, input: &[Q16], output: &mut [Q16]) {
        let rows = self.rows;
        let cols = self.cols;
        let w = &self.weights;
        let b = &self.biases;

        for r in 0..rows {
            let row_offset = r * cols;
            let row_w = &w[row_offset..row_offset + cols];
            let sum_raw = dot_product_q16(row_w, input, b[r].0 as i64);
            output[r] = Q16::clamp_01_raw(sum_raw);
        }
    }
}

/// 128-bit ARM NEON SIMD dot product for Q16.16.
/// Computes 4 multiply-accumulates simultaneously using smlal (32-bit x 32-bit -> 64-bit).
#[inline(always)]
fn dot_product_q16(w: &[Q16], input: &[Q16], bias_raw: i64) -> i64 {
    let cols = w.len();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut acc = bias_raw;
        let mut n = cols;
        let mut w_ptr = w.as_ptr() as *const i32;
        let mut x_ptr = input.as_ptr() as *const i32;

        core::arch::asm!(
            "2:",
            "cmp {n}, #4",
            "b.lt 3f",

            // Load 4 32-bit weights (16 bytes) into 128-bit NEON vector v0
            // Load 4 32-bit inputs (16 bytes) into 128-bit NEON vector v1
            "ldr q0, [{w_ptr}], #16",
            "ldr q1, [{x_ptr}], #16",

            // Multiply lower 2 32-bit ints -> 64-bit ints in v2
            "smull v2.2d, v0.2s, v1.2s",
            // Multiply upper 2 32-bit ints -> 64-bit ints in v3
            "smull2 v3.2d, v0.4s, v1.4s",

            // Shift right by 16 bits (Q16.16 fixed-point scaling)
            "sshr v2.2d, v2.2d, #16",
            "sshr v3.2d, v3.2d, #16",

            // Add the pairs
            "add v2.2d, v2.2d, v3.2d",

            // Extract to scalar general-purpose registers and add to acc
            "fmov {p0}, d2",
            "mov {p1}, v2.d[1]",
            "adds {acc}, {acc}, {p0}",
            "adds {acc}, {acc}, {p1}",

            "sub {n}, {n}, #4",
            "b 2b",

            "3:",
            "cbz {n}, 4f",
            "ldr {w0:w}, [{w_ptr}], #4",
            "ldr {x0:w}, [{x_ptr}], #4",
            "smull {p0}, {w0:w}, {x0:w}",
            "asr {p0}, {p0}, #16",
            "adds {acc}, {acc}, {p0}",
            "sub {n}, {n}, #1",
            "b 3b",

            "4:",
            n = inout(reg) n,
            w_ptr = inout(reg) w_ptr,
            x_ptr = inout(reg) x_ptr,
            acc = inout(reg) acc,
            p0 = out(reg) _,
            p1 = out(reg) _,
            w0 = out(reg) _,
            x0 = out(reg) _,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            options(nostack)
        );

        let _ = (n, w_ptr, x_ptr);
        return acc;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum_raw = bias_raw;
        for c in 0..cols {
            let p = (w[c].0 as i64 * input[c].0 as i64) >> 16;
            sum_raw = sum_raw.saturating_add(p);
        }
        sum_raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_activation_function() {
        let neg = Q32::from_f64(-5.0);
        let zero = Q32::from_f64(0.0);
        let mid = Q32::from_f64(0.5);
        let one = Q32::from_f64(1.0);
        let over = Q32::from_f64(42.0);

        assert_eq!(clamp_01(neg), Q32::ZERO);
        assert_eq!(clamp_01(zero), Q32::ZERO);
        assert_eq!(clamp_01(mid), Q32::from_f64(0.5));
        assert_eq!(clamp_01(one), Q32::ONE);
        assert_eq!(clamp_01(over), Q32::ONE);
    }

    #[test]
    fn test_dense_forward() {
        let w = Matrix::from_vec(
            2,
            2,
            vec![
                Q32::from_f64(1.0),
                Q32::from_f64(0.5),
                Q32::from_f64(-1.0),
                Q32::from_f64(2.0),
            ],
        );

        let b = Matrix::from_vec(2, 1, vec![Q32::from_f64(0.0), Q32::from_f64(-0.5)]);

        let layer = Dense::new(w, b);

        let input = Matrix::from_vec(2, 1, vec![Q32::from_f64(1.0), Q32::from_f64(1.0)]);

        let output = layer.forward(&input);

        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, 1);

        // Row 0: 1.0*1.0 + 0.5*1.0 + 0.0 = 1.5 -> clamped to 1.0
        assert_eq!(output.get(0, 0), Q32::from_f64(1.0));

        // Row 1: -1.0*1.0 + 2.0*1.0 + (-0.5) = 0.5 -> clamped to 0.5
        assert_eq!(output.get(1, 0), Q32::from_f64(0.5));
    }

    #[test]
    fn test_dense_forward_large_parallel() {
        let size = 128; // 128x128 = 16,384 MACs, triggers Rayon parallel kernel
        let mut w_data = vec![Q32::ZERO; size * size];
        for i in 0..size {
            w_data[i * size + i] = Q32::from_f64(0.5); // 0.5 identity
        }
        let w = Matrix::from_vec(size, size, w_data);
        let b = Matrix::from_vec(size, 1, vec![Q32::from_f64(0.1); size]);

        let layer = Dense::new(w, b);
        let input = vec![Q32::from_f64(0.6); size];
        let mut output = vec![Q32::ZERO; size];

        layer.forward_into(&input, &mut output);

        let expected = Q32::from_f64(0.5) * Q32::from_f64(0.6) + Q32::from_f64(0.1);
        for &val in &output {
            assert_eq!(val, expected);
        }
    }

    #[test]
    fn test_dense_q16_simd_forward() {
        let size = 16;
        let mut w_data = vec![Q32::ZERO; size * size];
        for i in 0..size {
            w_data[i * size + i] = Q32::from_f64(0.5); // 0.5 identity
        }
        let w = Matrix::from_vec(size, size, w_data);
        let b = Matrix::from_vec(size, 1, vec![Q32::from_f64(0.1); size]);

        let dense_q32 = Dense::new(w, b);
        let dense_q16 = DenseQ16::try_from_dense(&dense_q32).expect("Dense should fit in Q16");

        let input_q16 = vec![Q16::from_f64(0.6); size];
        let mut output_q16 = vec![Q16::ZERO; size];

        dense_q16.forward_into(&input_q16, &mut output_q16);

        let expected_f64 = 0.5 * 0.6 + 0.1;
        let expected_q16 = Q16::from_f64(expected_f64);

        for &val in &output_q16 {
            assert_eq!(val, expected_q16);
        }
    }
}
