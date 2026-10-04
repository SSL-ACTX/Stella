// crates/stella_core/src/math/matrix.rs

use crate::math::fix::Q32;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Add, Mul};
use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use rayon::prelude::*;

/// 2D Matrix backed by a flat contiguous vector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Q32>,
}

impl Matrix {
    /// Initializes a matrix of specified dimensions filled with Q32::ZERO.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Q32::ZERO; rows * cols],
        }
    }

    /// Creates an identity matrix of size N x N.
    pub fn identity(size: usize) -> Self {
        let mut m = Self::zeros(size, size);
        for i in 0..size {
            m.set(i, i, Q32::ONE);
        }
        m
    }

    /// Creates a matrix from a 1D vector. Panics if length does not match rows * cols.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<Q32>) -> Self {
        assert_eq!(rows * cols, data.len(), "Matrix dimension mismatch");
        Self { rows, cols, data }
    }

    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> Q32 {
        self.data[row * self.cols + col]
    }

    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize, val: Q32) {
        self.data[row * self.cols + col] = val;
    }

    /// Returns a transposed version of the matrix.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result.set(c, r, self.get(r, c));
            }
        }
        result
    }

    /// Extracts a square submatrix given a slice of row/column indices.
    pub fn submatrix(&self, indices: &[usize]) -> Self {
        let n = indices.len();
        let mut sub = Self::zeros(n, n);
        for (new_r, &orig_r) in indices.iter().enumerate() {
            for (new_c, &orig_c) in indices.iter().enumerate() {
                sub.set(new_r, new_c, self.get(orig_r, orig_c));
            }
        }
        sub
    }

    /// Approximates the spectral radius rho(W) = max |lambda_i| using the power iteration method.
    /// Returns the estimated dominant eigenvalue magnitude as f64.
    pub fn spectral_radius(&self, max_iterations: usize) -> f64 {
        assert_eq!(
            self.rows, self.cols,
            "Spectral radius requires square matrix"
        );
        let n = self.rows;
        if n == 0 {
            return 0.0;
        }

        // Initialize vector b_k with 1.0 / sqrt(n)
        let mut b = vec![1.0 / libm::sqrt(n as f64); n];

        let mut dominant_eigenvalue = 0.0;
        for _ in 0..max_iterations {
            // Compute y = W * b
            let mut y = vec![0.0; n];
            for r in 0..n {
                let mut sum = 0.0;
                for c in 0..n {
                    sum += self.get(r, c).to_f64() * b[c];
                }
                y[r] = sum;
            }

            // Calculate norm ||y||
            let mut norm_sq = 0.0;
            for &val in &y {
                norm_sq += val * val;
            }
            let norm = libm::sqrt(norm_sq);

            if norm < 1e-12 {
                return 0.0;
            }

            // Rayleigh quotient estimation: (b^T * y) / (b^T * b)
            let mut dot = 0.0;
            for i in 0..n {
                dot += b[i] * y[i];
            }
            dominant_eigenvalue = dot.abs();

            // Normalize b = y / ||y||
            for i in 0..n {
                b[i] = y[i] / norm;
            }
        }

        dominant_eigenvalue
    }
}

impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: Self) -> Matrix {
        assert_eq!(
            self.rows, rhs.rows,
            "Matrix row dimensions do not match for addition"
        );
        assert_eq!(
            self.cols, rhs.cols,
            "Matrix col dimensions do not match for addition"
        );

        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Matrix {
        assert_eq!(
            self.cols, rhs.rows,
            "Matrix dimensions do not match for multiplication"
        );

        let mut result = Matrix::zeros(self.rows, rhs.cols);

        #[cfg(feature = "std")]
        {
            let rhs_cols = rhs.cols;
            let self_cols = self.cols;

            // Only use Rayon if matrix is sufficiently large to offset thread pool overhead
            if self.rows * rhs_cols >= 256 {
                result
                    .data
                    .par_chunks_mut(rhs_cols)
                    .enumerate()
                    .for_each(|(r, row_slice)| {
                        for c in 0..rhs_cols {
                            let mut sum = Q32::ZERO;
                            for k in 0..self_cols {
                                let a = self.data[r * self_cols + k];
                                let b = rhs.data[k * rhs_cols + c];
                                sum += a * b;
                            }
                            row_slice[c] = sum;
                        }
                    });
                return result;
            }
        }

        // Sequential fallback for small matrices / no_std
        for r in 0..self.rows {
            let row_offset = r * self.cols;
            for c in 0..rhs.cols {
                let mut sum = Q32::ZERO;
                for k in 0..self.cols {
                    sum += self.data[row_offset + k] * rhs.get(k, c);
                }
                result.set(r, c, sum);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_matrix_multiplication() {
        let a = Matrix::from_vec(
            2,
            2,
            vec![
                Q32::from_f64(1.0),
                Q32::from_f64(2.0),
                Q32::from_f64(3.0),
                Q32::from_f64(4.0),
            ],
        );

        let b = Matrix::from_vec(
            2,
            2,
            vec![
                Q32::from_f64(2.0),
                Q32::from_f64(0.0),
                Q32::from_f64(1.0),
                Q32::from_f64(2.0),
            ],
        );

        let c = Matrix::from_vec(
            2,
            2,
            vec![
                Q32::from_f64(1.0),
                Q32::from_f64(1.0),
                Q32::from_f64(1.0),
                Q32::from_f64(1.0),
            ],
        );

        let ab = &a * &b;
        assert_eq!(
            ab.data,
            vec![
                Q32::from_f64(4.0),
                Q32::from_f64(4.0),
                Q32::from_f64(10.0),
                Q32::from_f64(8.0),
            ]
        );

        let ab_c = &ab * &c;
        let bc = &b * &c;
        let a_bc = &a * &bc;

        assert_eq!(ab_c, a_bc);
    }
}
