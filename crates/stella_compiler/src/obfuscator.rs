// crates/stella_compiler/src/obfuscator.rs

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};
use stella_core::layer::Dense;
use stella_core::math::{Matrix, Q32};

/// Cryptographically hardened isomorphic obfuscation key.
/// Combines:
/// 1. Hyperoctahedral group signed permutation (mapping + sign bitmask)
/// 2. Ghost dummy padding bounds
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObfuscationKey {
    /// Spatial permutation index mapping: clean_idx -> scrambled_idx
    pub mapping: Vec<usize>,
    /// Bipolar inversion bitmask: true indicates s_i' = 1.0 - s_i
    pub inversions: Vec<bool>,
    /// Total obfuscated state dimension
    pub total_size: usize,
    /// Original unpadded state dimension
    pub clean_size: usize,
    /// Time-varying chaotic coordinate manifold for continuous terminal rolling
    pub rolling: Option<crate::chaos::ChaoticManifold>,
}

/// Homomorphic continuous modular addition over [0.0, 1.0]:
/// x_obs = (x + 2*zeta) mod 2.0 / 2.0
#[inline]
pub fn continuous_mask_q32(x: Q32, zeta: Q32) -> Q32 {
    let two_zeta = zeta.0.wrapping_mul(2);
    let sum = x.0.wrapping_add(two_zeta);
    let rem = sum.rem_euclid(2 * Q32::ONE.to_raw());
    Q32(rem / 2)
}

/// Inverse homomorphic continuous modular addition:
/// x_real = (2*x_obs - 2*zeta) mod 2.0
#[inline]
pub fn continuous_unmask_q32(x_obs: Q32, zeta: Q32) -> Q32 {
    let two_obs = x_obs.0.wrapping_mul(2);
    let two_zeta = zeta.0.wrapping_mul(2);
    let diff = two_obs.wrapping_sub(two_zeta);
    let rem = diff.rem_euclid(2 * Q32::ONE.to_raw());
    Q32(rem)
}

#[inline]
pub fn continuous_mask_f64(x: f64, zeta: f64) -> f64 {
    let sum = x + 2.0 * zeta;
    let rem = sum.rem_euclid(2.0);
    rem / 2.0
}

#[inline]
pub fn continuous_unmask_f64(x_obs: f64, zeta: f64) -> f64 {
    let diff = 2.0 * x_obs - 2.0 * zeta;
    diff.rem_euclid(2.0)
}

impl ObfuscationKey {
    /// Serializes key to portable human-readable string format:
    /// `mapping_csv;inversions_hex;clean_size;total_size[;seed_x,seed_y]`
    pub fn to_key_string(&self) -> String {
        let map_str = self
            .mapping
            .iter()
            .map(|k| k.to_string())
            .collect::<Vec<String>>()
            .join(",");
        let inv_str: String = self
            .inversions
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect();
        if let Some(ref r) = self.rolling {
            format!(
                "{};{};{};{};{},{}",
                map_str, inv_str, self.clean_size, self.total_size, r.seed_x, r.seed_y
            )
        } else {
            format!(
                "{};{};{};{}",
                map_str, inv_str, self.clean_size, self.total_size
            )
        }
    }

    /// Parses an ObfuscationKey from string format (backward-compatible with pure CSV mappings).
    pub fn from_key_string(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if !s.contains(';') {
            // Backward-compatible with legacy comma-separated pure permutation keys
            let mapping: Vec<usize> = s
                .split(',')
                .filter_map(|tok| tok.trim().parse::<usize>().ok())
                .collect();
            let size = mapping.len();
            let inversions = vec![false; size];
            return Ok(Self {
                mapping,
                inversions,
                total_size: size,
                clean_size: size,
                rolling: None,
            });
        }

        let parts: Vec<&str> = s.split(';').collect();
        if parts.len() < 4 {
            return Err(
                "Invalid ObfuscationKey format: expected 4 or 5 semi-colon separated segments"
                    .into(),
            );
        }

        let mapping: Vec<usize> = parts[0]
            .split(',')
            .filter_map(|tok| tok.trim().parse::<usize>().ok())
            .collect();
        let inversions: Vec<bool> = parts[1].chars().map(|c| c == '1').collect();
        let clean_size = parts[2].parse::<usize>().map_err(|e| e.to_string())?;
        let total_size = parts[3].parse::<usize>().map_err(|e| e.to_string())?;

        let rolling = if parts.len() >= 5 && !parts[4].trim().is_empty() {
            let seeds: Vec<f64> = parts[4]
                .split(',')
                .filter_map(|tok| tok.trim().parse::<f64>().ok())
                .collect();
            if seeds.len() >= 2 {
                Some(crate::chaos::ChaoticManifold::new(seeds[0], seeds[1]))
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            mapping,
            inversions,
            total_size,
            clean_size,
            rolling,
        })
    }

    /// Encodes a clean input vector into the obfuscated state space.
    pub fn encode_state(&self, clean: &[Q32]) -> Vec<Q32> {
        let mut obf = vec![Q32::ZERO; self.total_size];
        for (clean_idx, &val) in clean.iter().enumerate() {
            if clean_idx < self.mapping.len() {
                let obf_idx = self.mapping[clean_idx];
                let encoded_val = if self.inversions[clean_idx] {
                    Q32::ONE - val
                } else {
                    val
                };
                obf[obf_idx] = encoded_val;
            }
        }
        obf
    }

    /// Decodes an obfuscated state vector back into clean variables.
    pub fn decode_state(&self, obf: &[f64]) -> Vec<f64> {
        let mut clean = vec![0.0; self.clean_size];
        for clean_idx in 0..self.clean_size {
            if clean_idx < self.mapping.len() {
                let obf_idx = self.mapping[clean_idx];
                let raw_val = obf.get(obf_idx).copied().unwrap_or(0.0);
                clean[clean_idx] = if self.inversions[clean_idx] {
                    1.0 - raw_val
                } else {
                    raw_val
                };
            }
        }
        clean
    }

    /// Couples clean output terminals with continuous time-varying chaotic masking bus:
    /// y_observed(t) = y_real \oplus_R \zeta(t)
    pub fn observe_terminals(&self, clean: &[f64], step: usize) -> Vec<f64> {
        let mut observed = clean.to_vec();
        if let Some(ref r) = self.rolling {
            let masks = r.mask_at_f64(step, self.clean_size);
            for i in 0..observed.len().min(masks.len()) {
                observed[i] = continuous_mask_f64(observed[i], masks[i]);
            }
        }
        observed
    }

    /// Decodes observed terminals using dynamic cryptographic keying:
    /// y_real = y_observed(t) \ominus_R \zeta(t)
    pub fn decode_terminals(&self, observed: &[f64], step: usize) -> Vec<f64> {
        let mut clean = observed.to_vec();
        if let Some(ref r) = self.rolling {
            let masks = r.mask_at_f64(step, self.clean_size);
            for i in 0..clean.len().min(masks.len()) {
                clean[i] = continuous_unmask_f64(clean[i], masks[i]);
            }
        }
        clean
    }

    /// End-to-end decode from raw obfuscated state vector + time-varying terminal unmasking:
    /// 1. Invert signed permutation basis
    /// 2. Remove continuous chaotic terminal mask \zeta(t)
    pub fn decode_observed_state(&self, obf: &[f64], step: usize) -> Vec<f64> {
        let clean_basis = self.decode_state(obf);
        self.decode_terminals(&clean_basis, step)
    }
}

/// Applies a signed permutation affine transformation to obfuscate layer weights and biases.
///
/// For each neuron $k$ with inversion bit $b_k \in \{0, 1\}$:
/// - State substitution: $s_k \to (1 - s_k)$ if $b_k = 1$
/// - For column $j$: $W'_{i, j} = -W_{i, j}$ and $B'_i += W_{i, j}$
/// - For row $i$: $W'_{i, j} = -W_{i, j}$ and $B'_i = 1 - B'_i$
/// - Followed by spatial permutation via $P$ and $P^{-1}$.
pub fn transform_signed_basis(layer: &Dense, key: &ObfuscationKey) -> Dense {
    let n = layer.weights.rows;
    assert_eq!(n, layer.weights.cols);
    assert_eq!(n, key.total_size);

    let mut w_mod = layer.weights.clone();
    let mut b_mod = layer.biases.clone();

    // 1. Column transformation (input state inversion: s_j -> 1 - s_j)
    for j in 0..n {
        if key.inversions[j] {
            for i in 0..n {
                let val = w_mod.get(i, j);
                b_mod.set(i, 0, b_mod.get(i, 0) + val);
                w_mod.set(i, j, -val);
            }
        }
    }

    // 2. Row transformation (output activation inversion: s_i' = 1 - s_i)
    for i in 0..n {
        if key.inversions[i] {
            for j in 0..n {
                let val = w_mod.get(i, j);
                w_mod.set(i, j, -val);
            }
            b_mod.set(i, 0, Q32::ONE - b_mod.get(i, 0));
        }
    }

    // 3. Spatial permutation: W_final = P_inv * W_mod * P, B_final = P_inv * B_mod
    let (p, p_inv) = build_permutation_matrices(&key.mapping);
    let intermediate = &w_mod * &p;
    let w_final = &p_inv * &intermediate;
    let b_final = &p_inv * &b_mod;

    Dense::new(w_final, b_final)
}

/// Applies a linear basis transformation to obfuscate the weights and biases of a layer.
/// Given an invertible matrix P and its inverse P_inv:
/// W' = P_inv * W * P
/// B' = P_inv * B
pub fn transform_basis(layer: &Dense, p: &Matrix, p_inv: &Matrix) -> Dense {
    assert_eq!(p.rows, p.cols, "Transformation matrix P must be square");
    assert_eq!(
        p_inv.rows, p_inv.cols,
        "Inverse transformation matrix P_inv must be square"
    );
    assert_eq!(p.rows, p_inv.rows, "P and P_inv dimensions must match");
    assert_eq!(
        layer.weights.rows, p.rows,
        "Layer dimensions must match transformation matrix"
    );

    let w_p = &layer.weights * p;
    let w_prime = p_inv * &w_p;
    let b_prime = p_inv * &layer.biases;

    Dense::new(w_prime, b_prime)
}

/// Builds permutation matrices P and P_inv from a permutation index mapping.
pub fn build_permutation_matrices(mapping: &[usize]) -> (Matrix, Matrix) {
    let n = mapping.len();
    let mut p = Matrix::zeros(n, n);
    let mut p_inv = Matrix::zeros(n, n);
    for (i, &mapped) in mapping.iter().enumerate() {
        p.set(i, mapped, Q32::ONE);
        p_inv.set(mapped, i, Q32::ONE);
    }
    (p, p_inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use stella_core::math::Q32;

    #[test]
    fn test_signed_permutation_isomorphism() {
        // Construct a small 3-neuron network
        let w = Matrix::from_vec(
            3,
            3,
            vec![
                Q32::from_f64(0.8),
                Q32::from_f64(-0.2),
                Q32::from_f64(0.1),
                Q32::from_f64(0.0),
                Q32::from_f64(0.5),
                Q32::from_f64(0.4),
                Q32::from_f64(-0.3),
                Q32::from_f64(0.2),
                Q32::from_f64(0.9),
            ],
        );
        let b = Matrix::from_vec(
            3,
            1,
            vec![Q32::from_f64(0.1), Q32::from_f64(0.0), Q32::from_f64(0.05)],
        );
        let clean_layer = Dense::new(w, b);

        let key = ObfuscationKey {
            mapping: vec![1, 2, 0],
            inversions: vec![true, false, true],
            clean_size: 3,
            total_size: 3,
            rolling: None,
        };

        let obf_layer = transform_signed_basis(&clean_layer, &key);

        // Verify that weight values are radically altered / inverted
        assert_ne!(clean_layer.weights, obf_layer.weights);

        // Run forward step on clean vs obfuscated
        let clean_state = vec![Q32::from_f64(0.4), Q32::from_f64(0.7), Q32::from_f64(0.2)];
        let mut clean_out = vec![Q32::ZERO; 3];
        clean_layer.forward_into(&clean_state, &mut clean_out);

        let obf_state = key.encode_state(&clean_state);
        let mut obf_out = vec![Q32::ZERO; 3];
        obf_layer.forward_into(&obf_state, &mut obf_out);

        let obf_out_f64: Vec<f64> = obf_out.iter().map(|q| q.to_f64()).collect();
        let decoded_out = key.decode_state(&obf_out_f64);

        for i in 0..3 {
            let clean_val = clean_out[i].to_f64();
            let dec_val = decoded_out[i];
            let diff = (clean_val - dec_val).abs();
            assert!(
                diff < 0.0001,
                "Neuron {} mismatch: clean={}, decoded={}",
                i,
                clean_val,
                dec_val
            );
        }
    }

    #[test]
    fn test_dynamic_terminal_rolling_bit_perfect_and_non_stationary() {
        let key = ObfuscationKey {
            mapping: vec![2, 0, 1],
            inversions: vec![false, true, false],
            clean_size: 3,
            total_size: 3,
            rolling: Some(crate::chaos::ChaoticManifold::new(0.42, 0.99)),
        };

        let clean_f64 = vec![0.0, 1.0, 0.45];

        // 1. Bit-perfect reversibility across 100 discrete time steps
        for step in 0..100 {
            let observed = key.observe_terminals(&clean_f64, step);
            let decoded = key.decode_terminals(&observed, step);

            for (i, &clean) in clean_f64.iter().enumerate() {
                let diff = (decoded[i] - clean).abs();
                assert!(
                    diff < 1e-6,
                    "Step {} neuron {} mismatch: orig={}, decoded={}",
                    step,
                    i,
                    clean,
                    decoded[i]
                );
            }
        }

        // 2. Non-stationarity: identical outputs at t=0 vs t=10 produce completely different observed states
        let obs_t0 = key.observe_terminals(&clean_f64, 0);
        let obs_t10 = key.observe_terminals(&clean_f64, 10);
        assert_ne!(
            obs_t0, obs_t10,
            "Terminal rolling must produce non-stationary observed orbits"
        );
    }
}
