// crates/stella_compiler/src/lib.rs
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod chaos;
pub mod codegen;
pub mod obfuscator;

use alloc::string::String;
use alloc::vec::Vec;
use bumpalo::Bump;

pub use chaos::HenonMap;
pub use codegen::{Codegen, CompiledUnit, ProbeHook};
pub use obfuscator::{
    build_permutation_matrices, transform_basis, transform_signed_basis, ObfuscationKey,
};
pub use stella_core::layer::Dense;
pub use stella_core::math::{Matrix, Q32};
pub use stella_core::vm::{PlasticityKind, PlasticityRule};
pub use stella_frontend::parse_synaptic;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct SymbolMeta {
    pub name: String,
    pub index: usize,
    pub width: usize,
    pub kind: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct StellaBinary {
    pub core: Dense,
    pub probes: Vec<ProbeHook>,
    pub initial_state: Option<Vec<Q32>>,
    pub plasticity_rules: Vec<PlasticityRule>,
    #[serde(default)]
    pub symbols: Vec<SymbolMeta>,
    /// Obfuscation key embedded in the binary for automatic probe decode and convergence masking.
    /// `None` for non-obfuscated binaries.
    #[serde(default)]
    pub obfuscation_key: Option<ObfuscationKey>,
}

#[derive(Debug, Clone)]
pub struct CompiledArtifact {
    pub core: Dense,
    pub permutation_key: Option<Vec<usize>>,
    pub obfuscation_key: Option<ObfuscationKey>,
    pub state_size: usize,
    pub initial_state: Vec<Q32>,
    pub probes: Vec<ProbeHook>,
    pub plasticity_rules: Vec<PlasticityRule>,
    pub symbols: Vec<SymbolMeta>,
}

impl CompiledArtifact {
    pub fn to_bytes(&self) -> Result<Vec<u8>, postcard::Error> {
        let bin = StellaBinary {
            core: self.core.clone(),
            probes: self.probes.clone(),
            initial_state: Some(self.initial_state.clone()),
            plasticity_rules: self.plasticity_rules.clone(),
            symbols: self.symbols.clone(),
            obfuscation_key: self.obfuscation_key.clone(),
        };
        postcard::to_allocvec(&bin)
    }
}

/// Full compiler pipeline:
/// Invokes `stella_frontend` (Lexer, Arena Parser, Symbol Layout) ->
/// `stella_compiler::codegen` (Continuous Matrix Synthesis) ->
/// optional chaotic basis transformation (Hénon attractor + rotational orbits).
pub fn compile(
    source: &str,
    obfuscate: bool,
    pad_size: Option<usize>,
    chaos_seed: Option<(f64, f64)>,
) -> Result<CompiledArtifact, String> {
    let bump = Bump::new();
    let (program, layout) = parse_synaptic(source, &bump)?;

    let codegen = Codegen::new(program, layout);
    let unit = codegen.compile()?;

    let orig_size = unit.core.weights.cols;
    let initial_state = unit.initial_state;
    let probes = unit.probes;
    let plasticity_rules = unit.plasticity_rules;

    let mut symbols_meta = Vec::new();
    for info in unit.layout.symbols.values() {
        let kind_str = match info.kind {
            stella_frontend::symbols::NeuronKind::TerminalIn => "TerminalIn".into(),
            stella_frontend::symbols::NeuronKind::TerminalOut => "TerminalOut".into(),
            stella_frontend::symbols::NeuronKind::Node(dyn_kind) => match dyn_kind {
                stella_frontend::ast::NodeDynamics::Standard => "Node(standard)".into(),
                stella_frontend::ast::NodeDynamics::Leak(_) => "Node(leak)".into(),
                stella_frontend::ast::NodeDynamics::Latch => "Node(latch)".into(),
                stella_frontend::ast::NodeDynamics::Hold => "Node(hold)".into(),
                stella_frontend::ast::NodeDynamics::Oscillator { .. } => "Node(oscillator)".into(),
                stella_frontend::ast::NodeDynamics::Plastic { .. } => "Node(plastic)".into(),
            },
            stella_frontend::symbols::NeuronKind::BasinContext => "BasinContext".into(),
            stella_frontend::symbols::NeuronKind::Scratch => "Scratch".into(),
        };
        symbols_meta.push(SymbolMeta {
            name: info.name.into(),
            index: info.index,
            width: info.width,
            kind: kind_str,
        });
    }

    let mut effective_obfuscate = obfuscate;
    let mut effective_pad = pad_size;
    let mut effective_seed = chaos_seed;

    for decl in program.declarations {
        match decl {
            stella_frontend::ast::Decl::Cloak { pad, seed } => {
                effective_obfuscate = true;
                if effective_pad.is_none() {
                    effective_pad = Some(*pad);
                }
                if effective_seed.is_none() {
                    effective_seed = *seed;
                }
            }
            stella_frontend::ast::Decl::AssertStable { target } => {
                // Determine target neuron indices to evaluate:
                // If a target subgraph name is specified, match neurons prefixed with that name or named that.
                // Otherwise, evaluate all internal recurrent dynamical neurons (excluding TerminalIn / TerminalOut persistent latches).
                let target_indices: Vec<usize> = if let Some(t) = target {
                    let mut idxs = Vec::new();
                    for info in unit.layout.symbols.values() {
                        if info.name == *t || info.name.starts_with(&alloc::format!("{}::", t)) {
                            for offset in 0..info.width {
                                idxs.push(info.index + offset);
                            }
                        }
                    }
                    idxs
                } else {
                    let mut idxs = Vec::new();
                    for info in unit.layout.symbols.values() {
                        match info.kind {
                            stella_frontend::symbols::NeuronKind::Node(
                                stella_frontend::ast::NodeDynamics::Hold,
                            )
                            | stella_frontend::symbols::NeuronKind::BasinContext => {
                                // Hold nodes and Basin attractor contexts maintain their energy via diagonal identity.
                                // Exclude from recurrent loop feedback checking unless explicitly targeted.
                            }
                            stella_frontend::symbols::NeuronKind::Node(_) => {
                                for offset in 0..info.width {
                                    idxs.push(info.index + offset);
                                }
                            }
                            _ => {}
                        }
                    }
                    idxs
                };

                let rho = if target_indices.is_empty() {
                    0.0
                } else {
                    let submatrix = unit.core.weights.submatrix(&target_indices);
                    submatrix.spectral_radius(100)
                };

                if rho >= 1.0 {
                    let target_msg = match target {
                        Some(t) => alloc::format!(" (subgraph/loop '{}')", t),
                        None => String::new(),
                    };
                    return Err(alloc::format!(
                        "Attractor stability assertion failed{}: spectral radius rho(W) = {:.4} >= 1.0 (recurrent divergence detected)",
                        target_msg, rho
                    ));
                }
            }
            _ => {}
        }
    }

    if !effective_obfuscate {
        return Ok(CompiledArtifact {
            core: unit.core,
            permutation_key: None,
            obfuscation_key: None,
            state_size: orig_size,
            initial_state,
            probes,
            plasticity_rules,
            symbols: symbols_meta,
        });
    }

    let target_size = effective_pad.unwrap_or(16).max(orig_size);

    // 1. Pad with irrational rotational chaos orbits + unidirectional ghost entanglement
    let mut w_padded = Matrix::zeros(target_size, target_size);
    let mut b_padded = Matrix::zeros(target_size, 1);

    for i in 0..orig_size {
        for j in 0..orig_size {
            w_padded.set(i, j, unit.core.weights.get(i, j));
        }
        b_padded.set(i, 0, unit.core.biases.get(i, 0));
    }

    let mut i = orig_size;
    let mut angle: f64 = 1.0;
    while i < target_size {
        if i + 1 < target_size {
            let cos_t = angle.cos();
            let sin_t = angle.sin();
            w_padded.set(i, i, Q32::from_f64(cos_t));
            w_padded.set(i, i + 1, Q32::from_f64(-sin_t));
            w_padded.set(i + 1, i, Q32::from_f64(sin_t));
            w_padded.set(i + 1, i + 1, Q32::from_f64(cos_t));

            let b_x = 0.5 - 0.5 * cos_t + 0.5 * sin_t + 0.123;
            let b_y = 0.5 - 0.5 * sin_t - 0.5 * cos_t + 0.321;
            b_padded.set(i, 0, Q32::from_f64(b_x));
            b_padded.set(i + 1, 0, Q32::from_f64(b_y));

            // Ghost Entanglement: Unidirectional feed-forward coupling from real logic into dummy attractors
            if orig_size > 0 {
                let real_src = (i * 7) % orig_size;
                let coupling_weight = Q32::from_f64(0.05 * (angle.sin().abs() + 0.01));
                w_padded.set(i, real_src, coupling_weight);
            }

            angle += 0.6180339887;
            i += 2;
        } else {
            w_padded.set(i, i, Q32::from_f64(-1.0));
            b_padded.set(i, 0, Q32::from_f64(1.0));
            if orig_size > 0 {
                w_padded.set(i, 0, Q32::from_f64(0.05));
            }
            i += 1;
        }
    }

    let padded_core = Dense::new(w_padded, b_padded);

    // 2. Chaotic permutation shuffle via Hénon strange attractor
    let (seed_x, seed_y) = effective_seed.unwrap_or((0.133742, 0.733199));
    let mut henon = HenonMap::new(seed_x, seed_y);
    let mut indices: Vec<usize> = (0..target_size).collect();
    for idx in (1..target_size).rev() {
        let rand_idx = henon.next_usize() % (idx + 1);
        indices.swap(idx, rand_idx);
    }

    // 3. Hyperoctahedral bipolar sign flips: B_N = Z_2^N x S_N
    // Deterministically select ~50% of neurons to have inverted sign polarity
    let mut inversions = alloc::vec![false; target_size];
    for idx in 0..target_size {
        inversions[idx] = (henon.next_usize() & 1) == 1;
    }

    let obf_key = ObfuscationKey {
        mapping: indices.clone(),
        inversions: inversions.clone(),
        total_size: target_size,
        clean_size: orig_size,
        rolling: Some(chaos::ChaoticManifold::new(seed_x, seed_y)),
    };

    let obf_core = transform_signed_basis(&padded_core, &obf_key);

    let mut initial_padded = alloc::vec![Q32::ZERO; target_size];
    for (i, &val) in initial_state.iter().enumerate() {
        if i < target_size {
            initial_padded[i] = val;
        }
    }
    let permuted_initial = obf_key.encode_state(&initial_padded);

    let obf_probes = probes
        .into_iter()
        .map(|p| ProbeHook {
            label: p.label,
            neuron_idx: indices[p.neuron_idx],
            gate_idx: p.gate_idx.map(|g| indices[g]),
        })
        .collect();

    let obf_plasticity = plasticity_rules
        .into_iter()
        .map(|r| PlasticityRule {
            pre: indices[r.pre],
            dest: indices[r.dest],
            rate: r.rate,
            decay: r.decay,
            kind: r.kind,
        })
        .collect();

    Ok(CompiledArtifact {
        core: obf_core,
        permutation_key: Some(indices),
        obfuscation_key: Some(obf_key),
        state_size: target_size,
        initial_state: permuted_initial,
        probes: obf_probes,
        plasticity_rules: obf_plasticity,
        symbols: symbols_meta,
    })
}
