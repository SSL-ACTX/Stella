// crates/stella_compiler/src/codegen/mod.rs

mod context;
mod expr;
mod flow;
mod memory;
mod spatial;

use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};
use stella_core::layer::Dense;
use stella_core::math::{Matrix, Q32};
use stella_core::vm::PlasticityRule;
use stella_frontend::ast::Program;
use stella_frontend::symbols::NeuronLayout;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeHook {
    pub label: Option<String>,
    pub neuron_idx: usize,
    pub gate_idx: Option<usize>,
}

pub struct CompiledUnit<'a> {
    pub core: Dense,
    pub layout: NeuronLayout<'a>,
    pub initial_state: Vec<Q32>,
    pub probes: Vec<ProbeHook>,
    pub plasticity_rules: Vec<PlasticityRule>,
}

pub struct Codegen<'a> {
    pub(super) program: Program<'a>,
    pub(super) layout: NeuronLayout<'a>,
    pub(super) w: Matrix,
    pub(super) b: Matrix,
    pub(super) cleared_rows: Vec<bool>,
    pub(super) next_scratch_idx: usize,
    pub(super) probes: Vec<ProbeHook>,
    pub(super) plasticity_rules: Vec<PlasticityRule>,
}
