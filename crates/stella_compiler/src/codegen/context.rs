use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use stella_core::layer::Dense;
use stella_core::math::{Matrix, Q32};
use stella_core::vm::{PlasticityKind, PlasticityRule};
use stella_frontend::ast::{
    ActivationKind, BinOp, Expr, NodeDynamics, NodeTarget, Program, UnaryOp,
};
use stella_frontend::symbols::{NeuronKind, NeuronLayout};

use super::{Codegen, CompiledUnit};

impl<'a> Codegen<'a> {
    pub fn new(program: Program<'a>, layout: NeuronLayout<'a>) -> Self {
        let size = layout.total_neurons;
        let w = Matrix::zeros(size, size);
        let b = Matrix::zeros(size, 1);

        let next_scratch_idx = layout.allocated_named_count;

        Self {
            program,
            layout,
            w,
            b,
            cleared_rows: vec![false; size],
            next_scratch_idx,
            probes: Vec::new(),
            plasticity_rules: Vec::new(),
        }
    }

    pub fn compile(mut self) -> Result<CompiledUnit<'a>, String> {
        let size = self.layout.total_neurons;
        let mut initial_state = vec![Q32::ZERO; size];

        // 1. Configure node physical dynamics and initial potentials
        for info in self.layout.symbols.values() {
            for offset in 0..info.width {
                let idx = info.index + offset;
                match info.kind {
                    NeuronKind::Node(dynamics) => match dynamics {
                        NodeDynamics::Leak(rate) => {
                            self.w.set(idx, idx, Q32::from_f64(rate));
                        }
                        NodeDynamics::Latch => {
                            self.w.set(idx, idx, Q32::ONE);
                        }
                        NodeDynamics::Hold => {
                            self.w.set(idx, idx, Q32::ONE);
                        }
                        NodeDynamics::Standard => {
                            self.w.set(idx, idx, Q32::ZERO);
                        }
                        NodeDynamics::Oscillator { period } => {
                            // 2D Rotation matrix around (0.5, 0.5):
                            // u (in-phase, index idx) and quad (quadrature, index idx + 1)
                            // theta = 2 * PI / period
                            // W = [ [cos(theta), -sin(theta)], [sin(theta), cos(theta)] ]
                            // B = [ 0.5*(1 - cos(theta) + sin(theta)), 0.5*(1 - sin(theta) - cos(theta)) ]
                            if offset == 0 {
                                let theta = 2.0 * core::f64::consts::PI / (period as f64);
                                let cos_t = libm::cos(theta);
                                let sin_t = libm::sin(theta);
                                let u_idx = idx;
                                let q_idx = idx + 1;

                                self.w.set(u_idx, u_idx, Q32::from_f64(cos_t));
                                self.w.set(u_idx, q_idx, Q32::from_f64(-sin_t));
                                self.b
                                    .set(u_idx, 0, Q32::from_f64(0.5 * (1.0 - cos_t + sin_t)));

                                self.w.set(q_idx, u_idx, Q32::from_f64(sin_t));
                                self.w.set(q_idx, q_idx, Q32::from_f64(cos_t));
                                self.b
                                    .set(q_idx, 0, Q32::from_f64(0.5 * (1.0 - sin_t - cos_t)));
                            }
                        }
                        NodeDynamics::Plastic { rate, decay } => {
                            self.w.set(idx, idx, Q32::ONE);
                            self.plasticity_rules.push(PlasticityRule {
                                pre: idx,
                                dest: idx,
                                rate: Q32::from_f64(rate),
                                decay: Q32::from_f64(decay),
                                kind: PlasticityKind::Hebbian,
                            });
                        }
                    },
                    NeuronKind::BasinContext => {
                        // Context neurons maintain themselves unless bifurcated
                        self.w.set(idx, idx, Q32::ONE);
                    }
                    NeuronKind::TerminalIn => {
                        self.w.set(idx, idx, Q32::ONE);
                    }
                    NeuronKind::TerminalOut | NeuronKind::Scratch => {}
                }

                if let NeuronKind::Node(NodeDynamics::Oscillator { .. }) = info.kind {
                    if offset == 0 {
                        initial_state[idx] = Q32::from_f64(info.initial_value.unwrap_or(1.0));
                    } else if offset == 1 {
                        initial_state[idx] = Q32::from_f64(0.5);
                    }
                } else if let Some(val) = info.initial_value {
                    initial_state[idx] = Q32::from_f64(val);
                }
            }
        }

        // 2. Compile global synaptic flows
        for flow in self.program.flows {
            self.compile_flow(flow, None)?;
        }

        // 3. Compile attractor basins and phase bifurcations
        for basin in self.program.basins {
            let from_ctx = *self
                .layout
                .basin_context_map
                .get(basin.name)
                .ok_or_else(|| format!("Unknown basin '{}'", basin.name))?;

            // Flows active in this basin
            for flow in basin.flows {
                self.compile_flow(flow, Some(from_ctx))?;
            }

            // Phase bifurcations (drifts)
            for bif in basin.bifurcations {
                let to_ctx = *self
                    .layout
                    .basin_context_map
                    .get(bif.target)
                    .ok_or_else(|| format!("Unknown target basin '{}'", bif.target))?;

                match bif.condition {
                    None => {
                        // Unconditional phase drift: source drains, target excites
                        self.add_w(from_ctx, from_ctx, -1.0);
                        self.add_w(to_ctx, from_ctx, 1.0);
                    }
                    Some(cond_expr) => {
                        // Bifurcation pulse wire
                        let pulse_wire = self.alloc_scratch();

                        let cond_neuron = self.resolve_expr_to_neuron(cond_expr)?;

                        // Pulse = AND(from_ctx, cond_neuron)
                        self.add_w(pulse_wire, from_ctx, 1.0);
                        self.add_w(pulse_wire, cond_neuron, 1.0);
                        self.add_b(pulse_wire, -1.0);

                        // Bifurcation drains source basin and excites target basin
                        self.add_w(from_ctx, pulse_wire, -1.0);
                        self.add_w(to_ctx, pulse_wire, 1.0);
                    }
                }
            }
        }

        // 4. Crop matrices to the exact allocated scratch size
        let final_size = self.next_scratch_idx;
        let mut final_w = Matrix::zeros(final_size, final_size);
        let mut final_b = Matrix::zeros(final_size, 1);
        for r in 0..final_size {
            for c in 0..final_size {
                final_w.set(r, c, self.w.get(r, c));
            }
            final_b.set(r, 0, self.b.get(r, 0));
        }

        self.layout.total_neurons = final_size;
        initial_state.truncate(final_size);

        Ok(CompiledUnit {
            core: Dense::new(final_w, final_b),
            layout: self.layout,
            initial_state,
            probes: self.probes,
            plasticity_rules: self.plasticity_rules,
        })
    }

    pub(super) fn alloc_scratch(&mut self) -> usize {
        let idx = self.next_scratch_idx;
        self.next_scratch_idx += 1;
        self.ensure_cleared(idx);
        idx
    }

    pub(super) fn eval_const(&self, expr: &Expr<'a>) -> Option<f64> {
        match expr {
            Expr::Number(n) => Some(*n),
            Expr::Ident(name) => self.layout.resolve_const(name),
            Expr::Unary { op, inner } => {
                let val = self.eval_const(inner)?;
                match op {
                    UnaryOp::Neg => Some(-val),
                    UnaryOp::Not => Some(if val == 0.0 { 1.0 } else { 0.0 }),
                }
            }
            Expr::Binary { op, left, right } => {
                let l = self.eval_const(left)?;
                let r = self.eval_const(right)?;
                match op {
                    BinOp::Add => Some(l + r),
                    BinOp::Sub => Some(l - r),
                    BinOp::Mul => Some(l * r),
                    BinOp::Div => {
                        if r != 0.0 {
                            Some(l / r)
                        } else {
                            None
                        }
                    }
                    BinOp::And => Some(if l > 0.0 && r > 0.0 { 1.0 } else { 0.0 }),
                    BinOp::Or => Some(if l > 0.0 || r > 0.0 { 1.0 } else { 0.0 }),
                    BinOp::Nand => Some(if !(l > 0.0 && r > 0.0) { 1.0 } else { 0.0 }),
                    BinOp::Nor => Some(if !(l > 0.0 || r > 0.0) { 1.0 } else { 0.0 }),
                    BinOp::Xor => Some(if (l > 0.0) ^ (r > 0.0) { 1.0 } else { 0.0 }),
                    BinOp::Gte => Some(if l >= r { 1.0 } else { 0.0 }),
                    BinOp::Lte => Some(if l <= r { 1.0 } else { 0.0 }),
                    BinOp::Gt => Some(if l > r { 1.0 } else { 0.0 }),
                    BinOp::Lt => Some(if l < r { 1.0 } else { 0.0 }),
                    BinOp::Eq => Some(if (l - r).abs() < 1e-6 { 1.0 } else { 0.0 }),
                    BinOp::Neq => Some(if (l - r).abs() >= 1e-6 { 1.0 } else { 0.0 }),
                }
            }
            Expr::Activation { kind, expr } => {
                let val = self.eval_const(expr)?;
                match kind {
                    ActivationKind::Saturate => Some(if val > 0.0 { 1.0 } else { 0.0 }),
                    ActivationKind::Clamp | ActivationKind::Relu => Some(val.clamp(0.0, 1.0)),
                    ActivationKind::Step => Some(if val >= 0.5 { 1.0 } else { 0.0 }),
                    ActivationKind::Inv => Some((1.0 - val).clamp(0.0, 1.0)),
                }
            }
            Expr::Index { .. }
            | Expr::MultiIndex { .. }
            | Expr::DynamicIndex { .. }
            | Expr::CircuitCall { .. }
            | Expr::Curve { .. }
            | Expr::Match { .. }
            | Expr::Conv2d { .. }
            | Expr::AvgPool2d { .. } => None,
        }
    }

    pub(super) fn resolve_expr_to_neuron(&mut self, expr: &Expr<'a>) -> Result<usize, String> {
        if let Some(val) = self.eval_const(expr) {
            let temp = self.alloc_scratch();
            self.add_b(temp, val);
            return Ok(temp);
        }

        match expr {
            Expr::Ident(name) => {
                if let Some(val) = self.layout.resolve_const(name) {
                    let temp = self.alloc_scratch();
                    self.add_b(temp, val);
                    Ok(temp)
                } else {
                    self.layout.resolve_ident(name)
                }
            }
            Expr::Index { name, index } => {
                let target = NodeTarget::indexed(name, *index);
                self.layout.resolve_target(&target)
            }
            Expr::MultiIndex { name, indices } => {
                let target = NodeTarget::multidim(name, indices);
                self.layout.resolve_target(&target)
            }
            Expr::Number(val) => {
                let temp = self.alloc_scratch();
                self.add_b(temp, *val);
                Ok(temp)
            }
            complex => {
                let temp = self.alloc_scratch();
                self.apply_expr_to_dest(temp, complex)?;
                Ok(temp)
            }
        }
    }

    pub(super) fn ensure_cleared(&mut self, dest: usize) {
        if !self.cleared_rows[dest] {
            let cols = self.layout.total_neurons;
            for c in 0..cols {
                self.w.set(dest, c, Q32::ZERO);
            }
            self.cleared_rows[dest] = true;
        }
    }

    pub(super) fn add_w(&mut self, r: usize, c: usize, val: f64) {
        let cur = self.w.get(r, c).to_f64();
        self.w.set(r, c, Q32::from_f64(cur + val));
    }

    pub(super) fn add_b(&mut self, r: usize, val: f64) {
        let cur = self.b.get(r, 0).to_f64();
        self.b.set(r, 0, Q32::from_f64(cur + val));
    }
}
