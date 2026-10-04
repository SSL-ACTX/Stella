use super::ProbeHook;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use stella_core::math::Q32;
use stella_core::vm::{PlasticityKind, PlasticityRule};
use stella_frontend::ast::{
    ActivationKind, BifurcateCond, BinOp, Expr, Flow, NodeTarget, PipelineStep,
};

use super::Codegen;

impl<'a> Codegen<'a> {
    fn resolve_effective_gate(
        &mut self,
        outer_gate: Option<usize>,
        when_cond: Option<&'a Expr<'a>>,
    ) -> Result<Option<usize>, String> {
        match (outer_gate, when_cond) {
            (Some(g), Some(cond)) => {
                let cond_neuron = self.resolve_expr_to_neuron(cond)?;
                let combined = self.alloc_scratch();
                self.add_w(combined, g, 1.0);
                self.add_w(combined, cond_neuron, 1.0);
                self.add_b(combined, -1.0);
                Ok(Some(combined))
            }
            (Some(g), None) => Ok(Some(g)),
            (None, Some(cond)) => {
                let cond_neuron = self.resolve_expr_to_neuron(cond)?;
                Ok(Some(cond_neuron))
            }
            (None, None) => Ok(None),
        }
    }

    fn add_synapse_connection(
        &mut self,
        src_idx: usize,
        dest_idx: usize,
        weight: f64,
        gate: Option<usize>,
    ) {
        if let Some(g) = gate {
            let gated_wire = self.alloc_scratch();
            self.add_w(gated_wire, g, 1.0);
            self.add_w(gated_wire, src_idx, 1.0);
            self.add_b(gated_wire, -1.0);
            self.add_w(dest_idx, gated_wire, weight);
        } else {
            self.add_w(dest_idx, src_idx, weight);
        }
    }

    fn connect_shunted_buses(
        &mut self,
        src: &NodeTarget<'a>,
        dest: &NodeTarget<'a>,
        weight: f64,
        shunt_neuron: usize,
        gate: Option<usize>,
    ) -> Result<(), String> {
        let src_w = self.layout.target_width(src)?;
        let dest_w = self.layout.target_width(dest)?;
        let s_base = self.layout.resolve_target(src)?;
        let d_base = self.layout.resolve_target(dest)?;

        let connect = |cg: &mut Self, s_idx: usize, d_idx: usize| {
            let shunted_wire = cg.alloc_scratch();
            if let Some(outer) = gate {
                cg.add_w(shunted_wire, outer, 1.0);
                cg.add_w(shunted_wire, s_idx, 1.0);
                cg.add_w(shunted_wire, shunt_neuron, -1.0);
                cg.add_b(shunted_wire, -1.0);
            } else {
                cg.add_w(shunted_wire, s_idx, 1.0);
                cg.add_w(shunted_wire, shunt_neuron, -1.0);
            }
            cg.add_w(d_idx, shunted_wire, weight);
        };

        if src_w == dest_w {
            for i in 0..src_w {
                connect(self, s_base + i, d_base + i);
            }
        } else if src_w == 1 {
            for i in 0..dest_w {
                connect(self, s_base, d_base + i);
            }
        } else {
            for i in 0..src_w {
                connect(self, s_base + i, d_base);
            }
        }
        Ok(())
    }

    fn combine_gate(&mut self, outer: Option<usize>, inner: usize) -> usize {
        if let Some(g) = outer {
            let combined = self.alloc_scratch();
            self.add_w(combined, g, 1.0);
            self.add_w(combined, inner, 1.0);
            self.add_b(combined, -1.0);
            combined
        } else {
            inner
        }
    }

    fn connect_buses(
        &mut self,
        src: &NodeTarget<'a>,
        dest: &NodeTarget<'a>,
        weight: f64,
        gate: Option<usize>,
    ) -> Result<(), String> {
        let src_w = self.layout.target_width(src)?;
        let dest_w = self.layout.target_width(dest)?;
        let s_base = self.layout.resolve_target(src)?;
        let d_base = self.layout.resolve_target(dest)?;

        if src_w == dest_w {
            for i in 0..src_w {
                self.add_synapse_connection(s_base + i, d_base + i, weight, gate);
            }
        } else if src_w == 1 {
            for i in 0..dest_w {
                self.add_synapse_connection(s_base, d_base + i, weight, gate);
            }
        } else if dest_w == 1 {
            for i in 0..src_w {
                self.add_synapse_connection(s_base + i, d_base, weight, gate);
            }
        } else {
            return Err(format!(
                "Mismatched bus widths in synapse: '{}' (width {}) ~> '{}' (width {})",
                src.name, src_w, dest.name, dest_w
            ));
        }
        Ok(())
    }

    pub(super) fn compile_flow(
        &mut self,
        flow: &Flow<'a>,
        gate: Option<usize>,
    ) -> Result<(), String> {
        match flow {
            Flow::Synapse {
                src,
                dest,
                weight,
                when,
            } => {
                let eff_gate = self.resolve_effective_gate(gate, *when)?;
                self.connect_buses(src, dest, *weight, eff_gate)?;
            }
            Flow::Inhibit { src, dest, when } => {
                let eff_gate = self.resolve_effective_gate(gate, *when)?;
                self.connect_buses(src, dest, -1.0, eff_gate)?;
            }
            Flow::Broadcast {
                src,
                dests,
                weight,
                when,
            } => {
                let eff_gate = self.resolve_effective_gate(gate, *when)?;
                for dest in *dests {
                    self.connect_buses(src, dest, *weight, eff_gate)?;
                }
            }
            Flow::FanIn {
                srcs,
                dest,
                weight,
                when,
            } => {
                let eff_gate = self.resolve_effective_gate(gate, *when)?;
                for src in *srcs {
                    self.connect_buses(src, dest, *weight, eff_gate)?;
                }
            }
            Flow::BroadcastInhibit { src, dests, when } => {
                let eff_gate = self.resolve_effective_gate(gate, *when)?;
                for dest in *dests {
                    self.connect_buses(src, dest, -1.0, eff_gate)?;
                }
            }
            Flow::FanInInhibit { srcs, dest, when } => {
                let eff_gate = self.resolve_effective_gate(gate, *when)?;
                for src in *srcs {
                    self.connect_buses(src, dest, -1.0, eff_gate)?;
                }
            }
            Flow::Drain(dest) => {
                let dest_idx = self.layout.resolve_target(dest)?;
                let dest_w = self.layout.target_width(dest)?;
                for i in 0..dest_w {
                    let idx = dest_idx + i;
                    if let Some(g) = gate {
                        self.add_w(idx, g, -1.0);
                    } else {
                        self.ensure_cleared(idx);
                    }
                }
            }
            Flow::Minimize(energy_expr) => {
                self.compile_energy_minimize(energy_expr, gate)?;
            }
            Flow::Assign { dest, expr } => {
                if let Some(addr_name) = dest.dynamic_index {
                    self.compile_dynamic_write(dest.name, addr_name, expr, gate)?;
                    return Ok(());
                }

                if let Expr::Conv2d {
                    input,
                    kernel,
                    stride,
                    padding,
                } = expr
                {
                    return self.compile_conv2d(dest, input, kernel, *stride, *padding, gate);
                }

                if let Expr::AvgPool2d {
                    input,
                    kernel_size,
                    stride,
                } = expr
                {
                    return self.compile_avgpool2d(dest, input, *kernel_size, *stride, gate);
                }

                if let Expr::Binary {
                    op: BinOp::MatMul,
                    left,
                    right,
                } = expr
                {
                    if let (Expr::Ident(vec_name), Expr::Ident(mat_name)) = (&**left, &**right) {
                        return self.compile_matmul(dest, vec_name, mat_name, gate);
                    }
                }

                if let Expr::Binary {
                    op: BinOp::OuterProduct,
                    left,
                    right,
                } = expr
                {
                    if let (Expr::Ident(l_name), Expr::Ident(r_name)) = (&**left, &**right) {
                        return self.compile_outer_product(dest, l_name, r_name, gate);
                    }
                }

                let dest_idx = self.layout.resolve_target(dest)?;
                let dest_w = self.layout.target_width(dest)?;
                if dest_w != 1 {
                    return Err(format!(
                        "Scalar assignment to multi-neuron bus '{}' requires indexing (e.g. {}[0] = ...)",
                        dest.name, dest.name
                    ));
                }

                if let Some(g) = gate {
                    // Check if self-accumulating: dest = dest + delta or dest = dest - delta
                    if let Expr::Binary {
                        op: BinOp::Add,
                        left,
                        right,
                    } = expr
                    {
                        let is_self = match left {
                            Expr::Ident(name) => {
                                *name == dest.name && dest.index.is_none() && dest.indices.is_none()
                            }
                            Expr::Index { name, index } => {
                                *name == dest.name && dest.index == Some(*index)
                            }
                            Expr::MultiIndex { name, indices } => {
                                *name == dest.name && dest.indices == Some(*indices)
                            }
                            _ => false,
                        };
                        if is_self {
                            if let Some(val) = self.eval_const(right) {
                                self.add_w(dest_idx, g, val);
                                return Ok(());
                            }
                        }
                    }
                    if let Expr::Binary {
                        op: BinOp::Sub,
                        left,
                        right,
                    } = expr
                    {
                        let is_self = match left {
                            Expr::Ident(name) => {
                                *name == dest.name && dest.index.is_none() && dest.indices.is_none()
                            }
                            Expr::Index { name, index } => {
                                *name == dest.name && dest.index == Some(*index)
                            }
                            Expr::MultiIndex { name, indices } => {
                                *name == dest.name && dest.indices == Some(*indices)
                            }
                            _ => false,
                        };
                        if is_self {
                            if let Some(val) = self.eval_const(right) {
                                self.add_w(dest_idx, g, -val);
                                return Ok(());
                            }
                        }
                    }

                    self.ensure_cleared(dest_idx);
                    if let Some(val) = self.eval_const(expr) {
                        self.add_w(dest_idx, g, val);
                    } else {
                        let src = self.resolve_expr_to_neuron(expr)?;
                        let gated_wire = self.alloc_scratch();
                        self.add_w(gated_wire, g, 1.0);
                        self.add_w(gated_wire, src, 1.0);
                        self.add_b(gated_wire, -1.0);
                        self.add_w(dest_idx, gated_wire, 1.0);
                    }
                } else {
                    self.ensure_cleared(dest_idx);
                    self.apply_expr_to_dest(dest_idx, expr)?;
                }
            }
            Flow::If {
                cond,
                then_flows,
                else_flows,
            } => {
                let cond_neuron = self.resolve_expr_to_neuron(cond)?;
                let then_gate = cond_neuron;

                let effective_then = self.combine_gate(gate, then_gate);

                for f in *then_flows {
                    self.compile_flow(f, Some(effective_then))?;
                }

                if let Some(elses) = else_flows {
                    let else_gate = self.alloc_scratch();
                    self.add_w(else_gate, then_gate, -1.0);
                    self.add_b(else_gate, 1.0);

                    let effective_else = self.combine_gate(gate, else_gate);

                    for f in *elses {
                        self.compile_flow(f, Some(effective_else))?;
                    }
                }
            }
            Flow::While { cond, body } => {
                let mut delta_opt = None;
                if let Expr::Binary {
                    op: BinOp::Lt,
                    left,
                    ..
                } = cond
                {
                    if let Expr::Ident(var) = left {
                        for f in *body {
                            if let Flow::Assign { dest, expr } = f {
                                if dest.name == *var && dest.index.is_none() {
                                    if let Expr::Binary {
                                        op: BinOp::Add,
                                        left: l,
                                        right: r,
                                    } = expr
                                    {
                                        if let Expr::Ident(l_name) = l {
                                            if l_name == var {
                                                delta_opt = self.eval_const(r);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let cond_neuron = if let (
                    Some(delta),
                    Expr::Binary {
                        op: BinOp::Lt,
                        left,
                        right,
                    },
                ) = (delta_opt, cond)
                {
                    if let Some(bound) = self.eval_const(right) {
                        let lookahead_bound = bound - delta * 1.5;
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        let dest = self.alloc_scratch();
                        self.add_w(dest, src1, -100.0);
                        self.add_b(dest, 100.0 * lookahead_bound - 0.01);
                        dest
                    } else {
                        self.resolve_expr_to_neuron(cond)?
                    }
                } else {
                    self.resolve_expr_to_neuron(cond)?
                };

                let effective_gate = self.combine_gate(gate, cond_neuron);

                for f in *body {
                    self.compile_flow(f, Some(effective_gate))?;
                }
            }
            Flow::Probe { label, expr } => {
                let target_neuron = self.resolve_expr_to_neuron(expr)?;
                self.probes.push(ProbeHook {
                    label: label.map(|s| s.into()),
                    neuron_idx: target_neuron,
                    gate_idx: gate,
                });
            }
            Flow::Emit {
                template,
                args,
                cond,
                on_stable,
            } => {
                let mut arg_indices = Vec::new();
                for a in *args {
                    arg_indices.push(self.resolve_expr_to_neuron(a)?);
                }

                let effective_gate = if let Some(c) = cond {
                    let cond_neuron = self.resolve_expr_to_neuron(c)?;
                    Some(self.combine_gate(gate, cond_neuron))
                } else {
                    gate
                };

                self.emits.push(super::EmitHook {
                    template: (*template).into(),
                    arg_indices,
                    gate_idx: effective_gate,
                    on_stable: *on_stable,
                });
            }
            Flow::Return(expr) => {
                let target = self.resolve_expr_to_neuron(expr)?;
                self.return_neuron = Some(target);
            }
            Flow::Plastic {
                pre,
                dest,
                rate,
                decay,
                kind,
                modulator,
            } => {
                let rule_kind = match *kind {
                    "anti_hebbian" | "anti" | "antihebbian" => PlasticityKind::AntiHebbian,
                    "oja" => PlasticityKind::Oja,
                    _ => PlasticityKind::Hebbian,
                };
                let s_base = self.layout.resolve_target(pre)?;
                let s_w = self.layout.target_width(pre)?;
                let d_base = self.layout.resolve_target(dest)?;
                let d_w = self.layout.target_width(dest)?;

                let mod_neuron = if let Some(m) = modulator {
                    Some(self.layout.resolve_target(m)?)
                } else {
                    None
                };

                let mut add_rule = |pre_idx: usize, dest_idx: usize| {
                    self.plasticity_rules.push(PlasticityRule {
                        pre: pre_idx,
                        dest: dest_idx,
                        rate: Q32::from_f64(*rate),
                        decay: Q32::from_f64(*decay),
                        kind: rule_kind,
                        modulator: mod_neuron,
                    });
                };

                if s_w == d_w {
                    for i in 0..s_w {
                        add_rule(s_base + i, d_base + i);
                    }
                } else if s_w == 1 {
                    for i in 0..d_w {
                        add_rule(s_base, d_base + i);
                    }
                } else {
                    for i in 0..s_w {
                        add_rule(s_base + i, d_base);
                    }
                }
            }
            Flow::GatedSynapse {
                src,
                dest,
                gate: gate_expr,
                weight,
            } => {
                let gate_neuron = self.resolve_expr_to_neuron(gate_expr)?;
                let eff_gate = self.combine_gate(gate, gate_neuron);
                self.connect_buses(src, dest, *weight, Some(eff_gate))?;
            }
            Flow::ShuntedSynapse {
                src,
                dest,
                shunt: shunt_expr,
                weight,
            } => {
                let shunt_neuron = self.resolve_expr_to_neuron(shunt_expr)?;
                self.connect_shunted_buses(src, dest, *weight, shunt_neuron, gate)?;
            }
            Flow::BroadcastGated {
                src,
                dests,
                gate: gate_expr,
                weight,
            } => {
                let gate_neuron = self.resolve_expr_to_neuron(gate_expr)?;
                let eff_gate = self.combine_gate(gate, gate_neuron);
                for dest in *dests {
                    self.connect_buses(src, dest, *weight, Some(eff_gate))?;
                }
            }
            Flow::BroadcastShunted {
                src,
                dests,
                shunt: shunt_expr,
                weight,
            } => {
                let shunt_neuron = self.resolve_expr_to_neuron(shunt_expr)?;
                for dest in *dests {
                    self.connect_shunted_buses(src, dest, *weight, shunt_neuron, gate)?;
                }
            }
            Flow::FanInGated {
                srcs,
                dest,
                gate: gate_expr,
                weight,
            } => {
                let gate_neuron = self.resolve_expr_to_neuron(gate_expr)?;
                let eff_gate = self.combine_gate(gate, gate_neuron);
                for src in *srcs {
                    self.connect_buses(src, dest, *weight, Some(eff_gate))?;
                }
            }
            Flow::FanInShunted {
                srcs,
                dest,
                shunt: shunt_expr,
                weight,
            } => {
                let shunt_neuron = self.resolve_expr_to_neuron(shunt_expr)?;
                for src in *srcs {
                    self.connect_shunted_buses(src, dest, *weight, shunt_neuron, gate)?;
                }
            }
            Flow::Bifurcate { expr, branches } => {
                let signal = self.resolve_expr_to_neuron(expr)?;
                let mut prev_conds = Vec::new();

                for branch in *branches {
                    let branch_cond = match branch.cond {
                        BifurcateCond::Lt(val) => {
                            let wire = self.alloc_scratch();
                            self.add_w(wire, signal, -100.0);
                            self.add_b(wire, 100.0 * val - 0.01);
                            prev_conds.push(wire);
                            wire
                        }
                        BifurcateCond::Lte(val) => {
                            let wire = self.alloc_scratch();
                            self.add_w(wire, signal, -100.0);
                            self.add_b(wire, 100.0 * val + 0.01);
                            prev_conds.push(wire);
                            wire
                        }
                        BifurcateCond::Gt(val) => {
                            let wire = self.alloc_scratch();
                            self.add_w(wire, signal, 100.0);
                            self.add_b(wire, -100.0 * val - 0.01);
                            prev_conds.push(wire);
                            wire
                        }
                        BifurcateCond::Gte(val) => {
                            let wire = self.alloc_scratch();
                            self.add_w(wire, signal, 100.0);
                            self.add_b(wire, -100.0 * val + 0.01);
                            prev_conds.push(wire);
                            wire
                        }
                        BifurcateCond::Range(low, high) => {
                            let low_cond = self.alloc_scratch();
                            self.add_w(low_cond, signal, 100.0);
                            self.add_b(low_cond, -100.0 * low - 0.01);

                            let high_cond = self.alloc_scratch();
                            self.add_w(high_cond, signal, -100.0);
                            self.add_b(high_cond, 100.0 * high + 0.01);

                            let combined = self.alloc_scratch();
                            self.add_w(combined, low_cond, 1.0);
                            self.add_w(combined, high_cond, 1.0);
                            self.add_b(combined, -1.0);
                            prev_conds.push(combined);
                            combined
                        }
                        BifurcateCond::Eq(val) => {
                            let low_cond = self.alloc_scratch();
                            self.add_w(low_cond, signal, 100.0);
                            self.add_b(low_cond, -100.0 * (val - 0.05));

                            let high_cond = self.alloc_scratch();
                            self.add_w(high_cond, signal, -100.0);
                            self.add_b(high_cond, 100.0 * (val + 0.05));

                            let combined = self.alloc_scratch();
                            self.add_w(combined, low_cond, 1.0);
                            self.add_w(combined, high_cond, 1.0);
                            self.add_b(combined, -1.0);
                            prev_conds.push(combined);
                            combined
                        }
                        BifurcateCond::When(cond_e) => {
                            let wire = self.resolve_expr_to_neuron(cond_e)?;
                            prev_conds.push(wire);
                            wire
                        }
                        BifurcateCond::Else => {
                            let wire = self.alloc_scratch();
                            for &pc in &prev_conds {
                                self.add_w(wire, pc, -1.0);
                            }
                            self.add_b(wire, 1.0);
                            wire
                        }
                    };

                    let effective_branch_gate = self.combine_gate(gate, branch_cond);

                    if let Some(target) = branch.target {
                        if let Some(&to_ctx) = self.layout.basin_context_map.get(target.name) {
                            if let Some(outer) = gate {
                                self.add_w(outer, effective_branch_gate, -1.0);
                            }
                            self.add_w(to_ctx, effective_branch_gate, 1.0);
                        } else {
                            let d_base = self.layout.resolve_target(&target)?;
                            self.add_w(d_base, effective_branch_gate, 1.0);
                        }
                    }

                    for f in branch.flows {
                        self.compile_flow(f, Some(effective_branch_gate))?;
                    }
                }
            }
            Flow::Compete {
                branches,
                threshold,
            } => {
                let mut candidate_neurons = Vec::new();

                for b in *branches {
                    for f in b.flows {
                        self.compile_flow(f, gate)?;
                    }
                    if let Some(head) = b.head {
                        if let Ok(dest_idx) = self.layout.resolve_target(&head) {
                            candidate_neurons.push(dest_idx);
                        }
                    }
                }

                // Cross-inhibitory lateral recurrent couplings
                for i in 0..candidate_neurons.len() {
                    let ci = candidate_neurons[i];
                    self.add_w(ci, ci, 1.0); // Self-sustain
                    if *threshold > 0.0 {
                        self.add_b(ci, -threshold * 0.5);
                    }
                    for j in 0..candidate_neurons.len() {
                        if i != j {
                            let cj = candidate_neurons[j];
                            self.add_w(ci, cj, -0.6); // Mutual lateral inhibition
                        }
                    }
                }
            }
            Flow::Relax {
                body,
                tolerance: _,
                timeout: _,
            } => {
                for f in *body {
                    self.compile_flow(f, gate)?;
                }
            }
            Flow::Superpose {
                branches,
                collapse_expr,
                dest,
            } => {
                for b in *branches {
                    for f in b.flows {
                        self.compile_flow(f, gate)?;
                    }
                }

                let collapse_neuron = self.resolve_expr_to_neuron(collapse_expr)?;
                let eff_collapse = self.combine_gate(gate, collapse_neuron);

                let d_base = self.layout.resolve_target(dest)?;
                self.add_w(d_base, eff_collapse, 1.0);
            }
            Flow::MultiBranch { src, branches } => {
                let s_base = self.layout.resolve_target(src)?;
                for branch in *branches {
                    let mut current_neuron = s_base;
                    for step in branch.steps {
                        match *step {
                            PipelineStep::Filter(act) => {
                                let filter_neuron = self.alloc_scratch();
                                match act {
                                    ActivationKind::Step => {
                                        self.add_w(filter_neuron, current_neuron, 100.0);
                                        self.add_b(filter_neuron, -50.0);
                                    }
                                    ActivationKind::Inv => {
                                        self.add_w(filter_neuron, current_neuron, -1.0);
                                        self.add_b(filter_neuron, 1.0);
                                    }
                                    ActivationKind::Clamp | ActivationKind::Relu => {
                                        self.add_w(filter_neuron, current_neuron, 1.0);
                                    }
                                    ActivationKind::Saturate => {
                                        self.add_w(filter_neuron, current_neuron, 100.0);
                                        self.add_b(filter_neuron, -50.0);
                                    }
                                }
                                current_neuron = filter_neuron;
                            }
                            PipelineStep::Scale(factor) => {
                                let scaled_neuron = self.alloc_scratch();
                                self.add_w(scaled_neuron, current_neuron, factor);
                                current_neuron = scaled_neuron;
                            }
                        }
                    }

                    for dest in branch.dests {
                        let d_base = self.layout.resolve_target(dest)?;
                        self.add_w(d_base, current_neuron, 1.0);
                    }
                }
            }
        }
        Ok(())
    }
}
