// crates/stella_frontend/src/allocator.rs

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::ast::{Decl, Expr, Flow, NodeDynamics, Program};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuronKind {
    TerminalIn,
    TerminalOut,
    Node(NodeDynamics),
    BasinContext,
    Scratch,
}

#[derive(Debug, Clone)]
pub struct NeuronInfo<'a> {
    pub name: &'a str,
    pub index: usize,
    pub width: usize,
    pub shape: Option<Vec<usize>>,
    pub kind: NeuronKind,
    pub initial_value: Option<f64>,
}

#[derive(Debug)]
pub struct NeuronLayout<'a> {
    pub symbols: BTreeMap<&'a str, NeuronInfo<'a>>,
    pub constants: BTreeMap<&'a str, f64>,
    pub matrix_constants: BTreeMap<&'a str, (usize, usize, Vec<f64>)>,
    pub total_neurons: usize,
    pub allocated_named_count: usize,
    pub in_terminal_count: usize,
    pub out_terminal_count: usize,
    pub basin_context_map: BTreeMap<&'a str, usize>,
}

fn count_flow_scratch(flows: &[Flow]) -> usize {
    let mut count = 0;
    for f in flows {
        count += match f {
            Flow::Synapse { when, .. } | Flow::Inhibit { when, .. } => {
                4 + when.map(count_expr_scratch).unwrap_or(0)
            }
            Flow::Broadcast { dests, when, .. } | Flow::BroadcastInhibit { dests, when, .. } => {
                4 * dests.len() + when.map(count_expr_scratch).unwrap_or(0)
            }
            Flow::FanIn { srcs, when, .. } | Flow::FanInInhibit { srcs, when, .. } => {
                4 * srcs.len() + when.map(count_expr_scratch).unwrap_or(0)
            }
            Flow::Drain(_) => 4,
            Flow::Assign { expr, .. } => 8 + count_expr_scratch(expr),
            Flow::If {
                cond,
                then_flows,
                else_flows,
            } => {
                8 + count_expr_scratch(cond)
                    + count_flow_scratch(then_flows)
                    + else_flows.map(|e| count_flow_scratch(e)).unwrap_or(0)
            }
            Flow::While { cond, body } => 8 + count_expr_scratch(cond) + count_flow_scratch(body),
            Flow::Probe { expr, .. } => 4 + count_expr_scratch(expr),
            Flow::Plastic { .. } => 4,
            Flow::GatedSynapse { gate, .. } => 8 + count_expr_scratch(gate),
            Flow::ShuntedSynapse { shunt, .. } => 8 + count_expr_scratch(shunt),
            Flow::BroadcastGated { dests, gate, .. } => 8 * dests.len() + count_expr_scratch(gate),
            Flow::BroadcastShunted { dests, shunt, .. } => {
                8 * dests.len() + count_expr_scratch(shunt)
            }
            Flow::FanInGated { srcs, gate, .. } => 8 * srcs.len() + count_expr_scratch(gate),
            Flow::FanInShunted { srcs, shunt, .. } => 8 * srcs.len() + count_expr_scratch(shunt),
            Flow::Bifurcate { expr, branches } => {
                let mut b_count = 8 + count_expr_scratch(expr);
                for b in *branches {
                    b_count += 6 + count_flow_scratch(b.flows);
                }
                b_count
            }
            Flow::Compete { branches, .. } => {
                let mut c_count = 8;
                for b in *branches {
                    c_count += 6 + count_flow_scratch(b.flows);
                }
                c_count
            }
            Flow::Relax { body, .. } => 8 + count_flow_scratch(body),
            Flow::Superpose {
                branches,
                collapse_expr,
                ..
            } => {
                let mut s_count = 8 + count_expr_scratch(collapse_expr);
                for b in *branches {
                    s_count += 6 + count_flow_scratch(b.flows);
                }
                s_count
            }
            Flow::MultiBranch { branches, .. } => {
                let mut c = 0;
                for b in *branches {
                    c += 8 + b.steps.len() * 4 + b.dests.len() * 2;
                }
                c
            }
        };
    }
    count
}

fn count_expr_scratch(expr: &Expr) -> usize {
    match expr {
        Expr::Number(_)
        | Expr::Ident(_)
        | Expr::Index { .. }
        | Expr::MultiIndex { .. }
        | Expr::DynamicIndex { .. } => 32,
        Expr::Unary { inner, .. } => 4 + count_expr_scratch(inner),
        Expr::Activation { expr, .. } => 4 + count_expr_scratch(expr),
        Expr::Binary { left, right, .. } => {
            6 + count_expr_scratch(left) + count_expr_scratch(right)
        }
        Expr::CircuitCall { args, .. } => {
            let mut c = 4;
            for a in *args {
                c += count_expr_scratch(a);
            }
            c
        }
        Expr::Curve { expr, points } => 16 + points.len() * 4 + count_expr_scratch(expr),
        Expr::Match { expr, pattern } => 16 + pattern.len() * 4 + count_expr_scratch(expr),
        Expr::Conv2d { input, .. } => 64 + count_expr_scratch(input),
        Expr::AvgPool2d { input, .. } => 64 + count_expr_scratch(input),
    }
}

impl<'a> NeuronLayout<'a> {
    pub fn allocate(program: &Program<'a>) -> Result<Self, String> {
        let mut symbols: BTreeMap<&'a str, NeuronInfo<'a>> = BTreeMap::new();
        let mut constants: BTreeMap<&'a str, f64> = BTreeMap::new();
        let mut matrix_constants: BTreeMap<&'a str, (usize, usize, Vec<f64>)> = BTreeMap::new();
        let mut next_idx = 0;

        let mut in_terminals = Vec::new();
        let mut out_terminals = Vec::new();
        let mut nodes = Vec::new();

        // 1. Separate declarations
        for decl in program.declarations {
            match *decl {
                Decl::TerminalIn { name, width, shape } => {
                    in_terminals.push((name, width, shape.map(|s| s.to_vec())))
                }
                Decl::TerminalOut { name, width, shape } => {
                    out_terminals.push((name, width, shape.map(|s| s.to_vec())))
                }
                Decl::Node {
                    name,
                    width,
                    shape,
                    dynamics,
                    initial,
                } => nodes.push((name, width, shape.map(|s| s.to_vec()), dynamics, initial)),
                Decl::Const(name, val) => {
                    if constants.insert(name, val).is_some() {
                        return Err(format!("Duplicate constant declaration '{}'", name));
                    }
                }
                Decl::ConstMatrix {
                    name,
                    rows,
                    cols,
                    data,
                } => {
                    if matrix_constants
                        .insert(name, (rows, cols, data.to_vec()))
                        .is_some()
                    {
                        return Err(format!("Duplicate matrix constant declaration '{}'", name));
                    }
                }
                Decl::Cloak { .. }
                | Decl::Circuit { .. }
                | Decl::Inst { .. }
                | Decl::TypeAlias { .. }
                | Decl::AssertStable { .. }
                | Decl::AssertBounded { .. } => {}
            }
        }

        // 2. Allocate input terminals (first N neurons for I/O mapping)
        let mut in_terminal_count = 0;
        for (name, width, shape) in in_terminals {
            if symbols.contains_key(name) {
                return Err(format!("Duplicate symbol declaration '{}'", name));
            }
            symbols.insert(
                name,
                NeuronInfo {
                    name,
                    index: next_idx,
                    width,
                    shape,
                    kind: NeuronKind::TerminalIn,
                    initial_value: None,
                },
            );
            next_idx += width;
            in_terminal_count += width;
        }

        // 3. Allocate output terminals
        let mut out_terminal_count = 0;
        for (name, width, shape) in out_terminals {
            if symbols.contains_key(name) {
                return Err(format!("Duplicate symbol declaration '{}'", name));
            }
            symbols.insert(
                name,
                NeuronInfo {
                    name,
                    index: next_idx,
                    width,
                    shape,
                    kind: NeuronKind::TerminalOut,
                    initial_value: None,
                },
            );
            next_idx += width;
            out_terminal_count += width;
        }

        // 4. Allocate internal neural nodes
        for (name, user_width, shape, dynamics, init) in nodes {
            if symbols.contains_key(name) {
                return Err(format!("Duplicate symbol declaration '{}'", name));
            }
            let actual_width = match dynamics {
                NodeDynamics::Oscillator { .. } => 2,
                _ => user_width,
            };
            symbols.insert(
                name,
                NeuronInfo {
                    name,
                    index: next_idx,
                    width: actual_width,
                    shape,
                    kind: NeuronKind::Node(dynamics),
                    initial_value: init,
                },
            );
            next_idx += actual_width;
        }

        // 5. Allocate basin attractor context neurons
        let mut basin_context_map = BTreeMap::new();
        for (basin_idx, basin) in program.basins.iter().enumerate() {
            let name = basin.name;
            if symbols.contains_key(name) {
                return Err(format!(
                    "Basin name '{}' clashes with an existing symbol",
                    name
                ));
            }
            let init_energy = if basin_idx == 0 { Some(1.0) } else { Some(0.0) };
            let ctx_neuron = NeuronInfo {
                name,
                index: next_idx,
                width: 1,
                shape: None,
                kind: NeuronKind::BasinContext,
                initial_value: init_energy,
            };
            symbols.insert(name, ctx_neuron);
            basin_context_map.insert(name, next_idx);
            next_idx += 1;
        }

        // 6. Generous headroom for bifurcation pulse wires, gating, and intermediate activations
        let allocated_named_count = next_idx;
        let mut headroom = 256;
        for basin in program.basins {
            headroom += basin.bifurcations.len() * 8;
            headroom += count_flow_scratch(basin.flows);
        }
        headroom += count_flow_scratch(program.flows);
        next_idx += headroom;

        Ok(Self {
            symbols,
            constants,
            matrix_constants,
            total_neurons: next_idx,
            allocated_named_count,
            in_terminal_count,
            out_terminal_count,
            basin_context_map,
        })
    }

    pub fn resolve_ident(&self, name: &str) -> Result<usize, String> {
        if let Some(info) = self.symbols.get(name) {
            return Ok(info.index);
        }

        if let Some(stripped) = name
            .strip_prefix("basin.")
            .or_else(|| name.strip_prefix("state."))
        {
            if let Some(&idx) = self.basin_context_map.get(stripped) {
                return Ok(idx);
            }
            let suffix = format!("::{}", stripped);
            for (&k, &idx) in &self.basin_context_map {
                if k == stripped || k.ends_with(&suffix) {
                    return Ok(idx);
                }
            }
            if let Some(info) = self.symbols.get(stripped) {
                return Ok(info.index);
            }
        }

        Err(format!("Unknown neural symbol '{}'", name))
    }

    pub fn resolve_target(&self, target: &crate::ast::NodeTarget) -> Result<usize, String> {
        let info = if let Some(info) = self.symbols.get(target.name) {
            info
        } else if let Some(stripped) = target
            .name
            .strip_prefix("basin.")
            .or_else(|| target.name.strip_prefix("state."))
        {
            if let Some(info) = self.symbols.get(stripped) {
                info
            } else {
                let suffix = format!("::{}", stripped);
                let found = self
                    .symbols
                    .iter()
                    .find(|(k, _)| **k == stripped || k.ends_with(&suffix));
                if let Some((_, info)) = found {
                    info
                } else {
                    return Err(format!("Unknown neural symbol '{}'", target.name));
                }
            }
        } else {
            return Err(format!("Unknown neural symbol '{}'", target.name));
        };

        if let Some(indices) = target.indices {
            // Multi-dimensional tensor coordinate indexing
            if let Some(ref shape) = info.shape {
                if indices.len() != shape.len() {
                    return Err(format!(
                        "Tensor '{}' has {} dimensions {:?}, but indexed with {} coordinates {:?}",
                        target.name,
                        shape.len(),
                        shape,
                        indices.len(),
                        indices
                    ));
                }
                let mut flat_offset = 0;
                let mut stride = 1;
                for i in (0..shape.len()).rev() {
                    let coord = indices[i];
                    let dim_size = shape[i];
                    if coord >= dim_size {
                        return Err(format!(
                            "Coordinate {} out of bounds for axis {} of tensor '{}' (dim size {})",
                            coord, i, target.name, dim_size
                        ));
                    }
                    flat_offset += coord * stride;
                    stride *= dim_size;
                }
                return Ok(info.index + flat_offset);
            } else {
                // Flatten row-major assuming 1D bus or error
                if indices.len() == 1 {
                    let idx = indices[0];
                    if idx >= info.width {
                        return Err(format!(
                            "Index {} out of bounds for bus '{}' (width {})",
                            idx, target.name, info.width
                        ));
                    }
                    return Ok(info.index + idx);
                } else {
                    return Err(format!(
                        "Symbol '{}' is not a multidimensional tensor",
                        target.name
                    ));
                }
            }
        }

        if let Some((start, end)) = target.slice {
            if start >= end {
                return Err(format!(
                    "Invalid slice range [{}..{}] for '{}'",
                    start, end, target.name
                ));
            }
            if end > info.width {
                return Err(format!(
                    "Slice range [{}..{}] exceeds width {} for '{}'",
                    start, end, info.width, target.name
                ));
            }
            return Ok(info.index + start);
        }

        match target.index {
            Some(idx) => {
                if idx >= info.width {
                    return Err(format!(
                        "Index {} out of bounds for bus '{}' (width {})",
                        idx, target.name, info.width
                    ));
                }
                Ok(info.index + idx)
            }
            None => Ok(info.index),
        }
    }

    pub fn target_width(&self, target: &crate::ast::NodeTarget) -> Result<usize, String> {
        let info = self
            .symbols
            .get(target.name)
            .ok_or_else(|| format!("Unknown neural symbol '{}'", target.name))?;
        if let Some((start, end)) = target.slice {
            if start >= end {
                return Err(format!(
                    "Invalid slice range [{}..{}] for '{}'",
                    start, end, target.name
                ));
            }
            if end > info.width {
                return Err(format!(
                    "Slice range [{}..{}] exceeds width {} for '{}'",
                    start, end, info.width, target.name
                ));
            }
            Ok(end - start)
        } else if target.index.is_some()
            || target.indices.is_some()
            || target.dynamic_index.is_some()
        {
            Ok(1)
        } else {
            Ok(info.width)
        }
    }

    pub fn resolve_const(&self, name: &str) -> Option<f64> {
        self.constants.get(name).copied()
    }
}
