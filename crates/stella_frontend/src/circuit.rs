// crates/stella_frontend/src/circuit.rs

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use bumpalo::Bump;

use crate::ast::*;

/// Expands all `circuit` definitions and `inst` instantiations in the program,
/// monomorphizing subgraphs into uniquely-scoped declarations and flows.
pub fn monomorphize<'a>(program: &Program<'a>, bump: &'a Bump) -> Result<Program<'a>, String> {
    // 1. Collect circuit definitions
    let mut circuits: BTreeMap<
        &'a str,
        (
            &'a [TemplateParam<'a>],
            &'a [CircuitParam<'a>],
            Option<CircuitParam<'a>>,
            &'a [Decl<'a>],
            &'a [Flow<'a>],
        ),
    > = BTreeMap::new();

    for decl in program.declarations {
        if let Decl::Circuit {
            name,
            template_params,
            params,
            return_param,
            declarations,
            flows,
        } = *decl
        {
            if circuits.contains_key(name) {
                return Err(format!("Duplicate circuit definition '{}'", name));
            }
            circuits.insert(
                name,
                (template_params, params, return_param, declarations, flows),
            );
        }
    }

    // 2. Desugar any expression-level circuit calls into synthetic instance declarations and flows
    let mut call_id = 0;
    let mut call_generated_decls = Vec::new();
    let mut desugared_flows = Vec::new();

    for flow in program.flows {
        let desugared = desugar_flow_circuit_calls(
            *flow,
            &circuits,
            &mut call_id,
            &mut call_generated_decls,
            bump,
        )?;
        desugared_flows.push(desugared);
    }

    let mut desugared_basins = Vec::new();
    for basin in program.basins {
        let mut b_flows = Vec::new();
        for f in basin.flows {
            let df = desugar_flow_circuit_calls(
                *f,
                &circuits,
                &mut call_id,
                &mut call_generated_decls,
                bump,
            )?;
            b_flows.push(df);
        }
        desugared_basins.push(Basin {
            name: basin.name,
            flows: bump.alloc_slice_copy(&b_flows),
            bifurcations: basin.bifurcations,
        });
    }

    let mut all_decls = Vec::new();
    for decl in program.declarations {
        all_decls.push(*decl);
    }
    for d in call_generated_decls {
        all_decls.push(d);
    }

    // If no circuits or instantiations exist, return the program untouched
    let has_instances = all_decls.iter().any(|d| matches!(d, Decl::Inst { .. }));

    if !has_instances {
        // Filter out Circuit declarations from the final program declarations
        let mut non_circuit_decls = Vec::new();
        for decl in &all_decls {
            if !matches!(decl, Decl::Circuit { .. }) {
                non_circuit_decls.push(*decl);
            }
        }
        return Ok(Program {
            declarations: bump.alloc_slice_copy(&non_circuit_decls),
            flows: bump.alloc_slice_copy(&desugared_flows),
            basins: bump.alloc_slice_copy(&desugared_basins),
        });
    }

    // 3. Expand instantiations recursively
    let mut expanded_decls = Vec::new();
    let mut expanded_flows = Vec::new();
    for flow in desugared_flows {
        expanded_flows.push(flow);
    }

    for decl in all_decls {
        match decl {
            Decl::Circuit { .. } => {
                // Circuits definitions themselves are not physical neurons; they are templates
            }
            Decl::Inst {
                name,
                circuit,
                template_args,
                args,
            } => {
                expand_instance(
                    name,
                    circuit,
                    template_args,
                    args,
                    &circuits,
                    &mut expanded_decls,
                    &mut expanded_flows,
                    bump,
                    1,
                )?;
            }
            other => {
                expanded_decls.push(other);
            }
        }
    }

    Ok(Program {
        declarations: bump.alloc_slice_copy(&expanded_decls),
        flows: bump.alloc_slice_copy(&expanded_flows),
        basins: bump.alloc_slice_copy(&desugared_basins),
    })
}

fn desugar_expr_circuit_calls<'a>(
    expr: &'a Expr<'a>,
    circuits: &BTreeMap<
        &'a str,
        (
            &'a [TemplateParam<'a>],
            &'a [CircuitParam<'a>],
            Option<CircuitParam<'a>>,
            &'a [Decl<'a>],
            &'a [Flow<'a>],
        ),
    >,
    call_id: &mut usize,
    out_decls: &mut Vec<Decl<'a>>,
    bump: &'a Bump,
) -> Result<&'a Expr<'a>, String> {
    match *expr {
        Expr::CircuitCall {
            circuit,
            template_args,
            args,
        } => {
            let &(_tmpl_params, params, return_param, _, _) = circuits
                .get(circuit)
                .ok_or_else(|| format!("Unknown circuit '{}' called as an expression", circuit))?;
            let ret = return_param.ok_or_else(|| {
                format!(
                    "Circuit '{}' called as an expression must declare a return parameter using '->'",
                    circuit
                )
            })?;

            let mut desugared_args = Vec::new();
            for a in args.iter() {
                desugared_args.push(desugar_expr_circuit_calls(
                    a, circuits, call_id, out_decls, bump,
                )?);
            }

            if desugared_args.len() != params.len() {
                return Err(format!(
                    "Circuit '{}' expects {} arguments, but got {}",
                    circuit,
                    params.len(),
                    desugared_args.len()
                ));
            }

            let inst_name = bump.alloc_str(&format!("_call_{}_{}", circuit, *call_id));
            *call_id += 1;

            let mut inst_args = Vec::new();
            for (i, &arg_expr) in desugared_args.iter().enumerate() {
                match *arg_expr {
                    Expr::Ident(id) => {
                        inst_args.push(NodeTarget::simple(id));
                    }
                    Expr::Index { name, index } => {
                        inst_args.push(NodeTarget::indexed(name, index));
                    }
                    Expr::MultiIndex { name, indices } => {
                        inst_args.push(NodeTarget::multidim(name, indices));
                    }
                    Expr::DynamicIndex { name, addr } => {
                        inst_args.push(NodeTarget::dynamic(name, addr));
                    }
                    _ => {
                        let scratch_name = bump.alloc_str(&format!("{}_arg_{}", inst_name, i));
                        out_decls.push(Decl::Node {
                            name: scratch_name,
                            width: 1,
                            shape: None,
                            dynamics: NodeDynamics::Standard,
                            initial: None,
                        });
                        out_decls.push(Decl::Inst {
                            name: scratch_name,
                            circuit: "",
                            template_args: &[],
                            args: &[],
                        }); // placeholder not needed
                    }
                }
            }

            let ret_target_name = bump.alloc_str(&format!("{}::{}", inst_name, ret.name));

            out_decls.push(Decl::Inst {
                name: inst_name,
                circuit,
                template_args,
                args: bump.alloc_slice_copy(&inst_args),
            });

            Ok(bump.alloc(Expr::Ident(ret_target_name)))
        }
        Expr::Binary { op, left, right } => {
            let l = desugar_expr_circuit_calls(left, circuits, call_id, out_decls, bump)?;
            let r = desugar_expr_circuit_calls(right, circuits, call_id, out_decls, bump)?;
            Ok(bump.alloc(Expr::Binary {
                op,
                left: l,
                right: r,
            }))
        }
        Expr::Unary { op, inner } => {
            let i = desugar_expr_circuit_calls(inner, circuits, call_id, out_decls, bump)?;
            Ok(bump.alloc(Expr::Unary { op, inner: i }))
        }
        Expr::Activation { kind, expr: inner } => {
            let e = desugar_expr_circuit_calls(inner, circuits, call_id, out_decls, bump)?;
            Ok(bump.alloc(Expr::Activation { kind, expr: e }))
        }
        Expr::Curve {
            expr: inner,
            points,
        } => {
            let e = desugar_expr_circuit_calls(inner, circuits, call_id, out_decls, bump)?;
            Ok(bump.alloc(Expr::Curve { expr: e, points }))
        }
        Expr::Match {
            expr: inner,
            pattern,
        } => {
            let e = desugar_expr_circuit_calls(inner, circuits, call_id, out_decls, bump)?;
            Ok(bump.alloc(Expr::Match { expr: e, pattern }))
        }
        Expr::Conv2d {
            input,
            kernel,
            stride,
            padding,
        } => {
            let i = desugar_expr_circuit_calls(input, circuits, call_id, out_decls, bump)?;
            Ok(bump.alloc(Expr::Conv2d {
                input: i,
                kernel,
                stride,
                padding,
            }))
        }
        Expr::AvgPool2d {
            input,
            kernel_size,
            stride,
        } => {
            let i = desugar_expr_circuit_calls(input, circuits, call_id, out_decls, bump)?;
            Ok(bump.alloc(Expr::AvgPool2d {
                input: i,
                kernel_size,
                stride,
            }))
        }
        _ => Ok(expr),
    }
}

fn desugar_flow_circuit_calls<'a>(
    flow: Flow<'a>,
    circuits: &BTreeMap<
        &'a str,
        (
            &'a [TemplateParam<'a>],
            &'a [CircuitParam<'a>],
            Option<CircuitParam<'a>>,
            &'a [Decl<'a>],
            &'a [Flow<'a>],
        ),
    >,
    call_id: &mut usize,
    out_decls: &mut Vec<Decl<'a>>,
    bump: &'a Bump,
) -> Result<Flow<'a>, String> {
    match flow {
        Flow::Assign { dest, expr } => {
            let new_expr = desugar_expr_circuit_calls(expr, circuits, call_id, out_decls, bump)?;
            Ok(Flow::Assign {
                dest,
                expr: new_expr,
            })
        }
        Flow::Synapse {
            src,
            dest,
            weight,
            when,
        } => {
            let new_when = match when {
                Some(w) => Some(desugar_expr_circuit_calls(
                    w, circuits, call_id, out_decls, bump,
                )?),
                None => None,
            };
            Ok(Flow::Synapse {
                src,
                dest,
                weight,
                when: new_when,
            })
        }
        Flow::Inhibit { src, dest, when } => {
            let new_when = match when {
                Some(w) => Some(desugar_expr_circuit_calls(
                    w, circuits, call_id, out_decls, bump,
                )?),
                None => None,
            };
            Ok(Flow::Inhibit {
                src,
                dest,
                when: new_when,
            })
        }
        Flow::Broadcast {
            src,
            dests,
            weight,
            when,
        } => {
            let new_when = match when {
                Some(w) => Some(desugar_expr_circuit_calls(
                    w, circuits, call_id, out_decls, bump,
                )?),
                None => None,
            };
            Ok(Flow::Broadcast {
                src,
                dests,
                weight,
                when: new_when,
            })
        }
        Flow::FanIn {
            srcs,
            dest,
            weight,
            when,
        } => {
            let new_when = match when {
                Some(w) => Some(desugar_expr_circuit_calls(
                    w, circuits, call_id, out_decls, bump,
                )?),
                None => None,
            };
            Ok(Flow::FanIn {
                srcs,
                dest,
                weight,
                when: new_when,
            })
        }
        Flow::BroadcastInhibit { src, dests, when } => {
            let new_when = match when {
                Some(w) => Some(desugar_expr_circuit_calls(
                    w, circuits, call_id, out_decls, bump,
                )?),
                None => None,
            };
            Ok(Flow::BroadcastInhibit {
                src,
                dests,
                when: new_when,
            })
        }
        Flow::FanInInhibit { srcs, dest, when } => {
            let new_when = match when {
                Some(w) => Some(desugar_expr_circuit_calls(
                    w, circuits, call_id, out_decls, bump,
                )?),
                None => None,
            };
            Ok(Flow::FanInInhibit {
                srcs,
                dest,
                when: new_when,
            })
        }
        Flow::GatedSynapse {
            src,
            dest,
            gate,
            weight,
        } => {
            let new_gate = desugar_expr_circuit_calls(gate, circuits, call_id, out_decls, bump)?;
            Ok(Flow::GatedSynapse {
                src,
                dest,
                gate: new_gate,
                weight,
            })
        }
        Flow::ShuntedSynapse {
            src,
            dest,
            shunt,
            weight,
        } => {
            let new_shunt = desugar_expr_circuit_calls(shunt, circuits, call_id, out_decls, bump)?;
            Ok(Flow::ShuntedSynapse {
                src,
                dest,
                shunt: new_shunt,
                weight,
            })
        }
        Flow::BroadcastGated {
            src,
            dests,
            gate,
            weight,
        } => {
            let new_gate = desugar_expr_circuit_calls(gate, circuits, call_id, out_decls, bump)?;
            Ok(Flow::BroadcastGated {
                src,
                dests,
                gate: new_gate,
                weight,
            })
        }
        Flow::BroadcastShunted {
            src,
            dests,
            shunt,
            weight,
        } => {
            let new_shunt = desugar_expr_circuit_calls(shunt, circuits, call_id, out_decls, bump)?;
            Ok(Flow::BroadcastShunted {
                src,
                dests,
                shunt: new_shunt,
                weight,
            })
        }
        Flow::FanInGated {
            srcs,
            dest,
            gate,
            weight,
        } => {
            let new_gate = desugar_expr_circuit_calls(gate, circuits, call_id, out_decls, bump)?;
            Ok(Flow::FanInGated {
                srcs,
                dest,
                gate: new_gate,
                weight,
            })
        }
        Flow::FanInShunted {
            srcs,
            dest,
            shunt,
            weight,
        } => {
            let new_shunt = desugar_expr_circuit_calls(shunt, circuits, call_id, out_decls, bump)?;
            Ok(Flow::FanInShunted {
                srcs,
                dest,
                shunt: new_shunt,
                weight,
            })
        }
        Flow::If {
            cond,
            then_flows,
            else_flows,
        } => {
            let new_cond = desugar_expr_circuit_calls(cond, circuits, call_id, out_decls, bump)?;
            let mut new_then = Vec::new();
            for f in then_flows.iter() {
                new_then.push(desugar_flow_circuit_calls(
                    *f, circuits, call_id, out_decls, bump,
                )?);
            }
            let new_else = match else_flows {
                Some(ef) => {
                    let mut arr = Vec::new();
                    for f in ef.iter() {
                        arr.push(desugar_flow_circuit_calls(
                            *f, circuits, call_id, out_decls, bump,
                        )?);
                    }
                    Some(bump.alloc_slice_copy(&arr) as &'a [Flow<'a>])
                }
                None => None,
            };
            Ok(Flow::If {
                cond: new_cond,
                then_flows: bump.alloc_slice_copy(&new_then),
                else_flows: new_else,
            })
        }
        Flow::While { cond, body } => {
            let new_cond = desugar_expr_circuit_calls(cond, circuits, call_id, out_decls, bump)?;
            let mut new_body = Vec::new();
            for f in body.iter() {
                new_body.push(desugar_flow_circuit_calls(
                    *f, circuits, call_id, out_decls, bump,
                )?);
            }
            Ok(Flow::While {
                cond: new_cond,
                body: bump.alloc_slice_copy(&new_body),
            })
        }
        Flow::Probe { label, expr } => {
            let new_expr = desugar_expr_circuit_calls(expr, circuits, call_id, out_decls, bump)?;
            Ok(Flow::Probe {
                label,
                expr: new_expr,
            })
        }
        Flow::Bifurcate { expr, branches } => {
            let new_expr = desugar_expr_circuit_calls(expr, circuits, call_id, out_decls, bump)?;
            Ok(Flow::Bifurcate {
                expr: new_expr,
                branches,
            })
        }
        Flow::Superpose {
            branches,
            collapse_expr,
            dest,
        } => {
            let new_expr =
                desugar_expr_circuit_calls(collapse_expr, circuits, call_id, out_decls, bump)?;
            Ok(Flow::Superpose {
                branches,
                collapse_expr: new_expr,
                dest,
            })
        }
        other => Ok(other),
    }
}

fn expand_instance<'a>(
    inst_name: &'a str,
    circuit_name: &'a str,
    template_args: &'a [f64],
    args: &'a [NodeTarget<'a>],
    circuits: &BTreeMap<
        &'a str,
        (
            &'a [TemplateParam<'a>],
            &'a [CircuitParam<'a>],
            Option<CircuitParam<'a>>,
            &'a [Decl<'a>],
            &'a [Flow<'a>],
        ),
    >,
    out_decls: &mut Vec<Decl<'a>>,
    out_flows: &mut Vec<Flow<'a>>,
    bump: &'a Bump,
    depth: usize,
) -> Result<(), String> {
    if depth > 32 {
        return Err(format!(
            "Circuit recursion depth exceeded (> 32) in instance '{}' of circuit '{}'",
            inst_name, circuit_name
        ));
    }

    let &(template_params, params, return_param, inner_decls, inner_flows) =
        circuits.get(circuit_name).ok_or_else(|| {
            format!(
                "Unknown circuit '{}' instantiated as '{}'",
                circuit_name, inst_name
            )
        })?;

    let expected_len = params.len();
    if return_param.is_some() {
        if args.len() != expected_len && args.len() != expected_len + 1 {
            return Err(format!(
                "Circuit '{}' expects {} or {} arguments, but got {} in instance '{}'",
                circuit_name,
                expected_len,
                expected_len + 1,
                args.len(),
                inst_name
            ));
        }
    } else if args.len() != expected_len {
        return Err(format!(
            "Circuit '{}' expects {} arguments, but got {} in instance '{}'",
            circuit_name,
            expected_len,
            args.len(),
            inst_name
        ));
    }

    // Build template value map: template_name -> f64
    let mut template_val_map: BTreeMap<&'a str, f64> = BTreeMap::new();
    for (i, t_param) in template_params.iter().enumerate() {
        if i < template_args.len() {
            template_val_map.insert(t_param.name, template_args[i]);
        } else if let Some(def_val) = t_param.default {
            template_val_map.insert(t_param.name, def_val);
        } else {
            return Err(format!(
                "Circuit '{}' missing required template argument '{}' in instance '{}'",
                circuit_name, t_param.name, inst_name
            ));
        }
    }

    // Build symbol shape map from inner and outer declarations to validate argument shapes
    let mut decl_shapes: BTreeMap<&str, (usize, Option<&'a [usize]>)> = BTreeMap::new();
    for d in out_decls.iter() {
        match *d {
            Decl::TerminalIn { name, width, shape }
            | Decl::TerminalOut { name, width, shape }
            | Decl::Node {
                name, width, shape, ..
            } => {
                decl_shapes.insert(name, (width, shape));
            }
            _ => {}
        }
    }

    // Build parameter substitution map: param_name -> NodeTarget and validate shape contracts
    let mut param_map: BTreeMap<&'a str, NodeTarget<'a>> = BTreeMap::new();
    for (param, &arg) in params.iter().zip(args.iter()) {
        if let Some(expected_shape) = param.shape {
            if let Some(&(arg_width, arg_shape)) = decl_shapes.get(arg.name) {
                if let Some(actual_shape) = arg_shape {
                    if actual_shape != expected_shape {
                        return Err(format!(
                            "Circuit '{}' argument '{}' shape mismatch: expected {:?}, got {:?}",
                            circuit_name, arg.name, expected_shape, actual_shape
                        ));
                    }
                } else if arg_width != param.width {
                    return Err(format!(
                        "Circuit '{}' argument '{}' width mismatch: expected {}, got {}",
                        circuit_name, arg.name, param.width, arg_width
                    ));
                }
            }
        } else if param.width > 1 {
            if let Some(&(arg_width, _)) = decl_shapes.get(arg.name) {
                if arg_width != param.width {
                    return Err(format!(
                        "Circuit '{}' argument '{}' width mismatch: expected {}, got {}",
                        circuit_name, arg.name, param.width, arg_width
                    ));
                }
            }
        }
        param_map.insert(param.name, arg);
    }

    if let Some(ret) = return_param {
        if args.len() == params.len() + 1 {
            param_map.insert(ret.name, args[params.len()]);
        } else {
            let scoped_ret = bump.alloc_str(&format!("{}::{}", inst_name, ret.name));
            out_decls.push(Decl::Node {
                name: scoped_ret,
                width: ret.width,
                shape: ret.shape,
                dynamics: NodeDynamics::Standard,
                initial: None,
            });
            param_map.insert(ret.name, NodeTarget::simple(scoped_ret));
        }
    }

    // Monomorphize inner declarations with prefix `inst_name::`
    for d in inner_decls {
        match *d {
            Decl::Node {
                name,
                width,
                shape,
                dynamics,
                initial,
            } => {
                let scoped_name = bump.alloc_str(&format!("{}::{}", inst_name, name));
                out_decls.push(Decl::Node {
                    name: scoped_name,
                    width,
                    shape,
                    dynamics,
                    initial,
                });
            }
            Decl::Const(name, val) => {
                let scoped_name = bump.alloc_str(&format!("{}::{}", inst_name, name));
                let final_val = template_val_map.get(name).copied().unwrap_or(val);
                out_decls.push(Decl::Const(scoped_name, final_val));
            }
            Decl::TerminalIn { name, width, shape } => {
                let scoped_name = bump.alloc_str(&format!("{}::{}", inst_name, name));
                out_decls.push(Decl::Node {
                    name: scoped_name,
                    width,
                    shape,
                    dynamics: NodeDynamics::Hold,
                    initial: None,
                });
            }
            Decl::TerminalOut { name, width, shape } => {
                let scoped_name = bump.alloc_str(&format!("{}::{}", inst_name, name));
                out_decls.push(Decl::Node {
                    name: scoped_name,
                    width,
                    shape,
                    dynamics: NodeDynamics::Standard,
                    initial: None,
                });
            }
            Decl::Inst {
                name: child_name,
                circuit: child_circuit,
                template_args: child_tmpl_args,
                args: child_args,
            } => {
                let scoped_inst_name = bump.alloc_str(&format!("{}::{}", inst_name, child_name));
                let mut remapped_child_args = Vec::new();
                for a in child_args {
                    remapped_child_args.push(remap_target(a, inst_name, &param_map, bump));
                }
                let child_args_slice = bump.alloc_slice_copy(&remapped_child_args);
                expand_instance(
                    scoped_inst_name,
                    child_circuit,
                    child_tmpl_args,
                    child_args_slice,
                    circuits,
                    out_decls,
                    out_flows,
                    bump,
                    depth + 1,
                )?;
            }
            Decl::Circuit { .. }
            | Decl::Cloak { .. }
            | Decl::ConstMatrix { .. }
            | Decl::TypeAlias { .. }
            | Decl::AssertStable { .. }
            | Decl::AssertBounded { .. } => {}
        }
    }

    // Monomorphize inner flows (substituting template parameter names if referenced in expressions)
    for f in inner_flows {
        out_flows.push(remap_flow_with_templates(
            f,
            inst_name,
            &param_map,
            &template_val_map,
            bump,
        ));
    }

    Ok(())
}

fn remap_target<'a>(
    target: &NodeTarget<'a>,
    prefix: &str,
    param_map: &BTreeMap<&'a str, NodeTarget<'a>>,
    bump: &'a Bump,
) -> NodeTarget<'a> {
    let remapped_dyn = target.dynamic_index.map(|d| {
        if let Some(&subst) = param_map.get(d) {
            subst.name
        } else {
            bump.alloc_str(&format!("{}::{}", prefix, d)) as &'a str
        }
    });

    if let Some(&subst) = param_map.get(target.name) {
        if target.index.is_some()
            || target.indices.is_some()
            || target.slice.is_some()
            || target.dynamic_index.is_some()
        {
            NodeTarget {
                name: subst.name,
                index: target.index,
                indices: target.indices,
                dynamic_index: remapped_dyn,
                slice: target.slice,
            }
        } else {
            subst
        }
    } else {
        // Internal circuit node
        let scoped_name = bump.alloc_str(&format!("{}::{}", prefix, target.name));
        NodeTarget {
            name: scoped_name,
            index: target.index,
            indices: target.indices,
            dynamic_index: remapped_dyn,
            slice: target.slice,
        }
    }
}

fn remap_target_slice<'a>(
    targets: &'a [NodeTarget<'a>],
    prefix: &str,
    param_map: &BTreeMap<&'a str, NodeTarget<'a>>,
    bump: &'a Bump,
) -> &'a [NodeTarget<'a>] {
    let mut list = Vec::with_capacity(targets.len());
    for t in targets {
        list.push(remap_target(t, prefix, param_map, bump));
    }
    bump.alloc_slice_copy(&list)
}

fn remap_flow_with_templates<'a>(
    flow: &Flow<'a>,
    prefix: &str,
    param_map: &BTreeMap<&'a str, NodeTarget<'a>>,
    template_vals: &BTreeMap<&'a str, f64>,
    bump: &'a Bump,
) -> Flow<'a> {
    match *flow {
        Flow::Synapse {
            src,
            dest,
            weight,
            when,
        } => Flow::Synapse {
            src: remap_target(&src, prefix, param_map, bump),
            dest: remap_target(&dest, prefix, param_map, bump),
            weight,
            when: when
                .map(|e| remap_expr_with_templates(e, prefix, param_map, template_vals, bump)),
        },
        Flow::Inhibit { src, dest, when } => Flow::Inhibit {
            src: remap_target(&src, prefix, param_map, bump),
            dest: remap_target(&dest, prefix, param_map, bump),
            when: when
                .map(|e| remap_expr_with_templates(e, prefix, param_map, template_vals, bump)),
        },
        Flow::Broadcast {
            src,
            dests,
            weight,
            when,
        } => Flow::Broadcast {
            src: remap_target(&src, prefix, param_map, bump),
            dests: remap_target_slice(dests, prefix, param_map, bump),
            weight,
            when: when
                .map(|e| remap_expr_with_templates(e, prefix, param_map, template_vals, bump)),
        },
        Flow::FanIn {
            srcs,
            dest,
            weight,
            when,
        } => Flow::FanIn {
            srcs: remap_target_slice(srcs, prefix, param_map, bump),
            dest: remap_target(&dest, prefix, param_map, bump),
            weight,
            when: when
                .map(|e| remap_expr_with_templates(e, prefix, param_map, template_vals, bump)),
        },
        Flow::BroadcastInhibit { src, dests, when } => Flow::BroadcastInhibit {
            src: remap_target(&src, prefix, param_map, bump),
            dests: remap_target_slice(dests, prefix, param_map, bump),
            when: when
                .map(|e| remap_expr_with_templates(e, prefix, param_map, template_vals, bump)),
        },
        Flow::FanInInhibit { srcs, dest, when } => Flow::FanInInhibit {
            srcs: remap_target_slice(srcs, prefix, param_map, bump),
            dest: remap_target(&dest, prefix, param_map, bump),
            when: when
                .map(|e| remap_expr_with_templates(e, prefix, param_map, template_vals, bump)),
        },
        Flow::Assign { dest, expr } => Flow::Assign {
            dest: remap_target(&dest, prefix, param_map, bump),
            expr: remap_expr_with_templates(expr, prefix, param_map, template_vals, bump),
        },
        Flow::Drain(target) => Flow::Drain(remap_target(&target, prefix, param_map, bump)),
        Flow::Plastic {
            pre,
            dest,
            rate,
            decay,
            kind,
        } => Flow::Plastic {
            pre: remap_target(&pre, prefix, param_map, bump),
            dest: remap_target(&dest, prefix, param_map, bump),
            rate,
            decay,
            kind,
        },
        Flow::Probe { label, expr } => Flow::Probe {
            label,
            expr: remap_expr_with_templates(expr, prefix, param_map, template_vals, bump),
        },
        Flow::GatedSynapse {
            src,
            dest,
            gate,
            weight,
        } => Flow::GatedSynapse {
            src: remap_target(&src, prefix, param_map, bump),
            dest: remap_target(&dest, prefix, param_map, bump),
            gate: remap_expr_with_templates(gate, prefix, param_map, template_vals, bump),
            weight,
        },
        Flow::ShuntedSynapse {
            src,
            dest,
            shunt,
            weight,
        } => Flow::ShuntedSynapse {
            src: remap_target(&src, prefix, param_map, bump),
            dest: remap_target(&dest, prefix, param_map, bump),
            shunt: remap_expr_with_templates(shunt, prefix, param_map, template_vals, bump),
            weight,
        },
        Flow::BroadcastGated {
            src,
            dests,
            gate,
            weight,
        } => Flow::BroadcastGated {
            src: remap_target(&src, prefix, param_map, bump),
            dests: remap_target_slice(dests, prefix, param_map, bump),
            gate: remap_expr_with_templates(gate, prefix, param_map, template_vals, bump),
            weight,
        },
        Flow::BroadcastShunted {
            src,
            dests,
            shunt,
            weight,
        } => Flow::BroadcastShunted {
            src: remap_target(&src, prefix, param_map, bump),
            dests: remap_target_slice(dests, prefix, param_map, bump),
            shunt: remap_expr_with_templates(shunt, prefix, param_map, template_vals, bump),
            weight,
        },
        Flow::FanInGated {
            srcs,
            dest,
            gate,
            weight,
        } => Flow::FanInGated {
            srcs: remap_target_slice(srcs, prefix, param_map, bump),
            dest: remap_target(&dest, prefix, param_map, bump),
            gate: remap_expr_with_templates(gate, prefix, param_map, template_vals, bump),
            weight,
        },
        Flow::FanInShunted {
            srcs,
            dest,
            shunt,
            weight,
        } => Flow::FanInShunted {
            srcs: remap_target_slice(srcs, prefix, param_map, bump),
            dest: remap_target(&dest, prefix, param_map, bump),
            shunt: remap_expr_with_templates(shunt, prefix, param_map, template_vals, bump),
            weight,
        },
        Flow::If {
            cond,
            then_flows,
            else_flows,
        } => {
            let mut rem_then = Vec::with_capacity(then_flows.len());
            for f in then_flows {
                rem_then.push(remap_flow_with_templates(
                    f,
                    prefix,
                    param_map,
                    template_vals,
                    bump,
                ));
            }
            let rem_else = else_flows.map(|efs| {
                let mut rem_e = Vec::with_capacity(efs.len());
                for f in efs {
                    rem_e.push(remap_flow_with_templates(
                        f,
                        prefix,
                        param_map,
                        template_vals,
                        bump,
                    ));
                }
                bump.alloc_slice_copy(&rem_e) as &'a [Flow<'a>]
            });
            Flow::If {
                cond: remap_expr_with_templates(cond, prefix, param_map, template_vals, bump),
                then_flows: bump.alloc_slice_copy(&rem_then),
                else_flows: rem_else,
            }
        }
        Flow::While { cond, body } => {
            let mut rem_body = Vec::with_capacity(body.len());
            for f in body {
                rem_body.push(remap_flow_with_templates(
                    f,
                    prefix,
                    param_map,
                    template_vals,
                    bump,
                ));
            }
            Flow::While {
                cond: remap_expr_with_templates(cond, prefix, param_map, template_vals, bump),
                body: bump.alloc_slice_copy(&rem_body),
            }
        }
        Flow::Bifurcate { expr, branches } => {
            let mut rem_branches = Vec::with_capacity(branches.len());
            for b in branches {
                let mut rem_flows = Vec::with_capacity(b.flows.len());
                for f in b.flows {
                    rem_flows.push(remap_flow_with_templates(
                        f,
                        prefix,
                        param_map,
                        template_vals,
                        bump,
                    ));
                }
                let rem_target = b.target.map(|t| remap_target(&t, prefix, param_map, bump));
                rem_branches.push(BifurcateBranch {
                    cond: b.cond,
                    target: rem_target,
                    flows: bump.alloc_slice_copy(&rem_flows),
                });
            }
            Flow::Bifurcate {
                expr: remap_expr_with_templates(expr, prefix, param_map, template_vals, bump),
                branches: bump.alloc_slice_copy(&rem_branches),
            }
        }
        Flow::Compete {
            branches,
            threshold,
        } => {
            let mut rem_branches = Vec::with_capacity(branches.len());
            for b in branches {
                let mut rem_flows = Vec::with_capacity(b.flows.len());
                for f in b.flows {
                    rem_flows.push(remap_flow_with_templates(
                        f,
                        prefix,
                        param_map,
                        template_vals,
                        bump,
                    ));
                }
                let rem_head = b.head.map(|h| remap_target(&h, prefix, param_map, bump));
                rem_branches.push(CompeteBranch {
                    name: b.name,
                    flows: bump.alloc_slice_copy(&rem_flows),
                    head: rem_head,
                });
            }
            Flow::Compete {
                branches: bump.alloc_slice_copy(&rem_branches),
                threshold,
            }
        }
        Flow::Relax {
            body,
            tolerance,
            timeout,
        } => {
            let mut rem_body = Vec::with_capacity(body.len());
            for f in body {
                rem_body.push(remap_flow_with_templates(
                    f,
                    prefix,
                    param_map,
                    template_vals,
                    bump,
                ));
            }
            Flow::Relax {
                body: bump.alloc_slice_copy(&rem_body),
                tolerance,
                timeout,
            }
        }
        Flow::Superpose {
            branches,
            collapse_expr,
            dest,
        } => {
            let mut rem_branches = Vec::with_capacity(branches.len());
            for b in branches {
                let mut rem_flows = Vec::with_capacity(b.flows.len());
                for f in b.flows {
                    rem_flows.push(remap_flow_with_templates(
                        f,
                        prefix,
                        param_map,
                        template_vals,
                        bump,
                    ));
                }
                rem_branches.push(SuperposeBranch {
                    name: b.name,
                    flows: bump.alloc_slice_copy(&rem_flows),
                });
            }
            Flow::Superpose {
                branches: bump.alloc_slice_copy(&rem_branches),
                collapse_expr: remap_expr_with_templates(
                    collapse_expr,
                    prefix,
                    param_map,
                    template_vals,
                    bump,
                ),
                dest: remap_target(&dest, prefix, param_map, bump),
            }
        }
        Flow::MultiBranch { src, branches } => {
            let mut rem_branches = Vec::with_capacity(branches.len());
            for b in branches.iter() {
                let mut rem_dests = Vec::with_capacity(b.dests.len());
                for d in b.dests.iter() {
                    rem_dests.push(remap_target(d, prefix, param_map, bump));
                }
                rem_branches.push(BranchPipeline {
                    steps: b.steps,
                    dests: bump.alloc_slice_copy(&rem_dests),
                });
            }
            Flow::MultiBranch {
                src: remap_target(&src, prefix, param_map, bump),
                branches: bump.alloc_slice_copy(&rem_branches),
            }
        }
    }
}

fn remap_expr_with_templates<'a>(
    expr: &'a Expr<'a>,
    prefix: &str,
    param_map: &BTreeMap<&'a str, NodeTarget<'a>>,
    template_vals: &BTreeMap<&'a str, f64>,
    bump: &'a Bump,
) -> &'a Expr<'a> {
    match *expr {
        Expr::Number(_) => expr,
        Expr::Ident(name) => {
            if let Some(&val) = template_vals.get(name) {
                bump.alloc(Expr::Number(val))
            } else if let Some(&subst) = param_map.get(name) {
                if let Some(idx) = subst.index {
                    bump.alloc(Expr::Index {
                        name: subst.name,
                        index: idx,
                    })
                } else {
                    bump.alloc(Expr::Ident(subst.name))
                }
            } else {
                let scoped = bump.alloc_str(&format!("{}::{}", prefix, name));
                bump.alloc(Expr::Ident(scoped))
            }
        }
        Expr::Index { name, index } => {
            if let Some(&subst) = param_map.get(name) {
                bump.alloc(Expr::Index {
                    name: subst.name,
                    index,
                })
            } else {
                let scoped = bump.alloc_str(&format!("{}::{}", prefix, name));
                bump.alloc(Expr::Index {
                    name: scoped,
                    index,
                })
            }
        }
        Expr::MultiIndex { name, indices } => {
            if let Some(&subst) = param_map.get(name) {
                bump.alloc(Expr::MultiIndex {
                    name: subst.name,
                    indices,
                })
            } else {
                let scoped = bump.alloc_str(&format!("{}::{}", prefix, name));
                bump.alloc(Expr::MultiIndex {
                    name: scoped,
                    indices,
                })
            }
        }
        Expr::DynamicIndex { name, addr } => {
            let remapped_name = if let Some(&subst) = param_map.get(name) {
                subst.name
            } else {
                bump.alloc_str(&format!("{}::{}", prefix, name)) as &'a str
            };
            let remapped_addr = if let Some(&subst) = param_map.get(addr) {
                subst.name
            } else {
                bump.alloc_str(&format!("{}::{}", prefix, addr)) as &'a str
            };
            bump.alloc(Expr::DynamicIndex {
                name: remapped_name,
                addr: remapped_addr,
            })
        }
        Expr::Binary { op, left, right } => bump.alloc(Expr::Binary {
            op,
            left: remap_expr_with_templates(left, prefix, param_map, template_vals, bump),
            right: remap_expr_with_templates(right, prefix, param_map, template_vals, bump),
        }),
        Expr::Unary { op, inner } => bump.alloc(Expr::Unary {
            op,
            inner: remap_expr_with_templates(inner, prefix, param_map, template_vals, bump),
        }),
        Expr::Activation { kind, expr: inner } => bump.alloc(Expr::Activation {
            kind,
            expr: remap_expr_with_templates(inner, prefix, param_map, template_vals, bump),
        }),
        Expr::CircuitCall {
            circuit,
            template_args,
            args,
        } => {
            let mut remapped_args = Vec::new();
            for a in args.iter() {
                remapped_args.push(remap_expr_with_templates(
                    a,
                    prefix,
                    param_map,
                    template_vals,
                    bump,
                ));
            }
            bump.alloc(Expr::CircuitCall {
                circuit,
                template_args,
                args: bump.alloc_slice_copy(&remapped_args),
            })
        }
        Expr::Curve {
            expr: inner,
            points,
        } => bump.alloc(Expr::Curve {
            expr: remap_expr_with_templates(inner, prefix, param_map, template_vals, bump),
            points,
        }),
        Expr::Match {
            expr: inner,
            pattern,
        } => bump.alloc(Expr::Match {
            expr: remap_expr_with_templates(inner, prefix, param_map, template_vals, bump),
            pattern,
        }),
        Expr::Conv2d {
            input,
            kernel,
            stride,
            padding,
        } => bump.alloc(Expr::Conv2d {
            input: remap_expr_with_templates(input, prefix, param_map, template_vals, bump),
            kernel,
            stride,
            padding,
        }),
        Expr::AvgPool2d {
            input,
            kernel_size,
            stride,
        } => bump.alloc(Expr::AvgPool2d {
            input: remap_expr_with_templates(input, prefix, param_map, template_vals, bump),
            kernel_size,
            stride,
        }),
    }
}
