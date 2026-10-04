// crates/stella_frontend/src/circuit.rs

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use bumpalo::Bump;

use crate::ast::*;
use crate::ast_walk::{walk_expr, AstFolder};

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

    let mut desugar_folder = CircuitDesugarFolder {
        circuits: &circuits,
        call_id: &mut call_id,
        out_decls: &mut call_generated_decls,
        bump,
    };

    for flow in program.flows {
        desugared_flows.push(desugar_folder.fold_flow(flow)?);
    }

    let mut desugared_basins = Vec::new();
    for basin in program.basins {
        let mut b_flows = Vec::new();
        for f in basin.flows {
            b_flows.push(desugar_folder.fold_flow(f)?);
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
    let mut current_decls = all_decls.clone();
    let mut expanded_flows = Vec::new();
    for flow in desugared_flows {
        expanded_flows.push(flow);
    }

    let mut expanded_decls = Vec::new();
    loop {
        let mut next_decls = Vec::new();
        let mut has_new_instances = false;

        for decl in current_decls {
            match decl {
                Decl::Circuit { .. } => {}
                Decl::Inst {
                    name,
                    circuit,
                    template_args,
                    args,
                } => {
                    has_new_instances = true;
                    expand_instance(
                        name,
                        circuit,
                        template_args,
                        args,
                        &circuits,
                        &all_decls,
                        &mut next_decls,
                        &mut expanded_flows,
                        bump,
                        1,
                    )?;
                }
                other => {
                    if !expanded_decls.contains(&other) {
                        expanded_decls.push(other);
                    }
                }
            }
        }

        for d in &next_decls {
            if !matches!(d, Decl::Inst { .. }) {
                if !expanded_decls.contains(d) {
                    expanded_decls.push(*d);
                }
                if !all_decls.contains(d) {
                    all_decls.push(*d);
                }
            }
        }

        if !has_new_instances {
            break;
        }
        current_decls = next_decls
            .into_iter()
            .filter(|d| matches!(d, Decl::Inst { .. }))
            .collect();
    }

    Ok(Program {
        declarations: bump.alloc_slice_copy(&expanded_decls),
        flows: bump.alloc_slice_copy(&expanded_flows),
        basins: bump.alloc_slice_copy(&desugared_basins),
    })
}

type CircuitTable<'a> = BTreeMap<
    &'a str,
    (
        &'a [TemplateParam<'a>],
        &'a [CircuitParam<'a>],
        Option<CircuitParam<'a>>,
        &'a [Decl<'a>],
        &'a [Flow<'a>],
    ),
>;

struct CircuitDesugarFolder<'a, 'b> {
    circuits: &'b CircuitTable<'a>,
    call_id: &'b mut usize,
    out_decls: &'b mut Vec<Decl<'a>>,
    bump: &'a Bump,
}

impl<'a, 'b> AstFolder<'a> for CircuitDesugarFolder<'a, 'b> {
    fn bump(&self) -> &'a Bump {
        self.bump
    }

    fn fold_expr(&mut self, expr: &'a Expr<'a>) -> Result<&'a Expr<'a>, String> {
        if let Expr::CircuitCall {
            circuit,
            template_args,
            args,
        } = *expr
        {
            let &(_tmpl_params, params, return_param, _, _) = self
                .circuits
                .get(circuit)
                .ok_or_else(|| format!("Unknown circuit '{}' called as an expression", circuit))?;
            let ret = return_param.ok_or_else(|| {
                format!(
                    "Circuit '{}' called as an expression must declare a return parameter using '->'",
                    circuit
                )
            })?;

            let mut desugared_args = Vec::with_capacity(args.len());
            for a in args.iter() {
                desugared_args.push(self.fold_expr(a)?);
            }

            if desugared_args.len() != params.len() {
                return Err(format!(
                    "Circuit '{}' expects {} arguments, but got {}",
                    circuit,
                    params.len(),
                    desugared_args.len()
                ));
            }

            let inst_name = self
                .bump
                .alloc_str(&format!("_call_{}_{}", circuit, *self.call_id));
            *self.call_id += 1;

            let mut inst_args = Vec::with_capacity(desugared_args.len());
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
                        let scratch_name = self.bump.alloc_str(&format!("{}_arg_{}", inst_name, i));
                        self.out_decls.push(Decl::Node {
                            name: scratch_name,
                            width: 1,
                            shape: None,
                            dynamics: NodeDynamics::Standard,
                            initial: None,
                        });
                        self.out_decls.push(Decl::Inst {
                            name: scratch_name,
                            circuit: "",
                            template_args: &[],
                            args: &[],
                        });
                    }
                }
            }

            let ret_target_name = self.bump.alloc_str(&format!("{}::{}", inst_name, ret.name));

            self.out_decls.push(Decl::Inst {
                name: inst_name,
                circuit,
                template_args,
                args: self.bump.alloc_slice_copy(&inst_args),
            });

            Ok(self.bump.alloc(Expr::Ident(ret_target_name)))
        } else {
            walk_expr(self, expr)
        }
    }
}

fn expand_instance<'a>(
    inst_name: &'a str,
    circuit_name: &'a str,
    template_args: &'a [f64],
    args: &'a [NodeTarget<'a>],
    circuits: &CircuitTable<'a>,
    all_decls: &[Decl<'a>],
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

    // Build symbol shape map from all and outer declarations to validate argument shapes
    let mut decl_shapes: BTreeMap<&str, (usize, Option<&'a [usize]>)> = BTreeMap::new();
    for d in all_decls.iter().chain(out_decls.iter()) {
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
                    all_decls,
                    out_decls,
                    out_flows,
                    bump,
                    depth + 1,
                )?;
            }
            Decl::Circuit { .. }
            | Decl::Import { .. }
            | Decl::Cloak { .. }
            | Decl::ConstMatrix { .. }
            | Decl::TypeAlias { .. }
            | Decl::AssertStable { .. }
            | Decl::AssertBounded { .. } => {}
        }
    }

    // Monomorphize inner flows using the AstFolder remapper
    let mut remapper = CircuitRemapFolder {
        prefix: inst_name,
        param_map: &param_map,
        template_vals: &template_val_map,
        bump,
    };

    for f in inner_flows {
        out_flows.push(remapper.fold_flow(f)?);
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

struct CircuitRemapFolder<'a, 'b> {
    prefix: &'b str,
    param_map: &'b BTreeMap<&'a str, NodeTarget<'a>>,
    template_vals: &'b BTreeMap<&'a str, f64>,
    bump: &'a Bump,
}

impl<'a, 'b> AstFolder<'a> for CircuitRemapFolder<'a, 'b> {
    fn bump(&self) -> &'a Bump {
        self.bump
    }

    fn fold_target(&mut self, target: &NodeTarget<'a>) -> Result<NodeTarget<'a>, String> {
        Ok(remap_target(target, self.prefix, self.param_map, self.bump))
    }

    fn fold_expr(&mut self, expr: &'a Expr<'a>) -> Result<&'a Expr<'a>, String> {
        match *expr {
            Expr::Number(_) => Ok(expr),
            Expr::Ident(name) => {
                if let Some(&val) = self.template_vals.get(name) {
                    Ok(self.bump.alloc(Expr::Number(val)))
                } else if let Some(&subst) = self.param_map.get(name) {
                    if let Some(idx) = subst.index {
                        Ok(self.bump.alloc(Expr::Index {
                            name: subst.name,
                            index: idx,
                        }))
                    } else {
                        Ok(self.bump.alloc(Expr::Ident(subst.name)))
                    }
                } else {
                    let scoped = self.bump.alloc_str(&format!("{}::{}", self.prefix, name));
                    Ok(self.bump.alloc(Expr::Ident(scoped)))
                }
            }
            Expr::Index { name, index } => {
                if let Some(&subst) = self.param_map.get(name) {
                    Ok(self.bump.alloc(Expr::Index {
                        name: subst.name,
                        index,
                    }))
                } else {
                    let scoped = self.bump.alloc_str(&format!("{}::{}", self.prefix, name));
                    Ok(self.bump.alloc(Expr::Index {
                        name: scoped,
                        index,
                    }))
                }
            }
            Expr::MultiIndex { name, indices } => {
                if let Some(&subst) = self.param_map.get(name) {
                    Ok(self.bump.alloc(Expr::MultiIndex {
                        name: subst.name,
                        indices,
                    }))
                } else {
                    let scoped = self.bump.alloc_str(&format!("{}::{}", self.prefix, name));
                    Ok(self.bump.alloc(Expr::MultiIndex {
                        name: scoped,
                        indices,
                    }))
                }
            }
            Expr::DynamicIndex { name, addr } => {
                let remapped_name = if let Some(&subst) = self.param_map.get(name) {
                    subst.name
                } else {
                    self.bump.alloc_str(&format!("{}::{}", self.prefix, name)) as &'a str
                };
                let remapped_addr = if let Some(&subst) = self.param_map.get(addr) {
                    subst.name
                } else {
                    self.bump.alloc_str(&format!("{}::{}", self.prefix, addr)) as &'a str
                };
                Ok(self.bump.alloc(Expr::DynamicIndex {
                    name: remapped_name,
                    addr: remapped_addr,
                }))
            }
            _ => walk_expr(self, expr),
        }
    }
}
