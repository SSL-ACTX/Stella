// crates/stella_frontend/src/parser/handlers/flows.rs
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use pest::iterators::Pair;

use crate::ast::*;
use crate::parser::{Lowerer, Rule};

impl<'a> Lowerer<'a> {
    pub fn lower_node_target(&self, pair: Pair<'a, Rule>) -> Result<NodeTarget<'a>, String> {
        if pair.as_rule() == Rule::expr {
            let expr = self.lower_expr(pair)?;
            return self.expr_to_target(expr);
        }
        if pair.as_rule() != Rule::node_target {
            return Err(format!("Expected node_target, got {:?}", pair.as_rule()));
        }
        let mut inner = pair.into_inner();
        let first_id = inner.next().unwrap().as_str().trim();

        let mut name = first_id;
        let mut target_index_pair = None;

        for p in inner {
            match p.as_rule() {
                Rule::ident => {
                    name = self
                        .bump
                        .alloc_str(&format!("{}.{}", first_id, p.as_str().trim()));
                }
                Rule::target_index => {
                    target_index_pair = Some(p);
                }
                _ => {}
            }
        }

        if let Some(t_idx) = target_index_pair {
            let inner = t_idx.into_inner().next().unwrap();
            let rule = inner.as_rule();
            match rule {
                Rule::int_lit => {
                    let idx = inner.as_str().trim().parse::<usize>().unwrap();
                    Ok(NodeTarget::indexed(name, idx))
                }
                Rule::slice_range => {
                    let mut nums = inner.into_inner();
                    let start = nums
                        .next()
                        .unwrap()
                        .as_str()
                        .trim()
                        .parse::<usize>()
                        .unwrap();
                    let end = nums
                        .next()
                        .unwrap()
                        .as_str()
                        .trim()
                        .parse::<usize>()
                        .unwrap();
                    Ok(NodeTarget::sliced(name, start, end))
                }
                Rule::multi_index => {
                    let mut dims = Vec::new();
                    for d in inner.clone().into_inner() {
                        dims.push(d.as_str().trim().parse::<usize>().unwrap());
                    }
                    if dims.is_empty() {
                        for part in inner.as_str().split(',') {
                            if let Ok(v) = part.trim().parse::<usize>() {
                                dims.push(v);
                            }
                        }
                    }
                    Ok(NodeTarget::multidim(
                        name,
                        self.bump.alloc_slice_copy(&dims),
                    ))
                }
                Rule::dynamic_addr | Rule::ident => {
                    let addr = inner.as_str().trim().trim_start_matches('@');
                    Ok(NodeTarget::dynamic(name, addr))
                }
                other => Err(format!("Unknown target index: {:?}", other)),
            }
        } else {
            Ok(NodeTarget::simple(name))
        }
    }

    pub fn expr_to_target(&self, expr: &Expr<'a>) -> Result<NodeTarget<'a>, String> {
        match expr {
            Expr::Ident(name) => Ok(NodeTarget::simple(name)),
            Expr::Index { name, index } => Ok(NodeTarget::indexed(name, *index)),
            Expr::MultiIndex { name, indices } => Ok(NodeTarget::multidim(name, indices)),
            Expr::DynamicIndex { name, addr } => Ok(NodeTarget::dynamic(name, addr)),
            _ => Err(format!(
                "Expression cannot be converted to node target: {:?}",
                expr
            )),
        }
    }

    pub fn lower_target_bundle(
        &self,
        pair: Pair<'a, Rule>,
    ) -> Result<&'a [NodeTarget<'a>], String> {
        let mut list = Vec::new();
        for item in pair.into_inner() {
            list.push(self.lower_node_target(item)?);
        }
        Ok(self.bump.alloc_slice_copy(&list))
    }

    pub fn lower_flow_stmt(&mut self, pair: Pair<'a, Rule>) -> Result<Vec<Flow<'a>>, String> {
        let inner = pair.into_inner().next().unwrap();
        match inner.as_rule() {
            Rule::synaptic_flow => self.lower_synaptic_flow(inner),
            Rule::assignment_flow => {
                let mut pairs = inner.into_inner();
                let dest = self.lower_node_target(pairs.next().unwrap())?;
                let op = pairs.next().unwrap().as_str();
                let right_expr = self.lower_expr(pairs.next().unwrap())?;

                let expr = match op {
                    "=" => right_expr,
                    "+=" => {
                        let left = self.target_to_expr(dest);
                        self.bump.alloc(Expr::Binary {
                            op: BinOp::Add,
                            left,
                            right: right_expr,
                        })
                    }
                    "-=" => {
                        let left = self.target_to_expr(dest);
                        self.bump.alloc(Expr::Binary {
                            op: BinOp::Sub,
                            left,
                            right: right_expr,
                        })
                    }
                    _ => right_expr,
                };

                let when_cond = if let Some(w) = pairs.next() {
                    Some(self.lower_expr(w.into_inner().next().unwrap())?)
                } else {
                    None
                };

                if let Some(cond) = when_cond {
                    let assign = self.bump.alloc_slice_copy(&[Flow::Assign { dest, expr }]);
                    Ok(vec![Flow::If {
                        cond,
                        then_flows: assign,
                        else_flows: None,
                    }])
                } else {
                    Ok(vec![Flow::Assign { dest, expr }])
                }
            }
            Rule::emit_stmt => {
                let mut pairs = inner.into_inner();
                let tmpl_pair = pairs.next().unwrap();
                let template = tmpl_pair.as_str().trim_matches('"');
                let mut args: Vec<&'a Expr<'a>> = Vec::new();
                let mut cond: Option<&'a Expr<'a>> = None;
                let mut on_stable = false;

                for p in pairs {
                    match p.as_rule() {
                        Rule::expr => {
                            args.push(self.lower_expr(p)?);
                        }
                        Rule::emit_cond => {
                            let text = p.as_str().trim();
                            if text.starts_with("on") {
                                let id_str = text.trim_start_matches("on").trim();
                                if id_str == "stable" {
                                    on_stable = true;
                                } else {
                                    cond = Some(
                                        self.bump.alloc(Expr::Ident(self.bump.alloc_str(id_str))),
                                    );
                                }
                            } else {
                                for sub in p.into_inner() {
                                    if sub.as_rule() == Rule::expr {
                                        cond = Some(self.lower_expr(sub)?);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }

                Ok(vec![Flow::Emit {
                    template,
                    args: self.bump.alloc_slice_copy(&args),
                    cond,
                    on_stable,
                }])
            }
            Rule::return_stmt => {
                let expr_pair = inner.into_inner().next().unwrap();
                let expr = self.lower_expr(expr_pair)?;
                Ok(vec![Flow::Return(expr)])
            }
            Rule::probe_stmt => {
                let mut pairs = inner.into_inner();
                let mut label = None;
                let first = pairs.next().unwrap();
                let expr_pair = if first.as_rule() == Rule::string_lit {
                    label = Some(first.as_str().trim_matches('"'));
                    pairs.next().unwrap()
                } else {
                    first
                };
                let expr = self.lower_expr(expr_pair)?;
                Ok(vec![Flow::Probe { label, expr }])
            }
            Rule::drain_stmt => {
                let mut pairs = inner.into_inner();
                let target = self.lower_node_target(pairs.next().unwrap())?;
                if let Some(w) = pairs.next() {
                    let cond = self.lower_expr(w.into_inner().next().unwrap())?;
                    let drain_slice = self.bump.alloc_slice_copy(&[Flow::Drain(target)]);
                    Ok(vec![Flow::If {
                        cond,
                        then_flows: drain_slice,
                        else_flows: None,
                    }])
                } else {
                    Ok(vec![Flow::Drain(target)])
                }
            }
            Rule::learn_stmt => {
                let mut pairs = inner.into_inner();
                let pre = self.lower_node_target(pairs.next().unwrap())?;
                let dest = self.lower_node_target(pairs.next().unwrap())?;
                let mut rate = 0.1;
                let mut decay = 0.01;
                let mut modulator = None;

                let mut parsed_numbers = Vec::new();
                for p in pairs {
                    match p.as_rule() {
                        Rule::number => {
                            parsed_numbers.push(p.as_str().parse::<f64>().unwrap());
                        }
                        Rule::plastic_mod => {
                            let mod_target =
                                self.lower_node_target(p.into_inner().next().unwrap())?;
                            modulator = Some(mod_target);
                        }
                        _ => {}
                    }
                }
                if let Some(r) = parsed_numbers.get(0) {
                    rate = *r;
                }
                if let Some(d) = parsed_numbers.get(1) {
                    decay = *d;
                }

                Ok(vec![Flow::Plastic {
                    pre,
                    dest,
                    rate,
                    decay,
                    kind: "hebbian",
                    modulator,
                }])
            }
            Rule::minimize_stmt => {
                let expr_pair = inner.into_inner().next().unwrap();
                let expr = self.lower_expr(expr_pair)?;
                Ok(vec![Flow::Minimize(expr)])
            }
            Rule::synapse_block => {
                let mut pairs = inner.into_inner();
                let pre = self.lower_node_target(pairs.next().unwrap())?;
                let dest = self.lower_node_target(pairs.next().unwrap())?;

                let mut rate = 0.05;
                let mut decay = 0.001;
                let mut kind = "hebbian";

                for prop in pairs {
                    let mut prop_inner = prop.into_inner();
                    let key = prop_inner.next().unwrap().as_str();
                    let val = prop_inner.next().unwrap().as_str();
                    match key {
                        "rate" => rate = val.parse::<f64>().unwrap_or(0.05),
                        "decay" => decay = val.parse::<f64>().unwrap_or(0.001),
                        "rule" => kind = self.bump.alloc_str(val),
                        _ => {}
                    }
                }

                Ok(vec![Flow::Plastic {
                    pre,
                    dest,
                    rate,
                    decay,
                    kind,
                    modulator: None,
                }])
            }
            Rule::if_flow => {
                let mut pairs = inner.into_inner();
                let cond = self.lower_expr(pairs.next().unwrap())?;
                let mut then_flows = Vec::new();
                let mut else_flows = None;

                let then_pair = pairs.next().unwrap();
                for p in then_pair.into_inner() {
                    let mut sub = self.lower_flow_stmt(p)?;
                    then_flows.append(&mut sub);
                }

                if let Some(else_pair) = pairs.next() {
                    let mut e_flows = Vec::new();
                    for p in else_pair.into_inner() {
                        let mut sub = self.lower_flow_stmt(p)?;
                        e_flows.append(&mut sub);
                    }
                    else_flows = Some(self.bump.alloc_slice_copy(&e_flows) as &'a [Flow<'a>]);
                }

                Ok(vec![Flow::If {
                    cond,
                    then_flows: self.bump.alloc_slice_copy(&then_flows),
                    else_flows,
                }])
            }
            Rule::while_flow => {
                let mut pairs = inner.into_inner();
                let cond = self.lower_expr(pairs.next().unwrap())?;
                let mut body = Vec::new();
                for p in pairs {
                    let mut sub = self.lower_flow_stmt(p)?;
                    body.append(&mut sub);
                }
                Ok(vec![Flow::While {
                    cond,
                    body: self.bump.alloc_slice_copy(&body),
                }])
            }
            Rule::compete_stmt => {
                let mut branches = Vec::new();
                let mut threshold = 0.5;

                for p in inner.into_inner() {
                    match p.as_rule() {
                        Rule::compete_branch => {
                            let mut b_pairs = p.into_inner();
                            let name = b_pairs.next().unwrap().as_str();
                            let mut branch_flows = Vec::new();
                            for f in b_pairs {
                                let mut sub = self.lower_flow_stmt(f)?;
                                branch_flows.append(&mut sub);
                            }
                            let mut head = None;
                            for f in &branch_flows {
                                match f {
                                    Flow::Assign { dest, .. } => {
                                        head = Some(*dest);
                                        break;
                                    }
                                    Flow::Synapse { dest, .. } => {
                                        head = Some(*dest);
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                            if head.is_none() {
                                head = Some(NodeTarget::simple(name));
                            }
                            branches.push(CompeteBranch {
                                name,
                                flows: self.bump.alloc_slice_copy(&branch_flows),
                                head,
                            });
                        }
                        Rule::number => {
                            threshold = p.as_str().parse::<f64>().unwrap_or(0.5);
                        }
                        _ => {}
                    }
                }

                Ok(vec![Flow::Compete {
                    branches: self.bump.alloc_slice_copy(&branches),
                    threshold,
                }])
            }
            Rule::relax_stmt => {
                let mut body = Vec::new();
                let mut tolerance = 0.001;
                let mut timeout = None;

                for p in inner.into_inner() {
                    match p.as_rule() {
                        Rule::flow_stmt => {
                            let mut sub = self.lower_flow_stmt(p)?;
                            body.append(&mut sub);
                        }
                        Rule::number => {
                            tolerance = p.as_str().parse::<f64>().unwrap();
                        }
                        Rule::int_lit => {
                            timeout = Some(p.as_str().parse::<usize>().unwrap());
                        }
                        _ => {}
                    }
                }

                Ok(vec![Flow::Relax {
                    body: self.bump.alloc_slice_copy(&body),
                    tolerance,
                    timeout,
                }])
            }
            Rule::superpose_stmt => {
                let mut branches = Vec::new();
                let mut collapse_expr = None;
                let mut dest = None;

                for p in inner.into_inner() {
                    match p.as_rule() {
                        Rule::superpose_branch => {
                            let mut b_pairs = p.into_inner();
                            let name = b_pairs.next().unwrap().as_str();
                            let mut branch_flows = Vec::new();
                            for f in b_pairs {
                                let mut sub = self.lower_flow_stmt(f)?;
                                branch_flows.append(&mut sub);
                            }
                            branches.push(SuperposeBranch {
                                name,
                                flows: self.bump.alloc_slice_copy(&branch_flows),
                            });
                        }
                        Rule::expr => {
                            collapse_expr = Some(self.lower_expr(p)?);
                        }
                        Rule::node_target => {
                            dest = Some(self.lower_node_target(p)?);
                        }
                        _ => {}
                    }
                }

                Ok(vec![Flow::Superpose {
                    branches: self.bump.alloc_slice_copy(&branches),
                    collapse_expr: collapse_expr.unwrap(),
                    dest: dest.unwrap(),
                }])
            }
            Rule::bifurcate_stmt => {
                let mut pairs = inner.into_inner();
                let expr = self.lower_expr(pairs.next().unwrap())?;
                let mut branches = Vec::new();

                for b in pairs {
                    let mut b_inner = b.into_inner();
                    let cond_pair = b_inner.next().unwrap();
                    let cond = self.lower_bifurcate_cond(cond_pair)?;

                    let mut target = None;
                    let mut b_flows = Vec::new();

                    for p in b_inner {
                        match p.as_rule() {
                            Rule::node_target => {
                                target = Some(self.lower_node_target(p)?);
                            }
                            Rule::flow_stmt => {
                                let mut sub = self.lower_flow_stmt(p)?;
                                b_flows.append(&mut sub);
                            }
                            _ => {}
                        }
                    }

                    branches.push(BifurcateBranch {
                        cond,
                        target,
                        flows: self.bump.alloc_slice_copy(&b_flows),
                    });
                }

                Ok(vec![Flow::Bifurcate {
                    expr,
                    branches: self.bump.alloc_slice_copy(&branches),
                }])
            }
            other => Err(format!("Unknown flow statement rule: {:?}", other)),
        }
    }

    pub fn lower_bifurcate_cond(
        &mut self,
        pair: Pair<'a, Rule>,
    ) -> Result<BifurcateCond<'a>, String> {
        let text = pair.as_str();
        if text.starts_with("<=") {
            let num = text[2..].trim().parse::<f64>().unwrap();
            Ok(BifurcateCond::Lte(num))
        } else if text.starts_with("<") {
            let num = text[1..].trim().parse::<f64>().unwrap();
            Ok(BifurcateCond::Lt(num))
        } else if text.starts_with(">=") {
            let num = text[2..].trim().parse::<f64>().unwrap();
            Ok(BifurcateCond::Gte(num))
        } else if text.starts_with(">") {
            let num = text[1..].trim().parse::<f64>().unwrap();
            Ok(BifurcateCond::Gt(num))
        } else if text.starts_with("==") || text.starts_with("=") {
            let num = text.trim_start_matches('=').trim().parse::<f64>().unwrap();
            Ok(BifurcateCond::Eq(num))
        } else if text.starts_with('[') {
            let inner = text.trim_matches('[').trim_matches(']');
            let parts: Vec<&str> = inner.split("..").collect();
            let low = parts[0].trim().parse::<f64>().unwrap();
            let high = parts[1].trim().parse::<f64>().unwrap();
            Ok(BifurcateCond::Range(low, high))
        } else if text == "else" {
            Ok(BifurcateCond::Else)
        } else if text.starts_with("when") {
            let expr_pair = pair.into_inner().next().unwrap();
            let c = self.lower_expr(expr_pair)?;
            Ok(BifurcateCond::When(c))
        } else {
            Err(format!("Unknown bifurcation condition: {}", text))
        }
    }

    pub fn lower_synaptic_flow(&mut self, pair: Pair<'a, Rule>) -> Result<Vec<Flow<'a>>, String> {
        let mut inner = pair.into_inner();
        let src_wrapper = inner.next().unwrap();
        let src_pair = if src_wrapper.as_rule() == Rule::flow_src {
            let mut s_inner = src_wrapper.into_inner();
            let first = s_inner.next().unwrap();
            if let Some(pipe_chain_pair) = s_inner.next() {
                // node_target followed by pipe_chain: convert to Expr
                let base_target = self.lower_node_target(first)?;
                let mut current_expr = self.target_to_expr(base_target);
                for step in pipe_chain_pair.into_inner() {
                    let mut step_inner = step.into_inner();
                    let step_first = step_inner.next().unwrap();
                    if step_first.as_rule() == Rule::activation_kind {
                        let kind = self.lower_activation_kind(step_first)?;
                        current_expr = self.bump.alloc(Expr::Activation {
                            kind,
                            expr: current_expr,
                        });
                        if let Some(scale_pair) = step_inner.next() {
                            let factor = scale_pair.as_str().parse::<f64>().unwrap_or(1.0);
                            current_expr = self.bump.alloc(Expr::Binary {
                                op: BinOp::Mul,
                                left: current_expr,
                                right: self.bump.alloc(Expr::Number(factor)),
                            });
                        }
                    } else if step_first.as_rule() == Rule::number {
                        let factor = step_first.as_str().parse::<f64>().unwrap_or(1.0);
                        current_expr = self.bump.alloc(Expr::Binary {
                            op: BinOp::Mul,
                            left: current_expr,
                            right: self.bump.alloc(Expr::Number(factor)),
                        });
                    }
                }
                // Wrap as synthetic pair or directly store expr:
                // We'll pass current_expr directly
                let mut flows = Vec::new();
                let mut when_cond = None;
                let mut links = Vec::new();
                for p in inner {
                    if p.as_rule() == Rule::when_cond {
                        when_cond = Some(self.lower_expr(p.into_inner().next().unwrap())?);
                    } else if p.as_rule() == Rule::flow_link {
                        links.push(p);
                    }
                }

                for link in links {
                    let mut link_inner = link.into_inner();
                    let conn = link_inner.next().unwrap();
                    let dest_wrapper = link_inner.next().unwrap();
                    let dest_pair = if dest_wrapper.as_rule() == Rule::flow_dest {
                        dest_wrapper.into_inner().next().unwrap()
                    } else {
                        dest_wrapper
                    };

                    let weight = self.extract_weight(&conn);

                    let final_expr = if (weight - 1.0).abs() > 1e-9 {
                        self.bump.alloc(Expr::Binary {
                            op: BinOp::Mul,
                            left: current_expr,
                            right: self.bump.alloc(Expr::Number(weight)),
                        }) as &'a Expr<'a>
                    } else {
                        current_expr
                    };

                    if dest_pair.as_rule() == Rule::target_bundle {
                        let dests = self.lower_target_bundle(dest_pair)?;
                        for dest in dests {
                            if let Some(cond) = when_cond {
                                let assign = self.bump.alloc_slice_copy(&[Flow::Assign {
                                    dest: *dest,
                                    expr: final_expr,
                                }]);
                                flows.push(Flow::If {
                                    cond,
                                    then_flows: assign,
                                    else_flows: None,
                                });
                            } else {
                                flows.push(Flow::Assign {
                                    dest: *dest,
                                    expr: final_expr,
                                });
                            }
                        }
                    } else {
                        let dest = self.lower_node_target(dest_pair)?;
                        if let Some(cond) = when_cond {
                            let assign = self.bump.alloc_slice_copy(&[Flow::Assign {
                                dest,
                                expr: final_expr,
                            }]);
                            flows.push(Flow::If {
                                cond,
                                then_flows: assign,
                                else_flows: None,
                            });
                        } else {
                            flows.push(Flow::Assign {
                                dest,
                                expr: final_expr,
                            });
                        }
                    }
                }
                return Ok(flows);
            } else {
                first
            }
        } else {
            src_wrapper
        };

        let mut flows = Vec::new();
        let mut when_cond = None;
        let mut links = Vec::new();

        for p in inner {
            if p.as_rule() == Rule::when_cond {
                when_cond = Some(self.lower_expr(p.into_inner().next().unwrap())?);
            } else if p.as_rule() == Rule::flow_link {
                links.push(p);
            }
        }

        let mut current_src_pair = src_pair;
        for link in links {
            let mut link_inner = link.into_inner();
            let conn = link_inner.next().unwrap();
            let dest_wrapper = link_inner.next().unwrap();
            let dest_pair = if dest_wrapper.as_rule() == Rule::flow_dest {
                dest_wrapper.into_inner().next().unwrap()
            } else {
                dest_wrapper
            };

            let mut plastic_mod = None;
            for p in link_inner.by_ref() {
                if p.as_rule() == Rule::plastic_mod {
                    plastic_mod = Some(self.lower_node_target(p.into_inner().next().unwrap())?);
                }
            }

            self.lower_flow_connection(
                current_src_pair.clone(),
                conn,
                dest_pair.clone(),
                when_cond,
                plastic_mod,
                &mut flows,
            )?;
            current_src_pair = dest_pair;
        }

        Ok(flows)
    }

    pub fn lower_flow_connection(
        &mut self,
        src_pair: Pair<'a, Rule>,
        conn: Pair<'a, Rule>,
        dest_pair: Pair<'a, Rule>,
        when: Option<&'a Expr<'a>>,
        plastic_mod: Option<NodeTarget<'a>>,
        flows: &mut Vec<Flow<'a>>,
    ) -> Result<(), String> {
        let is_bundle = dest_pair.as_rule() == Rule::target_bundle;
        let is_multibranch = dest_pair.as_rule() == Rule::multi_branch;
        let conn_str = conn.as_str();

        if is_multibranch {
            let src = self.lower_node_target(src_pair)?;
            let mut branches = Vec::new();
            for b in dest_pair.into_inner() {
                let mut b_inner = b.into_inner();
                let mut steps = Vec::new();
                let mut d_pair = b_inner.next().unwrap();
                if d_pair.as_rule() == Rule::pipe_chain {
                    for step in d_pair.into_inner() {
                        let mut step_inner = step.into_inner();
                        let first = step_inner.next().unwrap();
                        if first.as_rule() == Rule::activation_kind {
                            let act = self.lower_activation_kind(first)?;
                            steps.push(PipelineStep::Filter(act));
                            if let Some(num) = step_inner.next() {
                                steps.push(PipelineStep::Scale(
                                    num.as_str().parse::<f64>().unwrap(),
                                ));
                            }
                        } else if first.as_rule() == Rule::number {
                            steps.push(PipelineStep::Scale(first.as_str().parse::<f64>().unwrap()));
                        }
                    }
                    d_pair = b_inner.next().unwrap();
                }

                let dests = if d_pair.as_rule() == Rule::target_bundle
                    || d_pair.as_rule() == Rule::dest_bundle
                {
                    self.lower_target_bundle(d_pair)?
                } else {
                    let single = self.lower_node_target(d_pair)?;
                    self.bump.alloc_slice_copy(&[single]) as &'a [NodeTarget<'a>]
                };

                branches.push(BranchPipeline {
                    steps: self.bump.alloc_slice_copy(&steps),
                    dests,
                });
            }

            flows.push(Flow::MultiBranch {
                src,
                branches: self.bump.alloc_slice_copy(&branches),
            });
            return Ok(());
        }

        // Fan-in: target_bundle source
        if src_pair.as_rule() == Rule::target_bundle {
            let srcs = self.lower_target_bundle(src_pair)?;
            let dest = self.lower_node_target(dest_pair)?;
            let weight = self.extract_weight(&conn);

            if conn_str.contains("gate:") || conn_str.contains('?') {
                let gate_expr = self.extract_gate_or_shunt_expr(conn.clone())?;
                flows.push(Flow::FanInGated {
                    srcs,
                    dest,
                    gate: gate_expr,
                    weight,
                });
                return Ok(());
            }
            if conn_str.contains("shunt:") || conn_str.contains('!') {
                let shunt_expr = self.extract_gate_or_shunt_expr(conn.clone())?;
                flows.push(Flow::FanInShunted {
                    srcs,
                    dest,
                    shunt: shunt_expr,
                    weight,
                });
                return Ok(());
            }

            if conn_str.contains("~|>") {
                flows.push(Flow::FanInInhibit { srcs, dest, when });
            } else {
                flows.push(Flow::FanIn {
                    srcs,
                    dest,
                    weight,
                    when,
                });
            }
            return Ok(());
        }

        // Broadcast: destination bundle
        if is_bundle {
            let dests = self.lower_target_bundle(dest_pair)?;
            let src = self.lower_node_target(src_pair)?;
            let weight = self.extract_weight(&conn);

            if conn_str.contains("gate:") || conn_str.contains('?') {
                let gate_expr = self.extract_gate_or_shunt_expr(conn.clone())?;
                flows.push(Flow::BroadcastGated {
                    src,
                    dests,
                    gate: gate_expr,
                    weight,
                });
                return Ok(());
            }
            if conn_str.contains("shunt:") || conn_str.contains('!') {
                let shunt_expr = self.extract_gate_or_shunt_expr(conn.clone())?;
                flows.push(Flow::BroadcastShunted {
                    src,
                    dests,
                    shunt: shunt_expr,
                    weight,
                });
                return Ok(());
            }

            if conn_str.contains("~|>") {
                flows.push(Flow::BroadcastInhibit { src, dests, when });
            } else {
                flows.push(Flow::Broadcast {
                    src,
                    dests,
                    weight,
                    when,
                });
            }
            return Ok(());
        }

        // Single -> Single
        let dest = self.lower_node_target(dest_pair)?;

        if conn_str == "<~>" {
            let src = self.lower_node_target(src_pair)?;
            flows.push(Flow::Synapse {
                src,
                dest,
                weight: 1.0,
                when,
            });
            flows.push(Flow::Synapse {
                src: dest,
                dest: src,
                weight: 1.0,
                when,
            });
            return Ok(());
        }

        if conn_str == "<~|>" {
            let src = self.lower_node_target(src_pair)?;
            flows.push(Flow::Inhibit { src, dest, when });
            flows.push(Flow::Inhibit {
                src: dest,
                dest: src,
                when,
            });
            return Ok(());
        }

        if conn_str.starts_with("<|") && conn_str.ends_with("|>") {
            let src = self.lower_node_target(src_pair)?;
            let weight = if conn_str.contains('(') {
                let start = conn_str.find('(').unwrap() + 1;
                let end = conn_str.find(')').unwrap();
                conn_str[start..end].trim().parse::<f64>().unwrap_or(-0.8)
            } else {
                -0.8
            };
            flows.push(Flow::Synapse {
                src,
                dest,
                weight,
                when,
            });
            flows.push(Flow::Synapse {
                src: dest,
                dest: src,
                weight,
                when,
            });
            return Ok(());
        }

        if conn_str.starts_with("<~+(") || conn_str == "<~+~>" {
            let src = self.lower_node_target(src_pair)?;
            let (rate, decay) = if conn_str.contains('(') {
                let start = conn_str.find('(').unwrap() + 1;
                let end = conn_str.find(')').unwrap();
                let inner = &conn_str[start..end];
                let mut parts = inner.split(',');
                let r = parts.next().unwrap().trim().parse::<f64>().unwrap_or(0.05);
                let d = if let Some(decay_part) = parts.next() {
                    let d_clean = decay_part.trim().trim_start_matches("decay=").trim();
                    d_clean.parse::<f64>().unwrap_or(0.01)
                } else {
                    0.01
                };
                (r, d)
            } else {
                (0.05, 0.01)
            };
            flows.push(Flow::Plastic {
                pre: src,
                dest,
                rate,
                decay,
                kind: "hebbian",
                modulator: plastic_mod,
            });
            flows.push(Flow::Plastic {
                pre: dest,
                dest: src,
                rate,
                decay,
                kind: "hebbian",
                modulator: plastic_mod,
            });
            return Ok(());
        }

        if conn_str.starts_with("<~-(") || conn_str == "<~-~>" {
            let src = self.lower_node_target(src_pair)?;
            let (rate, decay) = if conn_str.contains('(') {
                let start = conn_str.find('(').unwrap() + 1;
                let end = conn_str.find(')').unwrap();
                let inner = &conn_str[start..end];
                let mut parts = inner.split(',');
                let r = parts.next().unwrap().trim().parse::<f64>().unwrap_or(0.02);
                let d = if let Some(decay_part) = parts.next() {
                    let d_clean = decay_part.trim().trim_start_matches("decay=").trim();
                    d_clean.parse::<f64>().unwrap_or(0.01)
                } else {
                    0.01
                };
                (r, d)
            } else {
                (0.02, 0.01)
            };
            flows.push(Flow::Plastic {
                pre: src,
                dest,
                rate,
                decay,
                kind: "anti_hebbian",
                modulator: plastic_mod,
            });
            flows.push(Flow::Plastic {
                pre: dest,
                dest: src,
                rate,
                decay,
                kind: "anti_hebbian",
                modulator: plastic_mod,
            });
            return Ok(());
        }

        if conn_str.starts_with("~+(") || conn_str == "~+>" {
            let src = self.lower_node_target(src_pair)?;
            let (rate, decay) = if conn_str.contains('(') {
                let start = conn_str.find('(').unwrap() + 1;
                let end = conn_str.find(')').unwrap();
                let inner = &conn_str[start..end];
                let mut parts = inner.split(',');
                let r = parts.next().unwrap().trim().parse::<f64>().unwrap_or(0.05);
                let d = if let Some(decay_part) = parts.next() {
                    let d_clean = decay_part.trim().trim_start_matches("decay=").trim();
                    d_clean.parse::<f64>().unwrap_or(0.01)
                } else {
                    0.01
                };
                (r, d)
            } else {
                (0.05, 0.01)
            };
            flows.push(Flow::Plastic {
                pre: src,
                dest,
                rate,
                decay,
                kind: "hebbian",
                modulator: plastic_mod,
            });
            return Ok(());
        }

        if conn_str.starts_with("~-(") || conn_str == "~->" {
            let src = self.lower_node_target(src_pair)?;
            let (rate, decay) = if conn_str.contains('(') {
                let start = conn_str.find('(').unwrap() + 1;
                let end = conn_str.find(')').unwrap();
                let inner = &conn_str[start..end];
                let mut parts = inner.split(',');
                let r = parts.next().unwrap().trim().parse::<f64>().unwrap_or(0.02);
                let d = if let Some(decay_part) = parts.next() {
                    let d_clean = decay_part.trim().trim_start_matches("decay=").trim();
                    d_clean.parse::<f64>().unwrap_or(0.01)
                } else {
                    0.01
                };
                (r, d)
            } else {
                (0.02, 0.01)
            };
            flows.push(Flow::Plastic {
                pre: src,
                dest,
                rate,
                decay,
                kind: "anti_hebbian",
                modulator: plastic_mod,
            });
            return Ok(());
        }

        if conn_str.contains("gate:") || conn_str.contains('?') {
            let gate_expr = self.extract_gate_or_shunt_expr(conn.clone())?;
            if let Ok(src) = self.lower_node_target(src_pair.clone()) {
                let weight = self.extract_weight(&conn);
                flows.push(Flow::GatedSynapse {
                    src,
                    dest,
                    gate: gate_expr,
                    weight,
                });
                return Ok(());
            }
        }

        if conn_str.contains("shunt:") || conn_str.contains('!') {
            let shunt_expr = self.extract_gate_or_shunt_expr(conn.clone())?;
            if let Ok(src) = self.lower_node_target(src_pair.clone()) {
                let weight = self.extract_weight(&conn);
                flows.push(Flow::ShuntedSynapse {
                    src,
                    dest,
                    shunt: shunt_expr,
                    weight,
                });
                return Ok(());
            }
        }

        if let Ok(src) = self.lower_node_target(src_pair.clone()) {
            let weight = self.extract_weight(&conn);
            if conn_str.contains("~|>") {
                flows.push(Flow::Inhibit { src, dest, when });
            } else {
                flows.push(Flow::Synapse {
                    src,
                    dest,
                    weight,
                    when,
                });
            }
            return Ok(());
        }

        if let Ok(expr) = self.lower_expr(src_pair.clone()) {
            flows.push(Flow::Assign { dest, expr });
            return Ok(());
        }

        let src = self.lower_node_target(src_pair)?;
        let weight = self.extract_weight(&conn);

        if conn_str.contains("~|>") {
            flows.push(Flow::Inhibit { src, dest, when });
        } else {
            flows.push(Flow::Synapse {
                src,
                dest,
                weight,
                when,
            });
        }

        Ok(())
    }

    pub fn extract_weight(&self, conn: &Pair<'a, Rule>) -> f64 {
        if conn.as_rule() == Rule::conn_weight {
            if let Some(num) = conn.clone().into_inner().next() {
                return num.as_str().parse::<f64>().unwrap_or(1.0);
            }
        }
        for p in conn.clone().into_inner() {
            if p.as_rule() == Rule::conn_weight {
                if let Some(num) = p.into_inner().next() {
                    return num.as_str().parse::<f64>().unwrap_or(1.0);
                }
            }
        }
        1.0
    }

    pub fn extract_gate_or_shunt_expr(&self, conn: Pair<'a, Rule>) -> Result<&'a Expr<'a>, String> {
        for p in conn.into_inner() {
            if p.as_rule() == Rule::gate_spec || p.as_rule() == Rule::shunt_spec {
                for sub in p.into_inner() {
                    if sub.as_rule() == Rule::expr {
                        return self.lower_expr(sub);
                    }
                }
            }
        }
        Err("Missing gate/shunt expression in connector".to_string())
    }

    pub fn target_to_expr(&self, target: NodeTarget<'a>) -> &'a Expr<'a> {
        if let Some(indices) = target.indices {
            self.bump.alloc(Expr::MultiIndex {
                name: target.name,
                indices,
            })
        } else if let Some(index) = target.index {
            self.bump.alloc(Expr::Index {
                name: target.name,
                index,
            })
        } else if let Some(addr) = target.dynamic_index {
            self.bump.alloc(Expr::DynamicIndex {
                name: target.name,
                addr,
            })
        } else {
            self.bump.alloc(Expr::Ident(target.name))
        }
    }
}
