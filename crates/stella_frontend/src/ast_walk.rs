// crates/stella_frontend/src/ast_walk.rs
use crate::ast::*;
use alloc::string::String;
use alloc::vec::Vec;
use bumpalo::Bump;

/// Generic AST folder trait to eliminate repetitive recursive tree traversals.
pub trait AstFolder<'a> {
    fn bump(&self) -> &'a Bump;

    fn fold_expr(&mut self, expr: &'a Expr<'a>) -> Result<&'a Expr<'a>, String> {
        walk_expr(self, expr)
    }

    fn fold_target(&mut self, target: &NodeTarget<'a>) -> Result<NodeTarget<'a>, String> {
        Ok(*target)
    }

    fn fold_flow(&mut self, flow: &Flow<'a>) -> Result<Flow<'a>, String> {
        walk_flow(self, flow)
    }
}

pub fn walk_expr<'a, F: AstFolder<'a> + ?Sized>(
    folder: &mut F,
    expr: &'a Expr<'a>,
) -> Result<&'a Expr<'a>, String> {
    let bump = folder.bump();
    match *expr {
        Expr::Number(_) => Ok(expr),
        Expr::Ident(_) => Ok(expr),
        Expr::Index { name, index } => Ok(bump.alloc(Expr::Index { name, index })),
        Expr::MultiIndex { name, indices } => Ok(bump.alloc(Expr::MultiIndex { name, indices })),
        Expr::DynamicIndex { name, addr } => Ok(bump.alloc(Expr::DynamicIndex { name, addr })),
        Expr::Binary { op, left, right } => {
            let l = folder.fold_expr(left)?;
            let r = folder.fold_expr(right)?;
            Ok(bump.alloc(Expr::Binary {
                op,
                left: l,
                right: r,
            }))
        }
        Expr::Unary { op, inner } => {
            let i = folder.fold_expr(inner)?;
            Ok(bump.alloc(Expr::Unary { op, inner: i }))
        }
        Expr::Activation { kind, expr: inner } => {
            let e = folder.fold_expr(inner)?;
            Ok(bump.alloc(Expr::Activation { kind, expr: e }))
        }
        Expr::CircuitCall {
            circuit,
            template_args,
            args,
        } => {
            let mut folded_args = Vec::with_capacity(args.len());
            for a in args {
                folded_args.push(folder.fold_expr(a)?);
            }
            Ok(bump.alloc(Expr::CircuitCall {
                circuit,
                template_args,
                args: bump.alloc_slice_copy(&folded_args),
            }))
        }
        Expr::Curve {
            expr: inner,
            points,
        } => {
            let e = folder.fold_expr(inner)?;
            Ok(bump.alloc(Expr::Curve { expr: e, points }))
        }
        Expr::Match {
            expr: inner,
            pattern,
        } => {
            let e = folder.fold_expr(inner)?;
            Ok(bump.alloc(Expr::Match { expr: e, pattern }))
        }
        Expr::Conv2d {
            input,
            kernel,
            stride,
            padding,
        } => {
            let inp = folder.fold_expr(input)?;
            Ok(bump.alloc(Expr::Conv2d {
                input: inp,
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
            let inp = folder.fold_expr(input)?;
            Ok(bump.alloc(Expr::AvgPool2d {
                input: inp,
                kernel_size,
                stride,
            }))
        }
    }
}

pub fn walk_flow<'a, F: AstFolder<'a> + ?Sized>(
    folder: &mut F,
    flow: &Flow<'a>,
) -> Result<Flow<'a>, String> {
    let bump = folder.bump();
    match *flow {
        Flow::Synapse {
            src,
            dest,
            weight,
            when,
        } => Ok(Flow::Synapse {
            src: folder.fold_target(&src)?,
            dest: folder.fold_target(&dest)?,
            weight,
            when: match when {
                Some(w) => Some(folder.fold_expr(w)?),
                None => None,
            },
        }),
        Flow::Inhibit { src, dest, when } => Ok(Flow::Inhibit {
            src: folder.fold_target(&src)?,
            dest: folder.fold_target(&dest)?,
            when: match when {
                Some(w) => Some(folder.fold_expr(w)?),
                None => None,
            },
        }),
        Flow::Broadcast {
            src,
            dests,
            weight,
            when,
        } => {
            let mut folded_dests = Vec::with_capacity(dests.len());
            for d in dests {
                folded_dests.push(folder.fold_target(d)?);
            }
            Ok(Flow::Broadcast {
                src: folder.fold_target(&src)?,
                dests: bump.alloc_slice_copy(&folded_dests),
                weight,
                when: match when {
                    Some(w) => Some(folder.fold_expr(w)?),
                    None => None,
                },
            })
        }
        Flow::FanIn {
            srcs,
            dest,
            weight,
            when,
        } => {
            let mut folded_srcs = Vec::with_capacity(srcs.len());
            for s in srcs {
                folded_srcs.push(folder.fold_target(s)?);
            }
            Ok(Flow::FanIn {
                srcs: bump.alloc_slice_copy(&folded_srcs),
                dest: folder.fold_target(&dest)?,
                weight,
                when: match when {
                    Some(w) => Some(folder.fold_expr(w)?),
                    None => None,
                },
            })
        }
        Flow::BroadcastInhibit { src, dests, when } => {
            let mut folded_dests = Vec::with_capacity(dests.len());
            for d in dests {
                folded_dests.push(folder.fold_target(d)?);
            }
            Ok(Flow::BroadcastInhibit {
                src: folder.fold_target(&src)?,
                dests: bump.alloc_slice_copy(&folded_dests),
                when: match when {
                    Some(w) => Some(folder.fold_expr(w)?),
                    None => None,
                },
            })
        }
        Flow::FanInInhibit { srcs, dest, when } => {
            let mut folded_srcs = Vec::with_capacity(srcs.len());
            for s in srcs {
                folded_srcs.push(folder.fold_target(s)?);
            }
            Ok(Flow::FanInInhibit {
                srcs: bump.alloc_slice_copy(&folded_srcs),
                dest: folder.fold_target(&dest)?,
                when: match when {
                    Some(w) => Some(folder.fold_expr(w)?),
                    None => None,
                },
            })
        }
        Flow::Assign { dest, expr } => Ok(Flow::Assign {
            dest: folder.fold_target(&dest)?,
            expr: folder.fold_expr(expr)?,
        }),
        Flow::Drain(target) => Ok(Flow::Drain(folder.fold_target(&target)?)),
        Flow::Minimize(expr) => Ok(Flow::Minimize(folder.fold_expr(expr)?)),
        Flow::Plastic {
            pre,
            dest,
            rate,
            decay,
            kind,
            modulator,
        } => Ok(Flow::Plastic {
            pre: folder.fold_target(&pre)?,
            dest: folder.fold_target(&dest)?,
            rate,
            decay,
            kind,
            modulator: modulator.map(|m| folder.fold_target(&m)).transpose()?,
        }),
        Flow::Probe { label, expr } => Ok(Flow::Probe {
            label,
            expr: folder.fold_expr(expr)?,
        }),
        Flow::Emit {
            template,
            args,
            cond,
            on_stable,
        } => {
            let mut folded_args = Vec::with_capacity(args.len());
            for &a in args.iter() {
                folded_args.push(folder.fold_expr(a)?);
            }
            let folded_cond = cond.map(|c| folder.fold_expr(c)).transpose()?;
            Ok(Flow::Emit {
                template,
                args: folder.bump().alloc_slice_copy(&folded_args),
                cond: folded_cond,
                on_stable,
            })
        }
        Flow::Return(expr) => Ok(Flow::Return(folder.fold_expr(expr)?)),
        Flow::GatedSynapse {
            src,
            dest,
            gate,
            weight,
        } => Ok(Flow::GatedSynapse {
            src: folder.fold_target(&src)?,
            dest: folder.fold_target(&dest)?,
            gate: folder.fold_expr(gate)?,
            weight,
        }),
        Flow::ShuntedSynapse {
            src,
            dest,
            shunt,
            weight,
        } => Ok(Flow::ShuntedSynapse {
            src: folder.fold_target(&src)?,
            dest: folder.fold_target(&dest)?,
            shunt: folder.fold_expr(shunt)?,
            weight,
        }),
        Flow::BroadcastGated {
            src,
            dests,
            gate,
            weight,
        } => {
            let mut folded_dests = Vec::with_capacity(dests.len());
            for d in dests {
                folded_dests.push(folder.fold_target(d)?);
            }
            Ok(Flow::BroadcastGated {
                src: folder.fold_target(&src)?,
                dests: bump.alloc_slice_copy(&folded_dests),
                gate: folder.fold_expr(gate)?,
                weight,
            })
        }
        Flow::BroadcastShunted {
            src,
            dests,
            shunt,
            weight,
        } => {
            let mut folded_dests = Vec::with_capacity(dests.len());
            for d in dests {
                folded_dests.push(folder.fold_target(d)?);
            }
            Ok(Flow::BroadcastShunted {
                src: folder.fold_target(&src)?,
                dests: bump.alloc_slice_copy(&folded_dests),
                shunt: folder.fold_expr(shunt)?,
                weight,
            })
        }
        Flow::FanInGated {
            srcs,
            dest,
            gate,
            weight,
        } => {
            let mut folded_srcs = Vec::with_capacity(srcs.len());
            for s in srcs {
                folded_srcs.push(folder.fold_target(s)?);
            }
            Ok(Flow::FanInGated {
                srcs: bump.alloc_slice_copy(&folded_srcs),
                dest: folder.fold_target(&dest)?,
                gate: folder.fold_expr(gate)?,
                weight,
            })
        }
        Flow::FanInShunted {
            srcs,
            dest,
            shunt,
            weight,
        } => {
            let mut folded_srcs = Vec::with_capacity(srcs.len());
            for s in srcs {
                folded_srcs.push(folder.fold_target(s)?);
            }
            Ok(Flow::FanInShunted {
                srcs: bump.alloc_slice_copy(&folded_srcs),
                dest: folder.fold_target(&dest)?,
                shunt: folder.fold_expr(shunt)?,
                weight,
            })
        }
        Flow::If {
            cond,
            then_flows,
            else_flows,
        } => {
            let mut new_then = Vec::with_capacity(then_flows.len());
            for f in then_flows {
                new_then.push(folder.fold_flow(f)?);
            }
            let new_else = match else_flows {
                Some(ef) => {
                    let mut arr = Vec::with_capacity(ef.len());
                    for f in ef {
                        arr.push(folder.fold_flow(f)?);
                    }
                    Some(bump.alloc_slice_copy(&arr) as &'a [Flow<'a>])
                }
                None => None,
            };
            Ok(Flow::If {
                cond: folder.fold_expr(cond)?,
                then_flows: bump.alloc_slice_copy(&new_then),
                else_flows: new_else,
            })
        }
        Flow::While { cond, body } => {
            let mut new_body = Vec::with_capacity(body.len());
            for f in body {
                new_body.push(folder.fold_flow(f)?);
            }
            Ok(Flow::While {
                cond: folder.fold_expr(cond)?,
                body: bump.alloc_slice_copy(&new_body),
            })
        }
        Flow::Compete {
            branches,
            threshold,
        } => {
            let mut new_branches = Vec::with_capacity(branches.len());
            for b in branches {
                let mut b_flows = Vec::with_capacity(b.flows.len());
                for f in b.flows {
                    b_flows.push(folder.fold_flow(f)?);
                }
                let head = match b.head {
                    Some(h) => Some(folder.fold_target(&h)?),
                    None => None,
                };
                new_branches.push(CompeteBranch {
                    name: b.name,
                    flows: bump.alloc_slice_copy(&b_flows),
                    head,
                });
            }
            Ok(Flow::Compete {
                branches: bump.alloc_slice_copy(&new_branches),
                threshold,
            })
        }
        Flow::Relax {
            body,
            tolerance,
            timeout,
        } => {
            let mut new_body = Vec::with_capacity(body.len());
            for f in body {
                new_body.push(folder.fold_flow(f)?);
            }
            Ok(Flow::Relax {
                body: bump.alloc_slice_copy(&new_body),
                tolerance,
                timeout,
            })
        }
        Flow::Superpose {
            branches,
            collapse_expr,
            dest,
        } => {
            let mut new_branches = Vec::with_capacity(branches.len());
            for b in branches {
                let mut b_flows = Vec::with_capacity(b.flows.len());
                for f in b.flows {
                    b_flows.push(folder.fold_flow(f)?);
                }
                new_branches.push(SuperposeBranch {
                    name: b.name,
                    flows: bump.alloc_slice_copy(&b_flows),
                });
            }
            Ok(Flow::Superpose {
                branches: bump.alloc_slice_copy(&new_branches),
                collapse_expr: folder.fold_expr(collapse_expr)?,
                dest: folder.fold_target(&dest)?,
            })
        }
        Flow::Bifurcate { expr, branches } => {
            let mut new_branches = Vec::with_capacity(branches.len());
            for b in branches {
                let mut b_flows = Vec::with_capacity(b.flows.len());
                for f in b.flows {
                    b_flows.push(folder.fold_flow(f)?);
                }
                let target = match b.target {
                    Some(t) => Some(folder.fold_target(&t)?),
                    None => None,
                };
                let cond = match b.cond {
                    BifurcateCond::When(w) => BifurcateCond::When(folder.fold_expr(w)?),
                    other => other,
                };
                new_branches.push(BifurcateBranch {
                    cond,
                    target,
                    flows: bump.alloc_slice_copy(&b_flows),
                });
            }
            Ok(Flow::Bifurcate {
                expr: folder.fold_expr(expr)?,
                branches: bump.alloc_slice_copy(&new_branches),
            })
        }
        Flow::MultiBranch { src, branches } => {
            let mut new_branches = Vec::with_capacity(branches.len());
            for b in branches {
                let mut b_dests = Vec::with_capacity(b.dests.len());
                for d in b.dests {
                    b_dests.push(folder.fold_target(d)?);
                }
                let mut b_steps = Vec::with_capacity(b.steps.len());
                for s in b.steps {
                    b_steps.push(*s);
                }
                new_branches.push(BranchPipeline {
                    steps: bump.alloc_slice_copy(&b_steps),
                    dests: bump.alloc_slice_copy(&b_dests),
                });
            }
            Ok(Flow::MultiBranch {
                src: folder.fold_target(&src)?,
                branches: bump.alloc_slice_copy(&new_branches),
            })
        }
    }
}
