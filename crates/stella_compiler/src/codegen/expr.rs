use alloc::string::String;
use stella_frontend::ast::{ActivationKind, BinOp, Expr, NodeTarget, UnaryOp};

use super::Codegen;

impl<'a> Codegen<'a> {
    pub(super) fn apply_expr_to_dest(
        &mut self,
        dest: usize,
        expr: &Expr<'a>,
    ) -> Result<(), String> {
        if let Some(c) = self.eval_const(expr) {
            self.add_b(dest, c);
            return Ok(());
        }

        match expr {
            Expr::Number(val) => {
                self.add_b(dest, *val);
            }
            Expr::Ident(name) => {
                if let Some(val) = self.layout.resolve_const(name) {
                    self.add_b(dest, val);
                } else {
                    let src = self.layout.resolve_ident(name)?;
                    self.add_w(dest, src, 1.0);
                }
            }
            Expr::Index { name, index } => {
                let target = NodeTarget::indexed(name, *index);
                let src = self.layout.resolve_target(&target)?;
                self.add_w(dest, src, 1.0);
            }
            Expr::MultiIndex { name, indices } => {
                let target = NodeTarget::multidim(name, indices);
                let src = self.layout.resolve_target(&target)?;
                self.add_w(dest, src, 1.0);
            }
            Expr::DynamicIndex { name, addr } => {
                self.compile_dynamic_read(dest, name, addr)?;
            }
            Expr::Curve {
                expr: inner,
                points,
            } => {
                self.compile_piecewise_curve(dest, inner, points)?;
            }
            Expr::Match {
                expr: inner,
                pattern,
            } => {
                let target_len = pattern.len();
                let src_w = match &**inner {
                    Expr::Ident(name) => {
                        self.layout.symbols.get(*name).map(|s| s.width).unwrap_or(1)
                    }
                    _ => 1,
                };
                let compare_len = target_len.min(src_w);

                if compare_len == 0 {
                    self.add_b(dest, 1.0);
                } else {
                    let mut src_base = 0;
                    if let Expr::Ident(name) = &**inner {
                        if let Some(info) = self.layout.symbols.get(*name) {
                            src_base = info.index;
                        }
                    }

                    let weight_scale = 1.0 / (compare_len as f64);
                    let bytes = pattern.as_bytes();

                    for k in 0..compare_len {
                        let expected_val = (bytes[k] as f64) / 255.0;
                        let src_neuron = src_base + k;

                        let diff = self.alloc_scratch();
                        self.add_w(diff, src_neuron, 1.0);
                        self.add_b(diff, -expected_val);

                        let up = self.alloc_scratch();
                        self.add_w(up, diff, 1.0);
                        self.add_b(up, 1.0);

                        let down = self.alloc_scratch();
                        self.add_w(down, diff, -1.0);
                        self.add_b(down, 1.0);

                        let match_k = self.alloc_scratch();
                        self.add_w(match_k, up, 1.0);
                        self.add_w(match_k, down, 1.0);
                        self.add_b(match_k, -1.0);

                        self.add_w(dest, match_k, weight_scale);
                    }
                }
            }
            Expr::Conv2d { .. } => {
                return Err("conv2d expression must be assigned to a tensor destination (e.g. out_feat = conv2d(in_img, K);)".into());
            }
            Expr::AvgPool2d { .. } => {
                return Err("avgpool2d expression must be assigned to a tensor destination (e.g. pooled = avgpool2d(img, 2, 2);)".into());
            }
            Expr::Unary { op, inner } => match op {
                UnaryOp::Neg => {
                    let src = self.resolve_expr_to_neuron(inner)?;
                    self.add_w(dest, src, -1.0);
                }
                UnaryOp::Not => {
                    let src = self.resolve_expr_to_neuron(inner)?;
                    self.add_w(dest, src, -1.0);
                    self.add_b(dest, 1.0);
                }
            },
            Expr::Activation { kind, expr: inner } => match kind {
                ActivationKind::Step => {
                    let src = self.resolve_expr_to_neuron(inner)?;
                    self.add_w(dest, src, 100.0);
                    self.add_b(dest, -50.0);
                }
                ActivationKind::Inv => {
                    let src = self.resolve_expr_to_neuron(inner)?;
                    self.add_w(dest, src, -1.0);
                    self.add_b(dest, 1.0);
                }
                ActivationKind::Clamp | ActivationKind::Relu => {
                    self.apply_expr_to_dest(dest, inner)?;
                }
                ActivationKind::Saturate => {
                    if let Expr::Binary {
                        op: BinOp::Sub,
                        left,
                        right,
                    } = inner
                    {
                        let s1 = self.resolve_expr_to_neuron(left)?;
                        let s2 = self.resolve_expr_to_neuron(right)?;
                        self.add_w(dest, s1, 100.0);
                        self.add_w(dest, s2, -100.0);
                    } else {
                        let src = self.resolve_expr_to_neuron(inner)?;
                        self.add_w(dest, src, 100.0);
                        self.add_b(dest, -50.0);
                    }
                }
            },
            Expr::Binary { op, left, right } => match op {
                BinOp::Add => {
                    let src1 = self.resolve_expr_to_neuron(left)?;
                    let src2 = self.resolve_expr_to_neuron(right)?;
                    self.add_w(dest, src1, 1.0);
                    self.add_w(dest, src2, 1.0);
                }
                BinOp::Sub => {
                    let src1 = self.resolve_expr_to_neuron(left)?;
                    let src2 = self.resolve_expr_to_neuron(right)?;
                    self.add_w(dest, src1, 1.0);
                    self.add_w(dest, src2, -1.0);
                }
                BinOp::Mul => {
                    if let Some(l) = self.eval_const(left) {
                        let src2 = self.resolve_expr_to_neuron(right)?;
                        self.add_w(dest, src2, l);
                    } else if let Some(r) = self.eval_const(right) {
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        self.add_w(dest, src1, r);
                    } else {
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        let src2 = self.resolve_expr_to_neuron(right)?;
                        self.add_w(dest, src1, 1.0);
                        self.add_w(dest, src2, 1.0);
                        self.add_b(dest, -1.0);
                    }
                }
                BinOp::Div => {
                    if let Some(r) = self.eval_const(right) {
                        if r == 0.0 {
                            return Err("Division by zero in expression".into());
                        }
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        self.add_w(dest, src1, 1.0 / r);
                    } else {
                        return Err(
                            "Continuous linear synapses require divisor of '/' to be a constant"
                                .into(),
                        );
                    }
                }
                BinOp::And => {
                    let src1 = self.resolve_expr_to_neuron(left)?;
                    let src2 = self.resolve_expr_to_neuron(right)?;
                    self.add_w(dest, src1, 1.0);
                    self.add_w(dest, src2, 1.0);
                    self.add_b(dest, -1.0);
                }
                BinOp::Or => {
                    let src1 = self.resolve_expr_to_neuron(left)?;
                    let src2 = self.resolve_expr_to_neuron(right)?;
                    self.add_w(dest, src1, 1.0);
                    self.add_w(dest, src2, 1.0);
                }
                BinOp::Nand => {
                    let src1 = self.resolve_expr_to_neuron(left)?;
                    let src2 = self.resolve_expr_to_neuron(right)?;
                    self.add_w(dest, src1, -1.0);
                    self.add_w(dest, src2, -1.0);
                    self.add_b(dest, 2.0);
                }
                BinOp::Nor => {
                    let src1 = self.resolve_expr_to_neuron(left)?;
                    let src2 = self.resolve_expr_to_neuron(right)?;
                    self.add_w(dest, src1, -1.0);
                    self.add_w(dest, src2, -1.0);
                    self.add_b(dest, 1.0);
                }
                BinOp::Xor => {
                    let src1 = self.resolve_expr_to_neuron(left)?;
                    let src2 = self.resolve_expr_to_neuron(right)?;
                    let h1 = self.alloc_scratch();
                    self.add_w(h1, src1, 1.0);
                    self.add_w(h1, src2, 1.0);

                    let h2 = self.alloc_scratch();
                    self.add_w(h2, src1, -1.0);
                    self.add_w(h2, src2, -1.0);
                    self.add_b(h2, 2.0);

                    self.add_w(dest, h1, 1.0);
                    self.add_w(dest, h2, 1.0);
                    self.add_b(dest, -1.0);
                }
                BinOp::Gte => {
                    if let Some(r) = self.eval_const(right) {
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        self.add_w(dest, src1, 100.0);
                        self.add_b(dest, -100.0 * r);
                    } else if let Some(l) = self.eval_const(left) {
                        let src2 = self.resolve_expr_to_neuron(right)?;
                        self.add_w(dest, src2, -100.0);
                        self.add_b(dest, 100.0 * l);
                    } else {
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        let src2 = self.resolve_expr_to_neuron(right)?;
                        self.add_w(dest, src1, 100.0);
                        self.add_w(dest, src2, -100.0);
                    }
                }
                BinOp::Lte => {
                    if let Some(r) = self.eval_const(right) {
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        self.add_w(dest, src1, -100.0);
                        self.add_b(dest, 100.0 * r);
                    } else if let Some(l) = self.eval_const(left) {
                        let src2 = self.resolve_expr_to_neuron(right)?;
                        self.add_w(dest, src2, 100.0);
                        self.add_b(dest, -100.0 * l);
                    } else {
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        let src2 = self.resolve_expr_to_neuron(right)?;
                        self.add_w(dest, src1, -100.0);
                        self.add_w(dest, src2, 100.0);
                    }
                }
                BinOp::Gt => {
                    if let Some(r) = self.eval_const(right) {
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        self.add_w(dest, src1, 100.0);
                        self.add_b(dest, -100.0 * r - 0.01);
                    } else if let Some(l) = self.eval_const(left) {
                        let src2 = self.resolve_expr_to_neuron(right)?;
                        self.add_w(dest, src2, -100.0);
                        self.add_b(dest, 100.0 * l - 0.01);
                    } else {
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        let src2 = self.resolve_expr_to_neuron(right)?;
                        self.add_w(dest, src1, 100.0);
                        self.add_w(dest, src2, -100.0);
                        self.add_b(dest, -0.01);
                    }
                }
                BinOp::Lt => {
                    if let Some(r) = self.eval_const(right) {
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        self.add_w(dest, src1, -100.0);
                        self.add_b(dest, 100.0 * r - 0.01);
                    } else if let Some(l) = self.eval_const(left) {
                        let src2 = self.resolve_expr_to_neuron(right)?;
                        self.add_w(dest, src2, 100.0);
                        self.add_b(dest, -100.0 * l - 0.01);
                    } else {
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        let src2 = self.resolve_expr_to_neuron(right)?;
                        self.add_w(dest, src1, -100.0);
                        self.add_w(dest, src2, 100.0);
                        self.add_b(dest, -0.01);
                    }
                }
                BinOp::Eq => {
                    let src1 = self.resolve_expr_to_neuron(left)?;
                    let src2 = self.resolve_expr_to_neuron(right)?;
                    self.add_w(dest, src1, 50.0);
                    self.add_w(dest, src2, 50.0);
                }
                BinOp::Neq => {
                    let h1 = self.alloc_scratch();
                    let h2 = self.alloc_scratch();
                    if let Some(r) = self.eval_const(right) {
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        self.add_w(h1, src1, 100.0);
                        self.add_b(h1, -100.0 * r - 0.01);

                        self.add_w(h2, src1, -100.0);
                        self.add_b(h2, 100.0 * r - 0.01);
                    } else if let Some(l) = self.eval_const(left) {
                        let src2 = self.resolve_expr_to_neuron(right)?;
                        self.add_w(h1, src2, -100.0);
                        self.add_b(h1, 100.0 * l - 0.01);

                        self.add_w(h2, src2, 100.0);
                        self.add_b(h2, -100.0 * l - 0.01);
                    } else {
                        let src1 = self.resolve_expr_to_neuron(left)?;
                        let src2 = self.resolve_expr_to_neuron(right)?;
                        self.add_w(h1, src1, 100.0);
                        self.add_w(h1, src2, -100.0);
                        self.add_b(h1, -0.01);

                        self.add_w(h2, src2, 100.0);
                        self.add_w(h2, src1, -100.0);
                        self.add_b(h2, -0.01);
                    }

                    self.add_w(dest, h1, 1.0);
                    self.add_w(dest, h2, 1.0);
                }
            },
            Expr::CircuitCall { circuit, .. } => {
                return Err(alloc::format!(
                    "Unexpanded circuit call '{}' reached codegen",
                    circuit
                ));
            }
        }
        Ok(())
    }
}
