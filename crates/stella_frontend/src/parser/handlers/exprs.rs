// crates/stella_frontend/src/parser/handlers/exprs.rs
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use pest::iterators::Pair;
use pest::pratt_parser::{Assoc, Op, PrattParser};

use crate::ast::*;
use crate::parser::{Lowerer, Rule};

pub fn make_pratt() -> PrattParser<Rule> {
    PrattParser::new()
        .op(Op::postfix(Rule::pipe_activation))
        .op(Op::infix(Rule::op_or, Assoc::Left) | Op::infix(Rule::op_nor, Assoc::Left))
        .op(Op::infix(Rule::op_and, Assoc::Left)
            | Op::infix(Rule::op_nand, Assoc::Left)
            | Op::infix(Rule::op_xor, Assoc::Left))
        .op(Op::infix(Rule::op_eq, Assoc::Left)
            | Op::infix(Rule::op_neq, Assoc::Left)
            | Op::infix(Rule::op_gte, Assoc::Left)
            | Op::infix(Rule::op_lte, Assoc::Left)
            | Op::infix(Rule::op_gt, Assoc::Left)
            | Op::infix(Rule::op_lt, Assoc::Left))
        .op(Op::infix(Rule::op_matmul, Assoc::Left) | Op::infix(Rule::op_outer_prod, Assoc::Left))
        .op(Op::infix(Rule::op_add, Assoc::Left) | Op::infix(Rule::op_sub, Assoc::Left))
        .op(Op::infix(Rule::op_mul, Assoc::Left) | Op::infix(Rule::op_div, Assoc::Left))
        .op(Op::prefix(Rule::op_neg) | Op::prefix(Rule::op_not))
        .op(Op::postfix(Rule::expr_call) | Op::postfix(Rule::expr_index))
}

impl<'a> Lowerer<'a> {
    pub fn lower_expr(&self, pair: Pair<'a, Rule>) -> Result<&'a Expr<'a>, String> {
        let actual_pair = if pair.as_rule() == Rule::primary {
            let inner = pair.into_inner().next().unwrap();
            if inner.as_rule() == Rule::expr {
                inner
            } else {
                return self.lower_primary(inner);
            }
        } else {
            pair
        };
        let pairs = actual_pair.into_inner();
        let bump = self.bump;
        let pratt = make_pratt();

        pratt
            .map_primary(|primary_pair| self.lower_primary(primary_pair))
            .map_prefix(|op_pair, child| {
                let inner = child?;
                match op_pair.as_rule() {
                    Rule::op_neg => Ok(bump.alloc(Expr::Unary {
                        op: UnaryOp::Neg,
                        inner,
                    }) as &'a Expr<'a>),
                    Rule::op_not => Ok(bump.alloc(Expr::Unary {
                        op: UnaryOp::Not,
                        inner,
                    }) as &'a Expr<'a>),
                    other => Err(format!("Unknown prefix op: {:?}", other)),
                }
            })
            .map_postfix(|child, op_pair| {
                let expr = child?;
                match op_pair.as_rule() {
                    Rule::pipe_activation => {
                        let mut inner = op_pair.into_inner();
                        let act_pair = inner.next().unwrap();
                        let kind_pair = if act_pair.as_rule() == Rule::activation_kind {
                            act_pair.into_inner().next().unwrap()
                        } else {
                            act_pair
                        };

                        match kind_pair.as_rule() {
                            Rule::act_curve => {
                                let mut points = Vec::new();
                                for pt in kind_pair.into_inner() {
                                    let mut nums = pt.into_inner();
                                    let x = nums.next().unwrap().as_str().parse::<f64>().unwrap();
                                    let y = nums.next().unwrap().as_str().parse::<f64>().unwrap();
                                    points.push((x, y));
                                }
                                Ok(bump.alloc(Expr::Curve {
                                    expr,
                                    points: bump.alloc_slice_copy(&points),
                                }) as &'a Expr<'a>)
                            }
                            Rule::act_match => {
                                let pattern_str = kind_pair
                                    .into_inner()
                                    .next()
                                    .unwrap()
                                    .as_str()
                                    .trim_matches('"');
                                Ok(bump.alloc(Expr::Match {
                                    expr,
                                    pattern: pattern_str,
                                }) as &'a Expr<'a>)
                            }
                            _ => {
                                let kind = self.lower_activation_kind(kind_pair)?;
                                let mut e =
                                    bump.alloc(Expr::Activation { kind, expr }) as &'a Expr<'a>;
                                if let Some(scale_pair) = inner.next() {
                                    let factor = scale_pair
                                        .as_str()
                                        .parse::<f64>()
                                        .map_err(|err| err.to_string())?;
                                    e = bump.alloc(Expr::Binary {
                                        op: BinOp::Mul,
                                        left: e,
                                        right: bump.alloc(Expr::Number(factor)),
                                    });
                                }
                                Ok(e)
                            }
                        }
                    }
                    Rule::expr_index => {
                        let idx_pair = op_pair.into_inner().next().unwrap();
                        let mut idx_inner = idx_pair.into_inner();
                        let first = idx_inner.next().unwrap();
                        let ident_name = match expr {
                            Expr::Ident(id) => *id,
                            _ => return Err("Expected identifier for indexed access".to_string()),
                        };

                        match first.as_rule() {
                            Rule::int_lit => {
                                let index = first.as_str().trim().parse::<usize>().unwrap();
                                Ok(bump.alloc(Expr::Index {
                                    name: ident_name,
                                    index,
                                }) as &'a Expr<'a>)
                            }
                            Rule::multi_index => {
                                let mut dims = Vec::new();
                                for d in first.clone().into_inner() {
                                    dims.push(d.as_str().trim().parse::<usize>().unwrap());
                                }
                                if dims.is_empty() {
                                    for part in first.as_str().split(',') {
                                        if let Ok(v) = part.trim().parse::<usize>() {
                                            dims.push(v);
                                        }
                                    }
                                }
                                Ok(bump.alloc(Expr::MultiIndex {
                                    name: ident_name,
                                    indices: bump.alloc_slice_copy(&dims),
                                }) as &'a Expr<'a>)
                            }
                            Rule::slice_range => {
                                let mut nums = first.into_inner();
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
                                Ok(bump.alloc(Expr::DynamicIndex {
                                    name: ident_name,
                                    addr: bump.alloc_str(&alloc::format!("{}..{}", start, end)),
                                }) as &'a Expr<'a>)
                            }
                            Rule::dynamic_addr | Rule::ident => {
                                let addr = first.as_str().trim().trim_start_matches('@');
                                Ok(bump.alloc(Expr::DynamicIndex {
                                    name: ident_name,
                                    addr,
                                }) as &'a Expr<'a>)
                            }
                            other => Err(format!("Unsupported expr index rule: {:?}", other)),
                        }
                    }
                    Rule::expr_call => {
                        let inner = op_pair.into_inner();
                        let mut template_args = Vec::new();
                        let mut args = Vec::new();

                        for p in inner {
                            match p.as_rule() {
                                Rule::template_args => {
                                    for num in p.into_inner() {
                                        template_args.push(num.as_str().parse::<f64>().unwrap());
                                    }
                                }
                                Rule::expr => {
                                    args.push(self.lower_expr(p)?);
                                }
                                _ => {}
                            }
                        }

                        let circuit_name = match expr {
                            Expr::Ident(id) => *id,
                            _ => "fn",
                        };

                        Ok(bump.alloc(Expr::CircuitCall {
                            circuit: circuit_name,
                            template_args: bump.alloc_slice_copy(&template_args),
                            args: bump.alloc_slice_copy(&args),
                        }) as &'a Expr<'a>)
                    }
                    other => Err(format!("Unknown postfix op: {:?}", other)),
                }
            })
            .map_infix(|lhs, op_pair, rhs| {
                let left = lhs?;
                let right = rhs?;
                let op = match op_pair.as_rule() {
                    Rule::op_or => BinOp::Or,
                    Rule::op_nor => BinOp::Nor,
                    Rule::op_and => BinOp::And,
                    Rule::op_nand => BinOp::Nand,
                    Rule::op_xor => BinOp::Xor,
                    Rule::op_eq => BinOp::Eq,
                    Rule::op_neq => BinOp::Neq,
                    Rule::op_gte => BinOp::Gte,
                    Rule::op_lte => BinOp::Lte,
                    Rule::op_gt => BinOp::Gt,
                    Rule::op_lt => BinOp::Lt,
                    Rule::op_add => BinOp::Add,
                    Rule::op_sub => BinOp::Sub,
                    Rule::op_mul => BinOp::Mul,
                    Rule::op_div => BinOp::Div,
                    Rule::op_matmul => BinOp::MatMul,
                    Rule::op_outer_prod => BinOp::OuterProduct,
                    other => return Err(format!("Unknown binary op: {:?}", other)),
                };
                Ok(bump.alloc(Expr::Binary { op, left, right }) as &'a Expr<'a>)
            })
            .parse(pairs)
    }

    pub fn lower_primary(&self, pair: Pair<'a, Rule>) -> Result<&'a Expr<'a>, String> {
        let inner = if pair.as_rule() == Rule::primary {
            pair.into_inner().next().unwrap()
        } else {
            pair
        };

        match inner.as_rule() {
            Rule::number => {
                let val = inner.as_str().parse::<f64>().unwrap();
                Ok(self.bump.alloc(Expr::Number(val)))
            }
            Rule::ident | Rule::scoped_ident => Ok(self.bump.alloc(Expr::Ident(inner.as_str()))),
            Rule::state_ident => {
                let text = inner.as_str();
                Ok(self.bump.alloc(Expr::Ident(self.bump.alloc_str(text))))
            }
            Rule::expr => self.lower_expr(inner),
            Rule::spatial_call => {
                let call = inner.into_inner().next().unwrap();
                match call.as_rule() {
                    Rule::conv2d_call => {
                        let mut call_inner = call.into_inner();
                        let input = self.lower_expr(call_inner.next().unwrap())?;
                        let mut kernel = "K";
                        let mut stride = 1;
                        let mut padding = PaddingMode::Valid;

                        for arg in call_inner {
                            for item in arg.into_inner() {
                                match item.as_rule() {
                                    Rule::ident => {
                                        let s = item.as_str();
                                        if s == "same" {
                                            padding = PaddingMode::Same;
                                        } else if s == "valid" {
                                            padding = PaddingMode::Valid;
                                        } else {
                                            kernel = s;
                                        }
                                    }
                                    Rule::int_lit => {
                                        stride = item.as_str().parse::<usize>().unwrap();
                                    }
                                    _ => {}
                                }
                            }
                        }

                        Ok(self.bump.alloc(Expr::Conv2d {
                            input,
                            kernel,
                            stride,
                            padding,
                        }))
                    }
                    Rule::avgpool2d_call => {
                        let mut call_inner = call.into_inner();
                        let input = self.lower_expr(call_inner.next().unwrap())?;
                        let mut kernel_size = 2;
                        let mut stride = 2;

                        let mut nums = Vec::new();
                        for arg in call_inner {
                            for item in arg.into_inner() {
                                if item.as_rule() == Rule::int_lit {
                                    nums.push(item.as_str().parse::<usize>().unwrap());
                                }
                            }
                        }
                        if let Some(&k) = nums.first() {
                            kernel_size = k;
                        }
                        if let Some(&s) = nums.get(1) {
                            stride = s;
                        }

                        Ok(self.bump.alloc(Expr::AvgPool2d {
                            input,
                            kernel_size,
                            stride,
                        }))
                    }
                    other => Err(format!("Unknown spatial call rule: {:?}", other)),
                }
            }
            other => Err(format!("Unknown primary rule: {:?}", other)),
        }
    }

    pub fn lower_activation_kind(&self, pair: Pair<'a, Rule>) -> Result<ActivationKind, String> {
        let rule = pair.as_rule();
        let target = if rule == Rule::activation_kind {
            pair.into_inner()
                .next()
                .ok_or_else(|| "Empty activation kind".to_string())?
        } else {
            pair
        };
        match target.as_rule() {
            Rule::act_saturate => Ok(ActivationKind::Saturate),
            Rule::act_clamp => Ok(ActivationKind::Clamp),
            Rule::act_step => Ok(ActivationKind::Step),
            Rule::act_inv => Ok(ActivationKind::Inv),
            Rule::act_relu => Ok(ActivationKind::Relu),
            Rule::act_curve | Rule::act_match => Ok(ActivationKind::Saturate),
            other => Err(format!("Unknown activation kind: {:?}", other)),
        }
    }
}
