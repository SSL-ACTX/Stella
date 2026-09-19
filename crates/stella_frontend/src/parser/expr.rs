use super::Parser;
use crate::ast::*;
use crate::lexer::TokenKind;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

impl<'a, 't> Parser<'a, 't> {
    pub fn parse_expr(&mut self) -> Result<&'a Expr<'a>, String> {
        let mut expr = self.parse_binary_expr(0)?;

        // Check for activation pipe `|> saturate` or `|> clamp`
        while self.peek_kind() == TokenKind::PipeGate {
            self.advance();
            match self.peek_kind() {
                TokenKind::Saturate => {
                    self.advance();
                    expr = self.bump.alloc(Expr::Activation {
                        kind: ActivationKind::Saturate,
                        expr,
                    });
                }
                TokenKind::Clamp => {
                    self.advance();
                    expr = self.bump.alloc(Expr::Activation {
                        kind: ActivationKind::Clamp,
                        expr,
                    });
                }
                TokenKind::Step => {
                    self.advance();
                    expr = self.bump.alloc(Expr::Activation {
                        kind: ActivationKind::Step,
                        expr,
                    });
                }
                TokenKind::Inv => {
                    self.advance();
                    expr = self.bump.alloc(Expr::Activation {
                        kind: ActivationKind::Inv,
                        expr,
                    });
                }
                TokenKind::Relu => {
                    self.advance();
                    expr = self.bump.alloc(Expr::Activation {
                        kind: ActivationKind::Relu,
                        expr,
                    });
                }
                TokenKind::Curve => {
                    self.advance();
                    self.expect(TokenKind::LParen)?;
                    self.expect(TokenKind::LBracket)?;
                    let mut points = Vec::new();
                    while self.peek_kind() != TokenKind::RBracket && !self.is_eof() {
                        let x = self.expect_number()?;
                        self.expect(TokenKind::FatArrow)?;
                        let y = self.expect_number()?;
                        points.push((x, y));
                        if self.peek_kind() == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.expect(TokenKind::RBracket)?;
                    self.expect(TokenKind::RParen)?;
                    expr = self.bump.alloc(Expr::Curve {
                        expr,
                        points: self.bump.alloc_slice_copy(&points),
                    });
                }
                TokenKind::Match => {
                    self.advance();
                    self.expect(TokenKind::LParen)?;
                    let pattern = self.expect_string()?;
                    self.expect(TokenKind::RParen)?;
                    expr = self.bump.alloc(Expr::Match {
                        expr,
                        pattern,
                    });
                }
                other => {
                    return Err(format!(
                        "Line {}: Expected filter ('saturate', 'clamp', 'step', 'inv', 'relu', 'curve', 'match') after '|>', got {:?}",
                        self.current_line(),
                        other
                    ))
                }
            }
        }

        Ok(expr)
    }

    fn parse_binary_expr(&mut self, min_prec: u8) -> Result<&'a Expr<'a>, String> {
        let mut left = self.parse_primary()?;

        while let Some((op, prec)) = self.peek_binop() {
            if prec < min_prec {
                break;
            }
            self.advance(); // consume op
            let right = self.parse_binary_expr(prec + 1)?;
            left = self.bump.alloc(Expr::Binary { op, left, right });
        }

        Ok(left)
    }

    fn peek_binop(&self) -> Option<(BinOp, u8)> {
        match self.peek_kind() {
            TokenKind::Or => Some((BinOp::Or, 1)),
            TokenKind::Nor => Some((BinOp::Nor, 1)),
            TokenKind::And => Some((BinOp::And, 2)),
            TokenKind::Nand => Some((BinOp::Nand, 2)),
            TokenKind::Xor => Some((BinOp::Xor, 2)),
            TokenKind::EqEq => Some((BinOp::Eq, 3)),
            TokenKind::Neq => Some((BinOp::Neq, 3)),
            TokenKind::Gte => Some((BinOp::Gte, 3)),
            TokenKind::Lte => Some((BinOp::Lte, 3)),
            TokenKind::Gt => Some((BinOp::Gt, 3)),
            TokenKind::Lt => Some((BinOp::Lt, 3)),
            TokenKind::Plus => Some((BinOp::Add, 4)),
            TokenKind::Minus => Some((BinOp::Sub, 4)),
            TokenKind::Star => Some((BinOp::Mul, 5)),
            TokenKind::Slash => Some((BinOp::Div, 5)),
            _ => None,
        }
    }

    fn parse_primary(&mut self) -> Result<&'a Expr<'a>, String> {
        match self.peek_kind() {
            TokenKind::Number(val) => {
                self.advance();
                Ok(self.bump.alloc(Expr::Number(val)))
            }
            TokenKind::Ident(id) => {
                self.advance();
                if self.peek_kind() == TokenKind::LBracket {
                    self.advance();
                    match self.peek_kind() {
                        TokenKind::Ident(addr_id) => {
                            self.advance();
                            self.expect(TokenKind::RBracket)?;
                            Ok(self.bump.alloc(Expr::DynamicIndex {
                                name: id,
                                addr: addr_id,
                            }))
                        }
                        _ => {
                            let first_idx = self.expect_number()? as usize;
                            if self.peek_kind() == TokenKind::Comma {
                                let mut dims = Vec::new();
                                dims.push(first_idx);
                                while self.peek_kind() == TokenKind::Comma {
                                    self.advance();
                                    dims.push(self.expect_number()? as usize);
                                }
                                self.expect(TokenKind::RBracket)?;
                                let indices = self.bump.alloc_slice_copy(&dims);
                                Ok(self.bump.alloc(Expr::MultiIndex { name: id, indices }))
                            } else {
                                self.expect(TokenKind::RBracket)?;
                                Ok(self.bump.alloc(Expr::Index {
                                    name: id,
                                    index: first_idx,
                                }))
                            }
                        }
                    }
                } else if self.peek_kind() == TokenKind::LParen
                    || (self.peek_kind() == TokenKind::Lt && self.is_template_call_ahead())
                {
                    let mut template_args = Vec::new();
                    if self.peek_kind() == TokenKind::Lt {
                        self.advance();
                        while self.peek_kind() != TokenKind::Gt && !self.is_eof() {
                            template_args.push(self.expect_number()?);
                            if self.peek_kind() == TokenKind::Comma {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        self.expect(TokenKind::Gt)?;
                    }

                    self.expect(TokenKind::LParen)?;
                    let mut args = Vec::new();
                    while self.peek_kind() != TokenKind::RParen && !self.is_eof() {
                        args.push(self.parse_expr()?);
                        if self.peek_kind() == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    Ok(self.bump.alloc(Expr::CircuitCall {
                        circuit: id,
                        template_args: self.bump.alloc_slice_copy(&template_args),
                        args: self.bump.alloc_slice_copy(&args),
                    }))
                } else if self.peek_kind() == TokenKind::Dot {
                    self.advance(); // .
                    let member = self.expect_ident()?;
                    let qualified = self.bump.alloc_str(&format!("{}.{}", id, member));
                    Ok(self.bump.alloc(Expr::Ident(qualified)))
                } else {
                    Ok(self.bump.alloc(Expr::Ident(id)))
                }
            }
            TokenKind::State | TokenKind::Basin => {
                let base = if self.peek_kind() == TokenKind::State {
                    "state"
                } else {
                    "basin"
                };
                self.advance();
                if self.peek_kind() == TokenKind::Dot {
                    self.advance(); // .
                    let member = self.expect_ident()?;
                    let qualified = self.bump.alloc_str(&format!("{}.{}", base, member));
                    Ok(self.bump.alloc(Expr::Ident(qualified)))
                } else {
                    Ok(self.bump.alloc(Expr::Ident(base)))
                }
            }
            TokenKind::Conv2d => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let input = self.parse_expr()?;
                self.expect(TokenKind::Comma)?;
                if self.peek_kind() == TokenKind::Ident("kernel") {
                    self.advance();
                    self.expect(TokenKind::Colon)?;
                }
                let kernel = self.expect_ident()?;
                let mut stride = 1;
                let mut padding = crate::ast::PaddingMode::Valid;

                while self.peek_kind() == TokenKind::Comma {
                    self.advance();
                    if self.peek_kind() == TokenKind::Ident("stride") {
                        self.advance();
                        self.expect(TokenKind::Colon)?;
                        stride = self.expect_number()? as usize;
                    } else if self.peek_kind() == TokenKind::Ident("padding") {
                        self.advance();
                        self.expect(TokenKind::Colon)?;
                        let pad_mode = self.expect_ident()?;
                        padding = match pad_mode {
                            "same" => crate::ast::PaddingMode::Same,
                            "valid" => crate::ast::PaddingMode::Valid,
                            other => {
                                return Err(format!(
                                    "Unknown padding mode '{}', expected 'same' or 'valid'",
                                    other
                                ))
                            }
                        };
                    } else {
                        stride = self.expect_number()? as usize;
                    }
                }
                self.expect(TokenKind::RParen)?;
                Ok(self.bump.alloc(Expr::Conv2d {
                    input,
                    kernel,
                    stride,
                    padding,
                }))
            }
            TokenKind::AvgPool2d => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let input = self.parse_expr()?;
                let mut kernel_size = 2;
                let mut stride = 2;

                while self.peek_kind() == TokenKind::Comma {
                    self.advance();
                    if self.peek_kind() == TokenKind::Ident("kernel_size") {
                        self.advance();
                        self.expect(TokenKind::Colon)?;
                        kernel_size = self.expect_number()? as usize;
                    } else if self.peek_kind() == TokenKind::Ident("stride") {
                        self.advance();
                        self.expect(TokenKind::Colon)?;
                        stride = self.expect_number()? as usize;
                    } else {
                        kernel_size = self.expect_number()? as usize;
                    }
                }
                self.expect(TokenKind::RParen)?;
                Ok(self.bump.alloc(Expr::AvgPool2d {
                    input,
                    kernel_size,
                    stride,
                }))
            }
            TokenKind::Not => {
                self.advance();
                let inner = self.parse_primary()?;
                Ok(self.bump.alloc(Expr::Unary {
                    op: UnaryOp::Not,
                    inner,
                }))
            }
            TokenKind::Minus => {
                self.advance();
                let inner = self.parse_primary()?;
                Ok(self.bump.alloc(Expr::Unary {
                    op: UnaryOp::Neg,
                    inner,
                }))
            }
            TokenKind::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }
            other => Err(format!(
                "Line {}: Expected expression, got {:?}",
                self.current_line(),
                other
            )),
        }
    }
}
