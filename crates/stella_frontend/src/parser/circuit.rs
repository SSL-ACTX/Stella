use super::Parser;
use crate::ast::*;
use crate::lexer::TokenKind;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

impl<'a, 't> Parser<'a, 't> {
    pub(super) fn parse_circuit(&mut self) -> Result<Decl<'a>, String> {
        self.expect(TokenKind::Circuit)?;
        let name = self.expect_ident()?;

        // Optional generic template parameters: <PARAM1, PARAM2 = 0.85>
        let mut template_params = Vec::new();
        if self.peek_kind() == TokenKind::Lt {
            self.advance();
            while self.peek_kind() != TokenKind::Gt && !self.is_eof() {
                let t_name = self.expect_ident()?;
                let default = if self.peek_kind() == TokenKind::Assign {
                    self.advance();
                    Some(self.expect_number()?)
                } else {
                    None
                };
                template_params.push(TemplateParam {
                    name: t_name,
                    default,
                });

                if self.peek_kind() == TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(TokenKind::Gt)?;
        }

        self.expect(TokenKind::LParen)?;

        let mut params = Vec::new();
        while self.peek_kind() != TokenKind::RParen && !self.is_eof() {
            let direction = match self.peek_kind() {
                TokenKind::In => {
                    self.advance();
                    ParamDirection::In
                }
                TokenKind::Out => {
                    self.advance();
                    ParamDirection::Out
                }
                _ => ParamDirection::In,
            };

            let (pname, width, shape) = self.parse_decl_name_and_width()?;
            let default = if self.peek_kind() == TokenKind::Assign {
                self.advance();
                Some(self.expect_number()?)
            } else {
                None
            };
            params.push(CircuitParam {
                direction,
                name: pname,
                width,
                shape,
                default,
            });

            if self.peek_kind() == TokenKind::Comma {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(TokenKind::RParen)?;

        // Optional return parameter: -> [out] return_name
        let mut return_param = None;
        if self.peek_kind() == TokenKind::RArrow {
            self.advance();
            if self.peek_kind() == TokenKind::Out {
                self.advance();
            }
            let (rname, rwidth, rshape) = self.parse_decl_name_and_width()?;
            return_param = Some(CircuitParam {
                direction: ParamDirection::Out,
                name: rname,
                width: rwidth,
                shape: rshape,
                default: None,
            });
        }

        self.expect(TokenKind::LBrace)?;

        let mut declarations = Vec::new();
        let mut flows = Vec::new();

        while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
            match self.peek_kind() {
                TokenKind::Node => {
                    self.advance();
                    let (n_name, width, shape) = self.parse_decl_name_and_width()?;
                    let mut dynamics = NodeDynamics::Standard;

                    if self.peek_kind() == TokenKind::Colon {
                        self.advance();
                        match self.peek_kind() {
                            TokenKind::Leak => {
                                self.advance();
                                self.expect(TokenKind::LParen)?;
                                let rate = self.expect_number()?;
                                self.expect(TokenKind::RParen)?;
                                dynamics = NodeDynamics::Leak(rate);
                            }
                            TokenKind::Latch => {
                                self.advance();
                                dynamics = NodeDynamics::Latch;
                            }
                            TokenKind::Hold => {
                                self.advance();
                                dynamics = NodeDynamics::Hold;
                            }
                            TokenKind::Plastic => {
                                self.advance();
                                let (rate, decay) = if self.peek_kind() == TokenKind::LParen {
                                    self.advance();
                                    let r = self.expect_number()?;
                                    let d = if self.peek_kind() == TokenKind::Comma {
                                        self.advance();
                                        self.expect_number()?
                                    } else {
                                        0.01
                                    };
                                    self.expect(TokenKind::RParen)?;
                                    (r, d)
                                } else {
                                    (0.1, 0.01)
                                };
                                dynamics = NodeDynamics::Plastic { rate, decay };
                            }
                            TokenKind::Oscillator | TokenKind::Orbit => {
                                self.advance();
                                self.expect(TokenKind::LParen)?;
                                let period = self.expect_number()? as usize;
                                self.expect(TokenKind::RParen)?;
                                dynamics = NodeDynamics::Oscillator {
                                    period: period.max(2),
                                };
                            }
                            other => {
                                return Err(format!(
                                    "Line {}: Expected 'leak(rate)', 'latch', 'hold', 'plastic', or 'oscillator(period)', got {:?}",
                                    self.current_line(),
                                    other
                                ))
                            }
                        }
                    }

                    let initial = if self.peek_kind() == TokenKind::Assign {
                        self.advance();
                        Some(self.expect_number()?)
                    } else {
                        None
                    };

                    self.expect(TokenKind::Semi)?;
                    declarations.push(Decl::Node {
                        name: n_name,
                        width,
                        shape,
                        dynamics,
                        initial,
                    });
                }
                TokenKind::Type => {
                    self.advance();
                    let name = self.expect_ident()?;
                    self.expect(TokenKind::Assign)?;
                    let shape_slice = self.parse_shape()?;
                    self.expect(TokenKind::Semi)?;
                    let width: usize = shape_slice.iter().product();
                    let shape_opt = if shape_slice.len() > 1 {
                        Some(shape_slice)
                    } else {
                        None
                    };
                    self.type_aliases.insert(name, (width, shape_opt));
                    declarations.push(Decl::TypeAlias {
                        name,
                        shape: shape_slice,
                    });
                }
                TokenKind::Const => {
                    self.advance();
                    let c_name = self.expect_ident()?;
                    self.expect(TokenKind::Assign)?;
                    let val = self.expect_number()?;
                    self.expect(TokenKind::Semi)?;
                    declarations.push(Decl::Const(c_name, val));
                }
                TokenKind::Inst => {
                    declarations.push(self.parse_inst()?);
                }
                _ => {
                    self.parse_flow_statement(&mut flows)?;
                }
            }
        }
        self.expect(TokenKind::RBrace)?;

        Ok(Decl::Circuit {
            name,
            template_params: self.bump.alloc_slice_copy(&template_params),
            params: self.bump.alloc_slice_copy(&params),
            return_param,
            declarations: self.bump.alloc_slice_copy(&declarations),
            flows: self.bump.alloc_slice_copy(&flows),
        })
    }

    pub(super) fn parse_inst(&mut self) -> Result<Decl<'a>, String> {
        self.expect(TokenKind::Inst)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::Assign)?;
        let circuit = self.expect_ident()?;

        // Optional template arguments: CircuitName<0.85, 2.0>(args...)
        let mut template_args = Vec::new();
        if self.peek_kind() == TokenKind::Lt {
            self.advance();
            while self.peek_kind() != TokenKind::Gt && !self.is_eof() {
                let val = self.expect_number()?;
                template_args.push(val);
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
            let target = self.parse_node_target()?;
            args.push(target);
            if self.peek_kind() == TokenKind::Comma {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::Semi)?;

        Ok(Decl::Inst {
            name,
            circuit,
            template_args: self.bump.alloc_slice_copy(&template_args),
            args: self.bump.alloc_slice_copy(&args),
        })
    }
}
