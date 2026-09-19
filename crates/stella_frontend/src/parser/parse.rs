use super::Parser;
use crate::ast::*;
use crate::lexer::{Token, TokenKind};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use bumpalo::Bump;

impl<'a, 't> Parser<'a, 't> {
    pub fn new(tokens: &'t [Token<'a>], bump: &'a Bump) -> Self {
        Self {
            tokens,
            cursor: 0,
            bump,
            type_aliases: BTreeMap::new(),
        }
    }

    pub fn parse(&mut self) -> Result<Program<'a>, String> {
        let mut declarations = Vec::new();
        let mut flows = Vec::new();
        let mut basins = Vec::new();

        while !self.is_eof() {
            match self.peek_kind() {
                TokenKind::Terminal => {
                    self.advance();
                    match self.peek_kind() {
                        TokenKind::In => {
                            self.advance();
                            let (name, width, shape) = self.parse_decl_name_and_width()?;
                            self.expect(TokenKind::Semi)?;
                            declarations.push(Decl::TerminalIn { name, width, shape });
                        }
                        TokenKind::Out => {
                            self.advance();
                            let (name, width, shape) = self.parse_decl_name_and_width()?;
                            self.expect(TokenKind::Semi)?;
                            declarations.push(Decl::TerminalOut { name, width, shape });
                        }
                        other => {
                            return Err(format!(
                                "Line {}: Expected 'in' or 'out' after 'terminal', got {:?}",
                                self.current_line(),
                                other
                            ))
                        }
                    }
                }
                TokenKind::In => {
                    self.advance();
                    let (name, width, shape) = self.parse_decl_name_and_width()?;
                    self.expect(TokenKind::Semi)?;
                    declarations.push(Decl::TerminalIn { name, width, shape });
                }
                TokenKind::Out => {
                    self.advance();
                    let (name, width, shape) = self.parse_decl_name_and_width()?;
                    self.expect(TokenKind::Semi)?;
                    declarations.push(Decl::TerminalOut { name, width, shape });
                }
                TokenKind::Node => {
                    self.advance();
                    let (name, width, shape) = self.parse_decl_name_and_width()?;
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
                        name,
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
                    let _annotated_shape = if self.peek_kind() == TokenKind::LBracket {
                        Some(self.parse_shape()?)
                    } else {
                        None
                    };
                    let name = self.expect_ident()?;
                    self.expect(TokenKind::Assign)?;
                    if self.peek_kind() == TokenKind::LBracket {
                        self.advance(); // [
                        let mut rows = 0;
                        let mut cols = 0;
                        let mut matrix_data = Vec::new();
                        while self.peek_kind() != TokenKind::RBracket && !self.is_eof() {
                            self.expect(TokenKind::LBracket)?;
                            let mut row_items = Vec::new();
                            while self.peek_kind() != TokenKind::RBracket && !self.is_eof() {
                                row_items.push(self.expect_number()?);
                                if self.peek_kind() == TokenKind::Comma {
                                    self.advance();
                                } else {
                                    break;
                                }
                            }
                            self.expect(TokenKind::RBracket)?;
                            if rows == 0 {
                                cols = row_items.len();
                            } else if row_items.len() != cols {
                                return Err(format!(
                                    "Line {}: Inconsistent row size in matrix constant '{}' (expected {}, got {})",
                                    self.current_line(),
                                    name,
                                    cols,
                                    row_items.len()
                                ));
                            }
                            matrix_data.extend(row_items);
                            rows += 1;
                            if self.peek_kind() == TokenKind::Comma {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        self.expect(TokenKind::RBracket)?;
                        self.expect(TokenKind::Semi)?;
                        declarations.push(Decl::ConstMatrix {
                            name,
                            rows,
                            cols,
                            data: self.bump.alloc_slice_copy(&matrix_data),
                        });
                    } else {
                        let val = self.expect_number()?;
                        self.expect(TokenKind::Semi)?;
                        declarations.push(Decl::Const(name, val));
                    }
                }
                TokenKind::Cloak => {
                    self.advance();
                    self.expect(TokenKind::LBrace)?;
                    let mut pad = 16;
                    let mut seed = None;
                    while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
                        match self.peek_kind() {
                            TokenKind::Pad => {
                                self.advance();
                                pad = self.expect_number()? as usize;
                                self.expect(TokenKind::Semi)?;
                            }
                            TokenKind::Seed => {
                                self.advance();
                                let x = self.expect_number()?;
                                self.expect(TokenKind::Comma)?;
                                let y = self.expect_number()?;
                                self.expect(TokenKind::Semi)?;
                                seed = Some((x, y));
                            }
                            other => {
                                return Err(format!(
                                    "Line {}: Expected 'pad' or 'seed' in cloak block, got {:?}",
                                    self.current_line(),
                                    other
                                ))
                            }
                        }
                    }
                    self.expect(TokenKind::RBrace)?;
                    declarations.push(Decl::Cloak { pad, seed });
                }
                TokenKind::Circuit => {
                    declarations.push(self.parse_circuit()?);
                }
                TokenKind::Inst => {
                    declarations.push(self.parse_inst()?);
                }
                TokenKind::Assert => {
                    self.advance();
                    match self.peek_kind() {
                        TokenKind::Stable => {
                            self.advance();
                            let target = if self.peek_kind() == TokenKind::LParen {
                                self.advance();
                                let id = self.expect_ident()?;
                                self.expect(TokenKind::RParen)?;
                                Some(id)
                            } else {
                                None
                            };
                            self.expect(TokenKind::Semi)?;
                            declarations.push(Decl::AssertStable { target });
                        }
                        TokenKind::Ident("bounded") => {
                            self.advance();
                            self.expect(TokenKind::LParen)?;
                            let target = self.parse_node_target()?;
                            self.expect(TokenKind::Comma)?;
                            let min = self.expect_number()?;
                            self.expect(TokenKind::Comma)?;
                            let max = self.expect_number()?;
                            self.expect(TokenKind::RParen)?;
                            self.expect(TokenKind::Semi)?;
                            declarations.push(Decl::AssertBounded { target, min, max });
                        }
                        other => {
                            return Err(format!(
                                "Line {}: Expected 'stable' or 'bounded' after 'assert', got {:?}",
                                self.current_line(),
                                other
                            ));
                        }
                    }
                }
                TokenKind::Basin | TokenKind::Phase => {
                    self.parse_basin_or_phase(&mut basins)?;
                }
                _ => {
                    self.parse_flow_statement(&mut flows)?;
                }
            }
        }

        let decl_slice = self.bump.alloc_slice_copy(&declarations);
        let flow_slice = self.bump.alloc_slice_copy(&flows);
        let basin_slice = self.bump.alloc_slice_copy(&basins);

        Ok(Program {
            declarations: decl_slice,
            flows: flow_slice,
            basins: basin_slice,
        })
    }
}
