use super::Parser;
use crate::ast::*;
use crate::lexer::TokenKind;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

impl<'a, 't> Parser<'a, 't> {
    pub(super) fn parse_basin_with_name(
        &mut self,
        name: &'a str,
        group: Option<&'a str>,
    ) -> Result<Basin<'a>, String> {
        self.expect(TokenKind::LBrace)?;

        let mut flows = Vec::new();
        let mut bifurcations = Vec::new();

        while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
            if self.peek_kind() == TokenKind::Drift {
                self.advance();
                if let TokenKind::Ident("to") = self.peek_kind() {
                    self.advance();
                }
                let raw_target = self.expect_ident()?;
                let target = if let Some(grp) = group {
                    if !raw_target.contains("::") {
                        self.bump
                            .alloc_str(&alloc::format!("{}::{}", grp, raw_target))
                    } else {
                        raw_target
                    }
                } else {
                    raw_target
                };
                if self.peek_kind() == TokenKind::When {
                    self.advance();
                    let cond = self.parse_expr()?;
                    self.expect(TokenKind::Semi)?;
                    bifurcations.push(Bifurcation {
                        target,
                        condition: Some(cond),
                    });
                } else {
                    self.expect(TokenKind::Semi)?;
                    bifurcations.push(Bifurcation {
                        target,
                        condition: None,
                    });
                }
            } else if self.peek_kind() == TokenKind::When {
                // `when cond drift Target;`
                self.advance();
                let cond = self.parse_expr()?;
                self.expect(TokenKind::Drift)?;
                if let TokenKind::Ident("to") = self.peek_kind() {
                    self.advance();
                }
                let raw_target = self.expect_ident()?;
                let target = if let Some(grp) = group {
                    if !raw_target.contains("::") {
                        self.bump
                            .alloc_str(&alloc::format!("{}::{}", grp, raw_target))
                    } else {
                        raw_target
                    }
                } else {
                    raw_target
                };
                self.expect(TokenKind::Semi)?;
                bifurcations.push(Bifurcation {
                    target,
                    condition: Some(cond),
                });
            } else {
                self.parse_flow_statement(&mut flows)?;
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Basin {
            name,
            flows: self.bump.alloc_slice_copy(&flows),
            bifurcations: self.bump.alloc_slice_copy(&bifurcations),
        })
    }

    pub(super) fn parse_basin_or_phase(
        &mut self,
        basins: &mut Vec<Basin<'a>>,
    ) -> Result<(), String> {
        if self.peek_kind() == TokenKind::Phase {
            self.advance();
            let phase_group_name = self.expect_ident()?;
            self.expect(TokenKind::LBrace)?;
            while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
                if self.peek_kind() == TokenKind::State || self.peek_kind() == TokenKind::Basin {
                    self.advance();
                    let raw_state_name = self.expect_ident()?;
                    let state_name = self.bump.alloc_str(&alloc::format!(
                        "{}::{}",
                        phase_group_name,
                        raw_state_name
                    ));
                    let b = self.parse_basin_with_name(state_name, Some(phase_group_name))?;
                    basins.push(b);
                } else {
                    return Err(format!(
                        "Line {}: Expected 'state <Name> {{ ... }}' inside phase block '{}', got {:?}",
                        self.current_line(),
                        phase_group_name,
                        self.peek_kind()
                    ));
                }
            }
            self.expect(TokenKind::RBrace)?;
            Ok(())
        } else {
            self.expect(TokenKind::Basin)?;
            let name = self.expect_ident()?;
            let b = self.parse_basin_with_name(name, None)?;
            basins.push(b);
            Ok(())
        }
    }

    pub(super) fn parse_flow_statement(&mut self, flows: &mut Vec<Flow<'a>>) -> Result<(), String> {
        match self.peek_kind() {
            TokenKind::If => {
                self.advance();
                let cond = self.parse_expr()?;
                self.expect(TokenKind::LBrace)?;
                let mut then_flows = Vec::new();
                while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
                    self.parse_flow_statement(&mut then_flows)?;
                }
                self.expect(TokenKind::RBrace)?;

                let else_flows = if self.peek_kind() == TokenKind::Else {
                    self.advance();
                    self.expect(TokenKind::LBrace)?;
                    let mut elses = Vec::new();
                    while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
                        self.parse_flow_statement(&mut elses)?;
                    }
                    self.expect(TokenKind::RBrace)?;
                    Some(self.bump.alloc_slice_copy(&elses) as &'a [Flow<'a>])
                } else {
                    None
                };

                flows.push(Flow::If {
                    cond,
                    then_flows: self.bump.alloc_slice_copy(&then_flows),
                    else_flows,
                });
                Ok(())
            }
            TokenKind::While => {
                self.advance();
                let cond = self.parse_expr()?;
                self.expect(TokenKind::LBrace)?;
                let mut body = Vec::new();
                while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
                    self.parse_flow_statement(&mut body)?;
                }
                self.expect(TokenKind::RBrace)?;

                flows.push(Flow::While {
                    cond,
                    body: self.bump.alloc_slice_copy(&body),
                });
                Ok(())
            }
            TokenKind::Bifurcate => {
                let f = self.parse_bifurcate()?;
                flows.push(f);
                Ok(())
            }
            TokenKind::Compete => {
                let f = self.parse_compete()?;
                flows.push(f);
                Ok(())
            }
            TokenKind::Relax => {
                let f = self.parse_relax()?;
                flows.push(f);
                Ok(())
            }
            TokenKind::Superpose => {
                let f = self.parse_superpose()?;
                flows.push(f);
                Ok(())
            }
            TokenKind::Drain => {
                self.advance();
                let target = self.parse_node_target()?;
                self.expect(TokenKind::Semi)?;
                flows.push(Flow::Drain(target));
                Ok(())
            }
            TokenKind::Learn => {
                self.advance();
                let pre = self.parse_node_target()?;
                self.expect(TokenKind::SynapseArrow)?;
                let dest = self.parse_node_target()?;
                let (rate, decay) = if self.peek_kind() == TokenKind::Colon {
                    self.advance();
                    let r = self.expect_number()?;
                    let d = if self.peek_kind() == TokenKind::Comma {
                        self.advance();
                        self.expect_number()?
                    } else {
                        0.01
                    };
                    (r, d)
                } else {
                    (0.1, 0.01)
                };
                self.expect(TokenKind::Semi)?;
                flows.push(Flow::Plastic {
                    pre,
                    dest,
                    rate,
                    decay,
                    kind: "hebbian",
                });
                Ok(())
            }
            TokenKind::Synapse => {
                self.advance();
                let pre = self.parse_node_target()?;
                self.expect(TokenKind::SynapseArrow)?;
                let dest = self.parse_node_target()?;
                self.expect(TokenKind::LBrace)?;

                let mut rate = 0.05;
                let mut decay = 0.001;
                let mut rule_kind = "hebbian";

                while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
                    let field = self.expect_ident()?;
                    self.expect(TokenKind::Colon)?;
                    match field {
                        "plastic" => {
                            let _val = self.expect_ident()?; // e.g. true
                        }
                        "rule" => {
                            rule_kind = self.expect_ident()?;
                        }
                        "rate" => {
                            rate = self.expect_number()?;
                        }
                        "decay" | "leak" => {
                            decay = self.expect_number()?;
                        }
                        other => {
                            return Err(format!(
                                "Line {}: Unknown property '{}' in synapse block",
                                self.current_line(),
                                other
                            ));
                        }
                    }
                    self.expect(TokenKind::Semi)?;
                }
                self.expect(TokenKind::RBrace)?;

                flows.push(Flow::Plastic {
                    pre,
                    dest,
                    rate,
                    decay,
                    kind: rule_kind,
                });
                Ok(())
            }
            TokenKind::Probe => {
                self.advance();
                let label = if let TokenKind::StringLit(s) = self.peek_kind() {
                    self.advance();
                    if self.peek_kind() == TokenKind::Comma {
                        self.advance();
                    }
                    Some(s)
                } else {
                    None
                };
                let expr = self.parse_expr()?;
                self.expect(TokenKind::Semi)?;
                flows.push(Flow::Probe { label, expr });
                Ok(())
            }
            TokenKind::LParen => {
                // Expression presynaptic stream: `(expr) ~> dest;`, `(expr) ~> (gate: g) ~> dest;`, etc.
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                match self.peek_kind() {
                    TokenKind::SynapseArrow => {
                        self.advance();
                        if self.peek_kind() == TokenKind::LParen
                            && self.peek_ahead_kind(1) == TokenKind::Gate
                        {
                            self.advance(); // (
                            self.advance(); // gate
                            self.expect(TokenKind::Colon)?;
                            let gate = self.parse_expr()?;
                            self.expect(TokenKind::RParen)?;
                            self.expect(TokenKind::SynapseArrow)?;
                            let gated_expr = self.bump.alloc(Expr::Binary {
                                op: BinOp::Mul,
                                left: expr,
                                right: self.bump.alloc(Expr::Activation {
                                    kind: ActivationKind::Saturate,
                                    expr: gate,
                                }),
                            });
                            if self.peek_kind() == TokenKind::LBracket {
                                let dests = self.parse_node_target_list()?;
                                self.expect(TokenKind::Semi)?;
                                for dest in dests.iter() {
                                    flows.push(Flow::Assign {
                                        dest: *dest,
                                        expr: gated_expr,
                                    });
                                }
                            } else {
                                let dest = self.parse_node_target()?;
                                self.expect(TokenKind::Semi)?;
                                flows.push(Flow::Assign {
                                    dest,
                                    expr: gated_expr,
                                });
                            }
                            Ok(())
                        } else if self.peek_kind() == TokenKind::LBracket
                            && self.peek_ahead_kind(1) == TokenKind::Shunt
                        {
                            self.advance(); // [
                            self.advance(); // shunt
                            self.expect(TokenKind::Colon)?;
                            let shunt = self.parse_expr()?;
                            self.expect(TokenKind::RBracket)?;
                            self.expect(TokenKind::SynapseArrow)?;
                            let inv_shunt = self.bump.alloc(Expr::Activation {
                                kind: ActivationKind::Inv,
                                expr: self.bump.alloc(Expr::Activation {
                                    kind: ActivationKind::Saturate,
                                    expr: shunt,
                                }),
                            });
                            let shunted_expr = self.bump.alloc(Expr::Binary {
                                op: BinOp::Mul,
                                left: expr,
                                right: inv_shunt,
                            });
                            if self.peek_kind() == TokenKind::LBracket {
                                let dests = self.parse_node_target_list()?;
                                self.expect(TokenKind::Semi)?;
                                for dest in dests.iter() {
                                    flows.push(Flow::Assign {
                                        dest: *dest,
                                        expr: shunted_expr,
                                    });
                                }
                            } else {
                                let dest = self.parse_node_target()?;
                                self.expect(TokenKind::Semi)?;
                                flows.push(Flow::Assign {
                                    dest,
                                    expr: shunted_expr,
                                });
                            }
                            Ok(())
                        } else if self.peek_kind() == TokenKind::LBracket {
                            let dests = self.parse_node_target_list()?;
                            self.expect(TokenKind::Semi)?;
                            for dest in dests.iter() {
                                flows.push(Flow::Assign {
                                    dest: *dest,
                                    expr,
                                });
                            }
                            Ok(())
                        } else {
                            let dest = self.parse_node_target()?;
                            self.expect(TokenKind::Semi)?;
                            flows.push(Flow::Assign { dest, expr });
                            Ok(())
                        }
                    }
                    TokenKind::SynapseWeightOpen => {
                        self.advance();
                        let weight = self.expect_number()?;
                        self.expect(TokenKind::SynapseWeightClose)?;
                        let scaled_expr = self.bump.alloc(Expr::Binary {
                            op: BinOp::Mul,
                            left: expr,
                            right: self.bump.alloc(Expr::Number(weight)),
                        });
                        if self.peek_kind() == TokenKind::LParen
                            && self.peek_ahead_kind(1) == TokenKind::Gate
                        {
                            self.advance(); // (
                            self.advance(); // gate
                            self.expect(TokenKind::Colon)?;
                            let gate = self.parse_expr()?;
                            self.expect(TokenKind::RParen)?;
                            self.expect(TokenKind::SynapseArrow)?;
                            let gated_expr = self.bump.alloc(Expr::Binary {
                                op: BinOp::Mul,
                                left: scaled_expr,
                                right: self.bump.alloc(Expr::Activation {
                                    kind: ActivationKind::Saturate,
                                    expr: gate,
                                }),
                            });
                            if self.peek_kind() == TokenKind::LBracket {
                                let dests = self.parse_node_target_list()?;
                                self.expect(TokenKind::Semi)?;
                                for dest in dests.iter() {
                                    flows.push(Flow::Assign {
                                        dest: *dest,
                                        expr: gated_expr,
                                    });
                                }
                            } else {
                                let dest = self.parse_node_target()?;
                                self.expect(TokenKind::Semi)?;
                                flows.push(Flow::Assign {
                                    dest,
                                    expr: gated_expr,
                                });
                            }
                            Ok(())
                        } else if self.peek_kind() == TokenKind::LBracket
                            && self.peek_ahead_kind(1) == TokenKind::Shunt
                        {
                            self.advance(); // [
                            self.advance(); // shunt
                            self.expect(TokenKind::Colon)?;
                            let shunt = self.parse_expr()?;
                            self.expect(TokenKind::RBracket)?;
                            self.expect(TokenKind::SynapseArrow)?;
                            let inv_shunt = self.bump.alloc(Expr::Activation {
                                kind: ActivationKind::Inv,
                                expr: self.bump.alloc(Expr::Activation {
                                    kind: ActivationKind::Saturate,
                                    expr: shunt,
                                }),
                            });
                            let shunted_expr = self.bump.alloc(Expr::Binary {
                                op: BinOp::Mul,
                                left: scaled_expr,
                                right: inv_shunt,
                            });
                            if self.peek_kind() == TokenKind::LBracket {
                                let dests = self.parse_node_target_list()?;
                                self.expect(TokenKind::Semi)?;
                                for dest in dests.iter() {
                                    flows.push(Flow::Assign {
                                        dest: *dest,
                                        expr: shunted_expr,
                                    });
                                }
                            } else {
                                let dest = self.parse_node_target()?;
                                self.expect(TokenKind::Semi)?;
                                flows.push(Flow::Assign {
                                    dest,
                                    expr: shunted_expr,
                                });
                            }
                            Ok(())
                        } else if self.peek_kind() == TokenKind::LBracket {
                            let dests = self.parse_node_target_list()?;
                            self.expect(TokenKind::Semi)?;
                            for dest in dests.iter() {
                                flows.push(Flow::Assign {
                                    dest: *dest,
                                    expr: scaled_expr,
                                });
                            }
                            Ok(())
                        } else {
                            let dest = self.parse_node_target()?;
                            self.expect(TokenKind::Semi)?;
                            flows.push(Flow::Assign {
                                dest,
                                expr: scaled_expr,
                            });
                            Ok(())
                        }
                    }
                    other => Err(format!(
                        "Line {}: Expected synaptic arrow ('~>' or '~[w]>') after expression in flow stream, got {:?}",
                        self.current_line(),
                        other
                    )),
                }
            }
            TokenKind::LBracket => {
                let srcs = self.parse_node_target_list()?;
                match self.peek_kind() {
                    TokenKind::SynapseArrow => {
                        self.advance();
                        if self.peek_kind() == TokenKind::LParen
                            && self.peek_ahead_kind(1) == TokenKind::Gate
                        {
                            self.advance(); // (
                            self.advance(); // gate
                            self.expect(TokenKind::Colon)?;
                            let gate = self.parse_expr()?;
                            self.expect(TokenKind::RParen)?;
                            self.expect(TokenKind::SynapseArrow)?;
                            let dest = self.parse_node_target()?;
                            self.expect(TokenKind::Semi)?;
                            flows.push(Flow::FanInGated {
                                srcs,
                                dest,
                                gate,
                                weight: 1.0,
                            });
                            Ok(())
                        } else if self.peek_kind() == TokenKind::LBracket
                            && self.peek_ahead_kind(1) == TokenKind::Shunt
                        {
                            self.advance(); // [
                            self.advance(); // shunt
                            self.expect(TokenKind::Colon)?;
                            let shunt = self.parse_expr()?;
                            self.expect(TokenKind::RBracket)?;
                            self.expect(TokenKind::SynapseArrow)?;
                            let dest = self.parse_node_target()?;
                            self.expect(TokenKind::Semi)?;
                            flows.push(Flow::FanInShunted {
                                srcs,
                                dest,
                                shunt,
                                weight: 1.0,
                            });
                            Ok(())
                        } else {
                            let dest = self.parse_node_target()?;
                            let when = self.parse_optional_when()?;
                            self.expect(TokenKind::Semi)?;
                            flows.push(Flow::FanIn {
                                srcs,
                                dest,
                                weight: 1.0,
                                when,
                            });
                            Ok(())
                        }
                    }
                    TokenKind::SynapseWeightOpen => {
                        self.advance();
                        let weight = self.expect_number()?;
                        self.expect(TokenKind::SynapseWeightClose)?;
                        if self.peek_kind() == TokenKind::LParen
                            && self.peek_ahead_kind(1) == TokenKind::Gate
                        {
                            self.advance(); // (
                            self.advance(); // gate
                            self.expect(TokenKind::Colon)?;
                            let gate = self.parse_expr()?;
                            self.expect(TokenKind::RParen)?;
                            self.expect(TokenKind::SynapseArrow)?;
                            let dest = self.parse_node_target()?;
                            self.expect(TokenKind::Semi)?;
                            flows.push(Flow::FanInGated {
                                srcs,
                                dest,
                                gate,
                                weight,
                            });
                            Ok(())
                        } else if self.peek_kind() == TokenKind::LBracket
                            && self.peek_ahead_kind(1) == TokenKind::Shunt
                        {
                            self.advance(); // [
                            self.advance(); // shunt
                            self.expect(TokenKind::Colon)?;
                            let shunt = self.parse_expr()?;
                            self.expect(TokenKind::RBracket)?;
                            self.expect(TokenKind::SynapseArrow)?;
                            let dest = self.parse_node_target()?;
                            self.expect(TokenKind::Semi)?;
                            flows.push(Flow::FanInShunted {
                                srcs,
                                dest,
                                shunt,
                                weight,
                            });
                            Ok(())
                        } else {
                            let dest = self.parse_node_target()?;
                            let when = self.parse_optional_when()?;
                            self.expect(TokenKind::Semi)?;
                            flows.push(Flow::FanIn {
                                srcs,
                                dest,
                                weight,
                                when,
                            });
                            Ok(())
                        }
                    }
                    TokenKind::InhibitArrow => {
                        self.advance();
                        let dest = self.parse_node_target()?;
                        let when = self.parse_optional_when()?;
                        self.expect(TokenKind::Semi)?;
                        flows.push(Flow::FanInInhibit { srcs, dest, when });
                        Ok(())
                    }
                    other => Err(format!(
                        "Line {}: Expected synaptic arrow after ident list, got {:?}",
                        self.current_line(),
                        other
                    )),
                }
            }
            TokenKind::Ident(_) | TokenKind::State | TokenKind::Basin => {
                let mut current_target = self.parse_node_target()?;

                // Check for scalar assignments first
                match self.peek_kind() {
                    TokenKind::Assign => {
                        self.advance();
                        let expr = self.parse_expr()?;
                        let when = self.parse_optional_when()?;
                        self.expect(TokenKind::Semi)?;
                        if let Some(cond) = when {
                            let cond_slice = self.bump.alloc_slice_copy(&[Flow::Assign {
                                dest: current_target,
                                expr,
                            }]);
                            flows.push(Flow::If {
                                cond,
                                then_flows: cond_slice,
                                else_flows: None,
                            });
                        } else {
                            flows.push(Flow::Assign {
                                dest: current_target,
                                expr,
                            });
                        }
                        return Ok(());
                    }
                    TokenKind::PlusAssign => {
                        self.advance();
                        let right = self.parse_expr()?;
                        let when = self.parse_optional_when()?;
                        self.expect(TokenKind::Semi)?;
                        let left = if let Some(indices) = current_target.indices {
                            self.bump.alloc(Expr::MultiIndex {
                                name: current_target.name,
                                indices,
                            })
                        } else if let Some(idx) = current_target.index {
                            self.bump.alloc(Expr::Index {
                                name: current_target.name,
                                index: idx,
                            })
                        } else {
                            self.bump.alloc(Expr::Ident(current_target.name))
                        };
                        let add_expr = self.bump.alloc(Expr::Binary {
                            op: BinOp::Add,
                            left,
                            right,
                        });
                        if let Some(cond) = when {
                            let cond_slice = self.bump.alloc_slice_copy(&[Flow::Assign {
                                dest: current_target,
                                expr: add_expr,
                            }]);
                            flows.push(Flow::If {
                                cond,
                                then_flows: cond_slice,
                                else_flows: None,
                            });
                        } else {
                            flows.push(Flow::Assign {
                                dest: current_target,
                                expr: add_expr,
                            });
                        }
                        return Ok(());
                    }
                    TokenKind::MinusAssign => {
                        self.advance();
                        let right = self.parse_expr()?;
                        let when = self.parse_optional_when()?;
                        self.expect(TokenKind::Semi)?;
                        let left = if let Some(indices) = current_target.indices {
                            self.bump.alloc(Expr::MultiIndex {
                                name: current_target.name,
                                indices,
                            })
                        } else if let Some(idx) = current_target.index {
                            self.bump.alloc(Expr::Index {
                                name: current_target.name,
                                index: idx,
                            })
                        } else {
                            self.bump.alloc(Expr::Ident(current_target.name))
                        };
                        let sub_expr = self.bump.alloc(Expr::Binary {
                            op: BinOp::Sub,
                            left,
                            right,
                        });
                        if let Some(cond) = when {
                            let cond_slice = self.bump.alloc_slice_copy(&[Flow::Assign {
                                dest: current_target,
                                expr: sub_expr,
                            }]);
                            flows.push(Flow::If {
                                cond,
                                then_flows: cond_slice,
                                else_flows: None,
                            });
                        } else {
                            flows.push(Flow::Assign {
                                dest: current_target,
                                expr: sub_expr,
                            });
                        }
                        return Ok(());
                    }
                    _ => {}
                }

                // Synaptic flow chain: one or more chained arrow transitions
                let mut loop_count = 0;
                while self.is_arrow_token() {
                    loop_count += 1;
                    match self.peek_kind() {
                        TokenKind::BiSynapseArrow => {
                            self.advance();
                            let next_target = self.parse_node_target()?;
                            let when = self.parse_optional_when()?;
                            flows.push(Flow::Synapse {
                                src: current_target,
                                dest: next_target,
                                weight: 1.0,
                                when,
                            });
                            flows.push(Flow::Synapse {
                                src: next_target,
                                dest: current_target,
                                weight: 1.0,
                                when,
                            });
                            current_target = next_target;
                        }
                        TokenKind::BiSynapseWeightOpen => {
                            self.advance();
                            let weight = self.expect_number()?;
                            self.expect(TokenKind::SynapseWeightClose)?;
                            let next_target = self.parse_node_target()?;
                            let when = self.parse_optional_when()?;
                            flows.push(Flow::Synapse {
                                src: current_target,
                                dest: next_target,
                                weight,
                                when,
                            });
                            flows.push(Flow::Synapse {
                                src: next_target,
                                dest: current_target,
                                weight,
                                when,
                            });
                            current_target = next_target;
                        }
                        TokenKind::BiInhibitArrow => {
                            self.advance();
                            let next_target = self.parse_node_target()?;
                            let when = self.parse_optional_when()?;
                            flows.push(Flow::Inhibit {
                                src: current_target,
                                dest: next_target,
                                when,
                            });
                            flows.push(Flow::Inhibit {
                                src: next_target,
                                dest: current_target,
                                when,
                            });
                            current_target = next_target;
                        }
                        TokenKind::SynapseArrow => {
                            if self.peek_ahead_kind(1) == TokenKind::LParen
                                && self.peek_ahead_kind(2) == TokenKind::Gate
                            {
                                self.advance(); // consume ~>
                                self.advance(); // consume (
                                self.advance(); // consume gate
                                self.expect(TokenKind::Colon)?;
                                let gate = self.parse_expr()?;
                                self.expect(TokenKind::RParen)?;
                                self.expect(TokenKind::SynapseArrow)?;
                                if self.peek_kind() == TokenKind::LBracket {
                                    let dests = self.parse_node_target_list()?;
                                    flows.push(Flow::BroadcastGated {
                                        src: current_target,
                                        dests,
                                        gate,
                                        weight: 1.0,
                                    });
                                    break;
                                } else {
                                    let next_target = self.parse_node_target()?;
                                    flows.push(Flow::GatedSynapse {
                                        src: current_target,
                                        dest: next_target,
                                        gate,
                                        weight: 1.0,
                                    });
                                    current_target = next_target;
                                }
                            } else if self.peek_ahead_kind(1) == TokenKind::LBracket
                                && self.peek_ahead_kind(2) == TokenKind::Shunt
                            {
                                self.advance(); // consume ~>
                                self.advance(); // consume [
                                self.advance(); // consume shunt
                                self.expect(TokenKind::Colon)?;
                                let shunt = self.parse_expr()?;
                                self.expect(TokenKind::RBracket)?;
                                self.expect(TokenKind::SynapseArrow)?;
                                if self.peek_kind() == TokenKind::LBracket {
                                    let dests = self.parse_node_target_list()?;
                                    flows.push(Flow::BroadcastShunted {
                                        src: current_target,
                                        dests,
                                        shunt,
                                        weight: 1.0,
                                    });
                                    break;
                                } else {
                                    let next_target = self.parse_node_target()?;
                                    flows.push(Flow::ShuntedSynapse {
                                        src: current_target,
                                        dest: next_target,
                                        shunt,
                                        weight: 1.0,
                                    });
                                    current_target = next_target;
                                }
                            } else if self.peek_ahead_kind(1) == TokenKind::LParen
                                && (self.peek_ahead_kind(2) == TokenKind::PipeGate
                                    || self.peek_ahead_kind(2) == TokenKind::Star
                                    || self.peek_ahead_kind(2) == TokenKind::SynapseArrow)
                            {
                                self.advance(); // consume ~>
                                self.advance(); // consume (
                                let mut branches = Vec::new();
                                while self.peek_kind() != TokenKind::RParen && !self.is_eof() {
                                    let mut steps = Vec::new();
                                    loop {
                                        if self.peek_kind() == TokenKind::PipeGate {
                                            self.advance();
                                            let act = match self.peek_kind() {
                                                TokenKind::Saturate => {
                                                    self.advance();
                                                    ActivationKind::Saturate
                                                }
                                                TokenKind::Clamp => {
                                                    self.advance();
                                                    ActivationKind::Clamp
                                                }
                                                TokenKind::Step => {
                                                    self.advance();
                                                    ActivationKind::Step
                                                }
                                                TokenKind::Inv => {
                                                    self.advance();
                                                    ActivationKind::Inv
                                                }
                                                TokenKind::Relu => {
                                                    self.advance();
                                                    ActivationKind::Relu
                                                }
                                                other => {
                                                    return Err(format!(
                                                        "Line {}: Expected filter name, got {:?}",
                                                        self.current_line(),
                                                        other
                                                    ))
                                                }
                                            };
                                            steps.push(crate::ast::PipelineStep::Filter(act));
                                        } else if self.peek_kind() == TokenKind::Star {
                                            self.advance();
                                            let factor = self.expect_number()?;
                                            steps.push(crate::ast::PipelineStep::Scale(factor));
                                        } else {
                                            break;
                                        }
                                    }

                                    self.expect(TokenKind::SynapseArrow)?;

                                    let dests = if self.peek_kind() == TokenKind::LParen {
                                        self.advance(); // (
                                        let mut d_list = Vec::new();
                                        while self.peek_kind() != TokenKind::RParen
                                            && !self.is_eof()
                                        {
                                            d_list.push(self.parse_node_target()?);
                                            if self.peek_kind() == TokenKind::Comma {
                                                self.advance();
                                            } else {
                                                break;
                                            }
                                        }
                                        self.expect(TokenKind::RParen)?;
                                        self.bump.alloc_slice_copy(&d_list)
                                            as &'a [NodeTarget<'a>]
                                    } else if self.peek_kind() == TokenKind::LBracket {
                                        self.parse_node_target_list()?
                                    } else {
                                        let single = self.parse_node_target()?;
                                        let mut d_list = Vec::new();
                                        d_list.push(single);
                                        self.bump.alloc_slice_copy(&d_list)
                                            as &'a [NodeTarget<'a>]
                                    };

                                    branches.push(crate::ast::BranchPipeline {
                                        steps: self.bump.alloc_slice_copy(&steps),
                                        dests,
                                    });

                                    if self.peek_kind() == TokenKind::Comma {
                                        self.advance();
                                    } else {
                                        break;
                                    }
                                }
                                self.expect(TokenKind::RParen)?;
                                flows.push(Flow::MultiBranch {
                                    src: current_target,
                                    branches: self.bump.alloc_slice_copy(&branches),
                                });
                                break;
                            } else if self.peek_ahead_kind(1) == TokenKind::LBracket {
                                self.advance();
                                let dests = self.parse_node_target_list()?;
                                let when = self.parse_optional_when()?;
                                flows.push(Flow::Broadcast {
                                    src: current_target,
                                    dests,
                                    weight: 1.0,
                                    when,
                                });
                                break;
                            } else {
                                self.advance();
                                let next_target = self.parse_node_target()?;
                                let when = self.parse_optional_when()?;
                                flows.push(Flow::Synapse {
                                    src: current_target,
                                    dest: next_target,
                                    weight: 1.0,
                                    when,
                                });
                                current_target = next_target;
                            }
                        }
                        TokenKind::SynapseWeightOpen => {
                            self.advance();
                            let weight = self.expect_number()?;
                            self.expect(TokenKind::SynapseWeightClose)?;
                            if self.peek_kind() == TokenKind::LParen
                                && self.peek_ahead_kind(1) == TokenKind::Gate
                            {
                                self.advance(); // consume (
                                self.advance(); // consume gate
                                self.expect(TokenKind::Colon)?;
                                let gate = self.parse_expr()?;
                                self.expect(TokenKind::RParen)?;
                                self.expect(TokenKind::SynapseArrow)?;
                                if self.peek_kind() == TokenKind::LBracket {
                                    let dests = self.parse_node_target_list()?;
                                    flows.push(Flow::BroadcastGated {
                                        src: current_target,
                                        dests,
                                        gate,
                                        weight,
                                    });
                                    break;
                                } else {
                                    let next_target = self.parse_node_target()?;
                                    flows.push(Flow::GatedSynapse {
                                        src: current_target,
                                        dest: next_target,
                                        gate,
                                        weight,
                                    });
                                    current_target = next_target;
                                }
                            } else if self.peek_kind() == TokenKind::LBracket
                                && self.peek_ahead_kind(1) == TokenKind::Shunt
                            {
                                self.advance(); // consume [
                                self.advance(); // consume shunt
                                self.expect(TokenKind::Colon)?;
                                let shunt = self.parse_expr()?;
                                self.expect(TokenKind::RBracket)?;
                                self.expect(TokenKind::SynapseArrow)?;
                                if self.peek_kind() == TokenKind::LBracket {
                                    let dests = self.parse_node_target_list()?;
                                    flows.push(Flow::BroadcastShunted {
                                        src: current_target,
                                        dests,
                                        shunt,
                                        weight,
                                    });
                                    break;
                                } else {
                                    let next_target = self.parse_node_target()?;
                                    flows.push(Flow::ShuntedSynapse {
                                        src: current_target,
                                        dest: next_target,
                                        shunt,
                                        weight,
                                    });
                                    current_target = next_target;
                                }
                            } else if self.peek_kind() == TokenKind::LBracket {
                                let dests = self.parse_node_target_list()?;
                                let when = self.parse_optional_when()?;
                                flows.push(Flow::Broadcast {
                                    src: current_target,
                                    dests,
                                    weight,
                                    when,
                                });
                                break;
                            } else {
                                let next_target = self.parse_node_target()?;
                                let when = self.parse_optional_when()?;
                                flows.push(Flow::Synapse {
                                    src: current_target,
                                    dest: next_target,
                                    weight,
                                    when,
                                });
                                current_target = next_target;
                            }
                        }
                        TokenKind::InhibitArrow => {
                            self.advance();
                            if self.peek_kind() == TokenKind::LBracket {
                                let dests = self.parse_node_target_list()?;
                                let when = self.parse_optional_when()?;
                                flows.push(Flow::BroadcastInhibit {
                                    src: current_target,
                                    dests,
                                    when,
                                });
                                break;
                            } else {
                                let next_target = self.parse_node_target()?;
                                let when = self.parse_optional_when()?;
                                flows.push(Flow::Inhibit {
                                    src: current_target,
                                    dest: next_target,
                                    when,
                                });
                                current_target = next_target;
                            }
                        }
                        _ => break,
                    }
                }

                if loop_count == 0 {
                    return Err(format!(
                        "Line {}: Expected flow operator ('~>', '<~>', '~[w]>', '<~[w]>', '~|>', '<~|>', '=', '+='), got {:?}",
                        self.current_line(),
                        self.peek_kind()
                    ));
                }

                self.expect(TokenKind::Semi)?;
                Ok(())
            }
            other => Err(format!(
                "Line {}: Expected flow statement, got {:?}",
                self.current_line(),
                other
            )),
        }
    }

    fn parse_bifurcate(&mut self) -> Result<Flow<'a>, String> {
        self.expect(TokenKind::Bifurcate)?;
        self.expect(TokenKind::LParen)?;
        let expr = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::LBrace)?;

        let mut branches = Vec::new();
        while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
            let cond = match self.peek_kind() {
                TokenKind::Lt => {
                    self.advance();
                    BifurcateCond::Lt(self.expect_number()?)
                }
                TokenKind::Lte => {
                    self.advance();
                    BifurcateCond::Lte(self.expect_number()?)
                }
                TokenKind::Gt => {
                    self.advance();
                    BifurcateCond::Gt(self.expect_number()?)
                }
                TokenKind::Gte => {
                    self.advance();
                    BifurcateCond::Gte(self.expect_number()?)
                }
                TokenKind::EqEq | TokenKind::Assign => {
                    self.advance();
                    BifurcateCond::Eq(self.expect_number()?)
                }
                TokenKind::LBracket => {
                    self.advance();
                    let low = self.expect_number()?;
                    self.expect(TokenKind::DotDot)?;
                    let high = self.expect_number()?;
                    self.expect(TokenKind::RBracket)?;
                    BifurcateCond::Range(low, high)
                }
                TokenKind::When => {
                    self.advance();
                    let c_expr = self.parse_expr()?;
                    BifurcateCond::When(c_expr)
                }
                TokenKind::Else => {
                    self.advance();
                    BifurcateCond::Else
                }
                other => {
                    return Err(format!(
                        "Line {}: Expected bifurcation condition (<, <=, >, >=, [low..high], when, else), got {:?}",
                        self.current_line(),
                        other
                    ));
                }
            };

            let mut target = None;
            let mut branch_flows = Vec::new();

            if self.peek_kind() == TokenKind::SynapseArrow {
                self.advance();
                if self.peek_kind() == TokenKind::Basin {
                    self.advance();
                }
                target = Some(self.parse_node_target()?);
                self.expect(TokenKind::Semi)?;
            } else if self.peek_kind() == TokenKind::Drift {
                self.advance();
                if self.peek_kind() == TokenKind::Basin {
                    self.advance();
                }
                target = Some(self.parse_node_target()?);
                self.expect(TokenKind::Semi)?;
            } else if self.peek_kind() == TokenKind::LBrace {
                self.advance();
                while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
                    self.parse_flow_statement(&mut branch_flows)?;
                }
                self.expect(TokenKind::RBrace)?;
            } else {
                return Err(format!(
                    "Line {}: Expected '~> target;' or '{{ flows... }}' after bifurcation condition",
                    self.current_line()
                ));
            }

            branches.push(BifurcateBranch {
                cond,
                target,
                flows: self.bump.alloc_slice_copy(&branch_flows),
            });
        }

        self.expect(TokenKind::RBrace)?;
        Ok(Flow::Bifurcate {
            expr,
            branches: self.bump.alloc_slice_copy(&branches),
        })
    }

    fn parse_compete(&mut self) -> Result<Flow<'a>, String> {
        self.expect(TokenKind::Compete)?;
        self.expect(TokenKind::LBrace)?;

        let mut branches = Vec::new();
        while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
            let name = self.expect_ident()?;
            let (branch_flows, head) = if self.peek_kind() == TokenKind::Colon {
                self.advance();
                let mut b_flows = Vec::new();
                if self.peek_kind() == TokenKind::LBrace {
                    self.advance();
                    while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
                        self.parse_flow_statement(&mut b_flows)?;
                    }
                    self.expect(TokenKind::RBrace)?;
                } else {
                    self.parse_flow_statement(&mut b_flows)?;
                }
                let h = b_flows
                    .iter()
                    .find_map(|f| match f {
                        Flow::Assign { dest, .. } => Some(*dest),
                        Flow::Synapse { dest, .. } => Some(*dest),
                        Flow::GatedSynapse { dest, .. } => Some(*dest),
                        Flow::ShuntedSynapse { dest, .. } => Some(*dest),
                        _ => None,
                    })
                    .or_else(|| Some(NodeTarget::simple(name)));
                (b_flows, h)
            } else {
                if self.peek_kind() == TokenKind::Semi {
                    self.advance();
                }
                (Vec::new(), Some(NodeTarget::simple(name)))
            };

            branches.push(CompeteBranch {
                name,
                flows: self.bump.alloc_slice_copy(&branch_flows),
                head,
            });
        }

        self.expect(TokenKind::RBrace)?;
        self.expect(TokenKind::Resolve)?;
        self.expect(TokenKind::WinnerTakeAll)?;

        let threshold = if self.peek_kind() == TokenKind::LParen {
            self.advance();
            let th = self.expect_number()?;
            self.expect(TokenKind::RParen)?;
            th
        } else {
            0.5
        };

        if self.peek_kind() == TokenKind::Semi {
            self.advance();
        }

        Ok(Flow::Compete {
            branches: self.bump.alloc_slice_copy(&branches),
            threshold,
        })
    }

    fn parse_relax(&mut self) -> Result<Flow<'a>, String> {
        self.expect(TokenKind::Relax)?;
        self.expect(TokenKind::LBrace)?;

        let mut body = Vec::new();
        while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
            self.parse_flow_statement(&mut body)?;
        }
        self.expect(TokenKind::RBrace)?;

        self.expect(TokenKind::Until)?;
        self.expect(TokenKind::Stable)?;
        self.expect(TokenKind::LParen)?;
        let tolerance = self.expect_number()?;
        self.expect(TokenKind::RParen)?;

        let mut timeout = None;
        if self.peek_kind() == TokenKind::Or {
            self.advance();
            self.expect(TokenKind::Timeout)?;
            self.expect(TokenKind::LParen)?;
            timeout = Some(self.expect_number()? as usize);
            self.expect(TokenKind::RParen)?;
        }

        if self.peek_kind() == TokenKind::Semi {
            self.advance();
        }

        Ok(Flow::Relax {
            body: self.bump.alloc_slice_copy(&body),
            tolerance,
            timeout,
        })
    }

    fn parse_superpose(&mut self) -> Result<Flow<'a>, String> {
        self.expect(TokenKind::Superpose)?;
        self.expect(TokenKind::LBrace)?;

        let mut branches = Vec::new();
        while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
            let name = self.expect_ident()?;
            self.expect(TokenKind::Colon)?;

            let mut branch_flows = Vec::new();
            if self.peek_kind() == TokenKind::LBrace {
                self.advance();
                while self.peek_kind() != TokenKind::RBrace && !self.is_eof() {
                    self.parse_flow_statement(&mut branch_flows)?;
                }
                self.expect(TokenKind::RBrace)?;
            } else {
                self.parse_flow_statement(&mut branch_flows)?;
            }

            branches.push(SuperposeBranch {
                name,
                flows: self.bump.alloc_slice_copy(&branch_flows),
            });
        }

        self.expect(TokenKind::RBrace)?;
        self.expect(TokenKind::CollapseOn)?;
        let collapse_expr = self.parse_expr()?;
        self.expect(TokenKind::SynapseArrow)?;
        let dest = self.parse_node_target()?;
        self.expect(TokenKind::Semi)?;

        Ok(Flow::Superpose {
            branches: self.bump.alloc_slice_copy(&branches),
            collapse_expr,
            dest,
        })
    }
}
