// crates/stella_frontend/src/parser/handlers/mod.rs
pub mod decls;
pub mod exprs;
pub mod flows;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use pest::iterators::Pair;

use crate::ast::*;
use crate::parser::{Lowerer, Rule};

impl<'a> Lowerer<'a> {
    pub fn lower_phase_block(&mut self, pair: Pair<'a, Rule>) -> Result<Vec<Basin<'a>>, String> {
        let mut inner = pair.into_inner();
        let phase_name = inner.next().unwrap().as_str();

        let mut basins = Vec::new();
        for state_pair in inner {
            let b = self.lower_basin_block(state_pair, Some(phase_name))?;
            basins.push(b);
        }

        Ok(basins)
    }

    pub fn lower_basin_block(
        &mut self,
        pair: Pair<'a, Rule>,
        group: Option<&'a str>,
    ) -> Result<Basin<'a>, String> {
        let mut inner = pair.into_inner();
        let raw_name = inner.next().unwrap().as_str();

        let name = if let Some(grp) = group {
            self.bump.alloc_str(&format!("{}::{}", grp, raw_name))
        } else {
            raw_name
        };

        let mut flows = Vec::new();
        let mut bifurcations = Vec::new();

        for item in inner {
            let item_inner = item.into_inner().next().unwrap();
            match item_inner.as_rule() {
                Rule::transition_stmt => {
                    let mut t_inner = item_inner.into_inner();
                    let first = t_inner.next().unwrap();
                    let (raw_target, condition) = if first.as_rule() == Rule::ident {
                        let target = first.as_str();
                        let cond = if let Some(w) = t_inner.next() {
                            Some(self.lower_expr(w.into_inner().next().unwrap())?)
                        } else {
                            None
                        };
                        (target, cond)
                    } else {
                        let cond = Some(self.lower_expr(first)?);
                        let target = t_inner.next().unwrap().as_str();
                        (target, cond)
                    };

                    let target = if let Some(grp) = group {
                        if !raw_target.contains("::") {
                            self.bump.alloc_str(&format!("{}::{}", grp, raw_target))
                        } else {
                            raw_target
                        }
                    } else {
                        raw_target
                    };

                    bifurcations.push(Bifurcation { target, condition });
                }
                Rule::flow_stmt => {
                    let mut sub = self.lower_flow_stmt(item_inner)?;
                    flows.append(&mut sub);
                }
                _ => {}
            }
        }

        Ok(Basin {
            name,
            flows: self.bump.alloc_slice_copy(&flows),
            bifurcations: self.bump.alloc_slice_copy(&bifurcations),
        })
    }
}
