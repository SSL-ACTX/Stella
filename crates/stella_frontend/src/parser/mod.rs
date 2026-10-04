// crates/stella_frontend/src/parser/mod.rs
// =============================================================================
// Stella Pest PEG Parser & AST Lowering Engine
// =============================================================================

pub mod handlers;

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use bumpalo::Bump;
use pest::iterators::Pairs;
use pest::Parser as PestParserTrait;
use pest_derive::Parser;

use crate::ast::*;

#[derive(Parser)]
#[grammar = "stella.pest"]
pub struct StellaParser;

/// Parses a Stella program source string into an arena-allocated zero-copy AST.
pub fn parse_program<'a>(source: &'a str, bump: &'a Bump) -> Result<Program<'a>, String> {
    let pairs = StellaParser::parse(Rule::program, source).map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new(bump);
    lowerer.lower_program(pairs)
}

pub struct Parser;
impl Parser {
    pub fn parse_source<'a>(source: &'a str, bump: &'a Bump) -> Result<Program<'a>, String> {
        parse_program(source, bump)
    }
}

pub(crate) struct Lowerer<'a> {
    pub(crate) bump: &'a Bump,
    pub(crate) type_aliases: BTreeMap<&'a str, (usize, Option<&'a [usize]>)>,
}

impl<'a> Lowerer<'a> {
    pub fn new(bump: &'a Bump) -> Self {
        Self {
            bump,
            type_aliases: BTreeMap::new(),
        }
    }

    pub fn lower_program(&mut self, mut pairs: Pairs<'a, Rule>) -> Result<Program<'a>, String> {
        let program_pair = pairs
            .next()
            .ok_or_else(|| "Empty program source".to_string())?;

        let mut declarations = Vec::new();
        let mut flows = Vec::new();
        let mut basins = Vec::new();

        for pair in program_pair.into_inner() {
            match pair.as_rule() {
                Rule::item => {
                    let inner = pair.into_inner().next().unwrap();
                    match inner.as_rule() {
                        Rule::attribute => {
                            if let Some(decl) = self.lower_attribute(inner)? {
                                declarations.push(decl);
                            }
                        }
                        Rule::decl => {
                            let mut decls = self.lower_decl(inner)?;
                            declarations.append(&mut decls);
                        }
                        Rule::phase_block => {
                            let mut b_list = self.lower_phase_block(inner)?;
                            basins.append(&mut b_list);
                        }
                        Rule::basin_block => {
                            let b = self.lower_basin_block(inner, None)?;
                            basins.push(b);
                        }
                        Rule::flow_stmt => {
                            let mut f_list = self.lower_flow_stmt(inner)?;
                            flows.append(&mut f_list);
                        }
                        Rule::EOI => break,
                        other => {
                            return Err(format!("Unexpected top-level rule: {:?}", other));
                        }
                    }
                }
                Rule::EOI => break,
                other => {
                    return Err(format!("Unexpected token in program root: {:?}", other));
                }
            }
        }

        Ok(Program {
            declarations: self.bump.alloc_slice_copy(&declarations),
            flows: self.bump.alloc_slice_copy(&flows),
            basins: self.bump.alloc_slice_copy(&basins),
        })
    }
}
