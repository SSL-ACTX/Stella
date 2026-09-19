// crates/stella_frontend/src/parser/mod.rs

mod circuit;
mod decl;
mod expr;
mod flow;
mod parse;
mod tokens;

use crate::lexer::Token;
use alloc::collections::BTreeMap;
use bumpalo::Bump;

pub struct Parser<'a, 't> {
    pub(super) tokens: &'t [Token<'a>],
    pub(super) cursor: usize,
    pub(super) bump: &'a Bump,
    pub(super) type_aliases: BTreeMap<&'a str, (usize, Option<&'a [usize]>)>,
}
