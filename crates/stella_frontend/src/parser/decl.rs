use super::Parser;
use crate::ast::*;
use crate::lexer::TokenKind;
use alloc::string::String;
use alloc::vec::Vec;

impl<'a, 't> Parser<'a, 't> {
    pub(super) fn parse_optional_when(&mut self) -> Result<Option<&'a Expr<'a>>, String> {
        if self.peek_kind() == TokenKind::When {
            self.advance();
            let cond = self.parse_expr()?;
            Ok(Some(cond))
        } else {
            Ok(None)
        }
    }

    pub(super) fn parse_shape(&mut self) -> Result<&'a [usize], String> {
        self.expect(TokenKind::LBracket)?;
        let mut dims = Vec::new();
        dims.push(self.expect_number()? as usize);
        while self.peek_kind() == TokenKind::Comma {
            self.advance();
            dims.push(self.expect_number()? as usize);
        }
        self.expect(TokenKind::RBracket)?;
        Ok(self.bump.alloc_slice_copy(&dims))
    }

    pub(super) fn parse_decl_name_and_width(
        &mut self,
    ) -> Result<(&'a str, usize, Option<&'a [usize]>), String> {
        if self.peek_kind() == TokenKind::LBracket {
            let dims = self.parse_shape()?;
            let total_width: usize = dims.iter().product();
            let shape = if dims.len() > 1 { Some(dims) } else { None };
            let name = self.expect_ident()?;
            Ok((name, total_width, shape))
        } else if self.peek_kind() == TokenKind::Bus || self.peek_kind() == TokenKind::Vec {
            self.advance();
            self.expect(TokenKind::LBracket)?;
            let width = self.expect_number()? as usize;
            self.expect(TokenKind::RBracket)?;
            let name = self.expect_ident()?;
            Ok((name, width, None))
        } else if self.peek_kind() == TokenKind::Tensor {
            self.advance();
            let dims = self.parse_shape()?;
            let name = self.expect_ident()?;
            let total_width: usize = dims.iter().product();
            Ok((name, total_width, Some(dims)))
        } else {
            let id = self.expect_ident()?;
            if let Some(&(width, shape)) = self.type_aliases.get(id) {
                let name = self.expect_ident()?;
                Ok((name, width, shape))
            } else if self.peek_kind() == TokenKind::LBracket {
                self.advance();
                let mut dims = Vec::new();
                dims.push(self.expect_number()? as usize);
                while self.peek_kind() == TokenKind::Comma {
                    self.advance();
                    dims.push(self.expect_number()? as usize);
                }
                self.expect(TokenKind::RBracket)?;
                let total_width = dims.iter().product();
                let shape = if dims.len() > 1 {
                    Some(self.bump.alloc_slice_copy(&dims) as &'a [usize])
                } else {
                    None
                };
                Ok((id, total_width, shape))
            } else {
                Ok((id, 1, None))
            }
        }
    }

    pub(super) fn parse_node_target(&mut self) -> Result<NodeTarget<'a>, String> {
        let name = self.expect_ident()?;
        if self.peek_kind() == TokenKind::LBracket {
            self.advance();
            match self.peek_kind() {
                TokenKind::Ident(addr_id) => {
                    self.advance();
                    self.expect(TokenKind::RBracket)?;
                    Ok(NodeTarget::dynamic(name, addr_id))
                }
                _ => {
                    let first_idx = self.expect_number()? as usize;
                    if self.peek_kind() == TokenKind::DotDot {
                        self.advance();
                        let end_idx = self.expect_number()? as usize;
                        self.expect(TokenKind::RBracket)?;
                        Ok(NodeTarget::sliced(name, first_idx, end_idx))
                    } else if self.peek_kind() == TokenKind::Comma {
                        let mut dims = Vec::new();
                        dims.push(first_idx);
                        while self.peek_kind() == TokenKind::Comma {
                            self.advance();
                            dims.push(self.expect_number()? as usize);
                        }
                        self.expect(TokenKind::RBracket)?;
                        let indices = self.bump.alloc_slice_copy(&dims);
                        Ok(NodeTarget::multidim(name, indices))
                    } else {
                        self.expect(TokenKind::RBracket)?;
                        Ok(NodeTarget::indexed(name, first_idx))
                    }
                }
            }
        } else {
            Ok(NodeTarget::simple(name))
        }
    }

    pub(super) fn parse_node_target_list(&mut self) -> Result<&'a [NodeTarget<'a>], String> {
        self.expect(TokenKind::LBracket)?;
        let mut list = Vec::new();
        while self.peek_kind() != TokenKind::RBracket && !self.is_eof() {
            let target = self.parse_node_target()?;
            list.push(target);
            if self.peek_kind() == TokenKind::Comma {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(TokenKind::RBracket)?;
        Ok(self.bump.alloc_slice_copy(&list))
    }
}
