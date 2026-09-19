use super::Parser;
use crate::lexer::{Token, TokenKind};
use alloc::format;
use alloc::string::String;

impl<'a, 't> Parser<'a, 't> {
    pub(super) fn is_arrow_token(&self) -> bool {
        matches!(
            self.peek_kind(),
            TokenKind::SynapseArrow
                | TokenKind::SynapseWeightOpen
                | TokenKind::InhibitArrow
                | TokenKind::BiSynapseArrow
                | TokenKind::BiSynapseWeightOpen
                | TokenKind::BiInhibitArrow
        )
    }

    pub fn peek_kind(&self) -> TokenKind<'a> {
        self.tokens
            .get(self.cursor)
            .map(|t| t.kind.clone())
            .unwrap_or(TokenKind::Eof)
    }

    pub(super) fn advance(&mut self) -> Option<&'t Token<'a>> {
        if self.cursor < self.tokens.len() {
            let tok = &self.tokens[self.cursor];
            self.cursor += 1;
            Some(tok)
        } else {
            None
        }
    }

    pub(super) fn expect(&mut self, expected: TokenKind<'a>) -> Result<(), String> {
        let cur = self.peek_kind();
        if cur == expected {
            self.advance();
            Ok(())
        } else {
            Err(format!(
                "Line {}: Expected {:?}, got {:?}",
                self.current_line(),
                expected,
                cur
            ))
        }
    }

    pub(super) fn expect_ident(&mut self) -> Result<&'a str, String> {
        match self.peek_kind() {
            TokenKind::Ident(id) => {
                self.advance();
                Ok(id)
            }
            TokenKind::Basin => {
                self.advance();
                Ok("basin")
            }
            TokenKind::Phase => {
                self.advance();
                Ok("phase")
            }
            TokenKind::State => {
                self.advance();
                Ok("state")
            }
            TokenKind::Node => {
                self.advance();
                Ok("node")
            }
            TokenKind::Hold => {
                self.advance();
                Ok("hold")
            }
            TokenKind::Latch => {
                self.advance();
                Ok("latch")
            }
            TokenKind::Leak => {
                self.advance();
                Ok("leak")
            }
            TokenKind::Plastic => {
                self.advance();
                Ok("plastic")
            }
            other => Err(format!(
                "Line {}: Expected identifier, got {:?}",
                self.current_line(),
                other
            )),
        }
    }

    pub(super) fn expect_number(&mut self) -> Result<f64, String> {
        match self.peek_kind() {
            TokenKind::Number(n) => {
                self.advance();
                Ok(n)
            }
            other => Err(format!(
                "Line {}: Expected number, got {:?}",
                self.current_line(),
                other
            )),
        }
    }

    pub(super) fn expect_string(&mut self) -> Result<&'a str, String> {
        match self.peek_kind() {
            TokenKind::StringLit(s) => {
                self.advance();
                Ok(s)
            }
            other => Err(format!(
                "Line {}: Expected string literal, got {:?}",
                self.current_line(),
                other
            )),
        }
    }

    pub(super) fn is_eof(&self) -> bool {
        self.peek_kind() == TokenKind::Eof
    }

    pub(super) fn current_line(&self) -> usize {
        self.tokens.get(self.cursor).map(|t| t.line).unwrap_or(1)
    }

    pub(super) fn peek_ahead_kind(&self, offset: usize) -> TokenKind<'a> {
        self.tokens
            .get(self.cursor + offset)
            .map(|t| t.kind.clone())
            .unwrap_or(TokenKind::Eof)
    }

    pub(super) fn is_template_call_ahead(&self) -> bool {
        let mut offset = 0;
        if self.peek_ahead_kind(offset) != TokenKind::Lt {
            return false;
        }
        offset += 1;
        while offset < 32 {
            match self.peek_ahead_kind(offset) {
                TokenKind::Gt => {
                    return matches!(self.peek_ahead_kind(offset + 1), TokenKind::LParen);
                }
                TokenKind::Number(_) | TokenKind::Comma | TokenKind::Minus => {
                    offset += 1;
                }
                _ => return false,
            }
        }
        false
    }
}
