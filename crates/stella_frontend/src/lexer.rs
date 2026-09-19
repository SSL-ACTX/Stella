// crates/stella_frontend/src/lexer.rs

use alloc::format;
use alloc::string::String;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind<'a> {
    // Keywords
    Terminal,
    In,
    Out,
    Node,
    Leak,
    Latch,
    Hold,
    Const,
    Basin,
    State,
    Drift,
    Goto,
    When,
    Drain,
    Saturate,
    Clamp,
    And,
    Or,
    Nand,
    Nor,
    Xor,
    Not,
    If,
    Else,
    While,
    Probe,
    Cloak,
    Pad,
    Seed,
    Step,
    Inv,
    Relu,
    Bus,
    Vec,
    Tensor,
    Circuit,
    Inst,
    Synapse,
    Plastic,
    Learn,
    Trace,
    Bifurcate,
    Compete,
    Resolve,
    WinnerTakeAll,
    Relax,
    Until,
    Stable,
    Timeout,
    Superpose,
    CollapseOn,
    Assert,
    Shunt,
    Gate,
    Oscillator,
    Orbit,
    Type,
    Curve,
    Match,
    Phase,
    Conv2d,
    AvgPool2d,

    StringLit(&'a str),
    Ident(&'a str),
    Number(f64),

    // Synaptic directed flow operators
    SynapseArrow,        // ~>
    SynapseWeightOpen,   // ~[
    SynapseWeightClose,  // ]>
    InhibitArrow,        // ~|>
    PipeGate,            // |>
    BiSynapseArrow,      // <~>
    BiSynapseWeightOpen, // <~[
    BiInhibitArrow,      // <~|>

    // Standard operators
    Plus,
    Minus,
    Star,
    Slash,
    Assign,
    PlusAssign,
    MinusAssign,
    EqEq,
    Neq,
    Gte,
    Lte,
    Gt,
    Lt,
    RArrow,   // ->
    FatArrow, // =>
    DotDot,   // ..
    Dot,      // .

    // Delimiters
    Semi,
    Comma,
    Colon,
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,

    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token<'a> {
    pub kind: TokenKind<'a>,
    pub line: usize,
}

pub struct Lexer<'a> {
    input: &'a str,
    chars: core::str::CharIndices<'a>,
    current_line: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.char_indices(),
            current_line: 1,
        }
    }

    pub fn tokenize(&mut self) -> Result<alloc::vec::Vec<Token<'a>>, String> {
        let mut tokens = alloc::vec::Vec::new();

        while let Some((idx, ch)) = self.chars.next() {
            if ch == '\n' {
                self.current_line += 1;
                continue;
            }
            if ch.is_whitespace() {
                continue;
            }

            // Comments (// or #)
            if (ch == '/' && self.peek() == Some('/')) || ch == '#' {
                while let Some((_, c)) = self.chars.next() {
                    if c == '\n' {
                        self.current_line += 1;
                        break;
                    }
                }
                continue;
            }

            let token = match ch {
                ';' => TokenKind::Semi,
                ',' => TokenKind::Comma,
                ':' => TokenKind::Colon,
                '{' => TokenKind::LBrace,
                '}' => TokenKind::RBrace,
                '(' => TokenKind::LParen,
                ')' => TokenKind::RParen,
                '[' => TokenKind::LBracket,
                ']' => {
                    if self.peek() == Some('>') {
                        self.chars.next();
                        TokenKind::SynapseWeightClose
                    } else {
                        TokenKind::RBracket
                    }
                }
                '*' => TokenKind::Star,
                '/' => TokenKind::Slash,
                '~' => match self.peek() {
                    Some('>') => {
                        self.chars.next();
                        TokenKind::SynapseArrow
                    }
                    Some('[') => {
                        self.chars.next();
                        TokenKind::SynapseWeightOpen
                    }
                    Some('|') => {
                        self.chars.next();
                        if self.peek() == Some('>') {
                            self.chars.next();
                            TokenKind::InhibitArrow
                        } else {
                            return Err(format!(
                                "Line {}: Expected '>' after '~|'",
                                self.current_line
                            ));
                        }
                    }
                    other => {
                        return Err(format!(
                            "Line {}: Unexpected character after '~': {:?}",
                            self.current_line, other
                        ))
                    }
                },
                '|' => {
                    if self.peek() == Some('>') {
                        self.chars.next();
                        TokenKind::PipeGate
                    } else {
                        return Err(format!(
                            "Line {}: Expected '>' after '|'",
                            self.current_line
                        ));
                    }
                }
                '+' => {
                    if self.peek() == Some('=') {
                        self.chars.next();
                        TokenKind::PlusAssign
                    } else {
                        TokenKind::Plus
                    }
                }
                '-' => {
                    if self.peek() == Some('=') {
                        self.chars.next();
                        TokenKind::MinusAssign
                    } else if self.peek() == Some('>') {
                        self.chars.next();
                        TokenKind::RArrow
                    } else if let Some(next_ch) = self.peek() {
                        if next_ch.is_ascii_digit() {
                            self.lex_number(idx)?
                        } else {
                            TokenKind::Minus
                        }
                    } else {
                        TokenKind::Minus
                    }
                }
                '=' => {
                    if self.peek() == Some('=') {
                        self.chars.next();
                        TokenKind::EqEq
                    } else if self.peek() == Some('>') {
                        self.chars.next();
                        TokenKind::FatArrow
                    } else {
                        TokenKind::Assign
                    }
                }
                '>' => {
                    if self.peek() == Some('=') {
                        self.chars.next();
                        TokenKind::Gte
                    } else {
                        TokenKind::Gt
                    }
                }
                '<' => {
                    if self.peek() == Some('=') {
                        self.chars.next();
                        TokenKind::Lte
                    } else if self.peek() == Some('~') {
                        self.chars.next();
                        match self.peek() {
                            Some('>') => {
                                self.chars.next();
                                TokenKind::BiSynapseArrow
                            }
                            Some('[') => {
                                self.chars.next();
                                TokenKind::BiSynapseWeightOpen
                            }
                            Some('|') => {
                                self.chars.next();
                                if self.peek() == Some('>') {
                                    self.chars.next();
                                    TokenKind::BiInhibitArrow
                                } else {
                                    return Err(format!(
                                        "Line {}: Expected '>' after '<~|'",
                                        self.current_line
                                    ));
                                }
                            }
                            other => {
                                return Err(format!(
                                    "Line {}: Unexpected character after '<~': {:?}",
                                    self.current_line, other
                                ));
                            }
                        }
                    } else {
                        TokenKind::Lt
                    }
                }
                '!' => {
                    if self.peek() == Some('=') {
                        self.chars.next();
                        TokenKind::Neq
                    } else {
                        TokenKind::Not
                    }
                }
                '.' => {
                    if self.peek() == Some('.') {
                        self.chars.next();
                        TokenKind::DotDot
                    } else {
                        TokenKind::Dot
                    }
                }
                '"' => self.lex_string(idx)?,
                c if c.is_ascii_digit() => self.lex_number(idx)?,
                c if c.is_ascii_alphabetic() || c == '_' => self.lex_ident(idx),
                other => {
                    return Err(format!(
                        "Line {}: Unexpected character '{}'",
                        self.current_line, other
                    ))
                }
            };

            tokens.push(Token {
                kind: token,
                line: self.current_line,
            });
        }

        tokens.push(Token {
            kind: TokenKind::Eof,
            line: self.current_line,
        });

        Ok(tokens)
    }

    fn peek(&self) -> Option<char> {
        self.chars.clone().next().map(|(_, c)| c)
    }

    fn lex_ident(&mut self, start: usize) -> TokenKind<'a> {
        let mut end = start;
        while let Some((idx, c)) = self.peek_char() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.chars.next();
                end = idx + c.len_utf8();
            } else {
                break;
            }
        }
        if end <= start {
            end = start
                + self.input[start..]
                    .chars()
                    .next()
                    .map_or(1, |c| c.len_utf8());
        }

        let slice = &self.input[start..end];
        match slice {
            "terminal" => TokenKind::Terminal,
            "in" | "input" => TokenKind::In,
            "out" | "output" => TokenKind::Out,
            "node" | "reg" => TokenKind::Node,
            "leak" | "decay" => TokenKind::Leak,
            "latch" => TokenKind::Latch,
            "hold" => TokenKind::Hold,
            "const" => TokenKind::Const,
            "basin" => TokenKind::Basin,
            "drift" | "goto" => TokenKind::Drift,
            "when" => TokenKind::When,
            "drain" => TokenKind::Drain,
            "saturate" => TokenKind::Saturate,
            "clamp" => TokenKind::Clamp,
            "and" => TokenKind::And,
            "or" => TokenKind::Or,
            "nand" => TokenKind::Nand,
            "nor" => TokenKind::Nor,
            "xor" => TokenKind::Xor,
            "not" => TokenKind::Not,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "probe" | "print" => TokenKind::Probe,
            "cloak" => TokenKind::Cloak,
            "pad" => TokenKind::Pad,
            "seed" => TokenKind::Seed,
            "step" => TokenKind::Step,
            "inv" | "invert" => TokenKind::Inv,
            "relu" | "rectify" => TokenKind::Relu,
            "bus" => TokenKind::Bus,
            "vec" => TokenKind::Vec,
            "tensor" => TokenKind::Tensor,
            "circuit" => TokenKind::Circuit,
            "inst" | "instance" => TokenKind::Inst,
            "synapse" => TokenKind::Synapse,
            "plastic" => TokenKind::Plastic,
            "learn" => TokenKind::Learn,
            "trace" => TokenKind::Trace,
            "bifurcate" => TokenKind::Bifurcate,
            "compete" => TokenKind::Compete,
            "resolve" => TokenKind::Resolve,
            "winner_take_all" | "wta" => TokenKind::WinnerTakeAll,
            "relax" => TokenKind::Relax,
            "until" => TokenKind::Until,
            "stable" => TokenKind::Stable,
            "timeout" => TokenKind::Timeout,
            "superpose" => TokenKind::Superpose,
            "collapse_on" | "collapse" => TokenKind::CollapseOn,
            "assert" => TokenKind::Assert,
            "shunt" => TokenKind::Shunt,
            "gate" => TokenKind::Gate,
            "oscillator" | "osc" => TokenKind::Oscillator,
            "orbit" => TokenKind::Orbit,
            "type" => TokenKind::Type,
            "curve" => TokenKind::Curve,
            "match" => TokenKind::Match,
            "phase" => TokenKind::Phase,
            "state" => TokenKind::State,
            "conv2d" => TokenKind::Conv2d,
            "avgpool2d" => TokenKind::AvgPool2d,
            ident => TokenKind::Ident(ident),
        }
    }

    fn lex_string(&mut self, start_idx: usize) -> Result<TokenKind<'a>, String> {
        let content_start = start_idx + 1;
        while let Some((idx, c)) = self.chars.next() {
            if c == '"' {
                let s = &self.input[content_start..idx];
                return Ok(TokenKind::StringLit(s));
            }
            if c == '\n' {
                self.current_line += 1;
            }
        }
        Err(format!(
            "Line {}: Unterminated string literal",
            self.current_line
        ))
    }

    fn lex_number(&mut self, start: usize) -> Result<TokenKind<'a>, String> {
        let mut end = start;
        let mut has_dot = false;

        while let Some((idx, c)) = self.peek_char() {
            if c.is_ascii_digit() {
                self.chars.next();
                end = idx + c.len_utf8();
            } else if c == '.' && !has_dot {
                // If followed by another '.', do not consume: it is a range operator '..'
                let mut peek_chars = self.chars.clone();
                peek_chars.next();
                if peek_chars.next().map(|(_, ch)| ch) == Some('.') {
                    break;
                }
                has_dot = true;
                self.chars.next();
                end = idx + c.len_utf8();
            } else {
                break;
            }
        }
        if end <= start {
            end = start
                + self.input[start..]
                    .chars()
                    .next()
                    .map_or(1, |c| c.len_utf8());
        }

        let slice = &self.input[start..end];
        let val: f64 = slice
            .parse()
            .map_err(|_| format!("Line {}: Invalid number '{}'", self.current_line, slice))?;

        Ok(TokenKind::Number(val))
    }

    fn peek_char(&self) -> Option<(usize, char)> {
        self.chars.clone().next()
    }
}
