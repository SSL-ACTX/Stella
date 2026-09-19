// crates/stella_frontend/src/lib.rs
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod ast;
pub mod circuit;
pub mod lexer;
pub mod parser;
pub mod symbols;

use alloc::string::String;
use bumpalo::Bump;

pub use ast::*;
pub use circuit::monomorphize;
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::Parser;
pub use symbols::{NeuronInfo, NeuronKind, NeuronLayout};

/// Front-end pipeline: tokenizes, parses into an arena-allocated zero-copy AST,
/// monomorphizes circuit subgraphs into continuous physical dynamics,
/// and performs semantic symbol resolution and layout planning.
pub fn parse_synaptic<'a>(
    source: &'a str,
    bump: &'a Bump,
) -> Result<(ast::Program<'a>, symbols::NeuronLayout<'a>), String> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;

    let mut parser = Parser::new(&tokens, bump);
    let raw_program = parser.parse()?;

    let program = circuit::monomorphize(&raw_program, bump)?;

    let layout = NeuronLayout::allocate(&program)?;
    Ok((program, layout))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_full_arithmetics_and_cfg() {
        let source = "
        const FACTOR = 2.5;
        terminal in a;
        terminal in b;
        terminal out out1;
        terminal out out2;
        node acc : latch = 0.0;

        out1 = a * FACTOR + b / 2.0 - (-a);
        out2 = (a xor b) and not (a nor b);

        if a != b {
            acc += 0.5;
        } else {
            acc -= 0.1;
        }

        while acc < 1.0 {
            acc += 0.2;
        }
        ";

        let bump = Bump::new();
        let (program, layout) = parse_synaptic(source, &bump).expect("Parsing failed");

        assert_eq!(layout.constants.get("FACTOR"), Some(&2.5));
        assert_eq!(layout.in_terminal_count, 2);
        assert_eq!(layout.out_terminal_count, 2);
        assert_eq!(program.flows.len(), 4);
    }

    #[test]
    fn test_parse_dynamical_control_flow() {
        let source = "
        terminal in sensor;
        terminal in enable;
        terminal in mute;
        terminal out alert;
        node state : hold;

        // Inline gated & shunted synaptic flow
        sensor ~> (gate: enable) ~> alert;
        sensor ~[0.75]> [shunt: mute] ~> state;

        // Continuous phase bifurcation
        bifurcate (sensor) {
            < 0.25 ~> alert;
            [0.25 .. 0.75] {
                state += 0.1;
            }
            > 0.75 {
                state = 1.0;
            }
            else ~> alert;
        }

        // Competitive lateral inhibition
        compete {
            primary: {
                state = sensor * 0.9;
            }
            backup: {
                state = 0.5;
            }
        } resolve winner_take_all(0.85);

        // Recurrent relaxation settling loop
        relax {
            state ~> alert;
        } until stable(0.001) or timeout(128);

        // State superposition & wave collapse
        superpose {
            path_a: {
                state += 0.2;
            }
            path_b: {
                state += 0.4;
            }
        } collapse_on (state |> saturate) ~> alert;
        ";

        let bump = Bump::new();
        let (program, layout) =
            parse_synaptic(source, &bump).expect("Parsing dynamical control flow failed");

        assert_eq!(layout.in_terminal_count, 3);
        assert_eq!(layout.out_terminal_count, 1);
        assert_eq!(program.flows.len(), 6);
    }

    #[test]
    fn test_parameterized_circuit_templates() {
        let source = "
        circuit Filter<alpha = 0.85, beta = 1.0>(in raw, out filtered) {
            node state : hold = 0.0;
            state = raw * alpha + state * (1.0 - alpha);
            filtered = state * beta;
        }

        terminal in sig;
        terminal out out_sig;

        inst f1 = Filter<0.92, 2.0>(sig, out_sig);
        ";

        let bump = Bump::new();
        let (program, layout) =
            parse_synaptic(source, &bump).expect("Parameterized circuit parsing failed");

        assert_eq!(layout.in_terminal_count, 1);
        assert_eq!(layout.out_terminal_count, 1);
        // Scoped node created: f1::state
        assert!(layout.symbols.contains_key("f1::state"));
        // Flows monomorphized: 2 assignment flows
        assert_eq!(program.flows.len(), 2);
    }

    #[test]
    fn test_multidimensional_tensors_and_receptive_fields() {
        let source = "
        terminal in tensor[4, 4] receptive_field;
        terminal out tensor[2, 2] pooled;
        node tensor[3, 3, 2] filter_bank;

        pooled[0, 1] = receptive_field[1, 2] * 0.75;
        pooled[1, 0] = filter_bank[2, 1, 0];
        ";

        let bump = Bump::new();
        let (program, layout) =
            parse_synaptic(source, &bump).expect("Tensor syntax parsing failed");

        assert_eq!(layout.in_terminal_count, 16);
        assert_eq!(layout.out_terminal_count, 4);

        let rf_info = layout.symbols.get("receptive_field").unwrap();
        assert_eq!(rf_info.width, 16);
        assert_eq!(rf_info.shape.as_deref(), Some(&[4, 4][..]));

        let pooled_info = layout.symbols.get("pooled").unwrap();
        assert_eq!(pooled_info.width, 4);
        assert_eq!(pooled_info.shape.as_deref(), Some(&[2, 2][..]));

        let fb_info = layout.symbols.get("filter_bank").unwrap();
        assert_eq!(fb_info.width, 18);
        assert_eq!(fb_info.shape.as_deref(), Some(&[3, 3, 2][..]));

        assert_eq!(program.flows.len(), 2);
    }

    #[test]
    fn test_universal_shapes_and_type_aliases() {
        let source = "
        type Byte = [8];
        type Matrix4x4 = [4, 4];

        terminal in Byte rx;
        terminal out [8] tx;
        node Matrix4x4 grid;
        node [16] buffer;

        circuit Inverter(in Byte a, out Byte b) {
            b = a * -1.0;
        }

        inst my_inv = Inverter(rx, tx);
        grid[1, 1] = rx[0];
        ";

        let bump = Bump::new();
        let (program, layout) =
            parse_synaptic(source, &bump).expect("Universal shape & type alias parsing failed");

        assert_eq!(layout.in_terminal_count, 8);
        assert_eq!(layout.out_terminal_count, 8);

        let rx_info = layout.symbols.get("rx").unwrap();
        assert_eq!(rx_info.width, 8);

        let tx_info = layout.symbols.get("tx").unwrap();
        assert_eq!(tx_info.width, 8);

        let grid_info = layout.symbols.get("grid").unwrap();
        assert_eq!(grid_info.width, 16);
        assert_eq!(grid_info.shape.as_deref(), Some(&[4, 4][..]));

        let buf_info = layout.symbols.get("buffer").unwrap();
        assert_eq!(buf_info.width, 16);

        assert!(program.flows.len() >= 2);
    }
}
