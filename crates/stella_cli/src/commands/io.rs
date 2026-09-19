use clap::Args;
use std::fs;
use std::path::PathBuf;

use stella_core::math::Q32;

#[derive(Args, Debug)]
pub struct IoArgs {
    /// Action mode: encode inputs for obfuscated VM, decode outputs, observe terminals, or unmask
    #[arg(value_name = "MODE")]
    pub mode: String,

    /// Path to private key file (.key) containing permutation mapping and chaotic manifold
    #[arg(short, long, value_name = "KEY_FILE")]
    pub key: PathBuf,

    /// Discrete clock cycle / step t for time-varying terminal rolling
    #[arg(short, long, default_value_t = 0)]
    pub step: usize,

    /// Values to encode or decode
    #[arg(value_name = "VALUES")]
    pub values: Vec<f64>,
}

pub fn handle_io(args: IoArgs) {
    let key_str = fs::read_to_string(&args.key).unwrap_or_else(|err| {
        eprintln!("Failed to open key file '{:?}': {}", args.key, err);
        std::process::exit(1);
    });

    let obf_key =
        stella_compiler::ObfuscationKey::from_key_string(&key_str).unwrap_or_else(|err| {
            eprintln!("Failed to parse key file '{:?}': {}", args.key, err);
            std::process::exit(1);
        });

    match args.mode.to_lowercase().as_str() {
        "encode" => {
            let q_inputs: Vec<Q32> = args.values.iter().map(|&v| Q32::from_f64(v)).collect();
            let encoded = obf_key.encode_state(&q_inputs);
            println!(
                "{}",
                encoded
                    .iter()
                    .map(|f| format!("{:.6}", f.to_f64()))
                    .collect::<Vec<String>>()
                    .join(" ")
            );
        }
        "decode" => {
            let decoded = if obf_key.rolling.is_some() && args.step > 0 {
                obf_key.decode_observed_state(&args.values, args.step)
            } else {
                obf_key.decode_state(&args.values)
            };
            println!("{:?}", decoded);
        }
        "observe" => {
            let observed = obf_key.observe_terminals(&args.values, args.step);
            println!("{:?}", observed);
        }
        "unmask" => {
            let clean = obf_key.decode_terminals(&args.values, args.step);
            println!("{:?}", clean);
        }
        _ => {
            eprintln!(
                "Unknown mode '{}'. Choose 'encode', 'decode', 'observe', or 'unmask'.",
                args.mode
            );
            std::process::exit(1);
        }
    }
}
