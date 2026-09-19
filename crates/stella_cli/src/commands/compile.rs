use clap::Args;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Args, Debug)]
pub struct CompileArgs {
    /// Path to the Stella Synaptic DSL file (.stl). If omitted, defaults to main.stl or first .stl found.
    #[arg(value_name = "SOURCE")]
    pub source: Option<PathBuf>,

    /// Output binary path (.stella)
    #[arg(short, long, value_name = "OUTPUT")]
    pub output: Option<PathBuf>,

    /// Enable chaotic polymorphic obfuscation (Hénon attractor permutation + rotational noise)
    #[arg(short = 'O', long)]
    pub obfuscate: bool,

    /// Minimum padded state vector size for obfuscation
    #[arg(long)]
    pub pad_size: Option<usize>,

    /// Path to export the private permutation key file (defaults to `<OUTPUT>.key`)
    #[arg(long, value_name = "KEY_FILE")]
    pub key: Option<PathBuf>,

    /// Enable verbose compiler diagnostics and matrix topology readout
    #[arg(short = 'v', long)]
    pub verbose: bool,
}

pub fn handle_compile(args: CompileArgs) {
    let source_path = match args.source {
        Some(p) => p,
        None => crate::discover_source().unwrap_or_else(|err| {
            eprintln!("error: {}", err);
            std::process::exit(1);
        }),
    };

    let source_text = fs::read_to_string(&source_path).unwrap_or_else(|err| {
        eprintln!("error: reading source file '{:?}': {}", source_path, err);
        std::process::exit(1);
    });

    let default_output = source_path.with_extension("stella");
    let out_stella = args.output.unwrap_or(default_output);
    let out_key = args.key.unwrap_or_else(|| out_stella.with_extension("key"));

    let start = Instant::now();
    let display_name = source_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("source");
    eprintln!("{:>12} {}", "Compiling", display_name);

    let chaos_seed = if args.obfuscate {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let seed_x = (nanos as f64 / 1_000_000_000.0) * 2.0 - 1.0;
        let seed_y = ((nanos % 1_000_000) as f64 / 1_000_000.0) * 2.0 - 1.0;
        Some((seed_x, seed_y))
    } else {
        None
    };

    let artifact =
        stella_compiler::compile(&source_text, args.obfuscate, args.pad_size, chaos_seed)
            .unwrap_or_else(|err| {
                eprintln!(
                    "error: compilation failed\n  --> {:?}\n  {}",
                    source_path, err
                );
                std::process::exit(1);
            });

    let bytes = artifact.to_bytes().expect("Postcard serialization failed");
    fs::write(&out_stella, &bytes).unwrap_or_else(|err| {
        eprintln!("error: failed to write output '{:?}': {}", out_stella, err);
        std::process::exit(1);
    });

    if let Some(ref key) = artifact.obfuscation_key {
        let key_str = key.to_key_string();
        fs::write(&out_key, key_str.as_bytes()).unwrap_or_else(|err| {
            eprintln!("error: failed to write key '{:?}': {}", out_key, err);
            std::process::exit(1);
        });
    } else if let Some(ref key) = artifact.permutation_key {
        let key_str = key
            .iter()
            .map(|k| k.to_string())
            .collect::<Vec<String>>()
            .join(",");
        fs::write(&out_key, key_str.as_bytes()).unwrap_or_else(|err| {
            eprintln!("error: failed to write key '{:?}': {}", out_key, err);
            std::process::exit(1);
        });
    }

    if args.verbose {
        eprintln!(
            "        Core: {} neurons, {} weights",
            artifact.state_size,
            artifact.state_size * artifact.state_size
        );
        eprintln!("      Output: {:?} ({} bytes)", out_stella, bytes.len());
        if artifact.obfuscation_key.is_some() || artifact.permutation_key.is_some() {
            eprintln!("         Key: {:?}", out_key);
        }
        for sym in &artifact.symbols {
            eprintln!(
                "      Symbol: {:<16} [N{:02}..N{:02}] ({})",
                sym.name,
                sym.index,
                sym.index + sym.width - 1,
                sym.kind
            );
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "{:>12} release target(s) in {:.2}s",
        "Finished",
        elapsed.as_secs_f64()
    );
}
