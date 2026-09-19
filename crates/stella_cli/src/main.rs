use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;

pub mod commands;

use commands::compile::{handle_compile, CompileArgs};
use commands::disasm::{handle_disasm, DisasmArgs};
use commands::io::{handle_io, IoArgs};
use commands::run::{handle_run, RunArgs};
use commands::swarm::{handle_swarm, SwarmArgs};

#[cfg(target_os = "android")]
use commands::radb::{handle_radb, RadbArgs};

#[derive(Parser, Debug)]
#[command(
    name = "stella",
    author = "Stella VM Team",
    version = "0.2.0",
    about = "Project Stella: Continuous-State Neural Virtual Machine Protector & Runtime",
    long_about = "A high-performance virtual machine protector translating continuous synaptic logic into fixed-point linear algebra attractors."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Compile/build a .stl Synaptic script into a .stella continuous neural binary
    #[command(alias = "build", alias = "c")]
    Compile(CompileArgs),

    /// Execute a compiled .stella binary (or run a .stl script directly)
    #[command(alias = "r")]
    Run(RunArgs),

    /// Disassemble / cryptanalyze an obfuscated .stella binary using original source
    Disasm(DisasmArgs),

    /// Encode input data or decode output state using a private permutation key
    Io(IoArgs),

    /// Execute a massive parallel neural swarm (GPU / tiled batch GEMM backend)
    Swarm(SwarmArgs),

    /// Built-in Pure-Rust Wireless ADB Engine (localhost UID 2000 hardware bridge)
    #[cfg(target_os = "android")]
    Radb(RadbArgs),
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile(args) => handle_compile(args),
        Commands::Run(args) => handle_run(args),
        Commands::Disasm(args) => handle_disasm(args),
        Commands::Io(args) => handle_io(args),
        Commands::Swarm(args) => handle_swarm(args),
        #[cfg(target_os = "android")]
        Commands::Radb(args) => handle_radb(args),
    }
}

pub fn discover_source() -> Result<PathBuf, String> {
    let candidates = ["main.stl", "src/main.stl", "index.stl"];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Ok(p);
        }
    }
    // Search current directory for any .stl file
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("stl") {
                return Ok(p);
            }
        }
    }
    Err("No .stl source file specified and none found in current directory (e.g. main.stl)".into())
}

pub fn discover_run_target() -> Result<PathBuf, String> {
    // Check for compiled binary first
    let bin_candidates = ["main.stella", "target/main.stella", "out.stella"];
    for c in &bin_candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Ok(p);
        }
    }
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("stella") {
                return Ok(p);
            }
        }
    }
    // Fall back to source file
    discover_source()
}
