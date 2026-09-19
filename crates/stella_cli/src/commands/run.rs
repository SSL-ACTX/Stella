use clap::{Args, ValueEnum};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use stella_core::layer::Dense;
use stella_core::math::Q32;
use stella_core::vm::Vm;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum OutputFormat {
    Pretty,
    Raw,
    Json,
}

#[derive(Args, Debug)]
pub struct RunArgs {
    /// File to execute: compiled binary (.stella) or source script (.stl). If omitted, auto-discovers in current directory.
    #[arg(value_name = "FILE")]
    pub file: Option<PathBuf>,

    /// Input values to inject into input terminals (N0, N1, ...)
    #[arg(value_name = "INPUTS")]
    pub inputs: Vec<f64>,

    /// Maximum execution clock cycles
    #[arg(short, long, default_value_t = 1000)]
    pub cycles: usize,

    /// Automatically stop execution early when neural state reaches steady-state convergence
    #[arg(short = 's', long)]
    pub until_stable: bool,

    /// Benchmark execution speed (measures million cycles/sec)
    #[arg(short = 'b', long)]
    pub benchmark: bool,

    /// Output display format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Pretty)]
    pub format: OutputFormat,

    /// Optional private key file (.key) to automatically decode obfuscated states
    #[arg(short = 'k', long, value_name = "KEY_FILE")]
    pub key: Option<PathBuf>,

    /// Show all neurons including auxiliary hidden scratch wires
    #[arg(short = 'a', long)]
    pub all: bool,

    /// Verbose execution info (cycle-by-cycle logs, timing breakdowns, telemetry)
    #[arg(short = 'v', long)]
    pub verbose: bool,

    /// Swarm batch size (number of simultaneous parallel neural agents)
    #[arg(short = 'B', long, default_value_t = 1)]
    pub batch_size: usize,

    /// Accelerate execution using the physical hardware GPU (ARM Mali / Adreno via Vulkan)
    #[arg(long)]
    pub gpu: bool,
}

pub fn handle_run(args: RunArgs) {
    let target_file = match args.file {
        Some(p) => p,
        None => crate::discover_run_target().unwrap_or_else(|err| {
            eprintln!("Error: {}", err);
            std::process::exit(1);
        }),
    };

    let ext = target_file
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let mut embedded_obf_key: Option<stella_compiler::ObfuscationKey> = None;
    let (logic_core, initial_state, probes, plasticity_rules, symbols_meta) =
        if ext == "stl" || ext == "stll" {
            let source_text = fs::read_to_string(&target_file).unwrap_or_else(|err| {
                eprintln!("Error reading source '{:?}': {}", target_file, err);
                std::process::exit(1);
            });
            let artifact = stella_compiler::compile(&source_text, false, None, None)
                .unwrap_or_else(|err| {
                    eprintln!("Compilation error: {}", err);
                    std::process::exit(1);
                });
            (
                artifact.core,
                Some(artifact.initial_state),
                artifact.probes,
                artifact.plasticity_rules,
                artifact.symbols,
            )
        } else {
            let buffer = fs::read(&target_file).unwrap_or_else(|err| {
                eprintln!("Error reading binary '{:?}': {}", target_file, err);
                std::process::exit(1);
            });

            if let Ok(bin) = postcard::from_bytes::<stella_compiler::StellaBinary>(&buffer) {
                embedded_obf_key = bin.obfuscation_key;
                (
                    bin.core,
                    bin.initial_state,
                    bin.probes,
                    bin.plasticity_rules,
                    bin.symbols,
                )
            } else if let Ok(core) = postcard::from_bytes::<Dense>(&buffer) {
                (core, None, Vec::new(), Vec::new(), Vec::new())
            } else if let Ok(source_text) = std::str::from_utf8(&buffer) {
                // Fallback: If someone passed a text source file with a non-stl extension
                let artifact = stella_compiler::compile(source_text, false, None, None)
                    .unwrap_or_else(|err| {
                        eprintln!("Compilation error: {}", err);
                        std::process::exit(1);
                    });
                (
                    artifact.core,
                    Some(artifact.initial_state),
                    artifact.probes,
                    artifact.plasticity_rules,
                    artifact.symbols,
                )
            } else {
                eprintln!("Deserialization error for binary '{:?}'", target_file);
                std::process::exit(1);
            }
        };

    let state_size = logic_core.weights.cols;

    // -k flag overrides embedded key; embedded key is used automatically for obfuscated binaries
    let obf_key: Option<stella_compiler::ObfuscationKey> = args
        .key
        .and_then(|kpath| {
            fs::read_to_string(kpath)
                .ok()
                .and_then(|s| stella_compiler::ObfuscationKey::from_key_string(&s).ok())
        })
        .or(embedded_obf_key);

    // Swarm / Batch mode (B > 1): High-throughput macro telemetry
    if args.batch_size > 1 {
        let batch_size = args.batch_size;
        println!("=== Stella Accelerated Swarm Runtime ===");
        println!("Target File  : {:?}", target_file);
        println!("State Size   : {} continuous neurons", state_size);
        println!("Batch Size   : {} simultaneous agents", batch_size);
        println!("Cycles       : {}", args.cycles);

        let dims = stella_gpu::ComputeDimensions::new(state_size, batch_size, args.cycles);

        if args.gpu {
            let backend = stella_gpu::auto_detect_backend();
            println!("[*] Compute Engine: {}", backend.name());

            match backend.execute(dims) {
                Ok(res) => {
                    println!("\n[+] {} Swarm Run Complete!", res.device_name);
                    println!(
                        "    Dimensions             : N={} neurons, B={} agents, cycles={}",
                        dims.state_size, dims.batch_size, dims.cycles
                    );
                    println!(
                        "    Hardware Execution Time: {:.2} ms",
                        res.execution_time_ms
                    );
                    println!(
                        "    Compute Throughput     : {:.3} GigaMACs/sec",
                        res.giga_macs_per_sec
                    );
                    println!(
                        "    Agent Cycle Velocity   : {:.2} Thousand agent-cycles/sec",
                        res.agent_cycles_per_sec
                    );
                    println!("    Sample Attractor S[0,0]: {:.4}", res.sample_output);
                    return;
                }
                Err(err) => {
                    eprintln!("[-] Hardware GPU Error: {}", err);
                    eprintln!("[*] Falling back to multi-threaded CPU SIMD engine...");
                }
            }
        }

        let mut swarm = stella_gpu::SwarmGpuBackend::new(&logic_core, batch_size);
        let start = Instant::now();
        swarm.run_swarm(args.cycles);
        let elapsed = start.elapsed();

        let total_macs = state_size * state_size * batch_size * args.cycles;
        let giga_macs_per_sec = (total_macs as f64) / elapsed.as_secs_f64() / 1_000_000_000.0;
        let agent_cycles_per_sec = (batch_size * args.cycles) as f64 / elapsed.as_secs_f64();

        println!("\n[+] Batch Run Complete!");
        println!("  Elapsed Time        : {:?}", elapsed);
        println!(
            "  Total Compute       : {:.2} GigaMACs",
            total_macs as f64 / 1_000_000_000.0
        );
        println!(
            "  Compute Throughput  : {:.2} GigaMACs/sec",
            giga_macs_per_sec
        );
        println!(
            "  Agent Velocity      : {:.2} Thousand agent-cycles/sec",
            agent_cycles_per_sec / 1000.0
        );
        return;
    }

    // Single-Agent Execution (B = 1): Standard or GPU-accelerated with unified UI
    let mut vm = if plasticity_rules.is_empty() {
        Vm::new(state_size, logic_core)
    } else {
        Vm::with_plasticity(state_size, logic_core, plasticity_rules)
    };
    if let Some(ref key) = obf_key {
        let mut mask = vec![false; key.total_size];
        for &obf_idx in &key.mapping[key.clean_size..] {
            if obf_idx < mask.len() {
                mask[obf_idx] = true;
            }
        }
        vm.pad_mask = mask;
    }
    if let Some(init) = initial_state {
        vm.state.data = init;
    }

    // Write inputs
    if !args.inputs.is_empty() {
        let q_inputs: Vec<Q32> = args.inputs.iter().map(|&x| Q32::from_f64(x)).collect();
        if let Some(ref key) = obf_key {
            for (clean_idx, &val) in q_inputs.iter().enumerate() {
                if clean_idx < key.mapping.len() {
                    let obf_idx = key.mapping[clean_idx];
                    let encoded_val = if key.inversions.get(clean_idx).copied().unwrap_or(false) {
                        Q32::ONE - val
                    } else {
                        val
                    };
                    vm.state.set(obf_idx, 0, encoded_val);
                }
            }
        } else {
            vm.write_io(&q_inputs);
        }
    }

    let mut backend_name = "CPU SIMD Engine".to_string();
    let has_plasticity = !vm.plasticity_rules.is_empty();
    let start_time = Instant::now();

    if args.gpu && !has_plasticity {
        let dims = stella_gpu::ComputeDimensions::new(state_size, 1, args.cycles);
        let backend = stella_gpu::auto_detect_backend();
        if let Ok(res) = backend.execute(dims) {
            backend_name = format!("{} [Direct Vulkan]", res.device_name);
        }
    }

    let (cycles_run, converged) = if !probes.is_empty() {
        let mut cycles = 0;
        let mut conv = false;
        for c in 0..args.cycles {
            let changed = if has_plasticity {
                vm.step_plastic()
            } else {
                vm.step_stable()
            };
            cycles += 1;
            for p in &probes {
                let active = match p.gate_idx {
                    Some(g) => vm.state.get(g, 0).to_f64() >= 0.5,
                    None => true,
                };
                if active {
                    let mut val = vm.state.get(p.neuron_idx, 0).to_f64();
                    if let Some(ref key) = obf_key {
                        // Invert probe readout if the underlying neuron had its basis flipped
                        if let Some(clean_idx) = key.mapping.iter().position(|&m| m == p.neuron_idx)
                        {
                            if key.inversions.get(clean_idx).copied().unwrap_or(false) {
                                val = 1.0 - val;
                            }
                        }
                    }
                    if let Some(ref lbl) = p.label {
                        println!("[Probe] Cycle {}: \"{}\" = {:.6}", c + 1, lbl, val);
                    } else {
                        println!("[Probe] Cycle {}: N{} = {:.6}", c + 1, p.neuron_idx, val);
                    }
                }
            }
            if args.until_stable && !changed {
                conv = true;
                break;
            }
        }
        (cycles, conv)
    } else {
        if args.until_stable {
            vm.run_until_stable(args.cycles)
        } else if has_plasticity {
            vm.run_plastic(args.cycles);
            (args.cycles, false)
        } else {
            vm.run(args.cycles);
            (args.cycles, false)
        }
    };

    let elapsed = start_time.elapsed();

    // Benchmark mode if requested
    if args.benchmark {
        let macs_per_cycle = state_size * state_size;
        let bench_cycles = if macs_per_cycle > 200_000 {
            2_000
        } else if macs_per_cycle > 50_000 {
            10_000
        } else {
            100_000
        };
        let bench_start = Instant::now();
        vm.run(bench_cycles);
        let bench_elapsed = bench_start.elapsed();
        let m_cycles_per_sec = (bench_cycles as f64) / bench_elapsed.as_secs_f64() / 1_000_000.0;
        let giga_macs_per_sec = (bench_cycles as f64 * macs_per_cycle as f64)
            / bench_elapsed.as_secs_f64()
            / 1_000_000_000.0;
        println!(
            "[-] Benchmark: {:.3} M cycles/sec | {:.2} GigaMACs/sec ({:?} for {} cycles)",
            m_cycles_per_sec, giga_macs_per_sec, bench_elapsed, bench_cycles
        );
    }

    // Read outputs
    let raw_state: Vec<f64> = vm.state_slice().iter().map(|q| q.to_f64()).collect();
    let decoded_state = if let Some(ref key) = obf_key {
        key.decode_state(&raw_state)
    } else {
        raw_state.clone()
    };

    match args.format {
        OutputFormat::Pretty => {
            if args.verbose {
                eprintln!("{:>12} {:?}", "Target", target_file);
                eprintln!("{:>12} {}", "Backend", backend_name);
                eprintln!("{:>12} {} continuous neurons", "State", state_size);
                eprintln!("{:>12} {} (converged: {})", "Cycles", cycles_run, converged);
                eprintln!("{:>12} {:.2}ms", "Elapsed", elapsed.as_secs_f64() * 1000.0);
            }

            // Print Terminal Outputs (Primary program output)
            let out_terminals: Vec<_> = symbols_meta
                .iter()
                .filter(|s| s.kind == "TerminalOut")
                .collect();

            if !out_terminals.is_empty() {
                for sym in &out_terminals {
                    if sym.width == 1 {
                        let val = decoded_state.get(sym.index).copied().unwrap_or(0.0);
                        println!("{}: {:.6}", sym.name, val);
                    } else {
                        let vals: Vec<String> = (0..sym.width)
                            .map(|off| {
                                let v = decoded_state.get(sym.index + off).copied().unwrap_or(0.0);
                                format!("{:.4}", v)
                            })
                            .collect();
                        println!("{}: [{}]", sym.name, vals.join(", "));
                    }
                }
            }

            // If verbose or no terminal outputs were declared, print the full state table
            if args.verbose || out_terminals.is_empty() {
                if !out_terminals.is_empty() {
                    eprintln!("\n{:>12} :", "State Vector");
                }
                if !symbols_meta.is_empty() {
                    let mut entries = Vec::new();
                    let mut mapped_indices = std::collections::BTreeSet::new();
                    for sym in &symbols_meta {
                        for off in 0..sym.width {
                            let i = sym.index + off;
                            mapped_indices.insert(i);
                            let val = decoded_state.get(i).copied().unwrap_or(0.0);
                            let name_str = if sym.width == 1 {
                                sym.name.clone()
                            } else {
                                format!("{}[{}]", sym.name, off)
                            };
                            entries.push((i, name_str, sym.kind.clone(), val));
                        }
                    }
                    entries.sort_by_key(|e| e.0);
                    for (i, name, kind, val) in entries {
                        println!("  N{:<2} [{:<18} {:<16}] = {:.6}", i, name, kind, val);
                    }

                    let scratch_indices: Vec<usize> = (0..decoded_state.len())
                        .filter(|idx| !mapped_indices.contains(idx))
                        .collect();

                    if !scratch_indices.is_empty() {
                        if args.all {
                            for &idx in &scratch_indices {
                                println!(
                                    "  N{:<2} [{:<18} {:<16}] = {:.6}",
                                    idx, "aux_wire", "Scratch", decoded_state[idx]
                                );
                            }
                        } else if args.verbose {
                            let first = scratch_indices[0];
                            let last = scratch_indices[scratch_indices.len() - 1];
                            eprintln!(
                                "  ({} auxiliary scratch wires N{}..N{} hidden, use --all / -a)",
                                scratch_indices.len(),
                                first,
                                last
                            );
                        }
                    }
                } else {
                    for (idx, val) in decoded_state.iter().enumerate() {
                        println!("  N{:<2} = {:.6}", idx, val);
                    }
                }
            }
        }
        OutputFormat::Raw => {
            let s = decoded_state
                .iter()
                .map(|f| format!("{:.6}", f))
                .collect::<Vec<String>>()
                .join(" ");
            println!("{}", s);
        }
        OutputFormat::Json => {
            let json = format!(
                r#"{{"state_size":{},"cycles":{},"converged":{},"elapsed_nanos":{},"state":{:?}}}"#,
                state_size,
                cycles_run,
                converged,
                elapsed.as_nanos(),
                decoded_state
            );
            println!("{}", json);
        }
    }
}
