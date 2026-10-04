use clap::{Args, ValueEnum};
use std::fs;
use std::io::{self, BufRead};
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

    /// Stream inputs from stdin line-by-line (time-series sensor feed / Unix pipe)
    #[arg(long)]
    pub stream: bool,
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
    let (logic_core, initial_state, probes, emits, return_neuron, plasticity_rules, symbols_meta) =
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
                artifact.emits,
                artifact.return_neuron,
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
                    bin.emits,
                    bin.return_neuron,
                    bin.plasticity_rules,
                    bin.symbols,
                )
            } else if let Ok(core) = postcard::from_bytes::<Dense>(&buffer) {
                (
                    core,
                    None,
                    Vec::new(),
                    Vec::new(),
                    None,
                    Vec::new(),
                    Vec::new(),
                )
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
                    artifact.emits,
                    artifact.return_neuron,
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

    let inject_inputs = |vm: &mut Vm, inputs: &[f64]| {
        if inputs.is_empty() {
            return;
        }
        let q_inputs: Vec<Q32> = inputs.iter().map(|&x| Q32::from_f64(x)).collect();
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
    };

    // --- Stream Mode: Continuous Time-Series Processing from stdin ---
    if args.stream {
        println!("● Stella Stream Processor");
        println!("  target     : {}", target_file.display());
        println!("  mode       : continuous stdin pipeline (line-by-line frames)");
        if obf_key.is_some() {
            println!("  security   : polymorphic basis obfuscated");
        }
        println!("  waiting for input frames (space/comma separated numbers, Ctrl+D to finish)...");
        println!();

        let stdin = io::stdin();
        let mut frame_idx = 0;

        let out_terminals: Vec<_> = symbols_meta
            .iter()
            .filter(|s| s.kind == "TerminalOut")
            .collect();

        for line_res in stdin.lock().lines() {
            let line = match line_res {
                Ok(l) => l,
                Err(_) => break,
            };
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Parse numbers from space or comma separated line
            let frame_inputs: Vec<f64> = trimmed
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.parse::<f64>().ok())
                .collect();

            if frame_inputs.is_empty() {
                continue;
            }

            frame_idx += 1;
            inject_inputs(&mut vm, &frame_inputs);

            // Run configured cycles for this frame
            let has_plasticity = !vm.plasticity_rules.is_empty();
            let mut cycles_this_frame = 0;
            let mut conv = false;

            for _ in 0..args.cycles {
                let changed = if has_plasticity {
                    vm.step_plastic()
                } else {
                    vm.step_stable()
                };
                cycles_this_frame += 1;

                if args.until_stable && !changed {
                    conv = true;
                    break;
                }
            }

            // Read decoded output state
            let raw: Vec<f64> = vm.state_slice().iter().map(|q| q.to_f64()).collect();
            let decoded = if let Some(ref key) = obf_key {
                key.decode_state(&raw)
            } else {
                raw
            };

            match args.format {
                OutputFormat::Raw => {
                    let out_vals: Vec<String> = if !out_terminals.is_empty() {
                        out_terminals
                            .iter()
                            .flat_map(|sym| {
                                (0..sym.width).map(|off| {
                                    format!(
                                        "{:.6}",
                                        decoded.get(sym.index + off).copied().unwrap_or(0.0)
                                    )
                                })
                            })
                            .collect()
                    } else {
                        decoded.iter().map(|v| format!("{:.6}", v)).collect()
                    };
                    println!("{}", out_vals.join(" "));
                }
                OutputFormat::Json => {
                    let mut json_out = format!(
                        r#"{{"frame":{},"cycles":{},"converged":{}"#,
                        frame_idx, cycles_this_frame, conv
                    );
                    if !out_terminals.is_empty() {
                        json_out.push_str(r#","outputs":{"#);
                        let mut first = true;
                        for sym in &out_terminals {
                            if !first {
                                json_out.push(',');
                            }
                            first = false;
                            if sym.width == 1 {
                                let v = decoded.get(sym.index).copied().unwrap_or(0.0);
                                json_out.push_str(&format!(r#""{}":{:.6}"#, sym.name, v));
                            } else {
                                let vals: Vec<String> = (0..sym.width)
                                    .map(|off| {
                                        format!(
                                            "{:.6}",
                                            decoded.get(sym.index + off).copied().unwrap_or(0.0)
                                        )
                                    })
                                    .collect();
                                json_out.push_str(&format!(
                                    r#""{}":[{}]"#,
                                    sym.name,
                                    vals.join(",")
                                ));
                            }
                        }
                        json_out.push('}');
                    } else {
                        json_out.push_str(&format!(r#","state":{:?}"#, decoded));
                    }
                    json_out.push('}');
                    println!("{}", json_out);
                }
                OutputFormat::Pretty => {
                    print!("  [Frame {:>3} / {}c] ", frame_idx, cycles_this_frame);
                    if !out_terminals.is_empty() {
                        for (i, sym) in out_terminals.iter().enumerate() {
                            if i > 0 {
                                print!("  |  ");
                            }
                            if sym.width == 1 {
                                let v = decoded.get(sym.index).copied().unwrap_or(0.0);
                                print!("{} = {:.4}", sym.name, v);
                            } else {
                                let vals: Vec<String> = (0..sym.width)
                                    .map(|off| {
                                        format!(
                                            "{:.3}",
                                            decoded.get(sym.index + off).copied().unwrap_or(0.0)
                                        )
                                    })
                                    .collect();
                                print!("{} = [{}]", sym.name, vals.join(", "));
                            }
                        }
                    } else {
                        print!("State[N0..N3] = ");
                        for (i, v) in decoded.iter().take(4).enumerate() {
                            if i > 0 {
                                print!(", ");
                            }
                            print!("{:.4}", v);
                        }
                    }
                    println!();
                }
            }
        }
        return;
    }

    // --- Standard Single Batch Injection ---
    inject_inputs(&mut vm, &args.inputs);

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

    let has_hooks = !probes.is_empty() || !emits.is_empty();
    let mut prev_emit_gates: Vec<bool> = vec![false; emits.len()];
    let mut emitted_messages: Vec<String> = Vec::new();

    let (cycles_run, converged) = if has_hooks {
        let mut cycles = 0;
        let mut conv = false;
        let mut prev_probe_values: Vec<Option<f64>> = vec![None; probes.len()];
        let mut probe_transitions: Vec<String> = Vec::new();

        for c in 0..args.cycles {
            let changed = if has_plasticity {
                vm.step_plastic()
            } else {
                vm.step_stable()
            };
            cycles += 1;

            // 1. Process Probes
            for (p_idx, p) in probes.iter().enumerate() {
                let active = match p.gate_idx {
                    Some(g) => vm.state.get(g, 0).to_f64() >= 0.5,
                    None => true,
                };
                if active {
                    let mut val = vm.state.get(p.neuron_idx, 0).to_f64();
                    if let Some(ref key) = obf_key {
                        if let Some(clean_idx) = key.mapping.iter().position(|&m| m == p.neuron_idx)
                        {
                            if key.inversions.get(clean_idx).copied().unwrap_or(false) {
                                val = 1.0 - val;
                            }
                        }
                    }

                    let prev = prev_probe_values[p_idx];
                    let changed_significantly = match prev {
                        None => true,
                        Some(pv) => (pv - val).abs() >= 1e-4,
                    };

                    if changed_significantly {
                        let name = p.label.as_deref().unwrap_or("unnamed");
                        if let Some(pv) = prev {
                            probe_transitions.push(format!(
                                "  cycle {:<3} : probe \"{}\" shifted {:.4} -> {:.4}",
                                c + 1,
                                name,
                                pv,
                                val
                            ));
                        } else if args.verbose {
                            probe_transitions.push(format!(
                                "  cycle {:<3} : probe \"{}\" initialized to {:.4}",
                                c + 1,
                                name,
                                val
                            ));
                        }
                        prev_probe_values[p_idx] = Some(val);
                    }
                }
            }

            // 2. Process Dynamic Emit statements (Rising edge trigger)
            for (e_idx, e) in emits.iter().enumerate() {
                if e.on_stable {
                    continue;
                }
                let active = match e.gate_idx {
                    Some(g) => vm.state.get(g, 0).to_f64() >= 0.5,
                    None => true,
                };
                let was_active = prev_emit_gates[e_idx];
                prev_emit_gates[e_idx] = active;

                if active && !was_active {
                    let mut rendered = e.template.clone();
                    for &arg_neuron in &e.arg_indices {
                        let mut val = vm.state.get(arg_neuron, 0).to_f64();
                        if let Some(ref key) = obf_key {
                            if let Some(clean_idx) =
                                key.mapping.iter().position(|&m| m == arg_neuron)
                            {
                                if key.inversions.get(clean_idx).copied().unwrap_or(false) {
                                    val = 1.0 - val;
                                }
                            }
                        }
                        if rendered.contains("{}") {
                            rendered = rendered.replacen("{}", &format!("{:.4}", val), 1);
                        } else if rendered.contains("{:.2}") {
                            rendered = rendered.replacen("{:.2}", &format!("{:.2}", val), 1);
                        } else if rendered.contains("{:.3}") {
                            rendered = rendered.replacen("{:.3}", &format!("{:.3}", val), 1);
                        } else if rendered.contains("{:.4}") {
                            rendered = rendered.replacen("{:.4}", &format!("{:.4}", val), 1);
                        } else {
                            rendered.push_str(&format!(" [{:.4}]", val));
                        }
                    }
                    let log_line = format!("  cycle {:<3} : {}", c + 1, rendered);
                    emitted_messages.push(log_line);
                }
            }

            if args.until_stable && !changed {
                conv = true;
                break;
            }
        }

        // Fire on_stable emit hooks if converged
        if conv {
            for e in &emits {
                if e.on_stable {
                    let mut rendered = e.template.clone();
                    for &arg_neuron in &e.arg_indices {
                        let mut val = vm.state.get(arg_neuron, 0).to_f64();
                        if let Some(ref key) = obf_key {
                            if let Some(clean_idx) =
                                key.mapping.iter().position(|&m| m == arg_neuron)
                            {
                                if key.inversions.get(clean_idx).copied().unwrap_or(false) {
                                    val = 1.0 - val;
                                }
                            }
                        }
                        if rendered.contains("{}") {
                            rendered = rendered.replacen("{}", &format!("{:.4}", val), 1);
                        } else {
                            rendered.push_str(&format!(" [{:.4}]", val));
                        }
                    }
                    emitted_messages.push(format!("  settled   : {}", rendered));
                }
            }
        }

        if !emitted_messages.is_empty() {
            println!("◆ Emitted Dynamic Signals");
            for line in &emitted_messages {
                println!("{}", line);
            }
            println!();
        }

        if !probe_transitions.is_empty() && (args.verbose || !conv) {
            println!("◆ Probe Dynamic Transitions");
            for line in &probe_transitions {
                println!("{}", line);
            }
            println!();
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

    let is_obfuscated = obf_key.is_some();
    let num_decoy_nodes = obf_key
        .as_ref()
        .map(|k| k.total_size.saturating_sub(k.clean_size))
        .unwrap_or(0);

    match args.format {
        OutputFormat::Pretty => {
            // Executive runtime banner & status indicators
            println!("● Stella Neural VM");
            println!("  target     : {}", target_file.display());
            println!("  backend    : {}", backend_name);
            if is_obfuscated {
                println!(
                    "  security   : polymorphic basis obfuscated ({} nodes, {} decoys masked)",
                    state_size, num_decoy_nodes
                );
            } else {
                println!(
                    "  security   : plain continuous topology ({} nodes)",
                    state_size
                );
            }
            if !vm.plasticity_rules.is_empty() {
                println!(
                    "  plasticity : active ({} synaptic adaptation rules)",
                    vm.plasticity_rules.len()
                );
            }
            if converged {
                println!(
                    "  dynamics   : converged to attractor equilibrium in {} cycles ({:.2} ms)",
                    cycles_run,
                    elapsed.as_secs_f64() * 1000.0
                );
            } else {
                println!(
                    "  dynamics   : executed {} cycles ({:.2} ms, limit reached)",
                    cycles_run,
                    elapsed.as_secs_f64() * 1000.0
                );
            }
            println!();

            // Terminal Outputs
            let out_terminals: Vec<_> = symbols_meta
                .iter()
                .filter(|s| s.kind == "TerminalOut")
                .collect();

            if !out_terminals.is_empty() {
                println!("◆ Terminal Outputs");
                for sym in &out_terminals {
                    if sym.width == 1 {
                        let val = decoded_state.get(sym.index).copied().unwrap_or(0.0);
                        let status_flag = if val >= 0.999 {
                            " [ACTIVE / 1.0]"
                        } else if val <= 0.001 {
                            " [INACTIVE / 0.0]"
                        } else {
                            ""
                        };
                        println!("  • {:<20} = {:.6}{}", sym.name, val, status_flag);
                    } else {
                        let vals: Vec<String> = (0..sym.width)
                            .map(|off| {
                                let v = decoded_state.get(sym.index + off).copied().unwrap_or(0.0);
                                format!("{:.4}", v)
                            })
                            .collect();
                        println!("  • {:<20} = [{}]", sym.name, vals.join(", "));
                    }
                }
                println!();
            }

            // Attractor Basin / Phase contexts
            let basin_nodes: Vec<_> = symbols_meta
                .iter()
                .filter(|s| s.kind == "BasinContext")
                .collect();

            if !basin_nodes.is_empty() {
                println!("◆ Attractor Basins & Phase States");
                for sym in &basin_nodes {
                    let val = decoded_state.get(sym.index).copied().unwrap_or(0.0);
                    let bullet = if val >= 0.5 {
                        "▲ [LOCKED]   "
                    } else {
                        "▽ [SUPPRESSED]"
                    };
                    println!("  {} {:<28} = {:.4}", bullet, sym.name, val);
                }
                println!();
            }

            // If verbose or no terminal outputs were declared, print grouped state hierarchy
            if args.verbose || out_terminals.is_empty() {
                println!("◆ Continuous State Inspection");

                let mut mapped_indices = std::collections::BTreeSet::new();

                // Group 1: Inputs
                let in_terminals: Vec<_> = symbols_meta
                    .iter()
                    .filter(|s| s.kind == "TerminalIn")
                    .collect();
                if !in_terminals.is_empty() {
                    println!("  Input Terminals:");
                    let mut in_entries = Vec::new();
                    for sym in &in_terminals {
                        for off in 0..sym.width {
                            let idx = sym.index + off;
                            mapped_indices.insert(idx);
                            let val = decoded_state.get(idx).copied().unwrap_or(0.0);
                            let name = if sym.width == 1 {
                                sym.name.clone()
                            } else {
                                format!("{}[{}]", sym.name, off)
                            };
                            in_entries.push((idx, name, val));
                        }
                    }
                    in_entries.sort_by_key(|e| e.0);
                    for (idx, name, val) in in_entries {
                        println!("    N{:<2}  {:<24} = {:.6}", idx, name, val);
                    }
                }

                // Group 2: Latent Attractor & Filter Nodes
                let latent_nodes: Vec<_> = symbols_meta
                    .iter()
                    .filter(|s| {
                        s.kind != "TerminalIn"
                            && s.kind != "TerminalOut"
                            && s.kind != "BasinContext"
                    })
                    .collect();
                if !latent_nodes.is_empty() {
                    println!("  Latent Neural Nodes:");
                    let mut latent_entries = Vec::new();
                    for sym in &latent_nodes {
                        for off in 0..sym.width {
                            let idx = sym.index + off;
                            mapped_indices.insert(idx);
                            let val = decoded_state.get(idx).copied().unwrap_or(0.0);
                            let name = if sym.width == 1 {
                                sym.name.clone()
                            } else {
                                format!("{}[{}]", sym.name, off)
                            };
                            latent_entries.push((idx, name, sym.kind.clone(), val));
                        }
                    }
                    latent_entries.sort_by_key(|e| e.0);
                    for (idx, name, kind, val) in latent_entries {
                        println!(
                            "    N{:<2}  {:<24} {:<15} = {:.6}",
                            idx,
                            name,
                            format!("({})", kind),
                            val
                        );
                    }
                }

                // Mark output terminals & basins as mapped
                for sym in &out_terminals {
                    for off in 0..sym.width {
                        mapped_indices.insert(sym.index + off);
                    }
                }
                for sym in &basin_nodes {
                    for off in 0..sym.width {
                        mapped_indices.insert(sym.index + off);
                    }
                }

                // Group 3: Auxiliary / Scratch wires
                let scratch_indices: Vec<usize> = (0..decoded_state.len())
                    .filter(|idx| !mapped_indices.contains(idx))
                    .collect();

                if !scratch_indices.is_empty() {
                    if args.all {
                        println!("  Auxiliary Scratch Wires:");
                        for &idx in &scratch_indices {
                            println!(
                                "    N{:<2}  {:<24} {:<15} = {:.6}",
                                idx, "aux_wire", "(Scratch)", decoded_state[idx]
                            );
                        }
                    } else {
                        let first = scratch_indices[0];
                        let last = scratch_indices[scratch_indices.len() - 1];
                        println!("  • {} auxiliary scratch wires (N{}..N{}) hidden (show with -a / --all)", scratch_indices.len(), first, last);
                    }
                }
                println!();
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

    // If a return statement was declared, exit with the quantized integer value (0 for 0.0, 1 for >= 0.5)
    if let Some(ret_idx) = return_neuron {
        let ret_val = decoded_state.get(ret_idx).copied().unwrap_or(0.0);
        let code = if ret_val >= 0.5 { 1 } else { 0 };
        std::process::exit(code);
    }
}
