use clap::Args;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Args, Debug)]
pub struct SwarmArgs {
    /// Target binary (.stella) or script (.stl)
    #[arg(value_name = "FILE")]
    pub file: Option<PathBuf>,

    /// Swarm batch size (number of simultaneous parallel neural agents)
    #[arg(short = 'b', long, default_value_t = 256)]
    pub batch_size: usize,

    /// Execution clock cycles for the swarm
    #[arg(short = 'c', long, default_value_t = 100)]
    pub cycles: usize,

    /// Accelerate swarm using physical hardware GPU (ARM Mali / Adreno via Vulkan)
    #[arg(long, alias = "hardware-gpu")]
    pub gpu: bool,
}

pub fn handle_swarm(args: SwarmArgs) {
    let target_file = args.file.unwrap_or_else(|| {
        crate::discover_run_target().unwrap_or_else(|err| {
            eprintln!("Error: {}", err);
            std::process::exit(1);
        })
    });

    println!("=== Stella GPU / Swarm Acceleration Backend ===");
    println!("Target Matrix : {:?}", target_file);
    println!("Swarm Batch   : {} simultaneous agents", args.batch_size);
    println!("Cycles        : {}", args.cycles);

    let (core, state_size) = if target_file.extension().and_then(|s| s.to_str()) == Some("stl") {
        let source_str = fs::read_to_string(&target_file).unwrap_or_else(|err| {
            eprintln!("Failed to read '{:?}': {}", target_file, err);
            std::process::exit(1);
        });
        let artifact =
            stella_compiler::compile(&source_str, false, None, None).unwrap_or_else(|err| {
                eprintln!("Compilation failed: {}", err);
                std::process::exit(1);
            });
        (artifact.core, artifact.state_size)
    } else {
        let buffer = fs::read(&target_file).unwrap_or_else(|err| {
            eprintln!("Failed to read '{:?}': {}", target_file, err);
            std::process::exit(1);
        });
        let binary: stella_compiler::StellaBinary =
            postcard::from_bytes(&buffer).unwrap_or_else(|err| {
                eprintln!("Deserialization failed: {}", err);
                std::process::exit(1);
            });
        let n = binary.core.weights.rows;
        (binary.core, n)
    };

    let dims = stella_gpu::ComputeDimensions::new(state_size, args.batch_size, args.cycles);

    if args.gpu {
        println!("[*] Initializing compute backend via GpuDispatcher...");
        let backend = stella_gpu::auto_detect_backend();
        println!("    Backend Selected: {}", backend.name());

        match backend.execute(dims) {
            Ok(res) => {
                println!("\n[+] {} Compute Finished Successfully!", res.device_name);
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
                    "    Swarm Agent Velocity   : {:.2} Thousand agent-cycles/sec",
                    res.agent_cycles_per_sec
                );
                println!(
                    "    Attractor Convergence  : S[0, 0] = {:.4}",
                    res.sample_output
                );
                return;
            }
            Err(err) => {
                eprintln!("[-] GPU Execution Error: {}", err);
                eprintln!("[*] Falling back to multi-threaded CPU SIMD engine...");
            }
        }
    }

    let mut swarm = stella_gpu::SwarmGpuBackend::new(&core, args.batch_size);

    let start = Instant::now();
    swarm.run_swarm(args.cycles);
    let elapsed = start.elapsed();

    let total_macs = state_size * state_size * args.batch_size * args.cycles;
    let giga_macs_per_sec = (total_macs as f64) / elapsed.as_secs_f64() / 1_000_000_000.0;
    let agent_cycles_per_sec = (args.batch_size * args.cycles) as f64 / elapsed.as_secs_f64();

    println!("\n[+] Swarm Simulation Complete!");
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
        "  Agent Cycle Velocity: {:.2} Thousand agent-cycles/sec",
        agent_cycles_per_sec / 1000.0
    );
}
