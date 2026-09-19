use clap::Args;
use std::fs;
use std::path::PathBuf;

use stella_compiler::obfuscator::transform_basis;
use stella_core::layer::Dense;

#[derive(Args, Debug)]
pub struct DisasmArgs {
    /// Original plain source script (.stl)
    #[arg(value_name = "SOURCE")]
    pub source: PathBuf,

    /// Obfuscated target binary (.stella)
    #[arg(value_name = "TARGET")]
    pub target: PathBuf,
}

pub fn handle_disasm(args: DisasmArgs) {
    println!("--- Stella Symbolic Permutation Disassembler ---");
    let target_buffer = fs::read(&args.target).unwrap_or_else(|err| {
        eprintln!("Failed to read target '{:?}': {}", args.target, err);
        std::process::exit(1);
    });
    let target_core: Dense = postcard::from_bytes(&target_buffer).unwrap_or_else(|err| {
        eprintln!("Deserialization failed for target: {}", err);
        std::process::exit(1);
    });

    let source_str = fs::read_to_string(&args.source).unwrap_or_else(|err| {
        eprintln!("Failed to read source '{:?}': {}", args.source, err);
        std::process::exit(1);
    });
    let clean_artifact =
        stella_compiler::compile(&source_str, false, None, None).unwrap_or_else(|err| {
            eprintln!("Parse error: {}", err);
            std::process::exit(1);
        });

    let size = clean_artifact.core.weights.rows;
    if target_core.weights.rows != size {
        eprintln!(
            "Size mismatch: Clean core is {}x{}, target is {}x{}",
            size, size, target_core.weights.rows, target_core.weights.cols
        );
        std::process::exit(1);
    }

    println!("Searching permutation space (size: {}!)...", size);
    let mut indices: Vec<usize> = (0..size).collect();
    let mut perms = Vec::new();
    generate_permutations(size, &mut indices, &mut perms);

    let mut found = false;
    for p_vec in perms {
        let (p, p_inv) = stella_compiler::obfuscator::build_permutation_matrices(&p_vec);
        let test_core = transform_basis(&clean_artifact.core, &p, &p_inv);
        if test_core.weights == target_core.weights && test_core.biases == target_core.biases {
            println!("\n[!] SUCCESS: Obfuscation permutation key recovered!");
            println!("Mapping: {:?}", p_vec);
            found = true;
            break;
        }
    }

    if !found {
        println!("[-] Permutation key not found in direct permutations (likely padded or non-isomorphic).");
    }
}

fn generate_permutations(n: usize, a: &mut Vec<usize>, res: &mut Vec<Vec<usize>>) {
    if n == 1 {
        res.push(a.clone());
        return;
    }
    for i in 0..n {
        generate_permutations(n - 1, a, res);
        if n % 2 == 0 {
            a.swap(i, n - 1);
        } else {
            a.swap(0, n - 1);
        }
    }
}
