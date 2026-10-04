// crates/stella_frontend/src/importer.rs

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use bumpalo::Bump;

use crate::ast::*;
use crate::parser::parse_program;

/// Trait abstracting file reading for `std` vs `no_std` environments.
pub trait ModuleResolver {
    fn resolve(&self, path: &str) -> Result<String, String>;
}

/// In-memory module resolver using a map of paths to source strings.
pub struct MemoryResolver {
    pub files: BTreeMap<String, String>,
}

impl MemoryResolver {
    pub fn new() -> Self {
        Self {
            files: BTreeMap::new(),
        }
    }

    pub fn add_file(&mut self, path: impl Into<String>, source: impl Into<String>) {
        self.files.insert(path.into(), source.into());
    }
}

impl ModuleResolver for MemoryResolver {
    fn resolve(&self, path: &str) -> Result<String, String> {
        self.files
            .get(path)
            .cloned()
            .ok_or_else(|| format!("Module '{}' not found in memory resolver", path))
    }
}

#[cfg(feature = "std")]
pub struct FsResolver;

#[cfg(feature = "std")]
impl ModuleResolver for FsResolver {
    fn resolve(&self, path: &str) -> Result<String, String> {
        std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read module '{}': {}", path, e))
    }
}

/// Recursively resolves all `<- "path"` / `use "path"` import directives in a Program,
/// parsing dependencies and prefixing/merging imported declarations into the arena AST.
pub fn resolve_imports<'a, R: ModuleResolver>(
    program: &Program<'a>,
    resolver: &R,
    bump: &'a Bump,
) -> Result<Program<'a>, String> {
    let mut visited = Vec::new();
    let mut all_decls = Vec::new();
    let mut all_flows = Vec::new();
    let mut all_basins = Vec::new();

    for f in program.flows {
        all_flows.push(*f);
    }
    for b in program.basins {
        all_basins.push(*b);
    }

    for decl in program.declarations {
        match *decl {
            Decl::Import { path, target } => {
                expand_import(path, target, resolver, &mut visited, &mut all_decls, bump)?;
            }
            other => {
                all_decls.push(other);
            }
        }
    }

    Ok(Program {
        declarations: bump.alloc_slice_copy(&all_decls),
        flows: bump.alloc_slice_copy(&all_flows),
        basins: bump.alloc_slice_copy(&all_basins),
    })
}

fn expand_import<'a, R: ModuleResolver>(
    path: &'a str,
    target: ImportTarget<'a>,
    resolver: &R,
    visited: &mut Vec<String>,
    out_decls: &mut Vec<Decl<'a>>,
    bump: &'a Bump,
) -> Result<(), String> {
    let path_str = path.into();
    if visited.contains(&path_str) {
        // Already imported, prevent circular dependency
        return Ok(());
    }
    visited.push(path_str);

    let src = resolver.resolve(path)?;
    // Allocate source in bump arena so parsed AST borrows &'a str safely
    let src_str: &'a str = bump.alloc_str(&src);
    let imported_ast = parse_program(src_str, bump)?;

    // Recursively resolve any nested imports inside the imported file
    for d in imported_ast.declarations {
        if let Decl::Import {
            path: sub_path,
            target: sub_target,
        } = *d
        {
            expand_import(sub_path, sub_target, resolver, visited, out_decls, bump)?;
        }
    }

    // Now export declarations according to target (Alias, Selective, Open)
    for d in imported_ast.declarations {
        if matches!(d, Decl::Import { .. }) {
            continue;
        }

        match target {
            ImportTarget::Alias(alias) => {
                // Prefix exported circuit and type names with `alias::`
                match *d {
                    Decl::Circuit {
                        name,
                        template_params,
                        params,
                        return_param,
                        declarations,
                        flows,
                    } => {
                        let scoped_name = bump.alloc_str(&format!("{}::{}", alias, name));
                        out_decls.push(Decl::Circuit {
                            name: scoped_name,
                            template_params,
                            params,
                            return_param,
                            declarations,
                            flows,
                        });
                    }
                    Decl::TypeAlias { name, shape } => {
                        let scoped_name = bump.alloc_str(&format!("{}::{}", alias, name));
                        out_decls.push(Decl::TypeAlias {
                            name: scoped_name,
                            shape,
                        });
                    }
                    Decl::Const(name, val) => {
                        let scoped_name = bump.alloc_str(&format!("{}::{}", alias, name));
                        out_decls.push(Decl::Const(scoped_name, val));
                    }
                    Decl::ConstMatrix {
                        name,
                        rows,
                        cols,
                        data,
                    } => {
                        let scoped_name = bump.alloc_str(&format!("{}::{}", alias, name));
                        out_decls.push(Decl::ConstMatrix {
                            name: scoped_name,
                            rows,
                            cols,
                            data,
                        });
                    }
                    other => {
                        out_decls.push(other);
                    }
                }
            }
            ImportTarget::Selective(names) => {
                let decl_name = match *d {
                    Decl::Circuit { name, .. } => Some(name),
                    Decl::TypeAlias { name, .. } => Some(name),
                    Decl::Const(name, _) => Some(name),
                    Decl::ConstMatrix { name, .. } => Some(name),
                    _ => None,
                };
                if let Some(n) = decl_name {
                    if names.contains(&n) {
                        out_decls.push(*d);
                    }
                }
            }
            ImportTarget::Open => {
                out_decls.push(*d);
            }
        }
    }

    Ok(())
}
