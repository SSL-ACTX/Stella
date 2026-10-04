// crates/stella_core/src/lib.rs
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod layer;
pub mod math;
pub mod vm;

// Re-export high-frequency types at the crate root for ergonomics
pub use layer::dense::{Dense, DenseQ16};
pub use math::fix::{Q16, Q32};
pub use math::matrix::Matrix;
pub use vm::engine::{PlasticityKind, PlasticityRule, Vm};
