//! The pure-Rust core of LoFI's chart kernel.
//!
//! The crate has no Python dependency and contains no unsafe code. The
//! `lofi-native` crate binds it to Python as the `lofi._native` extension
//! module.

pub mod build;
pub mod chart;
pub mod drawing;
pub mod program;
pub mod reachability;
pub mod rule;
pub mod semiring;
pub mod term;
