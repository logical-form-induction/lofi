//! Expose the Cargo profile of the build to the crate as the `lofi_profile` cfg.

use std::env;

fn main() -> Result<(), env::VarError> {
    let profile = env::var("PROFILE")?;
    println!("cargo::rerun-if-changed=build.rs");
    println!(r#"cargo::rustc-check-cfg=cfg(lofi_profile, values("debug", "release"))"#);
    println!(r#"cargo::rustc-cfg=lofi_profile="{profile}""#);
    Ok(())
}
