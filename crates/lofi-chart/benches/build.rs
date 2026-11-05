//! Instruction counts for reading the build information.
//!
//! The benchmarks live in a private module: gungraun's macros declare public
//! items, which would otherwise be undocumented items of the crate's interface.

use gungraun::main;

mod groups {
    use std::hint::black_box;

    use gungraun::{library_benchmark, library_benchmark_group};
    use lofi_chart::build::BuildInfo;

    #[library_benchmark]
    fn current() -> &'static str {
        black_box(BuildInfo::current()).version()
    }

    library_benchmark_group!(name = build; benchmarks = current);
}

use groups::build;

main!(library_benchmark_groups = build);
