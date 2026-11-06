//! Instruction counts for chart-kernel operations.
//!
//! The memo-hit benchmark records a measurement without a hard limit. A measured
//! baseline establishes the limit separately.

use gungraun::main;

mod groups {
    use std::collections::{HashMap, HashSet};
    use std::hint::black_box;

    use gungraun::{library_benchmark, library_benchmark_group};
    use lofi_chart::chart::{
        CategoryBounds, CategoryDimensions, CategoryMeta, ChartError, ChartProgram, NormalDecision,
        RuleSpec, Step, StepPlan,
    };

    fn memo_fixture() -> &'static ChartProgram {
        let mut program = ChartProgram::new(
            vec![RuleSpec {
                arity: 2,
                order: 0,
                complete: true,
            }],
            2,
            vec![0],
            Vec::new(),
            HashMap::from([((0, vec![1, 0]), NormalDecision::Admit(1))]),
            CategoryBounds::default(),
        );
        for category in 0..2 {
            let registered = program.register_category(
                category,
                CategoryMeta {
                    unary: Vec::new(),
                    viable: HashSet::new(),
                    joins: HashMap::new(),
                    factors: HashMap::new(),
                },
                CategoryDimensions { arity: 0, size: 1 },
            );
            let Ok(true) = registered else {
                unreachable!("the benchmark fixture registers dense admitted categories")
            };
        }
        program.remember(
            vec![0, 1],
            vec![1, 0],
            vec![0],
            vec![Step {
                rule: 0,
                output: 1,
                tag: 1,
            }],
        );
        Box::leak(Box::new(program))
    }

    #[library_benchmark(setup = memo_fixture)]
    fn memo_hit(program: &ChartProgram) -> Result<StepPlan, ChartError> {
        black_box(program).step_plan(
            black_box(&[0, 1]),
            black_box(&[1, 0]),
            black_box(&[0]),
            black_box(false),
        )
    }

    library_benchmark_group!(name = chart; benchmarks = memo_hit);
}

use groups::chart;

main!(library_benchmark_groups = chart);
