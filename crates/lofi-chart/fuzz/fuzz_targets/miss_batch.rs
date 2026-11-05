#![no_main]

use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

use lofi_chart::chart::{CategoryBounds, ChartError, ChartProgram, MissBatch, RuleSpec};

fuzz_target!(|data: &[u8]| {
    let [rules, tag_count, candidate_count, tail @ ..] = data else {
        return;
    };
    let candidate_count = usize::from(*candidate_count).min(tail.len());
    let (candidate_bytes, pushes) = tail.split_at(candidate_count);
    let candidates = candidate_bytes
        .iter()
        .map(|candidate| u16::from(*candidate))
        .collect::<Vec<_>>();
    let rules = usize::from(*rules);
    let tag_count = usize::from(*tag_count);
    let program = ChartProgram::new(
        vec![
            RuleSpec {
                arity: 2,
                order: 0,
                complete: false,
            };
            rules
        ],
        tag_count,
        Vec::new(),
        Vec::new(),
        HashMap::new(),
        CategoryBounds::default(),
    );
    let mut batch = MissBatch::new(&program, vec![0, 1], vec![0, 0], candidates.clone(), (0, 2));

    for row in pushes.chunks_exact(4) {
        let rule = u16::from_le_bytes([row[0], row[1]]);
        let output = u32::from(row[2]);
        let tag = row[3];
        let before = batch.results().len();
        let result = batch.push(rule, output, tag);
        let expected = if usize::from(rule) >= rules {
            Err(ChartError::UnknownRule { rule, rules })
        } else if !candidates.contains(&rule) {
            Err(ChartError::RuleNotCandidate { rule })
        } else if usize::from(tag) >= tag_count {
            Err(ChartError::UnknownTag {
                tag,
                tags: tag_count,
            })
        } else {
            Ok(())
        };

        assert_eq!(result, expected);
        assert_eq!(
            batch.results().len(),
            before + usize::from(expected.is_ok())
        );
    }
});
