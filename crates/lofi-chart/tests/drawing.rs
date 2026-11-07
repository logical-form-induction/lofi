//! Behavioral tests for scoped rule programs and the drawing program.

use std::sync::Arc;

use lofi_chart::chart::{
    CellExecution, InsideTable, ItemId, MissBatch, NormalDecision, Span, Step, UnaryClosure,
};
use lofi_chart::drawing::{
    DrawingHost, DrawingProgram, DrawingResolver, DrawingRule, EdgeScoring, ResidualRules,
};
use lofi_chart::program::{
    DrawError, HostError, RuleBoundary, Scope, ScopedRuleProgram, TemplateCache,
};
use lofi_chart::rule::{
    CompositionRestriction, CompositionRule, FunctorSpec, ProgramRule, RepresentativeKernel,
    RuleKernel, RuleProgram,
};
use lofi_chart::semiring::{Count, Counting};
use lofi_chart::term::{OperatorSpec, PayloadKind, SignatureSpec, Term, TermError, TermStore};

#[derive(Debug)]
struct Signature(TermStore);

impl Scope for Signature {
    fn store_for(&self, _term: &Term) -> Option<TermStore> {
        Some(self.0.clone())
    }
}

fn operator(name: &str, arguments: Vec<u16>, erased: u32) -> OperatorSpec {
    OperatorSpec {
        name: name.to_owned(),
        sort: 0,
        shifts: vec![vec![]; arguments.len()],
        arguments,
        payload: PayloadKind::None,
        erased,
        binding: vec![],
    }
}

fn store() -> Result<TermStore, TermError> {
    TermStore::new(
        SignatureSpec {
            name: "drawing".to_owned(),
            sorts: vec!["Cat".to_owned()],
            operators: vec![
                operator("S", vec![], 0),
                operator("NP", vec![], 1),
                operator("fwd", vec![0, 0], 2),
                operator("bwd", vec![0, 0], 3),
            ],
            vocabularies: vec![None],
            constant_cores: vec![None],
            constants: vec![None],
            bound: vec![None],
            free: vec![None],
            holes: vec![None],
            allows_vacuous_binding: true,
        },
        64,
    )
}

fn application() -> ProgramRule {
    ProgramRule {
        kernel: RuleKernel::Composition(CompositionRule {
            functors: FunctorSpec {
                operators: vec!["fwd".to_owned(), "bwd".to_owned()],
                forward: "fwd".to_owned(),
                backward: "bwd".to_owned(),
                result_path: vec![0],
                argument_path: vec![1],
            },
            order: Some(0),
            slash: "fwd".to_owned(),
            crossed: false,
            restriction: CompositionRestriction::Unrestricted,
        }),
        matching: 0,
        representatives: RepresentativeKernel::Identity,
        constants: Vec::new(),
    }
}

struct Fixture {
    scope: Arc<Signature>,
    program: DrawingProgram<Signature>,
    sentence: Vec<Term>,
}

fn fixture() -> Result<Fixture, Box<dyn std::error::Error>> {
    let store = store()?;
    let s = store.intern(0, vec![], None)?;
    let np = store.intern(1, vec![], None)?;
    let verb = store.intern(2, vec![s, np.clone()], None)?;
    let scope = Arc::new(Signature(store.clone()));
    let rules = ScopedRuleProgram::new(
        RuleProgram::new(vec![application()]),
        vec![RuleBoundary {
            representative_scope: None,
            store: Some(store),
            scope: Some(Arc::clone(&scope)),
        }],
        TemplateCache::default(),
    )
    .map_err(|error| error.to_string())?;
    let program = DrawingProgram::new(
        vec![DrawingRule {
            arity: 2,
            order: 0,
            complete: true,
            decisions: vec![(vec![0, 0], NormalDecision::Admit(0))],
        }],
        1,
        Arc::new(rules),
        false,
        Vec::new(),
        (None, None),
    )
    .map_err(|error| error.to_string())?;
    Ok(Fixture {
        scope,
        program,
        sentence: vec![verb, np],
    })
}

#[derive(Default)]
struct UnitHost {
    widths: usize,
}

impl ResidualRules for UnitHost {
    fn resolve(&mut self, batch: MissBatch) -> Result<Vec<Step>, HostError> {
        Ok(batch.results().to_vec())
    }
}

impl DrawingHost<Count> for UnitHost {
    fn admit_rows(
        &mut self,
        _rule: u16,
        _tags: &[u8],
        input_pairs: &[[u32; 2]],
        _output: u32,
    ) -> Result<Vec<bool>, HostError> {
        Ok(vec![true; input_pairs.len()])
    }

    fn weight(
        &mut self,
        _rule: u16,
        _inputs: &[u32],
        _output: u32,
        _span: Span,
        _split: Option<usize>,
    ) -> Result<Count, HostError> {
        Ok(Count::Small(1))
    }

    fn finish_width(&mut self) -> Result<(), HostError> {
        self.widths += 1;
        Ok(())
    }
}

#[test]
fn identifying_a_category_twice_returns_one_dense_identifier()
-> Result<(), Box<dyn std::error::Error>> {
    let Fixture {
        scope,
        program,
        sentence,
    } = fixture()?;
    let first = program
        .identify(sentence[0].clone(), Arc::clone(&scope))
        .map_err(|error| error.to_string())?;
    let again = program
        .identify(sentence[0].clone(), Arc::clone(&scope))
        .map_err(|error| error.to_string())?;
    assert_eq!(first, again);
    assert_eq!(
        program
            .lookup(&sentence[0])
            .map_err(|error| error.to_string())?,
        first
    );
    let identifier = first.ok_or("an unbounded program admits every category")?;
    let (term, held) = program
        .term(identifier)
        .map_err(|error| error.to_string())?;
    assert!(term.is_identical(&sentence[0]));
    assert!(Arc::ptr_eq(&held, &scope));
    Ok(())
}

#[test]
fn drawing_a_premise_pair_applies_the_complete_rule_and_memoizes_it()
-> Result<(), Box<dyn std::error::Error>> {
    let Fixture {
        scope,
        program,
        sentence,
    } = fixture()?;
    let mut inputs = Vec::new();
    for term in &sentence {
        inputs.push(
            program
                .identify(term.clone(), Arc::clone(&scope))
                .map_err(|error| error.to_string())?
                .ok_or("an unbounded program admits every category")?,
        );
    }
    let steps = program
        .steps(
            inputs.clone(),
            vec![0, 0],
            (0, 2),
            &[0],
            &mut UnitHost::default(),
        )
        .map_err(|error| error.to_string())?;
    assert_eq!(steps.len(), 1);
    let (conclusion, _) = program
        .term(steps[0].output)
        .map_err(|error| error.to_string())?;
    assert_eq!(conclusion.operator_name(), "S");
    assert_eq!(
        program
            .cached_outputs(0, &inputs)
            .map_err(|error| error.to_string())?,
        Some(vec![steps[0].output])
    );
    Ok(())
}

#[test]
fn an_inside_sweep_through_the_resolver_counts_one_derivation()
-> Result<(), Box<dyn std::error::Error>> {
    let Fixture {
        scope,
        program,
        sentence,
    } = fixture()?;
    let semiring = Counting;
    let mut table = InsideTable::new(2);
    for (position, term) in sentence.iter().enumerate() {
        let category = program
            .identify(term.clone(), Arc::clone(&scope))
            .map_err(|error| error.to_string())?
            .ok_or("an unbounded program admits every category")?;
        table
            .add(
                (position, position + 1),
                ItemId { category, tag: 0 },
                Count::Small(1),
                &semiring,
            )
            .map_err(|error| format!("{error:?}"))?;
    }
    let mut resolver = DrawingResolver::new(
        &program,
        UnitHost::default(),
        EdgeScoring::UnitEdges,
        &semiring,
    );
    let closure: UnaryClosure = table
        .sweep(&[2], 0, CellExecution::Attached, &semiring, &mut resolver)
        .map_err(|error| format!("{error:?}"))?;
    assert!(!closure.truncated);
    let derived = table
        .tagged_at((0, 2))
        .map_err(|error| format!("{error:?}"))?;
    assert_eq!(derived.len(), 1);
    assert_eq!(derived[0].1, Count::Small(1));
    Ok(())
}

#[test]
fn premises_from_another_scope_are_refused() -> Result<(), Box<dyn std::error::Error>> {
    let Fixture {
        scope, sentence, ..
    } = fixture()?;
    let stranger = Arc::new(Signature(store()?));
    let rules = ScopedRuleProgram::new(
        RuleProgram::new(vec![application()]),
        vec![RuleBoundary {
            representative_scope: None,
            store: None,
            scope: None,
        }],
        TemplateCache::default(),
    )
    .map_err(|error| error.to_string())?;
    let refused = rules.apply(
        &[0],
        &[
            (sentence[0].clone(), scope),
            (sentence[1].clone(), stranger),
        ],
    );
    assert!(matches!(refused, Err(DrawError::Invalid(_))));
    Ok(())
}

#[test]
fn a_program_refuses_a_repeated_child_tag_decision() -> Result<(), Box<dyn std::error::Error>> {
    let rules = ScopedRuleProgram::<Signature>::new(
        RuleProgram::new(vec![application()]),
        vec![RuleBoundary {
            representative_scope: None,
            store: None,
            scope: None,
        }],
        TemplateCache::default(),
    )
    .map_err(|error| error.to_string())?;
    let refused = DrawingProgram::new(
        vec![DrawingRule {
            arity: 2,
            order: 0,
            complete: true,
            decisions: vec![
                (vec![0, 0], NormalDecision::Admit(0)),
                (vec![0, 0], NormalDecision::Refuse),
            ],
        }],
        1,
        Arc::new(rules),
        false,
        Vec::new(),
        (None, None),
    );
    assert!(matches!(refused, Err(DrawError::Invalid(_))));
    Ok(())
}
