//! Behavioral tests for the inverse directions of the binary-functor rule kernels.

use lofi_chart::rule::{
    CompositionRestriction, CompositionRule, CompositionWitness, FunctorSpec, PremiseRestriction,
    RaisingRule, RepresentativeKernel, RepresentativeTable, RestrictedRule, RuleError, RuleKernel,
    SubstitutionRule,
};
use lofi_chart::term::{OperatorSpec, PayloadKind, SignatureSpec, Term, TermStore};

const NAMES: [&str; 5] = ["x", "y", "z0", "z1", "z2"];
const FORWARD: u32 = 5;
const BACKWARD: u32 = 6;

fn constant(name: &str, erased: u32) -> OperatorSpec {
    OperatorSpec {
        name: name.to_owned(),
        sort: 0,
        arguments: vec![],
        payload: PayloadKind::None,
        erased,
        binding: vec![],
        shifts: vec![],
    }
}

fn slash(name: &str, erased: u32) -> OperatorSpec {
    OperatorSpec {
        name: name.to_owned(),
        sort: 0,
        arguments: vec![0, 0],
        payload: PayloadKind::None,
        erased,
        binding: vec![],
        shifts: vec![vec![], vec![]],
    }
}

fn store() -> TermStore {
    let mut operators = NAMES
        .iter()
        .zip(0..)
        .map(|(name, erased)| constant(name, erased))
        .collect::<Vec<_>>();
    operators.push(slash("fwd", FORWARD));
    operators.push(slash("bwd", BACKWARD));
    TermStore::new(
        SignatureSpec {
            name: "rule-inverse".to_owned(),
            sorts: vec!["Cat".to_owned()],
            operators,
            vocabularies: vec![None],
            constant_cores: vec![None],
            constants: vec![None],
            bound: vec![None],
            free: vec![None],
            holes: vec![None],
            allows_vacuous_binding: true,
        },
        256,
    )
    .expect("the fixture signature is well formed")
}

struct Atoms {
    x: Term,
    y: Term,
    arguments: [Term; 3],
}

fn atoms(store: &TermStore) -> Atoms {
    let intern = |operator| {
        store
            .intern(operator, vec![], None)
            .expect("constants intern")
    };
    Atoms {
        x: intern(0),
        y: intern(1),
        arguments: [intern(2), intern(3), intern(4)],
    }
}

fn functors() -> FunctorSpec {
    FunctorSpec {
        operators: vec!["fwd".to_owned(), "bwd".to_owned()],
        forward: "fwd".to_owned(),
        backward: "bwd".to_owned(),
        result_path: vec![0],
        argument_path: vec![1],
    }
}

fn make(store: &TermStore, slash: u32, result: &Term, argument: &Term) -> Term {
    store
        .intern(slash, vec![result.clone(), argument.clone()], None)
        .expect("functors intern")
}

fn spine(store: &TermStore, foot: &Term, slashes: &[u32], arguments: &[Term]) -> Term {
    slashes
        .iter()
        .zip(arguments)
        .rev()
        .fold(foot.clone(), |built, (slash, argument)| {
            make(store, *slash, &built, argument)
        })
}

fn composition(order: Option<usize>, restriction: CompositionRestriction) -> CompositionRule {
    CompositionRule {
        functors: functors(),
        order,
        slash: "fwd".to_owned(),
        crossed: false,
        restriction,
    }
}

#[test]
fn composition_witnesses_name_the_orientation_and_the_carried_spine() {
    let store = store();
    let atoms = atoms(&store);
    let principal = make(&store, FORWARD, &atoms.x, &atoms.y);
    let container = spine(&store, &atoms.y, &[FORWARD, FORWARD], &atoms.arguments[..2]);
    let output = spine(&store, &atoms.x, &[FORWARD, FORWARD], &atoms.arguments[..2]);
    let rule = composition(None, CompositionRestriction::Steedman);
    let mut table = RepresentativeTable::default();
    let inputs = [principal, container];

    let witnesses = rule
        .witnesses(
            &store,
            &RepresentativeKernel::Identity,
            &mut table,
            &inputs,
            &output,
        )
        .expect("the step is well formed");

    assert_eq!(
        witnesses,
        vec![CompositionWitness {
            principal_slot: 0,
            container_slot: 1,
            spine: vec!["fwd".to_owned(), "fwd".to_owned()],
        }],
        "the unbounded rule composes at the order the container supplies"
    );
    assert_eq!(
        rule.applied_order(
            &store,
            &RepresentativeKernel::Identity,
            &mut table,
            &inputs,
            &output,
        ),
        Ok(Some(2)),
        "the applied order is the carried spine's length"
    );
    assert_eq!(
        rule.applied_order(
            &store,
            &RepresentativeKernel::Identity,
            &mut table,
            &inputs,
            &atoms.x,
        ),
        Ok(None),
        "a conclusion the rule does not draw has no order"
    );
}

#[test]
fn composition_witnesses_refuse_a_spine_the_restriction_forbids() {
    let store = store();
    let atoms = atoms(&store);
    let principal = make(&store, FORWARD, &atoms.x, &atoms.y);
    let container = spine(
        &store,
        &atoms.y,
        &[BACKWARD, FORWARD],
        &atoms.arguments[..2],
    );
    let output = spine(
        &store,
        &atoms.x,
        &[BACKWARD, FORWARD],
        &atoms.arguments[..2],
    );
    let inputs = [principal, container];
    let mut table = RepresentativeTable::default();

    for (restriction, licensed) in [
        (CompositionRestriction::Steedman, false),
        (CompositionRestriction::Kuhlmann, true),
        (CompositionRestriction::Unrestricted, true),
    ] {
        let witnesses = composition(Some(2), restriction)
            .witnesses(
                &store,
                &RepresentativeKernel::Identity,
                &mut table,
                &inputs,
                &output,
            )
            .expect("the step is well formed");
        assert_eq!(
            !witnesses.is_empty(),
            licensed,
            "{restriction:?} decides a mixed shallow frame"
        );
    }
}

#[test]
fn composition_residuals_recover_each_missing_premise() {
    let store = store();
    let atoms = atoms(&store);
    let principal = make(&store, FORWARD, &atoms.x, &atoms.y);
    let container = spine(&store, &atoms.y, &[FORWARD], &atoms.arguments[..1]);
    let output = spine(&store, &atoms.x, &[FORWARD], &atoms.arguments[..1]);
    let kernel = RuleKernel::Composition(composition(Some(1), CompositionRestriction::Steedman));
    let mut table = RepresentativeTable::default();

    let principals = kernel
        .residuate(
            &store,
            &RepresentativeKernel::Identity,
            &mut table,
            &output,
            &[None, Some(container.clone())],
        )
        .expect("the hole is supported");
    let containers = kernel
        .residuate(
            &store,
            &RepresentativeKernel::Identity,
            &mut table,
            &output,
            &[Some(principal.clone()), None],
        )
        .expect("the hole is supported");

    assert_eq!(principals, vec![principal], "the principal is recovered");
    assert_eq!(containers, vec![container], "the container is recovered");
    assert_eq!(
        kernel.residuate(
            &store,
            &RepresentativeKernel::Identity,
            &mut table,
            &output,
            &[None, None],
        ),
        Ok(Vec::new()),
        "two holes have no finite residual"
    );
}

#[test]
fn substitution_residuals_rebuild_the_conclusion_spine_around_the_argument() {
    let store = store();
    let atoms = atoms(&store);
    let shared = &atoms.arguments[0];
    let foot = make(&store, FORWARD, &atoms.x, &atoms.y);
    let functor = make(&store, FORWARD, &foot, shared);
    let container = make(&store, FORWARD, &atoms.y, shared);
    let output = make(&store, FORWARD, &atoms.x, shared);
    let kernel = RuleKernel::Substitution(SubstitutionRule {
        functors: functors(),
        order: Some(1),
        slash: "fwd".to_owned(),
        crossed: false,
    });
    let mut table = RepresentativeTable::default();

    assert_eq!(
        kernel.residuate(
            &store,
            &RepresentativeKernel::Identity,
            &mut table,
            &output,
            &[Some(functor.clone()), None],
        ),
        Ok(vec![container.clone()]),
        "the container is recovered from the functor and the conclusion"
    );
    assert_eq!(
        kernel.admits(
            &store,
            &RepresentativeKernel::Identity,
            &mut table,
            &[functor, container],
            &output,
        ),
        Ok(true),
        "the forward step is admitted"
    );
}

#[test]
fn undeclared_raising_targets_admit_and_residuate_without_enumerating() {
    let store = store();
    let atoms = atoms(&store);
    let inner = make(&store, BACKWARD, &atoms.x, &atoms.y);
    let raised = make(&store, FORWARD, &atoms.x, &inner);
    let open = RuleKernel::Raising(RaisingRule {
        functors: functors(),
        slash: "fwd".to_owned(),
        targets: None,
        arguments: None,
    });
    let finite = RuleKernel::Raising(RaisingRule {
        functors: functors(),
        slash: "fwd".to_owned(),
        targets: Some(vec![atoms.arguments[0].clone()]),
        arguments: None,
    });
    let mut table = RepresentativeTable::default();
    let identity = RepresentativeKernel::Identity;

    assert_eq!(
        open.apply(&store, &table, std::slice::from_ref(&atoms.y)),
        Ok(Vec::new()),
        "an open target set enumerates nothing"
    );
    assert_eq!(
        open.admits(
            &store,
            &identity,
            &mut table,
            std::slice::from_ref(&atoms.y),
            &raised
        ),
        Ok(true),
        "an open target set admits every raised shape"
    );
    assert_eq!(
        open.residuate(&store, &identity, &mut table, &raised, &[None]),
        Ok(vec![atoms.y.clone()]),
        "the raised input is recovered"
    );
    assert_eq!(
        finite.admits(&store, &identity, &mut table, &[atoms.y], &raised),
        Ok(false),
        "a target outside the declared set is refused"
    );
}

#[test]
fn declared_raising_arguments_bound_every_direction_of_the_rule() {
    let store = store();
    let atoms = atoms(&store);
    let target = atoms.arguments[0].clone();
    let functor = make(&store, FORWARD, &atoms.x, &atoms.y);
    let raise = |input: &Term| {
        let inner = make(&store, BACKWARD, &target, input);
        make(&store, FORWARD, &target, &inner)
    };
    let raised_atom = raise(&atoms.x);
    let raised_functor = raise(&functor);
    let rule = RuleKernel::Raising(RaisingRule {
        functors: functors(),
        slash: "fwd".to_owned(),
        targets: Some(vec![target.clone()]),
        arguments: Some(vec![atoms.x.clone()]),
    });
    let identity = RepresentativeKernel::Identity;
    let mut table = RepresentativeTable::default();
    for term in [&atoms.x, &functor, &target, &raised_atom, &raised_functor] {
        table.register_with(&identity, term);
    }

    assert_eq!(
        rule.apply(&store, &table, std::slice::from_ref(&atoms.x)),
        Ok(vec![raised_atom.clone()]),
        "a declared argument raises to every target"
    );
    assert_eq!(
        rule.apply(&store, &table, std::slice::from_ref(&functor)),
        Ok(Vec::new()),
        "an undeclared argument does not raise"
    );
    assert_eq!(
        rule.slot_viable(&table, &functor, 0),
        Ok(false),
        "an undeclared argument cannot fill the premise"
    );
    assert_eq!(
        rule.slot_viable(&table, &atoms.x, 0),
        Ok(true),
        "a declared argument fills the premise"
    );
    assert_eq!(
        rule.admits(
            &store,
            &identity,
            &mut table,
            std::slice::from_ref(&functor),
            &raised_functor
        ),
        Ok(false),
        "a raised undeclared argument is not admitted"
    );
    assert_eq!(
        rule.residuate(&store, &identity, &mut table, &raised_functor, &[None]),
        Ok(Vec::new()),
        "a raised undeclared argument has no residual"
    );
    assert_eq!(
        rule.residuate(&store, &identity, &mut table, &raised_atom, &[None]),
        Ok(vec![atoms.x]),
        "a raised declared argument is recovered"
    );
}

#[test]
fn a_restricted_kernel_admits_only_declared_primary_targets_and_secondaries() {
    let store = store();
    let atoms = atoms(&store);
    let z = atoms.arguments[0].clone();
    let allowed = make(
        &store,
        FORWARD,
        &make(&store, FORWARD, &atoms.x, &z),
        &atoms.y,
    );
    let refused = make(&store, FORWARD, &atoms.y, &atoms.y);
    let output = make(&store, FORWARD, &atoms.x, &z);
    let rule = RuleKernel::Restricted(Box::new(RestrictedRule {
        rule: RuleKernel::Composition(composition(Some(0), CompositionRestriction::Unrestricted)),
        restriction: PremiseRestriction {
            primary: 0,
            targets: Some(vec![atoms.x.clone()]),
            secondary: Some((1, vec![atoms.y.clone()])),
            descents: vec![("fwd".to_owned(), vec![0]), ("bwd".to_owned(), vec![0])],
        },
    }));
    let identity = RepresentativeKernel::Identity;
    let mut table = RepresentativeTable::default();
    for term in [&allowed, &refused, &output, &atoms.x, &atoms.y, &z] {
        table.register_with(&identity, term);
    }

    assert_eq!(
        rule.apply(&store, &table, &[allowed.clone(), atoms.y.clone()]),
        Ok(vec![output.clone()]),
        "a functor of the declared target applies to a declared secondary"
    );
    assert_eq!(
        rule.apply(&store, &table, &[refused.clone(), atoms.y.clone()]),
        Ok(Vec::new()),
        "a functor of another target does not apply"
    );
    assert_eq!(
        rule.slot_viable(&table, &refused, 0),
        Ok(false),
        "another target cannot fill the primary premise"
    );
    assert_eq!(
        rule.slot_viable(&table, &z, 1),
        Ok(false),
        "an undeclared category cannot fill the secondary premise"
    );
    assert_eq!(
        rule.join_keys(&table, &refused, 0),
        Ok(Vec::new()),
        "a refused primary premise has no join key"
    );
    assert_eq!(
        rule.admits(
            &store,
            &identity,
            &mut table,
            &[allowed.clone(), atoms.y.clone()],
            &output
        ),
        Ok(true),
        "the declared step is admitted"
    );
    assert_eq!(
        rule.admits(
            &store,
            &identity,
            &mut table,
            &[refused, atoms.y.clone()],
            &atoms.y
        ),
        Ok(false),
        "a step through another target is not admitted"
    );
    assert_eq!(
        rule.residuate(
            &store,
            &identity,
            &mut table,
            &output,
            &[None, Some(atoms.y.clone())]
        ),
        Ok(vec![allowed]),
        "the residual primary premise carries the declared target"
    );
    assert_eq!(
        rule.residuate(&store, &identity, &mut table, &output, &[None, Some(z)]),
        Ok(Vec::new()),
        "an undeclared secondary has no residual"
    );
}

#[test]
fn a_kernel_without_a_residual_refuses_residuation() {
    let store = store();
    let atoms = atoms(&store);
    let mut table = RepresentativeTable::default();

    assert!(
        matches!(
            RuleKernel::Empty.residuate(
                &store,
                &RepresentativeKernel::Identity,
                &mut table,
                &atoms.x,
                &[None],
            ),
            Err(RuleError::UnsupportedInverse(_))
        ),
        "an empty kernel declares no residual"
    );
}
