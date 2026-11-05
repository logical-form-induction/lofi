//! Cross-language fixtures for the authoritative native compiler boundary.

use std::error::Error;

use lofi_grammar::{
    CategorySyntaxKind, DeclarationEnvelope, GrammarCompiler, GrammarIr, LinkedConstructorNotation,
    LinkedOperator, LinkedPattern, LinkedRuleKind, RulePlanSource, canonical,
};

const PROPOSITIONAL: &[u8] = include_bytes!("fixtures/propositional-language.json");
const CCGBANK: &[u8] = include_bytes!("fixtures/ccgbank-grammar.json");
const RESEMBLING: &[u8] = include_bytes!("fixtures/resembling-grammar.json");

fn fixture_document(source: &[u8]) -> &[u8] {
    source.strip_suffix(b"\n").unwrap_or(source)
}

#[test]
fn packaged_propositional_and_ccgbank_envelopes_compile() -> Result<(), Box<dyn Error>> {
    let language = fixture_document(PROPOSITIONAL);
    let grammar = fixture_document(CCGBANK);
    let compiled = GrammarCompiler::compile(
        DeclarationEnvelope::new(language),
        DeclarationEnvelope::new(grammar),
    )?;

    assert_eq!(compiled.linked().ir().name(), "propositional:ccgbank");
    assert_eq!(compiled.rules().rules().len(), 36);
    Ok(())
}

#[test]
fn a_data_derived_resembling_lexicon_traverses_the_compile_route() -> Result<(), Box<dyn Error>> {
    let language = fixture_document(PROPOSITIONAL);
    let grammar = fixture_document(RESEMBLING);
    let compiled = GrammarCompiler::compile(
        DeclarationEnvelope::new(language),
        DeclarationEnvelope::new(grammar),
    )?;
    let linked = compiled.linked();
    let rules = compiled.rules();

    assert_eq!(linked.ir().name(), "propositional:unnamed");
    assert_eq!(rules.signature_fingerprint(), linked.ir().fingerprint());
    assert_eq!(rules.rules().len(), 2);
    Ok(())
}

#[test]
fn packaged_propositional_and_ccgbank_signatures_link_deterministically()
-> Result<(), Box<dyn Error>> {
    let language = fixture_document(PROPOSITIONAL);
    let grammar = fixture_document(CCGBANK);
    let first = GrammarCompiler::compile(
        DeclarationEnvelope::new(language),
        DeclarationEnvelope::new(grammar),
    )?;
    let second = GrammarCompiler::compile(
        DeclarationEnvelope::new(language),
        DeclarationEnvelope::new(grammar),
    )?;
    let first = first.linked();
    let second = second.linked();

    assert_eq!(first.ir(), second.ir());
    assert_eq!(first.ir().name(), "propositional:ccgbank");
    assert_eq!(first.ir().counts().sorts(), 3);
    assert_eq!(first.ir().counts().declarations(), 5);
    assert_eq!(first.ir().counts().operators(), 11);
    assert_eq!(first.ir().counts().vocabulary_constants(), 29);
    assert_eq!(first.ir().counts().category_constructors(), 2);
    assert_eq!(
        first.ir().fingerprint(),
        "c0a2ba213085abbc904539e2611712f2087a7331caecf924531f06cfef17ae43"
    );
    assert_eq!(
        first
            .ir()
            .operators()
            .iter()
            .map(LinkedOperator::name)
            .collect::<Vec<_>>(),
        [
            "#const[Tm]",
            "#bvar[Tm]",
            "#fvar[Tm]",
            "lam",
            "eval",
            "#const[Ty]",
            "#fvar[Ty]",
            "->",
            "#const[Cat]",
            "/",
            "\\",
        ]
    );
    assert_eq!(first.store().signature(), second.store().signature());
    assert_eq!(first.ir().open_sorts(), [0, 1]);
    assert_eq!(first.ir().classifiers().len(), 1);
    assert_eq!(first.ir().classifiers()[0].sort(), 0);
    assert_eq!(first.ir().classifiers()[0].classifier(), 1);
    assert!(!first.ir().has_holes());
    assert!(first.ir().allows_vacuous_binding());
    assert_eq!(first.ir().constant_family(), "constant");
    assert_eq!(first.ir().variable_family(), "variable");
    assert_eq!(first.ir().features(), []);
    let lam = first
        .ir()
        .operators()
        .iter()
        .find(|operator| operator.name() == "lam")
        .ok_or("language abstraction must be linked")?;
    assert_eq!(lam.binding()[0].annotation(), None);
    let categories = first
        .ir()
        .vocabularies()
        .iter()
        .find(|vocabulary| vocabulary.sort() == 2)
        .ok_or("category vocabulary must be linked")?;
    let sentence = categories
        .names()
        .iter()
        .position(|name| name == "S")
        .ok_or("bare sentence constant must be linked")?;
    let declarative = categories
        .names()
        .iter()
        .position(|name| name == "S[dcl]")
        .ok_or("featured sentence constant must be linked")?;
    assert_eq!(categories.cores()[declarative], sentence);
    assert_eq!(
        first.ir().category_syntax().kind(),
        CategorySyntaxKind::Declared
    );
    assert_eq!(first.ir().category_syntax().category_sort(), 2);
    assert_eq!(first.ir().category_syntax().constructors()[0].operator(), 9);
    assert!(matches!(
        first.ir().category_syntax().constructors()[0].notation(),
        LinkedConstructorNotation::Infix {
            spelling,
            precedence: 10,
            ..
        } if spelling == "/"
    ));
    assert_eq!(first.ir().category_syntax().functors()[0].constructor(), 9);
    Ok(())
}

#[test]
fn packaged_ccgbank_rules_expand_to_one_exact_dense_plan() -> Result<(), Box<dyn Error>> {
    let first = GrammarCompiler::compile(
        DeclarationEnvelope::new(fixture_document(PROPOSITIONAL)),
        DeclarationEnvelope::new(fixture_document(CCGBANK)),
    )?;
    let second = GrammarCompiler::compile(
        DeclarationEnvelope::new(fixture_document(PROPOSITIONAL)),
        DeclarationEnvelope::new(fixture_document(CCGBANK)),
    )?;
    let linked = first.linked();
    let first = first.rules();
    let second = second.rules();

    assert_eq!(first, second);
    assert_eq!(first.signature_fingerprint(), linked.ir().fingerprint());
    assert_eq!(first.rules().len(), 36);
    assert_eq!(first.explicit_rules(), 20);
    assert_eq!(first.theory().constructors().len(), 38);
    assert_eq!(first.theory().elaborated(), <[String; 0]>::default());
    assert_eq!(first.theory().constructors()[0].rule(), "n-to-np");
    assert_eq!(first.theory().constructors()[0].premises(), 1);
    assert_eq!(
        first.plugin_tags(),
        [
            "pattern",
            "absorption",
            "clausal-adjunct",
            "coordination",
            "application",
            "composition",
            "substitution",
            "type-raising",
        ]
    );
    assert!(
        first
            .rules()
            .iter()
            .enumerate()
            .all(|(index, rule)| { usize::try_from(rule.identifier()) == Ok(index) })
    );
    assert!(
        first.rules()[..20]
            .iter()
            .all(|rule| rule.location().source() == RulePlanSource::Explicit)
    );
    assert!(matches!(
        first.rules()[20].kind(),
        LinkedRuleKind::Application { functor: 0 }
    ));
    assert!(matches!(
        first.rules()[21].kind(),
        LinkedRuleKind::Application { functor: 1 }
    ));
    let LinkedRuleKind::Pattern { recipe } = first.rules()[0].kind() else {
        return Err("the first packaged rule lost its typed pattern recipe".into());
    };
    assert_eq!(recipe.name(), "n-to-np");
    assert_eq!(recipe.head_policy(), "pass-through");
    assert_eq!(recipe.head_slot(), Some(0));
    assert!(matches!(
        recipe.premises(),
        [LinkedPattern::Constant { .. }]
    ));
    assert!(matches!(
        recipe.conclusion(),
        LinkedPattern::Constant { .. }
    ));
    assert_eq!(
        first.rules()[20..]
            .iter()
            .map(lofi_grammar::LinkedRule::tag)
            .collect::<Vec<_>>(),
        [
            "application",
            "application",
            "composition",
            "composition",
            "composition",
            "composition",
            "composition",
            "composition",
            "composition",
            "composition",
            "substitution",
            "substitution",
            "substitution",
            "substitution",
            "type-raising",
            "type-raising",
        ]
    );
    Ok(())
}

/// Replace the `n-to-np` unary rule with an identity rule over one category.
fn identity_in_place_of_n_to_np(source: &str, category: &str) -> Result<String, Box<dyn Error>> {
    let mut document: serde_json::Value = serde_json::from_str(source)?;
    let rules = document
        .pointer_mut("/rules/rules")
        .and_then(serde_json::Value::as_array_mut)
        .ok_or("the fixture declares a rule list")?;
    let rule = rules
        .iter_mut()
        .find(|rule| rule.get("name").and_then(serde_json::Value::as_str) == Some("n-to-np"))
        .ok_or("the fixture declares the n-to-np rule")?;
    let constant = serde_json::json!({"name": category, "pattern": "constant", "sort": null});
    rule["name"] = serde_json::json!("identity");
    rule["premises"] = serde_json::json!([constant]);
    rule["conclusion"] = constant;
    let declaration: GrammarIr = serde_json::from_value(document)?;
    Ok(String::from_utf8(canonical::grammar_bytes(&declaration)?)?)
}

#[test]
fn rule_fingerprints_ignore_set_labels_and_distinguish_compiled_relations()
-> Result<(), Box<dyn Error>> {
    let source = String::from_utf8(fixture_document(CCGBANK).to_vec())?.replace("\"PP\"", "\"VP\"");
    let renamed = source.replace(
        "\"rules\":{\"name\":\"ccgbank\"",
        "\"rules\":{\"name\":\"presentation-only\"",
    );
    let np = identity_in_place_of_n_to_np(&source, "NP")?;
    let vp = identity_in_place_of_n_to_np(&source, "VP")?;
    let compile = |grammar: &[u8]| {
        GrammarCompiler::compile(
            DeclarationEnvelope::new(fixture_document(PROPOSITIONAL)),
            DeclarationEnvelope::new(grammar),
        )
    };
    let baseline = compile(source.as_bytes())?;
    let renamed = compile(renamed.as_bytes())?;
    let np = compile(np.as_bytes())?;
    let vp = compile(vp.as_bytes())?;

    assert_eq!(
        baseline.rules().fingerprint(),
        renamed.rules().fingerprint()
    );
    assert_eq!(baseline.rules().theory(), renamed.rules().theory());
    assert_ne!(np.rules().fingerprint(), vp.rules().fingerprint());
    assert_ne!(np.rules().theory(), vp.rules().theory());
    Ok(())
}

#[test]
fn runtime_categories_name_absorbed_adjunct_and_raising_atoms_once() -> Result<(), Box<dyn Error>> {
    let compiled = GrammarCompiler::compile(
        DeclarationEnvelope::new(fixture_document(PROPOSITIONAL)),
        DeclarationEnvelope::new(fixture_document(CCGBANK)),
    )?;

    assert_eq!(
        compiled.rules().runtime_categories(),
        [".", ",", ":", ";", "LRB", "RRB", "S", "NP", "S[dcl]"]
    );
    Ok(())
}

#[test]
fn only_pattern_recipes_without_targets_or_spines_have_direct_schemata()
-> Result<(), Box<dyn Error>> {
    let compiled = GrammarCompiler::compile(
        DeclarationEnvelope::new(fixture_document(PROPOSITIONAL)),
        DeclarationEnvelope::new(fixture_document(CCGBANK)),
    )?;
    let recipes = compiled
        .rules()
        .rules()
        .iter()
        .filter_map(|rule| match rule.kind() {
            LinkedRuleKind::Pattern { recipe } | LinkedRuleKind::Extension { recipe } => {
                Some(recipe)
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_ne!(recipes.len(), 0);
    for recipe in recipes {
        let direct = recipe.targets().is_none()
            && recipe.premises().iter().all(LinkedPattern::is_direct)
            && recipe.conclusion().is_direct();
        assert_eq!(recipe.has_direct_schemata(), direct, "{}", recipe.name());
        assert!(
            !recipe
                .premises()
                .iter()
                .chain([recipe.conclusion()])
                .any(
                    |pattern| matches!(pattern, LinkedPattern::Spine { .. }) && pattern.is_direct()
                )
        );
    }
    Ok(())
}
