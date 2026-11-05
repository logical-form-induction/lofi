//! The staged grammar compiler entry point.

use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

use crate::{
    CanonicalGrammarIr, CanonicalLanguageIr, CompileError, DeclarationKind, LinkDeclarations,
    LinkedGrammar, LinkedRulePlan, PreparedGrammarIr, canonical,
};

/// Canonical declaration bytes handed to the compiler.
#[derive(Clone, Copy, Debug)]
pub struct DeclarationEnvelope<'a> {
    canonical: &'a [u8],
}

impl<'a> DeclarationEnvelope<'a> {
    /// Wrap canonical declaration bytes.
    #[must_use]
    pub const fn new(canonical: &'a [u8]) -> Self {
        Self { canonical }
    }

    /// Return the canonical declaration bytes.
    #[must_use]
    pub const fn canonical_bytes(self) -> &'a [u8] {
        self.canonical
    }
}

impl PreparedGrammarIr {
    /// Return the prepared learner-language declaration.
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn language(&self) -> &CanonicalLanguageIr {
        &self.language
    }

    /// Return the prepared categorial-grammar declaration.
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn grammar(&self) -> &CanonicalGrammarIr {
        &self.grammar
    }

    fn fingerprint(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(self.language.canonical_bytes().len().to_le_bytes());
        digest.update(self.language.canonical_bytes());
        digest.update(self.grammar.canonical_bytes().len().to_le_bytes());
        digest.update(self.grammar.canonical_bytes());
        format!("{:x}", digest.finalize())
    }
}

/// The complete immutable result of one native grammar compile.
#[derive(Clone, Debug)]
pub struct CompiledGrammar {
    linked: LinkedGrammar,
    rules: LinkedRulePlan,
}

impl CompiledGrammar {
    /// Return the linked signature and adopted native term store.
    #[must_use]
    pub const fn linked(&self) -> &LinkedGrammar {
        &self.linked
    }

    /// Return the rules in final dense identifier order.
    #[must_use]
    pub const fn rules(&self) -> &LinkedRulePlan {
        &self.rules
    }

    /// Consume the compile result into its runtime-owned parts.
    #[must_use]
    pub fn into_parts(self) -> (LinkedGrammar, LinkedRulePlan) {
        (self.linked, self.rules)
    }
}

/// Decode canonical declarations into the first native compiler stage.
#[derive(Clone, Copy, Debug, Default)]
pub struct GrammarCompiler;

impl GrammarCompiler {
    /// Compile one canonical language and grammar declaration pair.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] when a canonical envelope, linked signature, or
    /// rule recipe is invalid.
    pub fn compile(
        language: DeclarationEnvelope<'_>,
        grammar: DeclarationEnvelope<'_>,
    ) -> Result<CompiledGrammar, CompileError> {
        let prepared = Self::prepare(language, grammar)?;
        let linked = Self::link(&prepared)?;
        let rules = Self::plan_rules(&prepared, &linked)?;
        Ok(CompiledGrammar { linked, rules })
    }

    /// Prepare typed IR from the canonical language and grammar declarations.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] when a declaration does not match its typed
    /// schema or its bytes are not the one compact ASCII canonical form.
    pub(crate) fn prepare(
        language: DeclarationEnvelope<'_>,
        grammar: DeclarationEnvelope<'_>,
    ) -> Result<PreparedGrammarIr, CompileError> {
        let language_ir = decode(
            DeclarationKind::Language,
            language.canonical,
            canonical::language_bytes,
        )?;
        let grammar_ir = decode(
            DeclarationKind::Grammar,
            grammar.canonical,
            canonical::grammar_bytes,
        )?;

        Ok(PreparedGrammarIr {
            language: CanonicalLanguageIr::new(language_ir, Box::from(language.canonical)),
            grammar: CanonicalGrammarIr::new(grammar_ir, Box::from(grammar.canonical)),
        })
    }

    /// Link prepared declarations into one native signature, category syntax, and term store.
    ///
    /// This stage does not compile lexical entries, rules, matching, or runtime grammar
    /// behavior. It consumes only the two signature declarations already prepared at the
    /// canonical boundary.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] when the two declarations cannot form one semantic signature
    /// or the requested term-store capacity cannot initialize the native store.
    pub(crate) fn link(prepared: &PreparedGrammarIr) -> Result<LinkedGrammar, CompileError> {
        let linked = prepared
            .language
            .declaration()
            .link_declarations(prepared.grammar.declaration())?;
        Ok(linked.with_declaration_fingerprint(prepared.fingerprint()))
    }

    /// Expand and link every rule declaration against one linked signature.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] when `linked` came from another prepared declaration pair,
    /// a rule names an absent dense signature member, or a generated inventory cannot be
    /// represented by the native rule identifier.
    pub(crate) fn plan_rules(
        prepared: &PreparedGrammarIr,
        linked: &LinkedGrammar,
    ) -> Result<LinkedRulePlan, CompileError> {
        prepared
            .grammar
            .declaration()
            .plan_rules(linked, prepared.fingerprint().as_str())
    }
}

fn decode<T>(
    declaration: DeclarationKind,
    source: &[u8],
    encode: fn(&T) -> Result<Vec<u8>, serde_json::Error>,
) -> Result<T, CompileError>
where
    T: DeserializeOwned,
{
    let decoded: T = serde_json::from_slice(source).map_err(|source| CompileError::Decode {
        declaration,
        source,
    })?;
    let encoded = encode(&decoded).map_err(|source| CompileError::Encode {
        declaration,
        source,
    })?;

    if encoded == source {
        Ok(decoded)
    } else {
        Err(CompileError::NonCanonical { declaration })
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::{DeclarationEnvelope, GrammarCompiler};
    use crate::{CompileError, DeclarationKind};

    const LANGUAGE: &[u8] = br#"{"abstraction":"lam","application":"eval","atomic_types":[{"kind":"type","name":"e"}],"binder_family":"binder","binders":{"style":"curry"},"bounds":{"Ty":{"depth":4,"free_ranks":0,"layout":"complete-tree","nodes":null,"order":"df"}},"category_sort":"Cat","constant_family":"constant","constants":[],"constructors":[],"function_type":"->","leaf_lexicon":null,"logic":{"domains":{},"equality":true,"functions":true,"intensional":false,"order":"first","postulates":[]},"name":"tiny","store_nodes":1000000,"term_sort":"Tm","type_sort":"Ty","variable_family":"variable"}"#;
    const GRAMMAR: &[u8] = br#"{"budget":{"beam":8,"beta_fuel":1000,"closure_nodes":2048,"closure_rounds":2,"depth":5,"mass":1.0,"max_category_arity":null,"max_category_size":null,"threads":1,"unary_rounds":1},"capacity_evidence":null,"formalism":"declared","goal":{"categories":["S"],"kind":"root","root_weights":null},"interpretations":[],"lexicon":{"entries":[],"kind":"inline"},"matching":{"kind":"identity"},"max_hypotheses":0,"name":"tiny","normal_form":{"kind":"none"},"rules":{"name":"none","rules":[]},"signature":{"allows_vacuous_binding":true,"category_sort":"Cat","classifiers":{},"conjunctions":null,"constructors":[{"kind":["Cat","Cat","Cat"],"name":"/","notation":{"associativity":"left","precedence":10,"spelling":"/","style":"infix"}},{"kind":["Cat","Cat","Cat"],"name":"\\","notation":{"associativity":"left","precedence":10,"spelling":"\\","style":"infix"}}],"feature_delimiter":"brackets","functors":[{"argument":1,"argument_side":"right","constructor":"/","name":"/","opposite":"\\","result":0},{"argument":1,"argument_side":"left","constructor":"\\","name":"\\","opposite":"/","result":0}],"holes":false,"name":"tiny","open_sorts":[],"operators":[],"primitives":["S"],"sorts":["Cat"],"syntax":"declared","vocabularies":{}}}"#;

    fn grammar_with_rule(rule: &str) -> Result<Vec<u8>, Box<dyn Error>> {
        let document: serde_json::Value = serde_json::from_str(rule)?;
        let theory = document
            .get("theory")
            .ok_or("the late rule fixture declares a theory")?;
        let schema = theory
            .get("schemata")
            .and_then(serde_json::Value::as_array)
            .and_then(|schemata| schemata.first())
            .ok_or("the late rule fixture declares one schema")?;
        let name = serde_json::to_string(
            theory
                .get("name")
                .ok_or("the late rule theory declares a name")?,
        )?;
        let premises = schema
            .get("premises")
            .ok_or("the late rule schema declares premises")?;
        let conclusion = schema
            .get("conclusion")
            .ok_or("the late rule schema declares a conclusion")?;
        let head_slot = premises
            .as_array()
            .filter(|premises| !premises.is_empty())
            .map_or("null", |_| "0");
        let runtime = format!(
            "\"rule\":\"late\",\"head_policy\":\"pass-through\",\"name\":{name},\"premises\":{},\"conclusion\":{},\"targets\":null,\"normal_form_role\":{{\"kind\":\"other\",\"direction\":null,\"principal_slot\":null,\"order\":null,\"crossed\":false}},\"runtime\":{{\"kind\":\"pattern\",\"head_slot\":{head_slot}}}",
            serde_json::to_string(premises)?,
            serde_json::to_string(conclusion)?,
        );
        let rule = rule.replace(
            "\"rule\":\"late\",\"head_policy\":\"pass-through\"",
            &runtime,
        );
        canonical_document(
            &String::from_utf8(GRAMMAR.to_vec())?
                .replace("\"rules\":[]}", &format!("\"rules\":[{rule}]}}")),
        )
    }

    fn canonical_document(document: &str) -> Result<Vec<u8>, Box<dyn Error>> {
        let value: serde_json::Value = serde_json::from_str(document)?;
        Ok(crate::canonical::value_bytes(&value)?)
    }

    fn grammar_with_operator_and_rule(
        operator: &str,
        rule: &str,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        canonical_document(
            &String::from_utf8(grammar_with_rule(rule)?)?
                .replace("\"operators\":[]", &format!("\"operators\":[{operator}]")),
        )
    }

    #[test]
    fn compiler_prepares_only_the_stage_it_implements() -> Result<(), Box<dyn Error>> {
        let prepared = GrammarCompiler::prepare(
            DeclarationEnvelope::new(LANGUAGE),
            DeclarationEnvelope::new(GRAMMAR),
        )?;

        assert_eq!(prepared.language().declaration().name(), "tiny");
        assert_eq!(prepared.grammar().declaration().name(), "tiny");
        assert_eq!(prepared.language().canonical_bytes(), LANGUAGE);
        assert_eq!(prepared.grammar().canonical_bytes(), GRAMMAR);
        Ok(())
    }

    #[test]
    fn compiler_refuses_a_language_declaration_that_does_not_decode() -> Result<(), Box<dyn Error>>
    {
        let Err(error) = GrammarCompiler::prepare(
            DeclarationEnvelope::new(b"not json"),
            DeclarationEnvelope::new(GRAMMAR),
        ) else {
            return Err("an undecodable language declaration was prepared".into());
        };

        assert!(matches!(
            error,
            CompileError::Decode {
                declaration: DeclarationKind::Language,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn compiler_refuses_noncanonical_whitespace() -> Result<(), Box<dyn Error>> {
        let mut language = LANGUAGE.to_vec();
        language.push(b' ');
        let Err(error) = GrammarCompiler::prepare(
            DeclarationEnvelope::new(&language),
            DeclarationEnvelope::new(GRAMMAR),
        ) else {
            return Err("trailing whitespace was accepted as canonical".into());
        };

        assert!(matches!(
            error,
            CompileError::NonCanonical {
                declaration: DeclarationKind::Language
            }
        ));
        Ok(())
    }

    #[test]
    fn compiler_refuses_a_complete_late_schema_with_a_free_result_index()
    -> Result<(), Box<dyn Error>> {
        let grammar = grammar_with_rule(
            r#"{"restriction":null,"matching":null,"introduces":null,"consumes":null,"rule":"late","head_policy":"pass-through","theory":{"forward_complete":true,"kind":"schemata","name":"dishonest","schemata":[{"conclusion":{"name":"Y","pattern":"term","sort":"Cat"},"premises":[{"name":"X","pattern":"term","sort":"Cat"}]}]}}"#,
        )?;
        let error = GrammarCompiler::compile(
            DeclarationEnvelope::new(LANGUAGE),
            DeclarationEnvelope::new(&grammar),
        )
        .err()
        .ok_or("a complete rule left its result index free")?;

        assert!(error.to_string().contains("index_Y"), "{error}");
        assert!(error.to_string().contains("free"), "{error}");
        Ok(())
    }

    #[test]
    fn compiler_refuses_a_nondeterministic_late_schema() -> Result<(), Box<dyn Error>> {
        let grammar = grammar_with_rule(
            r#"{"restriction":null,"matching":null,"introduces":null,"consumes":null,"rule":"late","head_policy":"pass-through","theory":{"forward_complete":false,"kind":"schemata","name":"nondeterministic","schemata":[{"conclusion":{"arguments":[],"operator":{"kind":"variable","name":"F"},"pattern":"node","payload":null},"premises":[]}]}}"#,
        )?;
        let error = GrammarCompiler::compile(
            DeclarationEnvelope::new(LANGUAGE),
            DeclarationEnvelope::new(&grammar),
        )
        .err()
        .ok_or("an operator-variable schema was accepted")?;

        assert!(
            error
                .to_string()
                .contains("nondeterministic operator variable"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn compiler_refuses_late_schema_sort_arity_and_payload_faults() -> Result<(), Box<dyn Error>> {
        let cases = [
            (
                r#"{"name":"X","pattern":"term","sort":"Nonexistent"}"#,
                "unknown sort \"Nonexistent\"",
            ),
            (
                r#"{"name":"X","pattern":"term","sort":"Ty"}"#,
                "not \"Cat\"",
            ),
            (
                r#"{"arguments":[],"operator":{"kind":"fixed","name":"/"},"pattern":"node","payload":null}"#,
                "arity 2 where sort \"Cat\" and arity 0 are required",
            ),
            (
                r##"{"arguments":[],"operator":{"kind":"fixed","name":"#const[Cat]"},"pattern":"node","payload":null}"##,
                "incompatible pattern payload None",
            ),
            (
                r#"{"arguments":[{"name":"X","pattern":"term","sort":"Cat"},{"name":"Y","pattern":"term","sort":"Cat"}],"operator":{"kind":"fixed","name":"/"},"pattern":"node","payload":{"kind":"fixed","value":0}}"#,
                "incompatible pattern payload Some",
            ),
        ];

        for (conclusion, message) in cases {
            let rule = format!(
                r#"{{"restriction":null,"matching":null,"introduces":null,"consumes":null,"rule":"late","head_policy":"pass-through","theory":{{"forward_complete":false,"kind":"schemata","name":"malformed","schemata":[{{"conclusion":{conclusion},"premises":[]}}]}}}}"#
            );
            let grammar = grammar_with_rule(&rule)?;
            let error = GrammarCompiler::compile(
                DeclarationEnvelope::new(LANGUAGE),
                DeclarationEnvelope::new(&grammar),
            )
            .err()
            .ok_or("a malformed late schema was accepted")?;

            assert!(error.to_string().contains(message), "{error}");
        }
        Ok(())
    }

    #[test]
    fn compiler_refuses_one_index_variable_at_incompatible_binder_scopes()
    -> Result<(), Box<dyn Error>> {
        let operator = r#"{"name":"bind","sort":"Cat","arg_sorts":["Cat"],"binding":[{"arguments":[0],"sort":"Cat","annotation":null}],"family":"binder","erases_to":null}"#;
        let rule = r#"{"restriction":null,"matching":null,"introduces":null,"consumes":null,"rule":"late","head_policy":"pass-through","theory":{"forward_complete":false,"kind":"schemata","name":"binder-fault","schemata":[{"conclusion":{"arguments":[{"name":"X","pattern":"term","sort":"Cat"}],"operator":{"kind":"fixed","name":"bind"},"pattern":"node","payload":null},"premises":[{"name":"X","pattern":"term","sort":"Cat"}]}]}}"#;
        let grammar = grammar_with_operator_and_rule(operator, rule)?;
        let error = GrammarCompiler::compile(
            DeclarationEnvelope::new(LANGUAGE),
            DeclarationEnvelope::new(&grammar),
        )
        .err()
        .ok_or("one index variable crossed incompatible binder scopes")?;

        assert!(error.to_string().contains("incompatible sorts"), "{error}");
        Ok(())
    }
}
