//! Typed intermediate representations of validated declarations.

use std::{collections::HashMap, fmt};

use indexmap::IndexMap;
use lofi_chart::term::{
    BindingSite as CoreBindingSite, OperatorId, OperatorSpec as CoreOperatorSpec, PayloadKind,
    SignatureSpec as CoreSignatureSpec, SortId, TermStore,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use serde_json::{Number, Value, json};
use sha2::{Digest, Sha256};

use crate::{
    CompileError, DeclarationKind, LinkComponent, LinkDeclarations, LinkError, LinkErrorKind,
    LinkLocation, LinkedAbsorptionRecipe, LinkedClausalAdjunctRecipe, LinkedCompositionDirection,
    LinkedCoordinationRecipe, LinkedNormalFormRole, LinkedOperatorPattern, LinkedPattern,
    LinkedPatternRecipe, LinkedPayloadPattern, LinkedRule, LinkedRuleKind, LinkedRulePlan,
    LinkedRuleRestriction, LinkedTheory, LinkedTheoryConstructor, RulePlanError, RulePlanErrorKind,
    RulePlanLocation, RulePlanSource, canonical,
};

/// A validated higher-order language declaration in its compiler-facing form.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageIr {
    name: String,
    atomic_types: Vec<AtomicTypeIr>,
    constructors: Vec<ConstructorIr>,
    constants: Vec<ConstantIr>,
    leaf_lexicon: Option<LeafLexiconIr>,
    binders: BinderIr,
    logic: LogicIr,
    function_type: String,
    term_sort: String,
    type_sort: String,
    category_sort: String,
    abstraction: String,
    application: String,
    binder_family: String,
    constant_family: String,
    variable_family: String,
    store_nodes: u64,
    bounds: IndexMap<String, SortBoundIr>,
}

impl LanguageIr {
    /// Return the declaration's document-level name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the declared native term-store capacity.
    #[must_use]
    pub const fn store_nodes(&self) -> u64 {
        self.store_nodes
    }
}

/// A validated categorial grammar declaration in its compiler-facing form.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GrammarIr {
    lexicon: LexiconIr,
    formalism: DeclaredFormalism,
    name: String,
    signature: SignatureIr,
    rules: RuleSetIr,
    interpretations: Vec<NamedInterpretationIr>,
    goal: GoalIr,
    matching: MatchingIr,
    normal_form: NormalFormIr,
    budget: BudgetIr,
    max_hypotheses: u64,
    capacity_evidence: Option<CapacityEvidenceIr>,
}

impl GrammarIr {
    /// Return the declaration's document-level name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the number of declared rule entries, each explicit or expanding.
    #[must_use]
    pub const fn declared_rule_families(&self) -> usize {
        self.rules.rules.len()
    }
}

/// Counts derived from one linked signature inventory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinkedGrammarCounts {
    sorts: usize,
    declarations: usize,
    operators: usize,
    vocabulary_constants: usize,
    category_constructors: usize,
}

impl LinkedGrammarCounts {
    /// Return the number of sorts in dense-code order.
    #[must_use]
    pub const fn sorts(self) -> usize {
        self.sorts
    }

    /// Return the number of source-level payload-free operator declarations.
    #[must_use]
    pub const fn declarations(self) -> usize {
        self.declarations
    }

    /// Return the complete operator count, including featured and kernel-derived operators.
    #[must_use]
    pub const fn operators(self) -> usize {
        self.operators
    }

    /// Return the total number of constants across all vocabularies.
    #[must_use]
    pub const fn vocabulary_constants(self) -> usize {
        self.vocabulary_constants
    }

    /// Return the number of category constructors linked to surface syntax.
    #[must_use]
    pub const fn category_constructors(self) -> usize {
        self.category_constructors
    }
}

/// One payload-free declaration paired with its final dense operator code.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedOperatorDeclaration {
    name: String,
    operator: OperatorId,
}

/// The non-term payload convention of a linked operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkedPayloadKind {
    /// The operator accepts no payload.
    None,
    /// The payload is a stable vocabulary identifier.
    Constant,
    /// The payload is a de Bruijn index.
    Index,
    /// The payload is a stable free-variable identifier.
    Free,
}

/// One linked binding site, including its source-level classifier annotation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedBindingSite {
    arguments: Vec<usize>,
    sort: SortId,
    annotation: Option<usize>,
}

impl LinkedBindingSite {
    /// Return the argument positions under this binding site.
    #[must_use]
    pub fn arguments(&self) -> &[usize] {
        &self.arguments
    }

    /// Return the dense code of the bound variable sort.
    #[must_use]
    pub const fn sort(&self) -> SortId {
        self.sort
    }

    /// Return the argument position holding the variable classifier, when declared.
    #[must_use]
    pub const fn annotation(&self) -> Option<usize> {
        self.annotation
    }
}

/// Complete linked metadata for one operator in dense-code order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedOperator {
    name: String,
    operator: OperatorId,
    sort: SortId,
    arguments: Vec<SortId>,
    family: String,
    payload: LinkedPayloadKind,
    erased: OperatorId,
    binding: Vec<LinkedBindingSite>,
}

impl LinkedOperator {
    /// Return the stable structural name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return this operator's dense code.
    #[must_use]
    pub const fn operator(&self) -> OperatorId {
        self.operator
    }

    /// Return the dense result-sort code.
    #[must_use]
    pub const fn sort(&self) -> SortId {
        self.sort
    }

    /// Return dense argument-sort codes in argument order.
    #[must_use]
    pub fn arguments(&self) -> &[SortId] {
        &self.arguments
    }

    /// Return the decoder-head family.
    #[must_use]
    pub fn family(&self) -> &str {
        &self.family
    }

    /// Return the non-term payload convention.
    #[must_use]
    pub const fn payload(&self) -> LinkedPayloadKind {
        self.payload
    }

    /// Return the dense code of the feature-erasure core.
    #[must_use]
    pub const fn erased(&self) -> OperatorId {
        self.erased
    }

    /// Return binding sites in declaration order, including classifier annotations.
    #[must_use]
    pub fn binding(&self) -> &[LinkedBindingSite] {
        &self.binding
    }
}

/// One classified sort and its classifier sort.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinkedClassifier {
    sort: SortId,
    classifier: SortId,
}

impl LinkedClassifier {
    /// Return the classified sort's dense code.
    #[must_use]
    pub const fn sort(self) -> SortId {
        self.sort
    }

    /// Return the classifier sort's dense code.
    #[must_use]
    pub const fn classifier(self) -> SortId {
        self.classifier
    }
}

/// Ordered feature values generated from one declared operator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedOperatorFeatures {
    name: String,
    operator: OperatorId,
    values: Vec<String>,
}

impl LinkedOperatorFeatures {
    /// Return the bare operator name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the bare operator's dense code.
    #[must_use]
    pub const fn operator(&self) -> OperatorId {
        self.operator
    }

    /// Return admitted feature values in declaration order.
    #[must_use]
    pub fn values(&self) -> &[String] {
        &self.values
    }
}

impl LinkedOperatorDeclaration {
    /// Return the stable structural name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the operator's dense core code.
    #[must_use]
    pub const fn operator(&self) -> OperatorId {
        self.operator
    }
}

/// One vocabulary and its direct feature-core identifiers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedVocabulary {
    sort: SortId,
    names: Vec<String>,
    cores: Vec<usize>,
}

impl LinkedVocabulary {
    /// Return the vocabulary sort's dense code.
    #[must_use]
    pub const fn sort(&self) -> SortId {
        self.sort
    }

    /// Return constant names in stable identifier order.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Return each constant's direct core identifier in stable identifier order.
    #[must_use]
    pub fn cores(&self) -> &[usize] {
        &self.cores
    }
}

/// The linked surface-notation mode for categories.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CategorySyntaxKind {
    /// Read categories using the declared constructor notations.
    Declared,
    /// Read categories as operator-headed s-expressions.
    SExpression,
}

/// Associativity of a linked infix category constructor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkedAssociativity {
    /// Group repeated operators to the left.
    Left,
    /// Group repeated operators to the right.
    Right,
    /// Do not admit an unparenthesized repeated operator.
    None,
}

/// Surface notation attached to one linked category constructor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkedConstructorNotation {
    /// A token preceding all constructor arguments.
    Prefix {
        /// The surface token.
        spelling: String,
    },
    /// A token between the two constructor arguments.
    Infix {
        /// The surface token.
        spelling: String,
        /// The positive binding precedence.
        precedence: u64,
        /// The repeated-operator grouping policy.
        associativity: LinkedAssociativity,
    },
    /// Literal parts surrounding and separating the constructor arguments.
    Mixfix {
        /// The literal parts in surface order.
        parts: Vec<String>,
        /// The positive binding precedence.
        precedence: u64,
    },
}

/// One category constructor paired with its dense operator code and notation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedCategoryConstructor {
    name: String,
    operator: OperatorId,
    notation: LinkedConstructorNotation,
}

impl LinkedCategoryConstructor {
    /// Return the constructor's stable name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the constructor's dense core code.
    #[must_use]
    pub const fn operator(&self) -> OperatorId {
        self.operator
    }

    /// Return the linked surface notation.
    #[must_use]
    pub const fn notation(&self) -> &LinkedConstructorNotation {
        &self.notation
    }
}

/// The side on which a functor view takes its argument.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkedArgumentSide {
    /// The argument is on the left.
    Left,
    /// The argument is on the right.
    Right,
    /// The same constructor can be viewed in either orientation.
    Either,
}

/// A validated functor view over a linked binary category constructor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedFunctor {
    name: String,
    constructor: OperatorId,
    opposite: String,
    argument_side: LinkedArgumentSide,
    result: usize,
    argument: usize,
}

impl LinkedFunctor {
    /// Return the rule-facing functor-view name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the selected constructor's dense operator code.
    #[must_use]
    pub const fn constructor(&self) -> OperatorId {
        self.constructor
    }

    /// Return the name of the mutually opposite view.
    #[must_use]
    pub fn opposite(&self) -> &str {
        &self.opposite
    }

    /// Return the view's argument side.
    #[must_use]
    pub const fn argument_side(&self) -> LinkedArgumentSide {
        self.argument_side
    }

    /// Return the constructor slot holding the functor result.
    #[must_use]
    pub const fn result(&self) -> usize {
        self.result
    }

    /// Return the constructor slot holding the functor argument.
    #[must_use]
    pub const fn argument(&self) -> usize {
        self.argument
    }
}

/// Category syntax resolved against one dense native signature.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedCategorySyntax {
    kind: CategorySyntaxKind,
    feature_delimiter: FeatureDelimiter,
    category_sort: SortId,
    constructors: Vec<LinkedCategoryConstructor>,
    functors: Vec<LinkedFunctor>,
}

impl LinkedCategorySyntax {
    /// Return the category text representation selected by the grammar.
    #[must_use]
    pub const fn kind(&self) -> CategorySyntaxKind {
        self.kind
    }

    /// Return the dense code of the category sort.
    #[must_use]
    pub const fn category_sort(&self) -> SortId {
        self.category_sort
    }

    const fn feature_delimiter(&self) -> FeatureDelimiter {
        self.feature_delimiter
    }

    /// Return category constructors in grammar declaration order.
    #[must_use]
    pub fn constructors(&self) -> &[LinkedCategoryConstructor] {
        &self.constructors
    }

    /// Return functor views in grammar declaration order.
    #[must_use]
    pub fn functors(&self) -> &[LinkedFunctor] {
        &self.functors
    }
}

/// Immutable linked metadata for one joint language and grammar signature.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedGrammarIr {
    name: String,
    sorts: Vec<String>,
    operators: Vec<LinkedOperator>,
    declarations: Vec<LinkedOperatorDeclaration>,
    vocabularies: Vec<LinkedVocabulary>,
    open_sorts: Vec<SortId>,
    classifiers: Vec<LinkedClassifier>,
    features: Vec<LinkedOperatorFeatures>,
    holes: bool,
    allows_vacuous_binding: bool,
    constant_family: String,
    variable_family: String,
    category_syntax: LinkedCategorySyntax,
    fingerprint: String,
    counts: LinkedGrammarCounts,
}

impl LinkedGrammarIr {
    /// Return the joint signature name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return sort names in dense-code order.
    #[must_use]
    pub fn sorts(&self) -> &[String] {
        &self.sorts
    }

    /// Return complete operator metadata in dense-code order.
    #[must_use]
    pub fn operators(&self) -> &[LinkedOperator] {
        &self.operators
    }

    /// Return payload-free source declarations in declaration order.
    #[must_use]
    pub fn declarations(&self) -> &[LinkedOperatorDeclaration] {
        &self.declarations
    }

    /// Return nonempty vocabularies in sort order.
    #[must_use]
    pub fn vocabularies(&self) -> &[LinkedVocabulary] {
        &self.vocabularies
    }

    /// Return sorts open to free variables in dense-sort order.
    #[must_use]
    pub fn open_sorts(&self) -> &[SortId] {
        &self.open_sorts
    }

    /// Return classifier relations in classified-sort order.
    #[must_use]
    pub fn classifiers(&self) -> &[LinkedClassifier] {
        &self.classifiers
    }

    /// Return operator feature declarations in target declaration order.
    #[must_use]
    pub fn features(&self) -> &[LinkedOperatorFeatures] {
        &self.features
    }

    /// Return whether the signature derives a hole operator for every sort.
    #[must_use]
    pub const fn has_holes(&self) -> bool {
        self.holes
    }

    /// Return whether binders may introduce unused variables.
    #[must_use]
    pub const fn allows_vacuous_binding(&self) -> bool {
        self.allows_vacuous_binding
    }

    /// Return the decoder family of kernel-derived constant operators.
    #[must_use]
    pub fn constant_family(&self) -> &str {
        &self.constant_family
    }

    /// Return the decoder family of kernel-derived variable operators.
    #[must_use]
    pub fn variable_family(&self) -> &str {
        &self.variable_family
    }

    /// Return category syntax resolved to the dense signature.
    #[must_use]
    pub const fn category_syntax(&self) -> &LinkedCategorySyntax {
        &self.category_syntax
    }

    /// Return the lowercase SHA-256 fingerprint matching the Python signature digest.
    #[must_use]
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Return all linked inventory counts.
    #[must_use]
    pub const fn counts(&self) -> LinkedGrammarCounts {
        self.counts
    }
}

/// A linked declaration stage backed by the native core term store.
#[derive(Clone, Debug)]
pub struct LinkedGrammar {
    ir: LinkedGrammarIr,
    store: TermStore,
    declaration_fingerprint: Option<String>,
}

impl LinkedGrammar {
    /// Return immutable metadata for the linked signature and syntax.
    #[must_use]
    pub const fn ir(&self) -> &LinkedGrammarIr {
        &self.ir
    }

    /// Return the initialized native term store.
    #[must_use]
    pub const fn store(&self) -> &TermStore {
        &self.store
    }

    /// Return the prepared declaration-pair fingerprint bound by [`GrammarCompiler`](crate::GrammarCompiler).
    #[must_use]
    pub fn declaration_fingerprint(&self) -> Option<&str> {
        self.declaration_fingerprint.as_deref()
    }

    pub(crate) fn with_declaration_fingerprint(mut self, fingerprint: String) -> Self {
        self.declaration_fingerprint = Some(fingerprint);
        self
    }
}

/// A language IR paired with its canonical bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalLanguageIr {
    declaration: LanguageIr,
    canonical: Box<[u8]>,
}

impl CanonicalLanguageIr {
    pub(crate) const fn new(declaration: LanguageIr, canonical: Box<[u8]>) -> Self {
        Self {
            declaration,
            canonical,
        }
    }

    /// Return the typed language declaration.
    #[must_use]
    pub const fn declaration(&self) -> &LanguageIr {
        &self.declaration
    }

    /// Return the declaration's one canonical byte form.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical
    }
}

/// A grammar IR paired with its canonical bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalGrammarIr {
    declaration: GrammarIr,
    canonical: Box<[u8]>,
}

impl CanonicalGrammarIr {
    pub(crate) const fn new(declaration: GrammarIr, canonical: Box<[u8]>) -> Self {
        Self {
            declaration,
            canonical,
        }
    }

    /// Return the typed grammar declaration.
    #[must_use]
    pub const fn declaration(&self) -> &GrammarIr {
        &self.declaration
    }

    /// Return the declaration's one canonical byte form.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum DeclaredFormalism {
    #[serde(rename = "declared")]
    Declared,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct AtomicTypeIr {
    name: String,
    kind: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ConstructorIr {
    name: String,
    kind: Vec<String>,
    notation: ConstructorNotationIr,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum ConstructorNotationIr {
    Prefix(PrefixConstructorNotationIr),
    Infix(InfixConstructorNotationIr),
    Mixfix(MixfixConstructorNotationIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PrefixConstructorNotationIr {
    style: PrefixStyle,
    spelling: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct InfixConstructorNotationIr {
    style: InfixStyle,
    spelling: String,
    precedence: u64,
    associativity: ConstructorAssociativity,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct MixfixConstructorNotationIr {
    style: MixfixStyle,
    parts: Vec<String>,
    precedence: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum PrefixStyle {
    #[serde(rename = "prefix")]
    Prefix,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum InfixStyle {
    #[serde(rename = "infix")]
    Infix,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum MixfixStyle {
    #[serde(rename = "mixfix")]
    Mixfix,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum ConstructorAssociativity {
    Left,
    Right,
    None,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ConstantIr {
    name: String,
    #[serde(rename = "type")]
    type_expression: String,
    quantifies: Vec<String>,
    classes: Vec<String>,
    notation: Option<ConstantNotationIr>,
    definition: Option<String>,
    semantics: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum ConstantNotationIr {
    Prefix(PrefixConstantNotationIr),
    Infix(InfixConstantNotationIr),
    Binder(BinderNotationIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PrefixConstantNotationIr {
    kind: PrefixStyle,
    ascii: String,
    unicode: String,
    precedence: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct InfixConstantNotationIr {
    kind: InfixStyle,
    ascii: String,
    unicode: String,
    precedence: u64,
    associativity: BinaryAssociativity,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct BinderNotationIr {
    kind: BinderKind,
    ascii: String,
    unicode: String,
    bodies: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum BinderKind {
    #[serde(rename = "binder")]
    Binder,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum BinaryAssociativity {
    Left,
    Right,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum LeafLexiconIr {
    Inline(InlineLeafLexiconIr),
    Path(PathLeafLexiconIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct InlineLeafLexiconIr {
    kind: InlineKind,
    entries: Vec<LeafConstantIr>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PathLeafLexiconIr {
    kind: PathKind,
    path: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct LeafConstantIr {
    name: String,
    #[serde(rename = "type")]
    type_expression: String,
    classes: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum InlineKind {
    #[serde(rename = "inline")]
    Inline,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum PathKind {
    #[serde(rename = "path")]
    Path,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct BinderIr {
    style: BinderStyle,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum BinderStyle {
    Church,
    Curry,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct LogicIr {
    order: LogicOrder,
    #[serde(skip_serializing_if = "Option::is_none")]
    intensional: Option<bool>,
    functions: bool,
    equality: bool,
    domains: IndexMap<String, u64>,
    postulates: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    variables: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    semantics: Option<String>,
    #[serde(default, skip_serializing_if = "Present::is_missing")]
    negation: Present<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    label_map: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prover: Option<IndexMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    labeling: Option<IndexMap<String, IndexMap<String, serde_json::Value>>>,
}

#[derive(Clone, Debug, PartialEq)]
enum Present<T> {
    Missing,
    Value(Option<T>),
}

impl<T> Present<T> {
    const fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }
}

impl<T> Default for Present<T> {
    fn default() -> Self {
        Self::Missing
    }
}

impl<'de, T> Deserialize<'de> for Present<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(Self::Value)
    }
}

impl<T> Serialize for Present<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Missing => serializer.serialize_none(),
            Self::Value(value) => value.serialize(serializer),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum LogicOrder {
    First,
    Second,
    Higher,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SortBoundIr {
    layout: BoundLayout,
    depth: u64,
    nodes: Option<u64>,
    order: TreeOrder,
    free_ranks: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum BoundLayout {
    #[serde(rename = "complete-tree")]
    CompleteTree,
    #[serde(rename = "preorder-arity")]
    PreorderArity,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum TreeOrder {
    Df,
    Bf,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum LexiconIr {
    Inline(InlineLexiconIr),
    Structured(StructuredLexiconIr),
    Generated(GeneratedLexiconIr),
    Resembling(ResemblingLexiconIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct InlineLexiconIr {
    kind: InlineKind,
    entries: Vec<LexicalEntryIr>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct LexicalEntryIr {
    form: String,
    category: String,
    meaning: Option<String>,
    weight: FiniteF64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct StructuredLexiconIr {
    kind: StructuredKind,
    forms: u64,
    joint: JointIr,
    ambiguity: AmbiguityLawIr,
    rank_law: RankLawIr,
    seed: SeedIr,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneratedLexiconIr {
    kind: GeneratedKind,
    forms: u64,
    categories: u64,
    category_shares: RankLawIr,
    coupling: FiniteF64,
    ambiguity: AmbiguityLawIr,
    rank_law: RankLawIr,
    seed: SeedIr,
    max_depth: u64,
    max_arguments: Option<CategoryComplexityBoundIr>,
    max_order: Option<CategoryComplexityBoundIr>,
    derivation_depth: u64,
    expansion: FiniteF64,
    patience: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ResemblingLexiconIr {
    kind: ResemblingKind,
    forms: u64,
    estimate: LexiconEstimateIr,
    seed: SeedIr,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct LexiconEstimateIr {
    digest: String,
    forms: u64,
    exponent: FiniteF64,
    rows: Vec<EstimatedClassIr>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct EstimatedClassIr {
    category: String,
    size: u64,
    share: FiniteF64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum StructuredKind {
    #[serde(rename = "structured")]
    Structured,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum GeneratedKind {
    #[serde(rename = "generated")]
    Generated,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum ResemblingKind {
    #[serde(rename = "resembling")]
    Resembling,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum SeedIr {
    Integer(u64),
    Text(String),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum RankLawIr {
    Uniform(UniformRankLawIr),
    PowerLaw(PowerLawRankLawIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct UniformRankLawIr {
    kind: UniformKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PowerLawRankLawIr {
    kind: PowerLawKind,
    exponent: FiniteF64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum UniformKind {
    #[serde(rename = "uniform")]
    Uniform,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum PowerLawKind {
    #[serde(rename = "power-law")]
    PowerLaw,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum AmbiguityLawIr {
    Parametric(ParametricAmbiguityLawIr),
    Histogram(HistogramAmbiguityLawIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ParametricAmbiguityLawIr {
    maximum: u64,
    placement: FiniteF64,
    spelling: ParametricSpelling,
    mean: FiniteF64,
    dispersion: FiniteF64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct HistogramAmbiguityLawIr {
    maximum: u64,
    placement: FiniteF64,
    spelling: HistogramSpelling,
    buckets: Vec<AmbiguityBucketIr>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct AmbiguityBucketIr {
    categories: u64,
    share: FiniteF64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum ParametricSpelling {
    #[serde(rename = "parametric")]
    Parametric,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum HistogramSpelling {
    #[serde(rename = "histogram")]
    Histogram,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum JointIr {
    Parametric(ParametricJointIr),
    Tabular(TabularJointIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ParametricJointIr {
    spelling: ParametricSpelling,
    categories: Vec<String>,
    category_shares: RankLawIr,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct TabularJointIr {
    spelling: TabularSpelling,
    rows: Vec<ClassIr>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ClassIr {
    category: String,
    share: FiniteF64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum TabularSpelling {
    #[serde(rename = "tabular")]
    Tabular,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CategoryComplexityBoundIr {
    value: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SignatureIr {
    name: String,
    sorts: Vec<String>,
    category_sort: String,
    primitives: Vec<String>,
    operators: Vec<OperatorIr>,
    constructors: Vec<ConstructorIr>,
    functors: Vec<FunctorIr>,
    syntax: SignatureSyntax,
    #[serde(default)]
    feature_delimiter: FeatureDelimiter,
    conjunctions: Option<ConjunctionsIr>,
    vocabularies: IndexMap<String, Vec<String>>,
    open_sorts: Vec<String>,
    classifiers: IndexMap<String, String>,
    allows_vacuous_binding: bool,
    holes: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct OperatorIr {
    name: String,
    sort: String,
    arg_sorts: Vec<String>,
    binding: Vec<BindingSiteIr>,
    family: String,
    erases_to: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct BindingSiteIr {
    arguments: Vec<u64>,
    sort: String,
    annotation: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct FunctorIr {
    name: String,
    constructor: String,
    argument_side: ArgumentSide,
    opposite: String,
    result: u64,
    argument: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum ArgumentSide {
    Left,
    Right,
    Either,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum SignatureSyntax {
    #[serde(rename = "declared")]
    Declared,
    #[serde(rename = "s-expression")]
    SExpression,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum FeatureDelimiter {
    #[default]
    Brackets,
    Colon,
}

impl FeatureDelimiter {
    const fn marker(self) -> char {
        match self {
            Self::Brackets => '[',
            Self::Colon => ':',
        }
    }

    fn name(self, target: &str, feature: &str) -> String {
        match self {
            Self::Brackets => format!("{target}[{feature}]"),
            Self::Colon => format!("{target}:{feature}"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ConjunctionsIr {
    name: String,
    markers: Vec<String>,
    pairs: Vec<ConjunctionPairIr>,
    marking: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ConjunctionPairIr {
    unmarked: String,
    marked: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RuleSetIr {
    name: String,
    rules: Vec<RuleEntryIr>,
}

/// One entry of the declared rule list: an expanding set or one rule.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum RuleEntryIr {
    Combinators(CombinatorSetIr),
    Coercions(CoercionSetIr),
    Rule(RuleIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum RuleIr {
    Application(ApplicationRuleIr),
    Composition(CompositionRuleIr),
    Substitution(SubstitutionRuleIr),
    TypeRaising(TypeRaisingRuleIr),
    ClausalAdjunct(ClausalAdjunctRuleIr),
    Pattern(PatternRuleIr),
    Absorption(AbsorptionRuleIr),
    Coordination(CoordinationRuleIr),
    Coercion(CoercionRuleIr),
    Opaque(OpaqueRuleIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct OpaqueRuleIr {
    restriction: Option<RuleRestrictionIr>,
    matching: Option<MatchingIr>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
    rule: String,
    head_policy: HeadPolicy,
    #[serde(flatten)]
    extension: IndexMap<String, OpaqueJsonValue>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum OpaqueJsonValue {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<OpaqueJsonValue>),
    Object(IndexMap<String, OpaqueJsonValue>),
}

impl OpaqueJsonValue {
    fn get(&self, name: &str) -> Option<&Self> {
        let Self::Object(fields) = self else {
            return None;
        };
        fields.get(name)
    }

    fn as_str(&self) -> Option<&str> {
        let Self::String(value) = self else {
            return None;
        };
        Some(value)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum OpaqueTheoryIr {
    Elaborated {
        name: String,
    },
    Schemata {
        name: String,
        forward_complete: bool,
        schemata: Vec<TheorySchemaIr>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum OpaqueRuntimeIr {
    Pattern {
        #[serde(default)]
        head_slot: Present<u64>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct OpaquePatternRecipeIr {
    name: String,
    premises: Vec<PatternIr>,
    conclusion: PatternIr,
    targets: Option<Vec<String>>,
    normal_form_role: NormalFormRoleIr,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TheorySchemaIr {
    premises: Vec<PatternIr>,
    conclusion: PatternIr,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ApplicationRuleIr {
    restriction: Option<RuleRestrictionIr>,
    matching: Option<MatchingIr>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
    rule: ApplicationRuleKind,
    view: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CompositionRuleIr {
    restriction: Option<RuleRestrictionIr>,
    matching: Option<MatchingIr>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
    rule: CompositionRuleKind,
    direction: CompositionRestriction,
    order: Option<u64>,
    view: String,
    crossed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SubstitutionRuleIr {
    restriction: Option<RuleRestrictionIr>,
    matching: Option<MatchingIr>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
    rule: SubstitutionRuleKind,
    order: Option<u64>,
    view: String,
    crossed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct TypeRaisingRuleIr {
    restriction: Option<RuleRestrictionIr>,
    matching: Option<MatchingIr>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
    rule: TypeRaisingRuleKind,
    view: String,
    targets: Option<Vec<String>>,
    arguments: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ClausalAdjunctRuleIr {
    restriction: Option<RuleRestrictionIr>,
    matching: Option<MatchingIr>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
    rule: ClausalAdjunctRuleKind,
    name: String,
    source_view: String,
    target_view: String,
    source_core: String,
    argument: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PatternRuleIr {
    restriction: Option<RuleRestrictionIr>,
    matching: Option<MatchingIr>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
    rule: PatternRuleKind,
    head_policy: HeadPolicy,
    name: String,
    premises: Vec<PatternIr>,
    conclusion: PatternIr,
    targets: Option<Vec<String>>,
    normal_form_role: NormalFormRoleIr,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct AbsorptionRuleIr {
    restriction: Option<RuleRestrictionIr>,
    matching: Option<MatchingIr>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
    rule: AbsorptionRuleKind,
    head_policy: HeadPolicy,
    name: String,
    premises: Vec<PatternIr>,
    conclusion: PatternIr,
    absorbing: Vec<String>,
    side: BinarySide,
    other_must_be_nonabsorbing: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CoordinationRuleIr {
    restriction: Option<RuleRestrictionIr>,
    matching: Option<MatchingIr>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
    rule: CoordinationRuleKind,
    head_policy: HeadPolicy,
    name: String,
    premises: Vec<PatternIr>,
    conclusion: PatternIr,
    phase: CoordinationPhase,
    core_relation: CoreRelation,
    conjunction_feature: String,
    targets: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CoercionRuleIr {
    restriction: Option<RuleRestrictionIr>,
    matching: Option<MatchingIr>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
    rule: CoercionRuleKind,
    label: String,
    member: u64,
    source: String,
    target: String,
}

macro_rules! marker_enum {
    ($name:ident, $wire:literal) => {
        #[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
        enum $name {
            #[serde(rename = $wire)]
            Value,
        }
    };
}

marker_enum!(ApplicationRuleKind, "application");
marker_enum!(CompositionRuleKind, "composition");
marker_enum!(SubstitutionRuleKind, "substitution");
marker_enum!(TypeRaisingRuleKind, "type-raising");
marker_enum!(ClausalAdjunctRuleKind, "clausal-adjunct");
marker_enum!(PatternRuleKind, "pattern");
marker_enum!(AbsorptionRuleKind, "absorption");
marker_enum!(CoordinationRuleKind, "coordination");
marker_enum!(CoercionRuleKind, "coercion");
marker_enum!(CombinatorSetRuleKind, "combinators");
marker_enum!(CoercionSetRuleKind, "coercions");

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum HeadPolicy {
    #[serde(rename = "functor")]
    Functor,
    #[serde(rename = "pass-through")]
    PassThrough,
    #[serde(rename = "non-absorbing")]
    NonAbsorbing,
    #[serde(rename = "left-conjunct")]
    LeftConjunct,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum CompositionRestriction {
    Steedman,
    Kuhlmann,
    Unrestricted,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RuleRestrictionIr {
    primary_targets: Vec<String>,
    secondary_patterns: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum MatchingIr {
    Identity(IdentityMatchingIr),
    Alpha(AlphaMatchingIr),
    FeatureErasure(FeatureErasureMatchingIr),
    Product(ProductMatchingIr),
    Subsumption(SubsumptionMatchingIr),
    Modal(ModalMatchingIr),
    Extension(ExtensionIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct IdentityMatchingIr {
    kind: IdentityKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct AlphaMatchingIr {
    kind: AlphaKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct FeatureErasureMatchingIr {
    kind: FeatureErasureKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ProductMatchingIr {
    kind: ProductKind,
    left: Box<MatchingIr>,
    right: Box<MatchingIr>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SubsumptionMatchingIr {
    kind: SubsumptionKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ModalMatchingIr {
    kind: ModalKind,
    modes: Vec<String>,
    order: Vec<ModeRelationIr>,
}

marker_enum!(IdentityKind, "identity");
marker_enum!(AlphaKind, "alpha");
marker_enum!(FeatureErasureKind, "feature-erasure");
marker_enum!(ProductKind, "product");
marker_enum!(SubsumptionKind, "subsumption");
marker_enum!(ModalKind, "modal");

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ModeRelationIr {
    lower: String,
    upper: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum PatternIr {
    Term(TermPatternIr),
    Constant(ConstantPatternIr),
    Node(NodePatternIr),
    Spine(SpinePatternIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct TermPatternIr {
    pattern: TermPatternKind,
    name: String,
    sort: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ConstantPatternIr {
    pattern: ConstantPatternKind,
    name: String,
    sort: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NodePatternIr {
    pattern: NodePatternKind,
    operator: OperatorPatternIr,
    arguments: Vec<PatternIr>,
    payload: Option<PayloadPatternIr>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SpinePatternIr {
    pattern: SpinePatternKind,
    name: String,
    core: Box<PatternIr>,
    operators: Vec<String>,
    slot: u64,
    min_order: u64,
    max_order: Option<u64>,
}

marker_enum!(TermPatternKind, "term");
marker_enum!(ConstantPatternKind, "constant");
marker_enum!(NodePatternKind, "node");
marker_enum!(SpinePatternKind, "spine");

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum OperatorPatternIr {
    Fixed(FixedOperatorPatternIr),
    Variable(VariableOperatorPatternIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct FixedOperatorPatternIr {
    kind: FixedKind,
    name: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct VariableOperatorPatternIr {
    kind: VariableKind,
    name: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum PayloadPatternIr {
    Fixed(FixedPayloadPatternIr),
    Variable(VariablePayloadPatternIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct FixedPayloadPatternIr {
    kind: FixedKind,
    value: i64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct VariablePayloadPatternIr {
    kind: VariableKind,
    name: String,
}

marker_enum!(FixedKind, "fixed");
marker_enum!(VariableKind, "variable");

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NormalFormRoleIr {
    kind: NormalFormRoleKind,
    direction: Option<NormalFormDirection>,
    principal_slot: Option<u64>,
    order: Option<u64>,
    crossed: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum NormalFormRoleKind {
    #[serde(rename = "composition")]
    Composition,
    #[serde(rename = "raising")]
    Raising,
    #[serde(rename = "coordination")]
    Coordination,
    #[serde(rename = "other")]
    Other,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum NormalFormDirection {
    #[serde(rename = "forward")]
    Forward,
    #[serde(rename = "backward")]
    Backward,
    #[serde(rename = "undirected")]
    Undirected,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum BinarySide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum CoordinationPhase {
    Mark,
    Merge,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum CoreRelation {
    Same,
    Different,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CombinatorSetIr {
    restriction: Option<RuleRestrictionIr>,
    matching: Option<MatchingIr>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
    rule: CombinatorSetRuleKind,
    application: bool,
    composition: Vec<Option<u64>>,
    crossed_composition: Vec<Option<u64>>,
    substitution: Vec<Option<u64>>,
    crossed_substitution: Vec<Option<u64>>,
    direction: CompositionRestriction,
    type_raising: bool,
    raising_targets: Option<Vec<String>>,
    raising_arguments: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CoercionSetIr {
    restriction: Option<RuleRestrictionIr>,
    matching: Option<MatchingIr>,
    introduces: Option<Vec<String>>,
    consumes: Option<Vec<String>>,
    rule: CoercionSetRuleKind,
    label: String,
    entries: Vec<String>,
}

const BUILTIN_RULE_TAGS: [&str; 11] = [
    "absorption",
    "application",
    "clausal-adjunct",
    "coercion",
    "coercions",
    "combinators",
    "composition",
    "coordination",
    "pattern",
    "substitution",
    "type-raising",
];

const fn linked_head_policy(policy: HeadPolicy) -> &'static str {
    match policy {
        HeadPolicy::Functor => "functor",
        HeadPolicy::PassThrough => "pass-through",
        HeadPolicy::NonAbsorbing => "non-absorbing",
        HeadPolicy::LeftConjunct => "left-conjunct",
    }
}

const fn linked_normal_form_kind(kind: NormalFormRoleKind) -> &'static str {
    match kind {
        NormalFormRoleKind::Composition => "composition",
        NormalFormRoleKind::Raising => "raising",
        NormalFormRoleKind::Coordination => "coordination",
        NormalFormRoleKind::Other => "other",
    }
}

const fn linked_normal_form_direction(direction: NormalFormDirection) -> &'static str {
    match direction {
        NormalFormDirection::Forward => "forward",
        NormalFormDirection::Backward => "backward",
        NormalFormDirection::Undirected => "undirected",
    }
}

fn linked_normal_form_role(role: &NormalFormRoleIr) -> LinkedNormalFormRole {
    LinkedNormalFormRole {
        kind: linked_normal_form_kind(role.kind),
        direction: role.direction.map(linked_normal_form_direction),
        principal_slot: role.principal_slot,
        order: role.order,
        crossed: role.crossed,
    }
}

const fn other_normal_form_role() -> LinkedNormalFormRole {
    LinkedNormalFormRole {
        kind: "other",
        direction: None,
        principal_slot: None,
        order: None,
        crossed: false,
    }
}

const fn coordination_normal_form_role() -> LinkedNormalFormRole {
    LinkedNormalFormRole {
        kind: "coordination",
        direction: None,
        principal_slot: None,
        order: None,
        crossed: false,
    }
}

fn linked_pattern(pattern: &PatternIr) -> LinkedPattern {
    match pattern {
        PatternIr::Term(pattern) => LinkedPattern::Term {
            name: pattern.name.clone(),
            sort: pattern.sort.clone(),
        },
        PatternIr::Constant(pattern) => LinkedPattern::Constant {
            name: pattern.name.clone(),
            sort: pattern.sort.clone(),
        },
        PatternIr::Node(pattern) => LinkedPattern::Node {
            operator: match &pattern.operator {
                OperatorPatternIr::Fixed(operator) => {
                    LinkedOperatorPattern::Fixed(operator.name.clone())
                }
                OperatorPatternIr::Variable(operator) => {
                    LinkedOperatorPattern::Variable(operator.name.clone())
                }
            },
            arguments: pattern.arguments.iter().map(linked_pattern).collect(),
            payload: pattern.payload.as_ref().map(|payload| match payload {
                PayloadPatternIr::Fixed(payload) => LinkedPayloadPattern::Fixed(payload.value),
                PayloadPatternIr::Variable(payload) => {
                    LinkedPayloadPattern::Variable(payload.name.clone())
                }
            }),
        },
        PatternIr::Spine(pattern) => LinkedPattern::Spine {
            name: pattern.name.clone(),
            core: Box::new(linked_pattern(&pattern.core)),
            operators: pattern.operators.clone(),
            slot: pattern.slot,
            min_order: pattern.min_order,
            max_order: pattern.max_order,
        },
    }
}

fn inferred_head_slot(
    policy: HeadPolicy,
    premises: &[PatternIr],
    conclusion: &PatternIr,
    default_sort: &str,
) -> Option<u64> {
    if premises.is_empty() {
        return None;
    }
    if premises.len() == 1 || matches!(policy, HeadPolicy::LeftConjunct) {
        return Some(0);
    }
    if matches!(policy, HeadPolicy::NonAbsorbing) {
        return Some(1);
    }
    let mut matching = premises
        .iter()
        .enumerate()
        .filter(|(_, premise)| patterns_equal(premise, conclusion, default_sort))
        .map(|(index, _)| index);
    let slot = matching.next()?;
    if matching.next().is_some() {
        return None;
    }
    u64::try_from(slot).ok()
}

fn patterns_equal(left: &PatternIr, right: &PatternIr, default_sort: &str) -> bool {
    match (left, right) {
        (PatternIr::Term(left), PatternIr::Term(right)) => {
            left.name == right.name
                && left.sort.as_deref().unwrap_or(default_sort)
                    == right.sort.as_deref().unwrap_or(default_sort)
        }
        (PatternIr::Constant(left), PatternIr::Constant(right)) => {
            left.name == right.name
                && left.sort.as_deref().unwrap_or(default_sort)
                    == right.sort.as_deref().unwrap_or(default_sort)
        }
        (PatternIr::Node(left), PatternIr::Node(right)) => {
            left.operator == right.operator
                && left.payload == right.payload
                && left.arguments.len() == right.arguments.len()
                && left
                    .arguments
                    .iter()
                    .zip(&right.arguments)
                    .all(|(left, right)| patterns_equal(left, right, default_sort))
        }
        (PatternIr::Spine(left), PatternIr::Spine(right)) => {
            left.name == right.name
                && left.operators == right.operators
                && left.slot == right.slot
                && left.min_order == right.min_order
                && left.max_order == right.max_order
                && patterns_equal(&left.core, &right.core, default_sort)
        }
        _ => false,
    }
}

fn linked_pattern_recipe(
    name: &str,
    premises: &[PatternIr],
    conclusion: &PatternIr,
    head_policy: HeadPolicy,
    head_slot: Option<u64>,
    targets: Option<&[String]>,
    normal_form_role: LinkedNormalFormRole,
) -> LinkedPatternRecipe {
    LinkedPatternRecipe {
        name: name.to_owned(),
        premises: premises.iter().map(linked_pattern).collect(),
        conclusion: linked_pattern(conclusion),
        head_policy: linked_head_policy(head_policy),
        head_slot,
        targets: targets.map(<[String]>::to_vec),
        normal_form_role,
    }
}

fn opaque_runtime(
    rule: &OpaqueRuleIr,
    location: RulePlanLocation,
) -> Result<OpaqueRuntimeIr, RulePlanError> {
    let value = rule.extension.get("runtime").ok_or_else(|| {
        theory_refusal(
            location,
            format!(
                "late rule {:?} does not declare a native runtime contract",
                rule.rule
            ),
        )
    })?;
    let document = serde_json::to_value(value).map_err(|error| {
        theory_refusal(
            location,
            format!(
                "late rule {:?} has an invalid runtime contract: {error}",
                rule.rule
            ),
        )
    })?;
    serde_json::from_value(document).map_err(|error| {
        theory_refusal(
            location,
            format!(
                "late rule {:?} has an invalid runtime contract: {error}",
                rule.rule
            ),
        )
    })
}

fn opaque_pattern_recipe(
    rule: &OpaqueRuleIr,
    grammar: &GrammarIr,
    linked: &LinkedGrammar,
    location: RulePlanLocation,
) -> Result<LinkedPatternRecipe, RulePlanError> {
    let head_slot = match opaque_runtime(rule, location)? {
        OpaqueRuntimeIr::Pattern {
            head_slot: Present::Value(head_slot),
        } => head_slot,
        OpaqueRuntimeIr::Pattern {
            head_slot: Present::Missing,
        } => {
            return Err(theory_refusal(
                location,
                format!(
                    "late pattern rule {:?} does not declare its physical head slot",
                    rule.rule
                ),
            ));
        }
    };
    let document = serde_json::to_value(&rule.extension).map_err(|error| {
        theory_refusal(
            location,
            format!(
                "late pattern rule {:?} has an invalid runtime recipe: {error}",
                rule.rule
            ),
        )
    })?;
    let recipe: OpaquePatternRecipeIr = serde_json::from_value(document).map_err(|error| {
        theory_refusal(
            location,
            format!(
                "late pattern rule {:?} has an invalid runtime recipe: {error}",
                rule.rule
            ),
        )
    })?;
    validate_patterns(
        recipe
            .premises
            .iter()
            .chain(std::iter::once(&recipe.conclusion)),
        grammar,
        linked,
        location,
    )?;
    validate_pattern_shapes(
        recipe
            .premises
            .iter()
            .chain(std::iter::once(&recipe.conclusion)),
        grammar,
        linked,
        location,
    )?;
    let theory_value = rule.extension.get("theory").ok_or_else(|| {
        theory_refusal(
            location,
            format!(
                "late rule {:?} does not declare a native theory contract",
                rule.rule
            ),
        )
    })?;
    let theory_document = serde_json::to_value(theory_value).map_err(|error| {
        theory_refusal(
            location,
            format!(
                "late rule {:?} has an invalid theory contract: {error}",
                rule.rule
            ),
        )
    })?;
    let theory: OpaqueTheoryIr = serde_json::from_value(theory_document).map_err(|error| {
        theory_refusal(
            location,
            format!(
                "late rule {:?} has an invalid theory contract: {error}",
                rule.rule
            ),
        )
    })?;
    let OpaqueTheoryIr::Schemata { name, schemata, .. } = theory else {
        return Err(theory_refusal(
            location,
            format!(
                "late rule {:?} declares a pattern runtime but only elaborated proof evidence",
                recipe.name
            ),
        ));
    };
    if name != recipe.name
        || schemata.len() != 1
        || schemata[0].premises != recipe.premises
        || schemata[0].conclusion != recipe.conclusion
    {
        return Err(theory_refusal(
            location,
            format!(
                "late pattern rule {:?} must declare one proof schema identical to its runtime relation",
                recipe.name
            ),
        ));
    }
    if head_slot.is_some_and(|slot| {
        usize::try_from(slot).map_or(true, |slot| slot >= recipe.premises.len())
    }) {
        return Err(theory_refusal(
            location,
            format!(
                "late pattern rule {:?} selects a head outside its premises",
                rule.rule
            ),
        ));
    }
    Ok(linked_pattern_recipe(
        &recipe.name,
        &recipe.premises,
        &recipe.conclusion,
        rule.head_policy,
        head_slot,
        recipe.targets.as_deref(),
        linked_normal_form_role(&recipe.normal_form_role),
    ))
}

impl RuleIr {
    fn tag(&self) -> &str {
        match self {
            Self::Application(_) => "application",
            Self::Composition(_) => "composition",
            Self::Substitution(_) => "substitution",
            Self::TypeRaising(_) => "type-raising",
            Self::ClausalAdjunct(_) => "clausal-adjunct",
            Self::Pattern(_) => "pattern",
            Self::Absorption(_) => "absorption",
            Self::Coordination(_) => "coordination",
            Self::Coercion(_) => "coercion",
            Self::Opaque(rule) => &rule.rule,
        }
    }

    fn matching(&self) -> Option<&MatchingIr> {
        match self {
            Self::Application(rule) => rule.matching.as_ref(),
            Self::Composition(rule) => rule.matching.as_ref(),
            Self::Substitution(rule) => rule.matching.as_ref(),
            Self::TypeRaising(rule) => rule.matching.as_ref(),
            Self::ClausalAdjunct(rule) => rule.matching.as_ref(),
            Self::Pattern(rule) => rule.matching.as_ref(),
            Self::Absorption(rule) => rule.matching.as_ref(),
            Self::Coordination(rule) => rule.matching.as_ref(),
            Self::Coercion(rule) => rule.matching.as_ref(),
            Self::Opaque(rule) => rule.matching.as_ref(),
        }
    }

    fn restriction(&self) -> Option<&RuleRestrictionIr> {
        match self {
            Self::Application(rule) => rule.restriction.as_ref(),
            Self::Composition(rule) => rule.restriction.as_ref(),
            Self::Substitution(rule) => rule.restriction.as_ref(),
            Self::TypeRaising(rule) => rule.restriction.as_ref(),
            Self::ClausalAdjunct(rule) => rule.restriction.as_ref(),
            Self::Pattern(rule) => rule.restriction.as_ref(),
            Self::Absorption(rule) => rule.restriction.as_ref(),
            Self::Coordination(rule) => rule.restriction.as_ref(),
            Self::Coercion(rule) => rule.restriction.as_ref(),
            Self::Opaque(rule) => rule.restriction.as_ref(),
        }
    }

    fn introduces(&self) -> Option<&[String]> {
        match self {
            Self::Application(rule) => rule.introduces.as_deref(),
            Self::Composition(rule) => rule.introduces.as_deref(),
            Self::Substitution(rule) => rule.introduces.as_deref(),
            Self::TypeRaising(rule) => rule.introduces.as_deref(),
            Self::ClausalAdjunct(rule) => rule.introduces.as_deref(),
            Self::Pattern(rule) => rule.introduces.as_deref(),
            Self::Absorption(rule) => rule.introduces.as_deref(),
            Self::Coordination(rule) => rule.introduces.as_deref(),
            Self::Coercion(rule) => rule.introduces.as_deref(),
            Self::Opaque(rule) => rule.introduces.as_deref(),
        }
    }

    fn consumes(&self) -> Option<&[String]> {
        match self {
            Self::Application(rule) => rule.consumes.as_deref(),
            Self::Composition(rule) => rule.consumes.as_deref(),
            Self::Substitution(rule) => rule.consumes.as_deref(),
            Self::TypeRaising(rule) => rule.consumes.as_deref(),
            Self::ClausalAdjunct(rule) => rule.consumes.as_deref(),
            Self::Pattern(rule) => rule.consumes.as_deref(),
            Self::Absorption(rule) => rule.consumes.as_deref(),
            Self::Coordination(rule) => rule.consumes.as_deref(),
            Self::Coercion(rule) => rule.consumes.as_deref(),
            Self::Opaque(rule) => rule.consumes.as_deref(),
        }
    }

    fn validate_references(
        &self,
        grammar: &GrammarIr,
        linked: &LinkedGrammar,
        location: RulePlanLocation,
    ) -> Result<(), RulePlanError> {
        match self {
            Self::Application(rule) => {
                functor_id(linked, &rule.view, location)?;
            }
            Self::Composition(rule) => {
                functor_id(linked, &rule.view, location)?;
            }
            Self::Substitution(rule) => {
                functor_id(linked, &rule.view, location)?;
            }
            Self::TypeRaising(rule) => {
                functor_id(linked, &rule.view, location)?;
            }
            Self::ClausalAdjunct(rule) => {
                functor_id(linked, &rule.source_view, location)?;
                functor_id(linked, &rule.target_view, location)?;
            }
            Self::Pattern(rule) => {
                validate_patterns(
                    rule.premises
                        .iter()
                        .chain(std::iter::once(&rule.conclusion)),
                    grammar,
                    linked,
                    location,
                )?;
                validate_pattern_shapes(
                    rule.premises
                        .iter()
                        .chain(std::iter::once(&rule.conclusion)),
                    grammar,
                    linked,
                    location,
                )?;
            }
            Self::Absorption(rule) => {
                validate_patterns(
                    rule.premises
                        .iter()
                        .chain(std::iter::once(&rule.conclusion)),
                    grammar,
                    linked,
                    location,
                )?;
                validate_pattern_shapes(
                    rule.premises
                        .iter()
                        .chain(std::iter::once(&rule.conclusion)),
                    grammar,
                    linked,
                    location,
                )?;
            }
            Self::Coordination(rule) => {
                validate_patterns(
                    rule.premises
                        .iter()
                        .chain(std::iter::once(&rule.conclusion)),
                    grammar,
                    linked,
                    location,
                )?;
                validate_pattern_shapes(
                    rule.premises
                        .iter()
                        .chain(std::iter::once(&rule.conclusion)),
                    grammar,
                    linked,
                    location,
                )?;
            }
            Self::Coercion(_) => {}
            Self::Opaque(rule) => {
                if BUILTIN_RULE_TAGS.contains(&rule.rule.as_str()) {
                    return Err(RulePlanError::located(
                        location,
                        RulePlanErrorKind::MalformedBuiltin {
                            tag: rule.rule.clone(),
                        },
                    ));
                }
            }
        }
        Ok(())
    }

    fn linked_kind(
        &self,
        grammar: &GrammarIr,
        linked: &LinkedGrammar,
        location: RulePlanLocation,
    ) -> Result<LinkedRuleKind, RulePlanError> {
        Ok(match self {
            Self::Application(rule) => LinkedRuleKind::Application {
                functor: functor_id(linked, &rule.view, location)?,
            },
            Self::Composition(rule) => LinkedRuleKind::Composition {
                functor: functor_id(linked, &rule.view, location)?,
                direction: linked_direction(rule.direction),
                order: rule.order,
                crossed: rule.crossed,
            },
            Self::Substitution(rule) => LinkedRuleKind::Substitution {
                functor: functor_id(linked, &rule.view, location)?,
                order: rule.order,
                crossed: rule.crossed,
            },
            Self::TypeRaising(rule) => LinkedRuleKind::TypeRaising {
                functor: functor_id(linked, &rule.view, location)?,
                targets: rule.targets.clone(),
                arguments: rule.arguments.clone(),
            },
            Self::ClausalAdjunct(rule) => LinkedRuleKind::ClausalAdjunct {
                recipe: LinkedClausalAdjunctRecipe {
                    name: rule.name.clone(),
                    source_view: rule.source_view.clone(),
                    target_view: rule.target_view.clone(),
                    source_core: rule.source_core.clone(),
                    argument: rule.argument.clone(),
                },
            },
            Self::Pattern(rule) => LinkedRuleKind::Pattern {
                recipe: linked_pattern_recipe(
                    &rule.name,
                    &rule.premises,
                    &rule.conclusion,
                    rule.head_policy,
                    inferred_head_slot(
                        rule.head_policy,
                        &rule.premises,
                        &rule.conclusion,
                        &grammar.signature.category_sort,
                    ),
                    rule.targets.as_deref(),
                    linked_normal_form_role(&rule.normal_form_role),
                ),
            },
            Self::Absorption(rule) => LinkedRuleKind::Absorption {
                recipe: LinkedAbsorptionRecipe {
                    pattern: linked_pattern_recipe(
                        &rule.name,
                        &rule.premises,
                        &rule.conclusion,
                        rule.head_policy,
                        Some(match rule.side {
                            BinarySide::Left => 1,
                            BinarySide::Right => 0,
                        }),
                        None,
                        other_normal_form_role(),
                    ),
                    absorbing: rule.absorbing.clone(),
                    side: match rule.side {
                        BinarySide::Left => "left",
                        BinarySide::Right => "right",
                    },
                },
            },
            Self::Coordination(rule) => LinkedRuleKind::Coordination {
                recipe: LinkedCoordinationRecipe {
                    pattern: linked_pattern_recipe(
                        &rule.name,
                        &rule.premises,
                        &rule.conclusion,
                        rule.head_policy,
                        inferred_head_slot(
                            rule.head_policy,
                            &rule.premises,
                            &rule.conclusion,
                            &grammar.signature.category_sort,
                        ),
                        rule.targets.as_deref(),
                        coordination_normal_form_role(),
                    ),
                    phase: match rule.phase {
                        CoordinationPhase::Mark => "mark",
                        CoordinationPhase::Merge => "merge",
                    },
                    core_relation: match rule.core_relation {
                        CoreRelation::Same => "same",
                        CoreRelation::Different => "different",
                    },
                    conjunction_feature: rule.conjunction_feature.clone(),
                },
            },
            Self::Coercion(rule) => LinkedRuleKind::Coercion {
                label: rule.label.clone(),
                member: rule.member,
                source: rule.source.clone(),
                target: rule.target.clone(),
            },
            Self::Opaque(rule) => LinkedRuleKind::Extension {
                recipe: opaque_pattern_recipe(rule, grammar, linked, location)?,
            },
        })
    }
}

fn validate_pattern_shapes<'a>(
    patterns: impl Iterator<Item = &'a PatternIr>,
    grammar: &GrammarIr,
    linked: &LinkedGrammar,
    location: RulePlanLocation,
) -> Result<(), RulePlanError> {
    let category_sort = linked
        .ir()
        .sorts()
        .iter()
        .position(|sort| sort == &grammar.signature.category_sort)
        .and_then(|index| u16::try_from(index).ok())
        .ok_or_else(|| {
            RulePlanError::located(
                location,
                RulePlanErrorKind::UnknownSort {
                    name: grammar.signature.category_sort.clone(),
                },
            )
        })?;

    for pattern in patterns {
        validate_pattern_shape(pattern, category_sort, linked, location)?;
    }
    Ok(())
}

fn validate_pattern_shape(
    pattern: &PatternIr,
    expected_sort: SortId,
    linked: &LinkedGrammar,
    location: RulePlanLocation,
) -> Result<(), RulePlanError> {
    let PatternIr::Node(pattern) = pattern else {
        return Ok(());
    };
    let OperatorPatternIr::Fixed(fixed) = &pattern.operator else {
        return Ok(());
    };
    let operator = linked
        .ir()
        .operators()
        .iter()
        .find(|operator| operator.name() == fixed.name)
        .ok_or_else(|| {
            RulePlanError::located(
                location,
                RulePlanErrorKind::UnknownOperator {
                    name: fixed.name.clone(),
                },
            )
        })?;

    if operator.sort() != expected_sort || operator.arguments().len() != pattern.arguments.len() {
        let expected_name = linked
            .ir()
            .sorts()
            .get(usize::from(expected_sort))
            .map_or("<unknown>", String::as_str);
        let received_name = linked
            .ir()
            .sorts()
            .get(usize::from(operator.sort()))
            .map_or("<unknown>", String::as_str);
        return Err(RulePlanError::located(
            location,
            RulePlanErrorKind::InvalidPatternShape {
                reason: format!(
                    "operator {:?} has sort {:?} and arity {} where sort {:?} and arity {} are required",
                    operator.name(),
                    received_name,
                    operator.arguments().len(),
                    expected_name,
                    pattern.arguments.len()
                ),
            },
        ));
    }

    let payload_matches = match operator.payload() {
        LinkedPayloadKind::None => pattern.payload.is_none(),
        LinkedPayloadKind::Constant | LinkedPayloadKind::Index | LinkedPayloadKind::Free => {
            pattern.payload.is_some()
        }
    };
    if !payload_matches {
        return Err(RulePlanError::located(
            location,
            RulePlanErrorKind::InvalidPatternShape {
                reason: format!(
                    "operator {:?} has incompatible pattern payload {:?}",
                    operator.name(),
                    pattern.payload
                ),
            },
        ));
    }

    for (argument, sort) in pattern.arguments.iter().zip(operator.arguments()) {
        validate_pattern_shape(argument, *sort, linked, location)?;
    }
    Ok(())
}

fn validate_patterns<'a>(
    patterns: impl Iterator<Item = &'a PatternIr>,
    grammar: &GrammarIr,
    linked: &LinkedGrammar,
    location: RulePlanLocation,
) -> Result<(), RulePlanError> {
    let mut pending = patterns.collect::<Vec<_>>();
    while let Some(pattern) = pending.pop() {
        match pattern {
            PatternIr::Term(pattern) => {
                let sort = pattern
                    .sort
                    .as_deref()
                    .unwrap_or(&grammar.signature.category_sort);
                if !linked
                    .ir()
                    .sorts()
                    .iter()
                    .any(|candidate| candidate == sort)
                {
                    return Err(RulePlanError::located(
                        location,
                        RulePlanErrorKind::UnknownSort {
                            name: sort.to_owned(),
                        },
                    ));
                }
            }
            PatternIr::Constant(pattern) => {
                let sort = pattern
                    .sort
                    .as_deref()
                    .unwrap_or(&grammar.signature.category_sort);
                let sort_id = linked
                    .ir()
                    .sorts()
                    .iter()
                    .position(|candidate| candidate == sort)
                    .ok_or_else(|| {
                        RulePlanError::located(
                            location,
                            RulePlanErrorKind::UnknownSort {
                                name: sort.to_owned(),
                            },
                        )
                    })?;
                let vocabulary = linked
                    .ir()
                    .vocabularies()
                    .iter()
                    .find(|vocabulary| usize::from(vocabulary.sort()) == sort_id);
                if vocabulary.is_none_or(|vocabulary| !vocabulary.names().contains(&pattern.name)) {
                    return Err(RulePlanError::located(
                        location,
                        RulePlanErrorKind::UnknownConstant {
                            sort: sort.to_owned(),
                            name: pattern.name.clone(),
                        },
                    ));
                }
            }
            PatternIr::Node(pattern) => {
                if let OperatorPatternIr::Fixed(operator) = &pattern.operator {
                    require_linked_operator(linked, &operator.name, location)?;
                }
                pending.extend(pattern.arguments.iter());
            }
            PatternIr::Spine(pattern) => {
                for operator in &pattern.operators {
                    require_linked_operator(linked, operator, location)?;
                }
                pending.push(&pattern.core);
            }
        }
    }
    Ok(())
}

fn require_linked_operator(
    linked: &LinkedGrammar,
    name: &str,
    location: RulePlanLocation,
) -> Result<(), RulePlanError> {
    if linked
        .ir()
        .operators()
        .iter()
        .any(|operator| operator.name() == name)
    {
        Ok(())
    } else {
        Err(RulePlanError::located(
            location,
            RulePlanErrorKind::UnknownOperator {
                name: name.to_owned(),
            },
        ))
    }
}

fn functor_id(
    linked: &LinkedGrammar,
    name: &str,
    location: RulePlanLocation,
) -> Result<u32, RulePlanError> {
    linked
        .ir()
        .category_syntax()
        .functors()
        .iter()
        .position(|functor| functor.name() == name)
        .ok_or_else(|| {
            RulePlanError::located(
                location,
                RulePlanErrorKind::UnknownFunctor {
                    name: name.to_owned(),
                },
            )
        })
        .and_then(|index| {
            u32::try_from(index).map_err(|_| {
                RulePlanError::located(
                    location,
                    RulePlanErrorKind::InventoryOverflow { size: index + 1 },
                )
            })
        })
}

struct RulePlanner<'a> {
    linked: &'a LinkedGrammar,
    tags: Vec<String>,
    rules: Vec<LinkedRule>,
}

fn linked_matching(
    matching: Option<&MatchingIr>,
    location: RulePlanLocation,
) -> Result<Option<String>, RulePlanError> {
    matching
        .map(|value| {
            serde_json::to_string(value).map_err(|error| {
                theory_refusal(
                    location,
                    format!("rule matching recipe cannot be encoded: {error}"),
                )
            })
        })
        .transpose()
}

/// Plan one declared restriction over the premises of the rule it restricts.
///
/// The primary premise of a functor rule is its functor and the secondary its
/// other premise; the primary premise of a unary rule is its one premise, and
/// a binary pattern rule names its principal slot through its composition
/// role. A rule with no such premise, a secondary restriction on a unary rule
/// and a category the signature does not parse are refused.
fn linked_restriction(
    value: Option<&RuleRestrictionIr>,
    kind: &LinkedRuleKind,
    linked: &LinkedGrammar,
    tag: &str,
    location: RulePlanLocation,
) -> Result<Option<LinkedRuleRestriction>, RulePlanError> {
    let Some(restriction) = value else {
        return Ok(None);
    };
    let refuse = |reason: String| {
        RulePlanError::located(
            location,
            RulePlanErrorKind::UnenforceableRestriction { reason },
        )
    };
    let unrestrictable = || {
        refuse(format!(
            "a {tag:?} rule has no primary premise, so its restriction cannot be enforced"
        ))
    };
    let (primary, secondary) = match kind {
        LinkedRuleKind::Application { functor }
        | LinkedRuleKind::Composition { functor, .. }
        | LinkedRuleKind::Substitution { functor, .. } => {
            match functor_or_refusal(linked, *functor, tag, location)?.argument_side() {
                LinkedArgumentSide::Right => (0, Some(1)),
                LinkedArgumentSide::Left => (1, Some(0)),
                LinkedArgumentSide::Either => {
                    return Err(refuse(format!(
                        "a {tag:?} rule's view takes its argument on either side, so it has no \
                         one primary premise and its restriction cannot be enforced"
                    )));
                }
            }
        }
        LinkedRuleKind::TypeRaising { .. } | LinkedRuleKind::Coercion { .. } => (0, None),
        LinkedRuleKind::Pattern { recipe } => match recipe.premises().len() {
            1 => (0, None),
            2 => match recipe.normal_form_role().principal_slot() {
                Some(0) => (0, Some(1)),
                Some(1) => (1, Some(0)),
                _ => return Err(unrestrictable()),
            },
            _ => return Err(unrestrictable()),
        },
        LinkedRuleKind::ClausalAdjunct { .. }
        | LinkedRuleKind::Absorption { .. }
        | LinkedRuleKind::Coordination { .. }
        | LinkedRuleKind::Extension { .. } => return Err(unrestrictable()),
    };
    if secondary.is_none() && !restriction.secondary_patterns.is_empty() {
        return Err(refuse(format!(
            "a {tag:?} rule has no secondary premise, so its secondary patterns cannot \
             be enforced"
        )));
    }
    for category in restriction
        .primary_targets
        .iter()
        .chain(&restriction.secondary_patterns)
    {
        category_pattern(category, linked, tag, location)?;
    }
    Ok(Some(LinkedRuleRestriction::new(
        restriction.primary_targets.clone(),
        restriction.secondary_patterns.clone(),
        primary,
        secondary.filter(|_| !restriction.secondary_patterns.is_empty()),
    )))
}

impl<'a> RulePlanner<'a> {
    const fn new(linked: &'a LinkedGrammar) -> Self {
        Self {
            linked,
            tags: Vec::new(),
            rules: Vec::new(),
        }
    }

    fn push(
        &mut self,
        tag: &str,
        location: RulePlanLocation,
        kind: LinkedRuleKind,
        matching: Option<&MatchingIr>,
        restriction: Option<&RuleRestrictionIr>,
        introduces: Option<&[String]>,
        consumes: Option<&[String]>,
    ) -> Result<(), RulePlanError> {
        let identifier = u32::try_from(self.rules.len()).map_err(|_| {
            RulePlanError::located(
                location,
                RulePlanErrorKind::InventoryOverflow {
                    size: self.rules.len() + 1,
                },
            )
        })?;
        let restriction = linked_restriction(restriction, &kind, self.linked, tag, location)?;
        if !self.tags.iter().any(|held| held == tag) {
            self.tags.push(tag.to_owned());
        }
        self.rules.push(LinkedRule::new(
            identifier,
            tag.to_owned(),
            location,
            kind,
            linked_matching(matching, location)?,
            restriction,
            introduces.map(<[String]>::to_vec),
            consumes.map(<[String]>::to_vec),
        ));
        Ok(())
    }

    fn push_combinators(
        &mut self,
        index: usize,
        declaration: &CombinatorSetIr,
    ) -> Result<(), RulePlanError> {
        let first = self.rules.len();
        let functors = self
            .linked
            .ir()
            .category_syntax()
            .functors()
            .iter()
            .map(|functor| functor.name().to_owned())
            .collect::<Vec<_>>();
        let mut member = 0;
        if declaration.application {
            for view in &functors {
                let location = RulePlanLocation::combinator(index, member);
                let functor = functor_id(self.linked, view, location)?;
                self.push(
                    "application",
                    location,
                    LinkedRuleKind::Application { functor },
                    declaration.matching.as_ref(),
                    declaration.restriction.as_ref(),
                    declaration.introduces.as_deref(),
                    declaration.consumes.as_deref(),
                )?;
                member += 1;
            }
        }
        for (orders, crossed) in [
            (&declaration.composition, false),
            (&declaration.crossed_composition, true),
        ] {
            for view in &functors {
                for order in orders {
                    let location = RulePlanLocation::combinator(index, member);
                    let functor = functor_id(self.linked, view, location)?;
                    self.push(
                        "composition",
                        location,
                        LinkedRuleKind::Composition {
                            functor,
                            direction: linked_direction(declaration.direction),
                            order: *order,
                            crossed,
                        },
                        declaration.matching.as_ref(),
                        declaration.restriction.as_ref(),
                        declaration.introduces.as_deref(),
                        declaration.consumes.as_deref(),
                    )?;
                    member += 1;
                }
            }
        }
        for (orders, crossed) in [
            (&declaration.substitution, false),
            (&declaration.crossed_substitution, true),
        ] {
            for view in &functors {
                for order in orders {
                    let location = RulePlanLocation::combinator(index, member);
                    let functor = functor_id(self.linked, view, location)?;
                    self.push(
                        "substitution",
                        location,
                        LinkedRuleKind::Substitution {
                            functor,
                            order: *order,
                            crossed,
                        },
                        declaration.matching.as_ref(),
                        declaration.restriction.as_ref(),
                        declaration.introduces.as_deref(),
                        declaration.consumes.as_deref(),
                    )?;
                    member += 1;
                }
            }
        }
        if declaration.type_raising {
            for view in &functors {
                let location = RulePlanLocation::combinator(index, member);
                let functor = functor_id(self.linked, view, location)?;
                self.push(
                    "type-raising",
                    location,
                    LinkedRuleKind::TypeRaising {
                        functor,
                        targets: declaration.raising_targets.clone(),
                        arguments: declaration.raising_arguments.clone(),
                    },
                    declaration.matching.as_ref(),
                    declaration.restriction.as_ref(),
                    declaration.introduces.as_deref(),
                    declaration.consumes.as_deref(),
                )?;
                member += 1;
            }
        }
        self.validate_combinator_relations(first)
    }

    fn validate_combinator_relations(&self, first: usize) -> Result<(), RulePlanError> {
        let expanded = &self.rules[first..];
        let mut pairs = Vec::new();
        let mut location = None;

        for (offset, rule) in expanded.iter().enumerate() {
            for earlier in &expanded[..offset] {
                if !generated_relations_coincide(self.linked, earlier.kind(), rule.kind()) {
                    continue;
                }
                location = Some(rule.location());
                pairs.push(format!(
                    "'{}' and '{}' differ only in {}",
                    python_single_quoted(&generated_rule_name(self.linked, earlier.kind())),
                    python_single_quoted(&generated_rule_name(self.linked, rule.kind())),
                    differing_generated_fields(earlier.kind(), rule.kind())
                ));
            }
        }

        location.map_or(Ok(()), |location| {
            Err(RulePlanError::located(
                location,
                RulePlanErrorKind::CoincidingRules { pairs },
            ))
        })
    }

    fn push_coercions(
        &mut self,
        index: usize,
        declaration: &CoercionSetIr,
    ) -> Result<(), RulePlanError> {
        for (member, entry) in declaration.entries.iter().enumerate() {
            let location = RulePlanLocation::coercion(index, member);
            let mut parts = entry.split("=>");
            let source = parts.next().map(str::trim).unwrap_or_default();
            let target = parts.next().map(str::trim).unwrap_or_default();
            if source.is_empty() || target.is_empty() || parts.next().is_some() {
                return Err(RulePlanError::located(
                    location,
                    RulePlanErrorKind::MalformedCoercion {
                        entry: entry.clone(),
                    },
                ));
            }
            let member_id = u64::try_from(member).map_err(|_| {
                RulePlanError::located(
                    location,
                    RulePlanErrorKind::InventoryOverflow { size: member + 1 },
                )
            })?;
            self.push(
                "coercion",
                location,
                LinkedRuleKind::Coercion {
                    label: declaration.label.clone(),
                    member: member_id,
                    source: source.to_owned(),
                    target: target.to_owned(),
                },
                declaration.matching.as_ref(),
                declaration.restriction.as_ref(),
                declaration.introduces.as_deref(),
                declaration.consumes.as_deref(),
            )?;
        }
        Ok(())
    }

    fn validate_relations(
        &self,
        grammar: &GrammarIr,
        relations: &[Value],
    ) -> Result<(), RulePlanError> {
        let mut pairs = Vec::new();
        let mut location = None;

        for (offset, rule) in self.rules.iter().enumerate() {
            for (earlier, relation) in self.rules[..offset].iter().zip(&relations[..offset]) {
                if *relation != relations[offset] {
                    continue;
                }
                location = Some(rule.location());
                pairs.push(format!(
                    "'{}' and '{}' differ only in {}",
                    python_single_quoted(&planned_rule_name(grammar, self.linked, earlier)),
                    python_single_quoted(&planned_rule_name(grammar, self.linked, rule)),
                    differing_declared_fields(grammar, earlier, rule),
                ));
            }
        }

        location.map_or(Ok(()), |location| {
            Err(RulePlanError::located(
                location,
                RulePlanErrorKind::CoincidingRules { pairs },
            ))
        })
    }
}

/// Name the declared fields on which the declarations of two coinciding rules
/// differ, in sorted order, as the Python boundary spells them.
///
/// Members expanded from one combinator set share their declaration, so their
/// generated relation parameters are compared instead.
fn differing_declared_fields(grammar: &GrammarIr, left: &LinkedRule, right: &LinkedRule) -> String {
    let entries = &grammar.rules.rules;
    let left_entry = &entries[left.location().index()];
    let right_entry = &entries[right.location().index()];
    if left.location().index() == right.location().index() {
        return differing_generated_fields(left.kind(), right.kind());
    }
    let (Ok(Value::Object(left_fields)), Ok(Value::Object(right_fields))) = (
        serde_json::to_value(left_entry),
        serde_json::to_value(right_entry),
    ) else {
        return "nothing".to_owned();
    };
    let mut fields = left_fields
        .keys()
        .chain(right_fields.keys())
        .filter(|key| left_fields.get(*key) != right_fields.get(*key))
        .map(|key| format!("'{}'", python_single_quoted(key)))
        .collect::<Vec<_>>();
    fields.sort();
    fields.dedup();
    if fields.is_empty() {
        "nothing".to_owned()
    } else {
        fields.join(", ")
    }
}

fn planned_rule_name(grammar: &GrammarIr, linked: &LinkedGrammar, rule: &LinkedRule) -> String {
    let entry = &grammar.rules.rules[rule.location().index()];
    match entry {
        RuleEntryIr::Rule(declared) => explicit_rule_name(declared),
        RuleEntryIr::Combinators(set) => qualified_name(
            generated_rule_name(linked, rule.kind()),
            set.restriction.as_ref(),
        ),
        RuleEntryIr::Coercions(set) => match rule.kind() {
            LinkedRuleKind::Coercion {
                label,
                member,
                source,
                target,
            } => qualified_name(
                format!("{label}:{member}:{source}=>{target}"),
                set.restriction.as_ref(),
            ),
            _ => "<unknown>".to_owned(),
        },
    }
}

fn effective_matching_value(
    grammar: &GrammarIr,
    rule: &LinkedRule,
) -> Result<Value, serde_json::Error> {
    rule.matching().map_or_else(
        || serde_json::to_value(&grammar.matching),
        serde_json::from_str,
    )
}

fn linked_pattern_value(pattern: &LinkedPattern) -> Value {
    match pattern {
        LinkedPattern::Term { name, sort } => {
            json!({"pattern": "term", "name": name, "sort": sort})
        }
        LinkedPattern::Constant { name, sort } => {
            json!({"pattern": "constant", "name": name, "sort": sort})
        }
        LinkedPattern::Node {
            operator,
            arguments,
            payload,
        } => {
            let operator = match operator {
                LinkedOperatorPattern::Fixed(name) => json!({"kind": "fixed", "name": name}),
                LinkedOperatorPattern::Variable(name) => {
                    json!({"kind": "variable", "name": name})
                }
            };
            let payload = payload.as_ref().map(|payload| match payload {
                LinkedPayloadPattern::Fixed(value) => json!({"kind": "fixed", "value": value}),
                LinkedPayloadPattern::Variable(name) => {
                    json!({"kind": "variable", "name": name})
                }
            });
            json!({
                "pattern": "node",
                "operator": operator,
                "arguments": arguments.iter().map(linked_pattern_value).collect::<Vec<_>>(),
                "payload": payload,
            })
        }
        LinkedPattern::Spine {
            name,
            core,
            operators,
            slot,
            min_order,
            max_order,
        } => json!({
            "pattern": "spine",
            "name": name,
            "core": linked_pattern_value(core),
            "operators": operators,
            "slot": slot,
            "min_order": min_order,
            "max_order": max_order,
        }),
    }
}

fn linked_pattern_relation_value(recipe: &LinkedPatternRecipe) -> Value {
    json!({
        "premises": recipe
            .premises()
            .iter()
            .map(linked_pattern_value)
            .collect::<Vec<_>>(),
        "conclusion": linked_pattern_value(recipe.conclusion()),
        "targets": recipe.targets(),
    })
}

fn linked_normal_form_value(role: &LinkedNormalFormRole) -> Value {
    json!({
        "kind": role.kind(),
        "direction": role.direction(),
        "principal_slot": role.principal_slot(),
        "order": role.order(),
        "crossed": role.crossed(),
    })
}

fn matching_erases_features(matching: &MatchingIr) -> bool {
    match matching {
        MatchingIr::FeatureErasure(_) => true,
        MatchingIr::Product(product) => {
            matching_erases_features(&product.left) || matching_erases_features(&product.right)
        }
        MatchingIr::Identity(_)
        | MatchingIr::Alpha(_)
        | MatchingIr::Subsumption(_)
        | MatchingIr::Modal(_)
        | MatchingIr::Extension(_) => false,
    }
}

fn rule_erases_features(grammar: &GrammarIr, rule: &LinkedRule) -> bool {
    let Some(recipe) = rule.matching() else {
        return matching_erases_features(&grammar.matching);
    };
    serde_json::from_str::<MatchingIr>(recipe)
        .is_ok_and(|matching| matching_erases_features(&matching))
}

fn erased_category_value(
    source: &str,
    linked: &LinkedGrammar,
    location: RulePlanLocation,
    erase: bool,
) -> Result<Value, RulePlanError> {
    let mut pattern = category_pattern(source, linked, "coercion", location)?;
    if erase {
        erase_category_constants(&mut pattern, linked);
    }
    serde_json::to_value(pattern)
        .map_err(|error| theory_refusal(location, format!("category cannot be encoded: {error}")))
}

fn erase_category_constants(pattern: &mut PatternIr, linked: &LinkedGrammar) {
    let PatternIr::Node(node) = pattern else {
        return;
    };
    if let Some(PayloadPatternIr::Fixed(payload)) = &mut node.payload {
        if let Some(vocabulary) =
            linked.ir().vocabularies().iter().find(|vocabulary| {
                vocabulary.sort() == linked.ir().category_syntax().category_sort()
            })
            && let Ok(identifier) = usize::try_from(payload.value)
            && let Some(core) = vocabulary.cores().get(identifier)
            && let Ok(core) = i64::try_from(*core)
        {
            payload.value = core;
        }
    }
    for argument in &mut node.arguments {
        erase_category_constants(argument, linked);
    }
}

fn planned_relation_value(
    grammar: &GrammarIr,
    linked: &LinkedGrammar,
    rule: &LinkedRule,
) -> Result<Value, RulePlanError> {
    let matching = effective_matching_value(grammar, rule).map_err(|error| {
        theory_refusal(
            rule.location(),
            format!("matching relation cannot be encoded: {error}"),
        )
    })?;
    let restriction = rule.restriction().map(|value| {
        json!({
            "primary_targets": value.primary_targets(),
            "secondary_patterns": value.secondary_patterns(),
        })
    });
    let relation = match rule.kind() {
        LinkedRuleKind::Application { functor } => {
            json!([
                "composition",
                functor,
                0,
                composition_spine(
                    linked,
                    *functor,
                    LinkedCompositionDirection::Unrestricted,
                    Some(0),
                    false,
                )
            ])
        }
        LinkedRuleKind::Composition {
            functor,
            direction,
            order,
            crossed,
        } => json!([
            "composition",
            functor,
            order,
            composition_spine(linked, *functor, *direction, *order, *crossed)
        ]),
        LinkedRuleKind::Substitution {
            functor,
            order,
            crossed,
        } => json!(["substitution", functor, order, crossed]),
        LinkedRuleKind::TypeRaising {
            functor,
            targets,
            arguments,
        } => {
            json!(["type-raising", functor, targets, arguments])
        }
        LinkedRuleKind::ClausalAdjunct { recipe } => json!([
            "clausal-adjunct",
            {
                "source_view": recipe.source_view(),
                "target_view": recipe.target_view(),
                "source_core": recipe.source_core(),
                "argument": recipe.argument(),
            }
        ]),
        LinkedRuleKind::Pattern { recipe } => {
            json!(["pattern", linked_pattern_relation_value(recipe)])
        }
        LinkedRuleKind::Absorption { recipe } => json!([
            "absorption",
            {
                "pattern": linked_pattern_relation_value(recipe.pattern()),
                "absorbing": recipe.absorbing(),
                "side": recipe.side(),
            }
        ]),
        LinkedRuleKind::Coordination { recipe } => json!([
            "coordination",
            {
                "pattern": linked_pattern_relation_value(recipe.pattern()),
                "phase": recipe.phase(),
                "core_relation": recipe.core_relation(),
                "conjunction_feature": recipe.conjunction_feature(),
            }
        ]),
        LinkedRuleKind::Extension { recipe } => {
            json!(["extension", linked_pattern_relation_value(recipe)])
        }
        LinkedRuleKind::Coercion { source, target, .. } => {
            let erase = rule_erases_features(grammar, rule);
            json!([
                "coercion",
                erased_category_value(source, linked, rule.location(), erase)?,
                erased_category_value(target, linked, rule.location(), erase)?,
            ])
        }
    };
    let (head_policy, head_slot, normal_form_role) = match rule.kind() {
        LinkedRuleKind::Application { .. }
        | LinkedRuleKind::Composition { .. }
        | LinkedRuleKind::Substitution { .. } => ("functor", None, None),
        LinkedRuleKind::TypeRaising { .. }
        | LinkedRuleKind::ClausalAdjunct { .. }
        | LinkedRuleKind::Coercion { .. } => ("pass-through", Some(0), None),
        LinkedRuleKind::Pattern { recipe } | LinkedRuleKind::Extension { recipe } => (
            recipe.head_policy(),
            recipe.head_slot(),
            Some(linked_normal_form_value(recipe.normal_form_role())),
        ),
        LinkedRuleKind::Absorption { recipe } => (
            recipe.pattern().head_policy(),
            recipe.pattern().head_slot(),
            Some(linked_normal_form_value(
                recipe.pattern().normal_form_role(),
            )),
        ),
        LinkedRuleKind::Coordination { recipe } => (
            recipe.pattern().head_policy(),
            recipe.pattern().head_slot(),
            Some(linked_normal_form_value(
                recipe.pattern().normal_form_role(),
            )),
        ),
    };
    Ok(json!({
        "plugin": rule.tag(),
        "relation": relation,
        "matching": matching,
        "restriction": restriction,
        "head_policy": head_policy,
        "head_slot": head_slot,
        "normal_form_role": normal_form_role,
        "introduces": rule.introduces(),
        "consumes": rule.consumes(),
    }))
}

fn python_single_quoted(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\'', "\\'")
}

fn generated_relations_coincide(
    linked: &LinkedGrammar,
    left: &LinkedRuleKind,
    right: &LinkedRuleKind,
) -> bool {
    match (left, right) {
        (
            LinkedRuleKind::Application { functor: left },
            LinkedRuleKind::Application { functor: right },
        ) => left == right,
        (
            LinkedRuleKind::Composition {
                functor: left_functor,
                direction: left_direction,
                order: left_order,
                crossed: left_crossed,
            },
            LinkedRuleKind::Composition {
                functor: right_functor,
                direction: right_direction,
                order: right_order,
                crossed: right_crossed,
            },
        ) => {
            left_functor == right_functor
                && left_order == right_order
                && composition_spine(
                    linked,
                    *left_functor,
                    *left_direction,
                    *left_order,
                    *left_crossed,
                ) == composition_spine(
                    linked,
                    *right_functor,
                    *right_direction,
                    *right_order,
                    *right_crossed,
                )
        }
        (
            LinkedRuleKind::Substitution {
                functor: left_functor,
                order: left_order,
                crossed: left_crossed,
            },
            LinkedRuleKind::Substitution {
                functor: right_functor,
                order: right_order,
                crossed: right_crossed,
            },
        ) => {
            left_functor == right_functor
                && left_order == right_order
                && left_crossed == right_crossed
        }
        (
            LinkedRuleKind::TypeRaising {
                functor: left_functor,
                targets: left_targets,
                arguments: left_arguments,
            },
            LinkedRuleKind::TypeRaising {
                functor: right_functor,
                targets: right_targets,
                arguments: right_arguments,
            },
        ) => {
            left_functor == right_functor
                && left_targets == right_targets
                && left_arguments == right_arguments
        }
        _ => false,
    }
}

fn composition_spine(
    linked: &LinkedGrammar,
    functor: u32,
    direction: LinkedCompositionDirection,
    order: Option<u64>,
    crossed: bool,
) -> Vec<(u32, Vec<u32>)> {
    let views = linked.ir().category_syntax().functors();
    let Some(view) = usize::try_from(functor)
        .ok()
        .and_then(|index| views.get(index))
    else {
        return Vec::new();
    };
    let opposite = views
        .iter()
        .position(|candidate| candidate.name() == view.opposite())
        .and_then(|index| u32::try_from(index).ok())
        .unwrap_or(functor);
    let mut both = vec![functor, opposite];
    both.sort_unstable();
    both.dedup();
    let deepest = match direction {
        LinkedCompositionDirection::Unrestricted => both.clone(),
        LinkedCompositionDirection::Steedman | LinkedCompositionDirection::Kuhlmann => {
            vec![if crossed { opposite } else { functor }]
        }
    };

    deepest
        .into_iter()
        .map(|slash| {
            let prefixes = if order == Some(1) {
                Vec::new()
            } else if direction == LinkedCompositionDirection::Steedman {
                vec![slash]
            } else {
                both.clone()
            };
            (slash, prefixes)
        })
        .collect()
}

fn generated_rule_name(linked: &LinkedGrammar, kind: &LinkedRuleKind) -> String {
    let view = |identifier: u32| {
        usize::try_from(identifier)
            .ok()
            .and_then(|index| linked.ir().category_syntax().functors().get(index))
            .map_or("<unknown>", LinkedFunctor::name)
    };
    let order =
        |value: Option<u64>| value.map_or_else(|| "any".to_owned(), |held| held.to_string());

    match kind {
        LinkedRuleKind::Application { functor } => format!("application:{}", view(*functor)),
        LinkedRuleKind::Composition {
            functor,
            direction,
            order: held_order,
            crossed,
        } => format!(
            "composition:{}:{}:{}:{}",
            view(*functor),
            order(*held_order),
            if *crossed { "crossed" } else { "harmonic" },
            format!("{direction:?}").to_lowercase()
        ),
        LinkedRuleKind::Substitution {
            functor,
            order: held_order,
            crossed,
        } => format!(
            "substitution:{}:{}:{}",
            view(*functor),
            order(*held_order),
            if *crossed { "crossed" } else { "harmonic" }
        ),
        LinkedRuleKind::TypeRaising {
            functor,
            targets,
            arguments,
        } => raising_name(&view(*functor), targets.as_deref(), arguments.as_deref()),
        LinkedRuleKind::ClausalAdjunct { .. }
        | LinkedRuleKind::Pattern { .. }
        | LinkedRuleKind::Absorption { .. }
        | LinkedRuleKind::Coordination { .. }
        | LinkedRuleKind::Extension { .. }
        | LinkedRuleKind::Coercion { .. } => "<explicit>".to_owned(),
    }
}

fn differing_generated_fields(left: &LinkedRuleKind, right: &LinkedRuleKind) -> String {
    let mut fields = Vec::new();

    if let (
        LinkedRuleKind::Composition {
            direction: left_direction,
            crossed: left_crossed,
            ..
        },
        LinkedRuleKind::Composition {
            direction: right_direction,
            crossed: right_crossed,
            ..
        },
    ) = (left, right)
    {
        if left_crossed != right_crossed {
            fields.push("'crossed'");
        }
        if left_direction != right_direction {
            fields.push("'direction'");
        }
    }

    if fields.is_empty() {
        "nothing".to_owned()
    } else {
        fields.join(", ")
    }
}

const fn linked_direction(direction: CompositionRestriction) -> LinkedCompositionDirection {
    match direction {
        CompositionRestriction::Steedman => LinkedCompositionDirection::Steedman,
        CompositionRestriction::Kuhlmann => LinkedCompositionDirection::Kuhlmann,
        CompositionRestriction::Unrestricted => LinkedCompositionDirection::Unrestricted,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TheoryVariableSort {
    Term { sort: SortId, binders: usize },
    Payload(LinkedPayloadKind),
}

fn theory_refusal(location: RulePlanLocation, reason: impl Into<String>) -> RulePlanError {
    RulePlanError::located(
        location,
        RulePlanErrorKind::InvalidPatternShape {
            reason: reason.into(),
        },
    )
}

fn encode_identifier(name: &str) -> String {
    let mut encoded = String::new();
    for character in name.chars() {
        if character.is_ascii_alphanumeric() {
            encoded.push(character);
        } else {
            encoded.push('_');
            encoded.push_str(&format!("{:x}", u32::from(character)));
            encoded.push('_');
        }
    }
    if encoded
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_digit())
    {
        let first = name.chars().next().unwrap_or_default();
        encoded.replace_range(..first.len_utf8(), &format!("_{:x}_", u32::from(first)));
    }
    encoded
}

/// Name a type-raising rule by its view, its targets and any declared arguments.
fn raising_name(view: &str, targets: Option<&[String]>, arguments: Option<&[String]>) -> String {
    let targets = targets.map_or_else(|| "*".to_owned(), |values| values.join(","));
    arguments.map_or_else(
        || format!("type-raising:{view}:targets={targets}"),
        |values| {
            format!(
                "type-raising:{view}:targets={targets}:arguments={}",
                values.join(",")
            )
        },
    )
}

fn qualified_name(name: String, restriction: Option<&RuleRestrictionIr>) -> String {
    let Some(restriction) = restriction else {
        return name;
    };
    let mut fields = Vec::new();
    for (label, values) in [
        ("targets", &restriction.primary_targets),
        ("patterns", &restriction.secondary_patterns),
    ] {
        if values.is_empty() {
            continue;
        }
        let mut sorted = values.clone();
        sorted.sort();
        fields.push(format!("{label}={}", sorted.join(",")));
    }
    format!("{name}:only({})", fields.join(";"))
}

fn explicit_rule_name(rule: &RuleIr) -> String {
    match rule {
        RuleIr::Application(rule) => qualified_name(
            format!("application:{}", rule.view),
            rule.restriction.as_ref(),
        ),
        RuleIr::Composition(rule) => qualified_name(
            format!(
                "composition:{}:{}:{}:{}",
                rule.view,
                rule.order
                    .map_or_else(|| "any".to_owned(), |order| order.to_string()),
                if rule.crossed { "crossed" } else { "harmonic" },
                format!("{:?}", rule.direction).to_lowercase()
            ),
            rule.restriction.as_ref(),
        ),
        RuleIr::Substitution(rule) => qualified_name(
            format!(
                "substitution:{}:{}:{}",
                rule.view,
                rule.order
                    .map_or_else(|| "any".to_owned(), |order| order.to_string()),
                if rule.crossed { "crossed" } else { "harmonic" }
            ),
            rule.restriction.as_ref(),
        ),
        RuleIr::TypeRaising(rule) => qualified_name(
            raising_name(
                &rule.view,
                rule.targets.as_deref(),
                rule.arguments.as_deref(),
            ),
            rule.restriction.as_ref(),
        ),
        RuleIr::ClausalAdjunct(rule) => rule.name.clone(),
        RuleIr::Pattern(rule) => rule.name.clone(),
        RuleIr::Absorption(rule) => rule.name.clone(),
        RuleIr::Coordination(rule) => rule.name.clone(),
        RuleIr::Coercion(rule) => format!(
            "{}:{}:{}=>{}",
            rule.label, rule.member, rule.source, rule.target
        ),
        RuleIr::Opaque(rule) => rule
            .extension
            .get("theory")
            .and_then(|value| value.get("name"))
            .and_then(OpaqueJsonValue::as_str)
            .unwrap_or(&rule.rule)
            .to_owned(),
    }
}

fn theory_variable(
    variables: &mut IndexMap<String, TheoryVariableSort>,
    name: String,
    sort: TheoryVariableSort,
    rule: &str,
    location: RulePlanLocation,
) -> Result<(), RulePlanError> {
    if let Some(held) = variables.get(&name) {
        if *held != sort {
            return Err(theory_refusal(
                location,
                format!("rule {rule:?} uses index variable {name:?} at incompatible sorts"),
            ));
        }
    } else {
        variables.insert(name, sort);
    }
    Ok(())
}

fn theory_pattern(
    pattern: &PatternIr,
    expected_sort: SortId,
    binders: usize,
    linked: &LinkedGrammar,
    variables: &mut IndexMap<String, TheoryVariableSort>,
    rule: &str,
    location: RulePlanLocation,
) -> Result<Value, RulePlanError> {
    match pattern {
        PatternIr::Term(pattern) => {
            if let Some(sort) = &pattern.sort {
                let expected = linked
                    .ir()
                    .sorts()
                    .get(usize::from(expected_sort))
                    .map_or("<unknown>", String::as_str);
                if sort != expected {
                    return Err(theory_refusal(
                        location,
                        format!(
                            "rule {rule:?} uses index variable {:?} at sort {sort:?}, not {expected:?}",
                            pattern.name
                        ),
                    ));
                }
            }
            let name = format!("index_{}", encode_identifier(&pattern.name));
            theory_variable(
                variables,
                name.clone(),
                TheoryVariableSort::Term {
                    sort: expected_sort,
                    binders,
                },
                rule,
                location,
            )?;
            Ok(json!({
                "kind": "variable",
                "name": name,
                "sort": linked.ir().sorts().get(usize::from(expected_sort)),
                "binders": binders,
            }))
        }
        PatternIr::Constant(pattern) => {
            let expected = linked
                .ir()
                .sorts()
                .get(usize::from(expected_sort))
                .map_or("<unknown>", String::as_str);
            if pattern.sort.as_deref().is_some_and(|sort| sort != expected) {
                return Err(theory_refusal(
                    location,
                    format!(
                        "rule {rule:?} uses constant {:?} at the wrong sort",
                        pattern.name
                    ),
                ));
            }
            let identifier = linked
                .ir()
                .vocabularies()
                .iter()
                .find(|vocabulary| vocabulary.sort() == expected_sort)
                .and_then(|vocabulary| {
                    vocabulary
                        .names()
                        .iter()
                        .position(|name| name == &pattern.name)
                });
            let Some(identifier) = identifier else {
                return Err(theory_refusal(
                    location,
                    format!(
                        "rule {rule:?} names undeclared constant {:?} at sort {expected:?}",
                        pattern.name
                    ),
                ));
            };
            Ok(json!({
                "kind": "node",
                "operator": format!("#const[{expected}]"),
                "sort": expected,
                "binders": binders,
                "payload": {"kind": "fixed", "value": identifier},
                "arguments": [],
            }))
        }
        PatternIr::Spine(pattern) => Err(theory_refusal(
            location,
            format!(
                "rule {rule:?} has nondeterministic spine context {:?}",
                pattern.name
            ),
        )),
        PatternIr::Node(pattern) => {
            let OperatorPatternIr::Fixed(fixed) = &pattern.operator else {
                let OperatorPatternIr::Variable(variable) = &pattern.operator else {
                    unreachable!();
                };
                return Err(theory_refusal(
                    location,
                    format!(
                        "rule {rule:?} has nondeterministic operator variable {:?}",
                        variable.name
                    ),
                ));
            };
            let operator = linked
                .ir()
                .operators()
                .iter()
                .find(|operator| operator.name() == fixed.name)
                .ok_or_else(|| {
                    theory_refusal(
                        location,
                        format!("rule {rule:?} names unknown operator {:?}", fixed.name),
                    )
                })?;
            if operator.sort() != expected_sort
                || operator.arguments().len() != pattern.arguments.len()
            {
                return Err(theory_refusal(
                    location,
                    format!(
                        "rule {rule:?} applies operator {:?} at the wrong sort or arity",
                        fixed.name
                    ),
                ));
            }
            let payload = match (&pattern.payload, operator.payload()) {
                (None, LinkedPayloadKind::None) => Value::Null,
                (Some(PayloadPatternIr::Fixed(value)), LinkedPayloadKind::Constant)
                | (Some(PayloadPatternIr::Fixed(value)), LinkedPayloadKind::Index)
                | (Some(PayloadPatternIr::Fixed(value)), LinkedPayloadKind::Free) => {
                    json!({"kind": "fixed", "value": value.value})
                }
                (
                    Some(PayloadPatternIr::Variable(variable)),
                    payload @ (LinkedPayloadKind::Constant
                    | LinkedPayloadKind::Index
                    | LinkedPayloadKind::Free),
                ) => {
                    theory_variable(
                        variables,
                        format!("payload_{}", encode_identifier(&variable.name)),
                        TheoryVariableSort::Payload(payload),
                        rule,
                        location,
                    )?;
                    json!({
                        "kind": "variable",
                        "name": format!("payload_{}", encode_identifier(&variable.name)),
                        "payload": format!("{payload:?}").to_lowercase(),
                    })
                }
                (None, _) => {
                    return Err(theory_refusal(
                        location,
                        format!(
                            "rule {rule:?} omits the payload required by operator {:?}",
                            fixed.name
                        ),
                    ));
                }
                (Some(_), LinkedPayloadKind::None) => {
                    return Err(theory_refusal(
                        location,
                        format!(
                            "rule {rule:?} gives operator {:?} a payload it does not accept",
                            fixed.name
                        ),
                    ));
                }
            };
            let mut arguments = Vec::with_capacity(pattern.arguments.len());
            for (position, (argument, sort)) in pattern
                .arguments
                .iter()
                .zip(operator.arguments())
                .enumerate()
            {
                let scoped = operator
                    .binding()
                    .iter()
                    .filter(|site| site.arguments().contains(&position))
                    .count();
                arguments.push(theory_pattern(
                    argument, *sort, scoped, linked, variables, rule, location,
                )?);
            }
            Ok(json!({
                "kind": "node",
                "operator": fixed.name,
                "sort": linked.ir().sorts().get(usize::from(expected_sort)),
                "binders": binders,
                "payload": payload,
                "arguments": arguments,
            }))
        }
    }
}

#[derive(Clone)]
struct TheoryFrame {
    operator: String,
    slot: usize,
    arguments: Vec<PatternIr>,
}

type TheoryContext = Vec<TheoryFrame>;

fn collect_theory_spines(
    pattern: &PatternIr,
    occurrences: &mut IndexMap<String, Vec<SpinePatternIr>>,
) {
    match pattern {
        PatternIr::Spine(spine) => {
            occurrences
                .entry(spine.name.clone())
                .or_default()
                .push(spine.clone());
            collect_theory_spines(&spine.core, occurrences);
        }
        PatternIr::Node(node) => {
            for argument in &node.arguments {
                collect_theory_spines(argument, occurrences);
            }
        }
        PatternIr::Term(_) | PatternIr::Constant(_) => {}
    }
}

fn theory_pattern_is_deterministic(pattern: &PatternIr) -> bool {
    match pattern {
        PatternIr::Term(_) | PatternIr::Constant(_) => true,
        PatternIr::Spine(_) => false,
        PatternIr::Node(node) => {
            matches!(node.operator, OperatorPatternIr::Fixed(_))
                && !matches!(node.payload, Some(PayloadPatternIr::Variable(_)))
                && node.arguments.iter().all(theory_pattern_is_deterministic)
        }
    }
}

fn repeat_theory_frames(
    operators: &[TheoryFrame],
    order: usize,
    built: &mut TheoryContext,
    contexts: &mut Vec<TheoryContext>,
) {
    if built.len() == order {
        contexts.push(built.clone());
        return;
    }
    for operator in operators {
        built.push(operator.clone());
        repeat_theory_frames(operators, order, built, contexts);
        built.pop();
    }
}

fn theory_contexts(
    occurrences: &[SpinePatternIr],
    linked: &LinkedGrammar,
) -> Option<Vec<TheoryContext>> {
    if occurrences
        .iter()
        .any(|occurrence| occurrence.max_order.is_none())
    {
        return None;
    }
    let first = occurrences.first()?;
    let low = occurrences
        .iter()
        .map(|occurrence| occurrence.min_order)
        .max()
        .unwrap_or(0);
    let mut high = occurrences
        .iter()
        .filter_map(|occurrence| occurrence.max_order)
        .min()
        .unwrap_or(0);
    if occurrences
        .iter()
        .any(|occurrence| occurrence.slot != first.slot)
    {
        high = high.min(0);
    }

    let slot = usize::try_from(first.slot).ok()?;
    let mut frames = Vec::new();
    for name in &first.operators {
        if frames
            .iter()
            .any(|frame: &TheoryFrame| frame.operator == *name)
            || occurrences
                .iter()
                .any(|occurrence| !occurrence.operators.contains(name))
        {
            continue;
        }
        let Some(operator) = linked
            .ir()
            .operators()
            .iter()
            .find(|operator| operator.name() == name)
        else {
            continue;
        };
        if slot >= operator.arguments().len() {
            continue;
        }
        let arguments = operator
            .arguments()
            .iter()
            .enumerate()
            .filter(|(position, _)| *position != slot)
            .map(|(position, sort)| {
                PatternIr::Term(TermPatternIr {
                    pattern: TermPatternKind::Value,
                    name: format!("{}#{{depth}}.{position}", first.name),
                    sort: linked.ir().sorts().get(usize::from(*sort)).cloned(),
                })
            })
            .collect();
        frames.push(TheoryFrame {
            operator: name.clone(),
            slot,
            arguments,
        });
    }

    let mut contexts = Vec::new();
    for order in low..=high {
        let Ok(order) = usize::try_from(order) else {
            return Some(Vec::new());
        };
        repeat_theory_frames(&frames, order, &mut Vec::new(), &mut contexts);
    }
    Some(contexts)
}

fn instantiate_theory_pattern(
    pattern: &PatternIr,
    contexts: &IndexMap<String, TheoryContext>,
) -> PatternIr {
    match pattern {
        PatternIr::Term(_) | PatternIr::Constant(_) => pattern.clone(),
        PatternIr::Node(node) => PatternIr::Node(NodePatternIr {
            pattern: node.pattern,
            operator: node.operator.clone(),
            arguments: node
                .arguments
                .iter()
                .map(|argument| instantiate_theory_pattern(argument, contexts))
                .collect(),
            payload: node.payload.clone(),
        }),
        PatternIr::Spine(spine) => {
            let mut built = instantiate_theory_pattern(&spine.core, contexts);
            for (depth, frame) in contexts[&spine.name].iter().enumerate().rev() {
                let arguments = (0..=frame.arguments.len())
                    .map(|position| {
                        if position == frame.slot {
                            built.clone()
                        } else {
                            let sibling = position - usize::from(position > frame.slot);
                            let mut sibling = frame.arguments[sibling].clone();
                            if let PatternIr::Term(term) = &mut sibling {
                                term.name = term.name.replace("{depth}", &depth.to_string());
                            }
                            sibling
                        }
                    })
                    .collect();
                built = PatternIr::Node(NodePatternIr {
                    pattern: NodePatternKind::Value,
                    operator: OperatorPatternIr::Fixed(FixedOperatorPatternIr {
                        kind: FixedKind::Value,
                        name: frame.operator.clone(),
                    }),
                    arguments,
                    payload: None,
                });
            }
            built
        }
    }
}

fn theory_schemata(
    premises: &[PatternIr],
    conclusion: &PatternIr,
    linked: &LinkedGrammar,
) -> Option<Vec<(Vec<PatternIr>, PatternIr)>> {
    let mut occurrences = IndexMap::new();
    for pattern in premises.iter().chain(std::iter::once(conclusion)) {
        collect_theory_spines(pattern, &mut occurrences);
    }
    let mut choices = Vec::new();
    for (name, held) in occurrences {
        let contexts = theory_contexts(&held, linked)?;
        choices.push((name, contexts));
    }

    let mut environments = vec![IndexMap::new()];
    for (name, contexts) in choices {
        let mut expanded = Vec::new();
        for environment in &environments {
            for context in &contexts {
                let mut chosen = environment.clone();
                chosen.insert(name.clone(), context.clone());
                expanded.push(chosen);
            }
        }
        environments = expanded;
    }
    let schemata = environments
        .into_iter()
        .map(|contexts| {
            (
                premises
                    .iter()
                    .map(|pattern| instantiate_theory_pattern(pattern, &contexts))
                    .collect::<Vec<_>>(),
                instantiate_theory_pattern(conclusion, &contexts),
            )
        })
        .collect::<Vec<_>>();
    schemata
        .iter()
        .all(|(premises, conclusion)| {
            premises.iter().all(theory_pattern_is_deterministic)
                && theory_pattern_is_deterministic(conclusion)
        })
        .then_some(schemata)
}

fn theory_constructor(
    rule: &str,
    serial: usize,
    premises: &[PatternIr],
    conclusion: &PatternIr,
    forward_complete: bool,
    category_sort: SortId,
    linked: &LinkedGrammar,
    location: RulePlanLocation,
) -> Result<LinkedTheoryConstructor, RulePlanError> {
    let mut variables = IndexMap::new();
    let mut premise_indices = Vec::with_capacity(premises.len());
    for premise in premises {
        let expression = theory_pattern(
            premise,
            category_sort,
            0,
            linked,
            &mut variables,
            rule,
            location,
        )?;
        premise_indices.push(theory_expression(expression, location)?);
    }
    let inferred = variables.keys().cloned().collect::<Vec<_>>();
    let conclusion_index = theory_expression(
        theory_pattern(
            conclusion,
            category_sort,
            0,
            linked,
            &mut variables,
            rule,
            location,
        )?,
        location,
    )?;
    let declared = variables
        .keys()
        .filter(|name| !inferred.contains(name))
        .cloned()
        .collect::<Vec<_>>();
    if forward_complete && !declared.is_empty() {
        return Err(theory_refusal(
            location,
            format!(
                "rule {rule:?} leaves result index variables {} free",
                declared.join(", ")
            ),
        ));
    }
    Ok(LinkedTheoryConstructor::new(
        rule.to_owned(),
        format!("rule_{serial}_{}", encode_identifier(rule)),
        inferred,
        declared,
        premises.len(),
        premise_indices,
        conclusion_index,
    ))
}

fn theory_expression(value: Value, location: RulePlanLocation) -> Result<String, RulePlanError> {
    let bytes = canonical::value_bytes(&value).map_err(|error| {
        theory_refusal(location, format!("theory index cannot be encoded: {error}"))
    })?;
    String::from_utf8(bytes).map_err(|error| {
        theory_refusal(
            location,
            format!("theory index is not valid UTF-8: {error}"),
        )
    })
}

fn pattern_theory(
    name: &str,
    premises: &[PatternIr],
    conclusion: &PatternIr,
    targets: Option<&[String]>,
    forward_complete: bool,
    category_sort: SortId,
    linked: &LinkedGrammar,
    location: RulePlanLocation,
    constructors: &mut Vec<LinkedTheoryConstructor>,
    elaborated: &mut Vec<String>,
) -> Result<(), RulePlanError> {
    let Some(schemata) = theory_schemata(premises, conclusion, linked) else {
        elaborated.push(name.to_owned());
        return Ok(());
    };
    if schemata.is_empty() {
        elaborated.push(name.to_owned());
        return Ok(());
    }

    let mut serial = 0;
    for (premises, conclusion) in schemata {
        if let Some(targets) = targets {
            let template = theory_constructor(
                name,
                serial,
                &premises,
                &conclusion,
                false,
                category_sort,
                linked,
                location,
            )?;
            for target in targets {
                constructors.push(LinkedTheoryConstructor::new(
                    name.to_owned(),
                    format!("rule_{serial}_{}", encode_identifier(name)),
                    template.inferred().to_vec(),
                    Vec::new(),
                    premises.len(),
                    template.premise_indices().to_vec(),
                    theory_expression(
                        json!({
                            "template": template.conclusion_index(),
                            "target": target,
                        }),
                        location,
                    )?,
                ));
                serial += 1;
            }
        } else {
            constructors.push(theory_constructor(
                name,
                serial,
                &premises,
                &conclusion,
                forward_complete,
                category_sort,
                linked,
                location,
            )?);
            serial += 1;
        }
    }
    Ok(())
}

fn functor_or_refusal<'a>(
    linked: &'a LinkedGrammar,
    identifier: u32,
    rule: &str,
    location: RulePlanLocation,
) -> Result<&'a LinkedFunctor, RulePlanError> {
    usize::try_from(identifier)
        .ok()
        .and_then(|index| linked.ir().category_syntax().functors().get(index))
        .ok_or_else(|| {
            theory_refusal(
                location,
                format!("rule {rule:?} lost its linked functor view"),
            )
        })
}

fn checked_schema_power(
    base: usize,
    exponent: u64,
    rule: &str,
    location: RulePlanLocation,
) -> Result<usize, RulePlanError> {
    (0..exponent).try_fold(1_usize, |count, _| {
        count.checked_mul(base).ok_or_else(|| {
            theory_refusal(
                location,
                format!("rule {rule:?} expands to too many proof schemata"),
            )
        })
    })
}

fn theory_term(name: impl Into<String>) -> PatternIr {
    PatternIr::Term(TermPatternIr {
        pattern: TermPatternKind::Value,
        name: name.into(),
        sort: None,
    })
}

fn theory_functor(
    linked: &LinkedGrammar,
    slash: &str,
    result: PatternIr,
    argument: PatternIr,
    rule: &str,
    location: RulePlanLocation,
) -> Result<PatternIr, RulePlanError> {
    let view = linked
        .ir()
        .category_syntax()
        .functors()
        .iter()
        .find(|view| view.name() == slash)
        .ok_or_else(|| {
            theory_refusal(
                location,
                format!("rule {rule:?} lost functor view {slash:?}"),
            )
        })?;
    let operator = linked
        .ir()
        .operators()
        .iter()
        .find(|operator| operator.operator() == view.constructor())
        .ok_or_else(|| {
            theory_refusal(
                location,
                format!("rule {rule:?} lost the operator for functor view {slash:?}"),
            )
        })?;
    let arguments = if view.result() == 0 {
        vec![result, argument]
    } else {
        vec![argument, result]
    };
    Ok(PatternIr::Node(NodePatternIr {
        pattern: NodePatternKind::Value,
        operator: OperatorPatternIr::Fixed(FixedOperatorPatternIr {
            kind: FixedKind::Value,
            name: operator.name().to_owned(),
        }),
        arguments,
        payload: None,
    }))
}

fn theory_spine(
    linked: &LinkedGrammar,
    slashes: &[String],
    foot: PatternIr,
    arguments: &[PatternIr],
    rule: &str,
    location: RulePlanLocation,
) -> Result<PatternIr, RulePlanError> {
    slashes
        .iter()
        .zip(arguments)
        .rev()
        .try_fold(foot, |held, (slash, argument)| {
            theory_functor(linked, slash, held, argument.clone(), rule, location)
        })
}

fn slash_sequences(prefixes: &[String], deepest: &str, order: u64) -> Vec<Vec<String>> {
    if order == 0 {
        return vec![Vec::new()];
    }
    let mut sequences = vec![Vec::new()];
    for _ in 1..order {
        sequences = sequences
            .into_iter()
            .flat_map(|prefix| {
                prefixes.iter().map(move |slash| {
                    let mut extended = prefix.clone();
                    extended.push(slash.clone());
                    extended
                })
            })
            .collect();
    }
    for sequence in &mut sequences {
        sequence.push(deepest.to_owned());
    }
    sequences
}

fn theory_slashes(view: &LinkedFunctor) -> Vec<String> {
    let mut slashes = vec![view.name().to_owned(), view.opposite().to_owned()];
    slashes.sort();
    slashes.dedup();
    slashes
}

fn fixed_theory_node(name: impl Into<String>, arguments: Vec<PatternIr>) -> PatternIr {
    PatternIr::Node(NodePatternIr {
        pattern: NodePatternKind::Value,
        operator: OperatorPatternIr::Fixed(FixedOperatorPatternIr {
            kind: FixedKind::Value,
            name: name.into(),
        }),
        arguments,
        payload: None,
    })
}

fn category_constant_pattern(linked: &LinkedGrammar, name: &str) -> Option<PatternIr> {
    let category_sort = linked.ir().category_syntax().category_sort();
    let vocabulary = linked
        .ir()
        .vocabularies()
        .iter()
        .find(|vocabulary| vocabulary.sort() == category_sort)?;
    let identifier = vocabulary
        .names()
        .iter()
        .position(|candidate| candidate == name)?;
    let operator = linked.ir().operators().iter().find(|operator| {
        operator.sort() == category_sort && operator.payload() == LinkedPayloadKind::Constant
    })?;
    Some(PatternIr::Node(NodePatternIr {
        pattern: NodePatternKind::Value,
        operator: OperatorPatternIr::Fixed(FixedOperatorPatternIr {
            kind: FixedKind::Value,
            name: operator.name().to_owned(),
        }),
        arguments: Vec::new(),
        payload: Some(PayloadPatternIr::Fixed(FixedPayloadPatternIr {
            kind: FixedKind::Value,
            value: i64::try_from(identifier).ok()?,
        })),
    }))
}

struct CategoryPatternParser<'a> {
    source: &'a str,
    offset: usize,
    linked: &'a LinkedGrammar,
    rule: &'a str,
    location: RulePlanLocation,
}

impl<'a> CategoryPatternParser<'a> {
    const fn new(
        source: &'a str,
        linked: &'a LinkedGrammar,
        rule: &'a str,
        location: RulePlanLocation,
    ) -> Self {
        Self {
            source,
            offset: 0,
            linked,
            rule,
            location,
        }
    }

    fn refusal(&self, reason: impl Into<String>) -> RulePlanError {
        theory_refusal(
            self.location,
            format!(
                "rule {:?} has invalid category {:?}: {}",
                self.rule,
                self.source,
                reason.into()
            ),
        )
    }

    fn skip_space(&mut self) {
        while let Some(character) = self.source[self.offset..].chars().next() {
            if !character.is_whitespace() {
                break;
            }
            self.offset += character.len_utf8();
        }
    }

    fn starts(&mut self, token: &str) -> bool {
        self.skip_space();
        self.source[self.offset..].starts_with(token)
    }

    fn take(&mut self, token: &str) -> Result<(), RulePlanError> {
        if !self.starts(token) {
            return Err(self.refusal(format!("expected {token:?}")));
        }
        self.offset += token.len();
        Ok(())
    }

    fn boundary(&self, offset: usize) -> bool {
        if offset == self.source.len() {
            return true;
        }
        let remainder = &self.source[offset..];
        remainder.chars().next().is_some_and(|character| {
            character.is_whitespace()
                || "()[],".contains(character)
                || character
                    == self
                        .linked
                        .ir()
                        .category_syntax()
                        .feature_delimiter()
                        .marker()
        }) || self
            .linked
            .ir()
            .category_syntax()
            .constructors()
            .iter()
            .any(|constructor| match constructor.notation() {
                LinkedConstructorNotation::Prefix { spelling }
                | LinkedConstructorNotation::Infix { spelling, .. } => {
                    remainder.starts_with(spelling)
                }
                LinkedConstructorNotation::Mixfix { parts, .. } => parts
                    .iter()
                    .filter(|part| !part.is_empty())
                    .any(|part| remainder.starts_with(part)),
            })
    }

    fn feature(&mut self, mut pattern: PatternIr) -> Result<PatternIr, RulePlanError> {
        loop {
            self.skip_space();
            let delimiter = self.linked.ir().category_syntax().feature_delimiter();
            if !self.source[self.offset..].starts_with(delimiter.marker()) {
                return Ok(pattern);
            }
            let end = match delimiter {
                FeatureDelimiter::Brackets => {
                    let Some(relative) = self.source[self.offset + 1..].find(']') else {
                        return Err(self.refusal("feature has no closing bracket"));
                    };
                    self.offset + relative + 2
                }
                FeatureDelimiter::Colon => {
                    let mut end = self.offset + 1;
                    while end < self.source.len() && !self.boundary(end) {
                        let character = self.source[end..]
                            .chars()
                            .next()
                            .ok_or_else(|| self.refusal("feature is empty"))?;
                        end += character.len_utf8();
                    }
                    if end == self.offset + 1 {
                        return Err(self.refusal("feature is empty"));
                    }
                    end
                }
            };
            let suffix = &self.source[self.offset..end];
            let PatternIr::Node(node) = &mut pattern else {
                return Err(self.refusal("a feature follows a non-node category"));
            };
            let OperatorPatternIr::Fixed(operator) = &mut node.operator else {
                return Err(self.refusal("a feature follows a variable operator"));
            };
            let featured = format!("{}{suffix}", operator.name);
            let valid = self.linked.ir().operators().iter().any(|candidate| {
                candidate.name() == featured
                    && candidate.sort() == self.linked.ir().category_syntax().category_sort()
                    && candidate.arguments().len() == node.arguments.len()
                    && candidate.payload() == LinkedPayloadKind::None
            });
            if !valid {
                return Err(self.refusal(format!(
                    "featured operator {featured:?} is not declared by the signature"
                )));
            }
            operator.name = featured;
            self.offset = end;
        }
    }

    fn atom(&mut self) -> Result<PatternIr, RulePlanError> {
        self.skip_space();
        let start = self.offset;
        let category_sort = self.linked.ir().category_syntax().category_sort();
        let mut names = self
            .linked
            .ir()
            .operators()
            .iter()
            .filter(|operator| {
                operator.sort() == category_sort
                    && operator.arguments().is_empty()
                    && operator.payload() == LinkedPayloadKind::None
            })
            .map(|operator| (operator.name(), false))
            .chain(
                self.linked
                    .ir()
                    .vocabularies()
                    .iter()
                    .filter(|vocabulary| vocabulary.sort() == category_sort)
                    .flat_map(|vocabulary| vocabulary.names())
                    .map(|name| (name.as_str(), true)),
            )
            .collect::<Vec<_>>();
        names.sort_by_key(|(name, _)| std::cmp::Reverse(name.len()));
        for (name, constant) in names {
            if self.source[start..].starts_with(name) && self.boundary(start + name.len()) {
                self.offset += name.len();
                let pattern = if constant {
                    category_constant_pattern(self.linked, name).ok_or_else(|| {
                        self.refusal(format!("constant {name:?} cannot be linked"))
                    })?
                } else {
                    fixed_theory_node(name, Vec::new())
                };
                return self.feature(pattern);
            }
        }
        Err(self.refusal("operator is not declared by the signature"))
    }

    fn primary(&mut self) -> Result<PatternIr, RulePlanError> {
        if self.starts("(") {
            self.take("(")?;
            let held = self.expression(1)?;
            self.take(")")?;
            return self.feature(held);
        }
        let mut prefixes = self
            .linked
            .ir()
            .category_syntax()
            .constructors()
            .iter()
            .filter_map(|constructor| match constructor.notation() {
                LinkedConstructorNotation::Prefix { spelling } => {
                    Some((spelling.as_str(), constructor))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        prefixes.sort_by_key(|(spelling, _)| std::cmp::Reverse(spelling.len()));
        for (spelling, constructor) in prefixes {
            if !self.starts(spelling) {
                continue;
            }
            self.take(spelling)?;
            self.take("(")?;
            let arity = self
                .linked
                .ir()
                .operators()
                .iter()
                .find(|operator| operator.operator() == constructor.operator())
                .ok_or_else(|| self.refusal("category constructor lost its linked operator"))?
                .arguments()
                .len();
            let mut arguments = Vec::with_capacity(arity);
            for position in 0..arity {
                arguments.push(self.expression(1)?);
                if position + 1 < arity {
                    self.take(",")?;
                }
            }
            self.take(")")?;
            return self.feature(fixed_theory_node(constructor.name(), arguments));
        }
        let mut mixfix = self
            .linked
            .ir()
            .category_syntax()
            .constructors()
            .iter()
            .filter_map(|constructor| match constructor.notation() {
                LinkedConstructorNotation::Mixfix { parts, precedence } => {
                    Some((parts, *precedence, constructor))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        mixfix.sort_by_key(|(parts, _, _)| std::cmp::Reverse(parts.first().map_or(0, String::len)));
        for (parts, precedence, constructor) in mixfix {
            let first = parts.first().map_or("", String::as_str);
            if !self.starts(first) {
                continue;
            }
            self.take(first)?;
            let mut arguments = Vec::with_capacity(parts.len().saturating_sub(1));
            for part in &parts[1..] {
                arguments.push(self.expression(precedence)?);
                if !part.is_empty() {
                    self.take(part)?;
                }
            }
            return self.feature(fixed_theory_node(constructor.name(), arguments));
        }
        self.atom()
    }

    fn infix(&mut self) -> Option<(&'a LinkedCategoryConstructor, u64)> {
        self.skip_space();
        let mut choices = self
            .linked
            .ir()
            .category_syntax()
            .constructors()
            .iter()
            .filter_map(|constructor| match constructor.notation() {
                LinkedConstructorNotation::Infix {
                    spelling,
                    precedence,
                    ..
                } if self.source[self.offset..].starts_with(spelling) => {
                    Some((constructor, *precedence))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        choices.sort_by_key(|(constructor, _)| match constructor.notation() {
            LinkedConstructorNotation::Infix { spelling, .. } => std::cmp::Reverse(spelling.len()),
            _ => std::cmp::Reverse(0),
        });
        choices.into_iter().next()
    }

    fn expression(&mut self, minimum: u64) -> Result<PatternIr, RulePlanError> {
        let mut left = self.primary()?;
        let mut last_nonassociative = None;
        while let Some((constructor, precedence)) = self.infix() {
            if precedence < minimum {
                break;
            }
            let LinkedConstructorNotation::Infix {
                spelling,
                associativity,
                ..
            } = constructor.notation()
            else {
                unreachable!();
            };
            if last_nonassociative == Some(precedence) {
                return Err(self.refusal("nonassociative constructor is chained"));
            }
            self.take(spelling)?;
            let right_minimum = if *associativity == LinkedAssociativity::Right {
                precedence
            } else {
                precedence
                    .checked_add(1)
                    .ok_or_else(|| self.refusal("infix precedence is too large"))?
            };
            let right = self.expression(right_minimum)?;
            left = self.feature(fixed_theory_node(constructor.name(), vec![left, right]))?;
            last_nonassociative =
                (*associativity == LinkedAssociativity::None).then_some(precedence);
        }
        Ok(left)
    }

    fn quoted(&mut self) -> Result<String, RulePlanError> {
        self.skip_space();
        let mut stream =
            serde_json::Deserializer::from_str(&self.source[self.offset..]).into_iter::<String>();
        let value = stream
            .next()
            .transpose()
            .map_err(|error| self.refusal(format!("expected a quoted name: {error}")))?
            .ok_or_else(|| self.refusal("expected a quoted name"))?;
        self.offset += stream.byte_offset();
        Ok(value)
    }

    fn sexpression(&mut self) -> Result<PatternIr, RulePlanError> {
        self.take("(")?;
        let name = self.quoted()?;
        let operator = self
            .linked
            .ir()
            .operators()
            .iter()
            .find(|operator| operator.name() == name)
            .ok_or_else(|| self.refusal(format!("operator {name:?} is not declared")))?;
        if operator.sort() != self.linked.ir().category_syntax().category_sort() {
            return Err(self.refusal(format!("operator {name:?} is not a category operator")));
        }
        let payload = match operator.payload() {
            LinkedPayloadKind::None => None,
            LinkedPayloadKind::Constant => {
                let constant = self.quoted()?;
                let PatternIr::Node(node) = category_constant_pattern(self.linked, &constant)
                    .ok_or_else(|| {
                        self.refusal(format!("{constant:?} is not a category constant"))
                    })?
                else {
                    unreachable!();
                };
                node.payload
            }
            LinkedPayloadKind::Index | LinkedPayloadKind::Free => {
                return Err(self.refusal("a ground category cannot contain a variable payload"));
            }
        };
        let mut arguments = Vec::with_capacity(operator.arguments().len());
        for _ in operator.arguments() {
            arguments.push(self.sexpression()?);
        }
        self.take(")")?;
        Ok(PatternIr::Node(NodePatternIr {
            pattern: NodePatternKind::Value,
            operator: OperatorPatternIr::Fixed(FixedOperatorPatternIr {
                kind: FixedKind::Value,
                name,
            }),
            arguments,
            payload,
        }))
    }

    fn complete(mut self) -> Result<PatternIr, RulePlanError> {
        let pattern = match self.linked.ir().category_syntax().kind() {
            CategorySyntaxKind::Declared => self.expression(1)?,
            CategorySyntaxKind::SExpression => self.sexpression()?,
        };
        self.skip_space();
        if self.offset != self.source.len() {
            return Err(self.refusal("found trailing input"));
        }
        Ok(pattern)
    }
}

fn category_pattern(
    source: &str,
    linked: &LinkedGrammar,
    rule: &str,
    location: RulePlanLocation,
) -> Result<PatternIr, RulePlanError> {
    CategoryPatternParser::new(source, linked, rule, location).complete()
}

fn composition_constructors(
    name: &str,
    functor_identifier: u32,
    order: u64,
    direction: LinkedCompositionDirection,
    crossed: bool,
    linked: &LinkedGrammar,
    location: RulePlanLocation,
    constructors: &mut Vec<LinkedTheoryConstructor>,
) -> Result<(), RulePlanError> {
    let functor = functor_or_refusal(linked, functor_identifier, name, location)?;
    let orientations: &[bool] = match functor.argument_side() {
        LinkedArgumentSide::Right => &[true],
        LinkedArgumentSide::Left => &[false],
        LinkedArgumentSide::Either => &[true, false],
    };
    let result = theory_term("X");
    let argument = theory_term("Y");
    let principal = theory_functor(
        linked,
        functor.name(),
        result.clone(),
        argument.clone(),
        name,
        location,
    )?;
    let slashes = theory_slashes(functor);
    let deepest = if direction == LinkedCompositionDirection::Unrestricted {
        slashes.clone()
    } else if crossed {
        vec![functor.opposite().to_owned()]
    } else {
        vec![functor.name().to_owned()]
    };
    let arguments = (0..order)
        .map(|index| theory_term(format!("Z{index}")))
        .collect::<Vec<_>>();
    let mut pairs = Vec::new();
    if order == 0 {
        pairs.push((argument, result));
    } else {
        for last in deepest {
            let prefixes = if direction == LinkedCompositionDirection::Steedman {
                vec![last.clone()]
            } else {
                slashes.clone()
            };
            for sequence in slash_sequences(&prefixes, &last, order) {
                pairs.push((
                    theory_spine(
                        linked,
                        &sequence,
                        argument.clone(),
                        &arguments,
                        name,
                        location,
                    )?,
                    theory_spine(
                        linked,
                        &sequence,
                        result.clone(),
                        &arguments,
                        name,
                        location,
                    )?,
                ));
            }
        }
    }
    let mut serial = 0;
    let category_sort = linked.ir().category_syntax().category_sort();
    for (container, conclusion) in pairs {
        for principal_first in orientations {
            let premises = if *principal_first {
                vec![principal.clone(), container.clone()]
            } else {
                vec![container.clone(), principal.clone()]
            };
            constructors.push(theory_constructor(
                name,
                serial,
                &premises,
                &conclusion,
                true,
                category_sort,
                linked,
                location,
            )?);
            serial += 1;
        }
    }
    Ok(())
}

fn substitution_constructors(
    name: &str,
    functor_identifier: u32,
    order: u64,
    crossed: bool,
    linked: &LinkedGrammar,
    location: RulePlanLocation,
    constructors: &mut Vec<LinkedTheoryConstructor>,
) -> Result<(), RulePlanError> {
    let functor = functor_or_refusal(linked, functor_identifier, name, location)?;
    let orientations: &[bool] = match functor.argument_side() {
        LinkedArgumentSide::Right => &[true],
        LinkedArgumentSide::Left => &[false],
        LinkedArgumentSide::Either => &[true, false],
    };
    let result = theory_term("X");
    let argument = theory_term("Y");
    let foot = theory_functor(
        linked,
        functor.name(),
        result.clone(),
        argument.clone(),
        name,
        location,
    )?;
    let slashes = theory_slashes(functor);
    let deepest = if crossed {
        functor.opposite()
    } else {
        functor.name()
    };
    let arguments = (0..order)
        .map(|index| theory_term(format!("Z{index}")))
        .collect::<Vec<_>>();
    let category_sort = linked.ir().category_syntax().category_sort();
    let mut serial = 0;
    for sequence in slash_sequences(&slashes, deepest, order) {
        let principal = theory_spine(linked, &sequence, foot.clone(), &arguments, name, location)?;
        let container = theory_spine(
            linked,
            &sequence,
            argument.clone(),
            &arguments,
            name,
            location,
        )?;
        let conclusion = theory_spine(
            linked,
            &sequence,
            result.clone(),
            &arguments,
            name,
            location,
        )?;
        for principal_first in orientations {
            let premises = if *principal_first {
                vec![principal.clone(), container.clone()]
            } else {
                vec![container.clone(), principal.clone()]
            };
            constructors.push(theory_constructor(
                name,
                serial,
                &premises,
                &conclusion,
                true,
                category_sort,
                linked,
                location,
            )?);
            serial += 1;
        }
    }
    Ok(())
}

fn type_raising_constructors(
    name: &str,
    functor_identifier: u32,
    targets: Option<&[String]>,
    arguments: Option<&[String]>,
    linked: &LinkedGrammar,
    location: RulePlanLocation,
    constructors: &mut Vec<LinkedTheoryConstructor>,
) -> Result<(), RulePlanError> {
    let functor = functor_or_refusal(linked, functor_identifier, name, location)?;
    let inputs = if let Some(arguments) = arguments {
        arguments
            .iter()
            .map(|argument| category_pattern(argument, linked, name, location))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![theory_term("X")]
    };
    let forward_complete = targets.is_some();
    let targets = if let Some(targets) = targets {
        targets
            .iter()
            .map(|target| category_pattern(target, linked, name, location))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![theory_term("T")]
    };
    let pairs = targets
        .iter()
        .flat_map(|target| inputs.iter().map(move |input| (target, input)));
    for (serial, (target, input)) in pairs.enumerate() {
        let inner = theory_functor(
            linked,
            functor.opposite(),
            target.clone(),
            input.clone(),
            name,
            location,
        )?;
        let conclusion = theory_functor(
            linked,
            functor.name(),
            target.clone(),
            inner,
            name,
            location,
        )?;
        constructors.push(theory_constructor(
            name,
            serial,
            std::slice::from_ref(input),
            &conclusion,
            forward_complete,
            linked.ir().category_syntax().category_sort(),
            linked,
            location,
        )?);
    }
    Ok(())
}

fn clausal_adjunct_constructor(
    name: &str,
    rule: &ClausalAdjunctRuleIr,
    linked: &LinkedGrammar,
    location: RulePlanLocation,
) -> Result<LinkedTheoryConstructor, RulePlanError> {
    let source = linked
        .ir()
        .category_syntax()
        .functors()
        .iter()
        .find(|functor| functor.name() == rule.source_view)
        .ok_or_else(|| theory_refusal(location, "clausal source view was not linked"))?;
    let target = linked
        .ir()
        .category_syntax()
        .functors()
        .iter()
        .find(|functor| functor.name() == rule.target_view)
        .ok_or_else(|| theory_refusal(location, "clausal target view was not linked"))?;
    let premise = theory_functor(
        linked,
        source.name(),
        category_pattern(&rule.source_core, linked, name, location)?,
        category_pattern(&rule.argument, linked, name, location)?,
        name,
        location,
    )?;
    let variable = theory_term("X");
    let conclusion = theory_functor(
        linked,
        target.name(),
        variable.clone(),
        variable,
        name,
        location,
    )?;
    theory_constructor(
        name,
        0,
        &[premise],
        &conclusion,
        false,
        linked.ir().category_syntax().category_sort(),
        linked,
        location,
    )
}

fn coercion_constructor(
    name: &str,
    source: &str,
    target: &str,
    linked: &LinkedGrammar,
    location: RulePlanLocation,
) -> Result<LinkedTheoryConstructor, RulePlanError> {
    theory_constructor(
        name,
        0,
        &[category_pattern(source, linked, name, location)?],
        &category_pattern(target, linked, name, location)?,
        true,
        linked.ir().category_syntax().category_sort(),
        linked,
        location,
    )
}

fn explicit_theory(
    source: &RuleIr,
    linked: &LinkedGrammar,
    location: RulePlanLocation,
    constructors: &mut Vec<LinkedTheoryConstructor>,
    elaborated: &mut Vec<String>,
) -> Result<(), RulePlanError> {
    let name = explicit_rule_name(source);
    let category_sort = linked.ir().category_syntax().category_sort();
    match source {
        RuleIr::Pattern(rule) => pattern_theory(
            &name,
            &rule.premises,
            &rule.conclusion,
            rule.targets.as_deref(),
            rule.targets.is_some(),
            category_sort,
            linked,
            location,
            constructors,
            elaborated,
        )?,
        RuleIr::Absorption(rule) => pattern_theory(
            &name,
            &rule.premises,
            &rule.conclusion,
            None,
            true,
            category_sort,
            linked,
            location,
            constructors,
            elaborated,
        )?,
        RuleIr::Coordination(rule) => pattern_theory(
            &name,
            &rule.premises,
            &rule.conclusion,
            rule.targets.as_deref(),
            rule.phase == CoordinationPhase::Merge || rule.targets.is_some(),
            category_sort,
            linked,
            location,
            constructors,
            elaborated,
        )?,
        RuleIr::Application(rule) => {
            let functor = linked
                .ir()
                .category_syntax()
                .functors()
                .iter()
                .position(|functor| functor.name() == rule.view)
                .and_then(|index| u32::try_from(index).ok())
                .ok_or_else(|| {
                    theory_refusal(
                        location,
                        format!("rule {name:?} lost its linked functor view"),
                    )
                })?;
            composition_constructors(
                &name,
                functor,
                0,
                LinkedCompositionDirection::Unrestricted,
                false,
                linked,
                location,
                constructors,
            )?;
        }
        RuleIr::Composition(rule) => {
            if let Some(order) = rule.order {
                let functor = linked
                    .ir()
                    .category_syntax()
                    .functors()
                    .iter()
                    .position(|functor| functor.name() == rule.view)
                    .and_then(|index| u32::try_from(index).ok())
                    .ok_or_else(|| {
                        theory_refusal(
                            location,
                            format!("rule {name:?} lost its linked functor view"),
                        )
                    })?;
                let slashes = linked.ir().category_syntax().functors().len();
                match rule.direction {
                    CompositionRestriction::Steedman => 1,
                    CompositionRestriction::Kuhlmann => {
                        checked_schema_power(slashes, order.saturating_sub(1), &name, location)?
                    }
                    CompositionRestriction::Unrestricted => {
                        checked_schema_power(slashes, order, &name, location)?
                    }
                };
                composition_constructors(
                    &name,
                    functor,
                    order,
                    linked_direction(rule.direction),
                    rule.crossed,
                    linked,
                    location,
                    constructors,
                )?;
            } else {
                elaborated.push(name);
            }
        }
        RuleIr::Substitution(rule) => {
            let Some(order) = rule.order else {
                elaborated.push(name);
                return Ok(());
            };
            let functor = linked
                .ir()
                .category_syntax()
                .functors()
                .iter()
                .position(|functor| functor.name() == rule.view)
                .and_then(|index| u32::try_from(index).ok())
                .ok_or_else(|| {
                    theory_refusal(
                        location,
                        format!("rule {name:?} lost its linked functor view"),
                    )
                })?;
            checked_schema_power(
                linked.ir().category_syntax().functors().len(),
                order.saturating_sub(1),
                &name,
                location,
            )?;
            substitution_constructors(
                &name,
                functor,
                order,
                rule.crossed,
                linked,
                location,
                constructors,
            )?;
        }
        RuleIr::TypeRaising(rule) => {
            let functor = linked
                .ir()
                .category_syntax()
                .functors()
                .iter()
                .position(|functor| functor.name() == rule.view)
                .and_then(|index| u32::try_from(index).ok())
                .ok_or_else(|| theory_refusal(location, "raising view was not linked"))?;
            type_raising_constructors(
                &name,
                functor,
                rule.targets.as_deref(),
                rule.arguments.as_deref(),
                linked,
                location,
                constructors,
            )?;
        }
        RuleIr::ClausalAdjunct(rule) => {
            constructors.push(clausal_adjunct_constructor(&name, rule, linked, location)?);
        }
        RuleIr::Coercion(rule) => {
            constructors.push(coercion_constructor(
                &name,
                &rule.source,
                &rule.target,
                linked,
                location,
            )?);
        }
        RuleIr::Opaque(rule) => {
            let value = rule.extension.get("theory").ok_or_else(|| {
                theory_refusal(
                    location,
                    format!(
                        "late rule {:?} does not declare a native theory contract",
                        rule.rule
                    ),
                )
            })?;
            let document = serde_json::to_value(value).map_err(|error| {
                theory_refusal(
                    location,
                    format!(
                        "late rule {:?} has an invalid theory contract: {error}",
                        rule.rule
                    ),
                )
            })?;
            let theory: OpaqueTheoryIr = serde_json::from_value(document).map_err(|error| {
                theory_refusal(
                    location,
                    format!(
                        "late rule {:?} has an invalid theory contract: {error}",
                        rule.rule
                    ),
                )
            })?;
            match theory {
                OpaqueTheoryIr::Elaborated { name } => {
                    return Err(theory_refusal(
                        location,
                        format!(
                            "late rule {name:?} declares a pattern runtime but only elaborated proof evidence"
                        ),
                    ));
                }
                OpaqueTheoryIr::Schemata {
                    name,
                    forward_complete,
                    schemata,
                } => {
                    if schemata.len() != 1 {
                        return Err(theory_refusal(
                            location,
                            format!(
                                "late pattern rule {name:?} must declare exactly one deterministic schema"
                            ),
                        ));
                    }
                    for (serial, schema) in schemata.iter().enumerate() {
                        constructors.push(theory_constructor(
                            &name,
                            serial,
                            &schema.premises,
                            &schema.conclusion,
                            forward_complete,
                            category_sort,
                            linked,
                            location,
                        )?);
                    }
                }
            }
        }
    }
    Ok(())
}

fn compile_theory(
    grammar: &GrammarIr,
    linked: &LinkedGrammar,
    rules: &[LinkedRule],
) -> Result<LinkedTheory, RulePlanError> {
    let mut constructors = Vec::new();
    let mut elaborated = Vec::new();
    for rule in rules {
        let location = rule.location();
        match location.source() {
            RulePlanSource::Explicit => {
                if let RuleEntryIr::Rule(source) = &grammar.rules.rules[location.index()] {
                    explicit_theory(source, linked, location, &mut constructors, &mut elaborated)?;
                }
            }
            RulePlanSource::Combinator => {
                let name = generated_rule_name(linked, rule.kind());
                match rule.kind() {
                    LinkedRuleKind::Application { functor } => {
                        composition_constructors(
                            &name,
                            *functor,
                            0,
                            LinkedCompositionDirection::Unrestricted,
                            false,
                            linked,
                            location,
                            &mut constructors,
                        )?;
                    }
                    LinkedRuleKind::Composition { order: None, .. } => elaborated.push(name),
                    LinkedRuleKind::Composition {
                        functor,
                        direction,
                        order: Some(order),
                        crossed,
                    } => {
                        let slashes = linked.ir().category_syntax().functors().len();
                        match direction {
                            LinkedCompositionDirection::Steedman => 1,
                            LinkedCompositionDirection::Kuhlmann => checked_schema_power(
                                slashes,
                                order.saturating_sub(1),
                                &name,
                                location,
                            )?,
                            LinkedCompositionDirection::Unrestricted => {
                                checked_schema_power(slashes, *order, &name, location)?
                            }
                        };
                        composition_constructors(
                            &name,
                            *functor,
                            *order,
                            *direction,
                            *crossed,
                            linked,
                            location,
                            &mut constructors,
                        )?;
                    }
                    LinkedRuleKind::Substitution {
                        functor,
                        order: Some(order),
                        crossed,
                    } => {
                        checked_schema_power(
                            linked.ir().category_syntax().functors().len(),
                            order.saturating_sub(1),
                            &name,
                            location,
                        )?;
                        substitution_constructors(
                            &name,
                            *functor,
                            *order,
                            *crossed,
                            linked,
                            location,
                            &mut constructors,
                        )?;
                    }
                    LinkedRuleKind::Substitution { order: None, .. } => elaborated.push(name),
                    LinkedRuleKind::TypeRaising {
                        functor,
                        targets,
                        arguments,
                    } => {
                        type_raising_constructors(
                            &name,
                            *functor,
                            targets.as_deref(),
                            arguments.as_deref(),
                            linked,
                            location,
                            &mut constructors,
                        )?;
                    }
                    LinkedRuleKind::ClausalAdjunct { .. }
                    | LinkedRuleKind::Pattern { .. }
                    | LinkedRuleKind::Absorption { .. }
                    | LinkedRuleKind::Coordination { .. }
                    | LinkedRuleKind::Extension { .. }
                    | LinkedRuleKind::Coercion { .. } => unreachable!(),
                }
            }
            RulePlanSource::Coercion => {
                let LinkedRuleKind::Coercion {
                    label,
                    member,
                    source,
                    target,
                } = rule.kind()
                else {
                    unreachable!();
                };
                let name = format!("{label}:{member}:{source}=>{target}");
                constructors.push(coercion_constructor(
                    &name, source, target, linked, location,
                )?);
            }
        }
    }
    Ok(LinkedTheory::new(constructors, elaborated))
}

impl GrammarIr {
    pub(crate) fn plan_rules(
        &self,
        linked: &LinkedGrammar,
        prepared_fingerprint: &str,
    ) -> Result<LinkedRulePlan, CompileError> {
        if linked.declaration_fingerprint() != Some(prepared_fingerprint) {
            return Err(CompileError::RulePlan {
                source: RulePlanError::global(RulePlanErrorKind::SignatureFingerprint {
                    expected: prepared_fingerprint.to_owned(),
                    received: linked
                        .declaration_fingerprint()
                        .unwrap_or("unbound")
                        .to_owned(),
                }),
            });
        }
        self.goal
            .refuse_malformed_builtins()
            .and_then(|()| self.matching.refuse_malformed_builtins())
            .and_then(|()| self.normal_form.refuse_malformed_builtins())
            .map_err(|source| CompileError::RulePlan { source })?;
        let mut planner = RulePlanner::new(linked);
        for (index, entry) in self.rules.rules.iter().enumerate() {
            let rule = match entry {
                RuleEntryIr::Combinators(combinators) => {
                    planner
                        .push_combinators(index, combinators)
                        .map_err(|source| CompileError::RulePlan { source })?;
                    continue;
                }
                RuleEntryIr::Coercions(coercions) => {
                    planner
                        .push_coercions(index, coercions)
                        .map_err(|source| CompileError::RulePlan { source })?;
                    continue;
                }
                RuleEntryIr::Rule(rule) => rule,
            };
            let location = RulePlanLocation::explicit(index);
            if let Some(matching) = rule.matching() {
                matching
                    .refuse_malformed_builtins()
                    .map_err(|source| CompileError::RulePlan { source })?;
            }
            rule.validate_references(self, linked, location)
                .map_err(|source| CompileError::RulePlan { source })?;
            let kind = rule
                .linked_kind(self, linked, location)
                .map_err(|source| CompileError::RulePlan { source })?;
            planner
                .push(
                    rule.tag(),
                    location,
                    kind,
                    rule.matching(),
                    rule.restriction(),
                    rule.introduces(),
                    rule.consumes(),
                )
                .map_err(|source| CompileError::RulePlan { source })?;
        }
        let rows = planner
            .rules
            .iter()
            .map(|rule| planned_relation_value(self, linked, rule))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| CompileError::RulePlan { source })?;
        planner
            .validate_relations(self, &rows)
            .map_err(|source| CompileError::RulePlan { source })?;
        let theory = compile_theory(self, linked, &planner.rules)
            .map_err(|source| CompileError::RulePlan { source })?;
        let semantic = json!({
            "signature": linked.ir().fingerprint(),
            "plan": rows,
            "theory": theory_fingerprint_value(&theory),
        });
        let bytes = canonical::value_bytes(&semantic).map_err(|source| CompileError::Encode {
            declaration: DeclarationKind::Grammar,
            source,
        })?;
        let fingerprint = sha256_hex(&bytes);
        Ok(LinkedRulePlan::new(
            linked.ir().fingerprint().to_owned(),
            fingerprint,
            planner.tags,
            planner.rules,
            theory,
        ))
    }
}

fn theory_fingerprint_value(theory: &LinkedTheory) -> Value {
    json!({
        "constructors": theory.constructors().iter().map(|constructor| json!({
            "inferred": constructor.inferred(),
            "declared": constructor.declared(),
            "premise_indices": constructor.premise_indices(),
            "conclusion_index": constructor.conclusion_index(),
        })).collect::<Vec<_>>(),
        "elaborated": theory.elaborated().len(),
    })
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum NormalFormIr {
    None(NoNormalFormIr),
    Eisner(EisnerNormalFormIr),
    UndirectedEisner(UndirectedEisnerNormalFormIr),
    HockenmaierBisk(HockenmaierBiskNormalFormIr),
    AllOf(AllOfNormalFormIr),
    Extension(ExtensionIr),
}

/// A declaration of a family member another distribution installs.
///
/// The compiler checks only its tag: the installed plugin owns the rest of
/// its fields.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct ExtensionIr {
    kind: String,
    #[serde(flatten)]
    extension: IndexMap<String, OpaqueJsonValue>,
}

const BUILTIN_NORMAL_FORM_KINDS: [&str; 5] = [
    "all-of",
    "eisner",
    "hockenmaier-bisk",
    "none",
    "undirected-eisner",
];

const BUILTIN_MATCHING_KINDS: [&str; 6] = [
    "alpha",
    "feature-erasure",
    "identity",
    "modal",
    "product",
    "subsumption",
];

const BUILTIN_GOAL_KINDS: [&str; 2] = ["cover", "root"];

fn refuse_builtin_extension(
    family: &'static str,
    extension: &ExtensionIr,
    builtins: &[&str],
) -> Result<(), RulePlanError> {
    if builtins.contains(&extension.kind.as_str()) {
        Err(RulePlanError::global(
            RulePlanErrorKind::MalformedBuiltinComponent {
                family,
                kind: extension.kind.clone(),
            },
        ))
    } else {
        Ok(())
    }
}

impl NormalFormIr {
    fn refuse_malformed_builtins(&self) -> Result<(), RulePlanError> {
        match self {
            Self::AllOf(all) => all
                .normal_forms
                .iter()
                .try_for_each(Self::refuse_malformed_builtins),
            Self::Extension(extension) => {
                refuse_builtin_extension("normal form", extension, &BUILTIN_NORMAL_FORM_KINDS)
            }
            Self::None(_)
            | Self::Eisner(_)
            | Self::UndirectedEisner(_)
            | Self::HockenmaierBisk(_) => Ok(()),
        }
    }
}

impl MatchingIr {
    fn refuse_malformed_builtins(&self) -> Result<(), RulePlanError> {
        match self {
            Self::Product(product) => {
                product.left.refuse_malformed_builtins()?;
                product.right.refuse_malformed_builtins()
            }
            Self::Extension(extension) => {
                refuse_builtin_extension("matching", extension, &BUILTIN_MATCHING_KINDS)
            }
            Self::Identity(_)
            | Self::Alpha(_)
            | Self::FeatureErasure(_)
            | Self::Subsumption(_)
            | Self::Modal(_) => Ok(()),
        }
    }
}

impl GoalIr {
    fn refuse_malformed_builtins(&self) -> Result<(), RulePlanError> {
        match self {
            Self::Extension(extension) => {
                refuse_builtin_extension("goal", extension, &BUILTIN_GOAL_KINDS)
            }
            Self::Root(_) | Self::Cover(_) => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NoNormalFormIr {
    kind: NoNormalFormKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct EisnerNormalFormIr {
    kind: EisnerKind,
    include_crossed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct UndirectedEisnerNormalFormIr {
    kind: UndirectedEisnerKind,
    include_crossed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct HockenmaierBiskNormalFormIr {
    kind: HockenmaierBiskKind,
    order_bound: u64,
    track_child_order: Flag,
    composition_primary: Flag,
    unit_primary: Flag,
    raised_secondary: Flag,
    raised_application: Flag,
    coordination_raising: Flag,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct AllOfNormalFormIr {
    kind: AllOfKind,
    normal_forms: Vec<NormalFormIr>,
}

marker_enum!(NoNormalFormKind, "none");
marker_enum!(EisnerKind, "eisner");
marker_enum!(UndirectedEisnerKind, "undirected-eisner");
marker_enum!(HockenmaierBiskKind, "hockenmaier-bisk");
marker_enum!(AllOfKind, "all-of");

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum GoalIr {
    Root(RootGoalIr),
    Cover(CoverGoalIr),
    Extension(ExtensionIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RootGoalIr {
    categories: Vec<String>,
    kind: RootGoalKind,
    root_weights: Option<Vec<FiniteF64>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CoverGoalIr {
    categories: Vec<String>,
    kind: CoverGoalKind,
    max_fragments: u64,
    penalty: FiniteF64,
    penalty_form: PenaltyForm,
    tiling: Tiling,
}

marker_enum!(RootGoalKind, "root");
marker_enum!(CoverGoalKind, "cover");

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum PenaltyForm {
    #[serde(rename = "after-first")]
    AfterFirst,
    #[serde(rename = "each-fragment")]
    EachFragment,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum Tiling {
    #[serde(rename = "fewest-fragments")]
    FewestFragments,
    #[serde(rename = "lexical")]
    Lexical,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct BudgetIr {
    depth: u64,
    mass: FiniteF64,
    beam: u64,
    unary_rounds: u64,
    threads: u64,
    beta_fuel: u64,
    closure_rounds: u64,
    closure_nodes: u64,
    max_category_arity: Option<u64>,
    max_category_size: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CapacityEvidenceIr {
    rule_family: RuleFamily,
    bounded_degree: bool,
    prefix_closed: bool,
    application_target_restrictions: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum RuleFamily {
    #[serde(rename = "pure_ccg")]
    PureCcg,
    #[serde(rename = "prefix_closed_vw_ccg")]
    PrefixClosedVwCcg,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NamedInterpretationIr {
    name: String,
    reading: InterpretationIr,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum InterpretationIr {
    Category(CategoryInterpretationIr),
    String(StringInterpretationIr),
    Dependency(DependencyInterpretationIr),
    Meaning(MeaningInterpretationIr),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CategoryInterpretationIr {
    kind: CategoryInterpretationKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct StringInterpretationIr {
    kind: StringInterpretationKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct DependencyInterpretationIr {
    kind: DependencyInterpretationKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct MeaningInterpretationIr {
    kind: MeaningInterpretationKind,
}

marker_enum!(CategoryInterpretationKind, "category");
marker_enum!(StringInterpretationKind, "string");
marker_enum!(DependencyInterpretationKind, "dependency");
marker_enum!(MeaningInterpretationKind, "meaning");

#[derive(Clone)]
struct JointBinding {
    arguments: Vec<usize>,
    sort: String,
    annotation: Option<usize>,
}

#[derive(Clone)]
struct JointOperator {
    name: String,
    sort: String,
    arguments: Vec<String>,
    binding: Vec<JointBinding>,
    family: String,
    payload: PayloadKind,
    erases_to: Option<String>,
    location: LinkLocation,
}

#[derive(Clone)]
struct JointVocabulary {
    names: Vec<String>,
    cores: IndexMap<String, String>,
    location: LinkLocation,
}

struct FeaturedAtom {
    target: String,
    location: LinkLocation,
}

struct JointSignature<'a> {
    name: String,
    sorts: Vec<String>,
    declarations: Vec<JointOperator>,
    features: Vec<(String, Vec<String>)>,
    vocabularies: IndexMap<String, JointVocabulary>,
    open_sorts: Vec<String>,
    classifiers: IndexMap<String, String>,
    allows_vacuous_binding: bool,
    holes: bool,
    constant_family: String,
    variable_family: String,
    category_sort: String,
    syntax: SignatureSyntax,
    feature_delimiter: FeatureDelimiter,
    constructors: &'a [ConstructorIr],
    functors: &'a [FunctorIr],
}

impl LinkDeclarations for LanguageIr {
    fn link_declarations(&self, grammar: &GrammarIr) -> Result<LinkedGrammar, CompileError> {
        let joint = JointSignature::extend(self, grammar)
            .map_err(|source| CompileError::Link { source })?;
        joint.finish(self.store_nodes)
    }
}

impl<'a> JointSignature<'a> {
    fn extend(language: &LanguageIr, grammar: &'a GrammarIr) -> Result<Self, LinkError> {
        let mut joint = Self::from_language(language)?;
        joint.extend_sorts(&grammar.signature)?;
        joint.extend_declarations(&grammar.signature)?;
        joint.extend_vocabularies(&grammar.signature)?;
        joint.extend_openness(&grammar.signature)?;
        joint.name = format!("{}:{}", language.name, grammar.signature.name);
        joint.allows_vacuous_binding = grammar.signature.allows_vacuous_binding;
        joint.holes = grammar.signature.holes;
        joint
            .category_sort
            .clone_from(&grammar.signature.category_sort);
        joint.syntax = grammar.signature.syntax;
        joint.feature_delimiter = grammar.signature.feature_delimiter;
        joint.constructors = &grammar.signature.constructors;
        joint.functors = &grammar.signature.functors;
        Ok(joint)
    }

    fn from_language(language: &LanguageIr) -> Result<Self, LinkError> {
        let mut sorts = Vec::new();
        for (index, sort) in [
            language.term_sort.as_str(),
            language.type_sort.as_str(),
            language.category_sort.as_str(),
        ]
        .into_iter()
        .enumerate()
        {
            push_sort(
                &mut sorts,
                sort,
                language_indexed(LinkComponent::Sort, index),
            )?;
        }
        let constant_family = language.constant_family.clone();
        let variable_family = language.variable_family.clone();
        if constant_family == variable_family
            || constant_family == "hole"
            || variable_family == "hole"
        {
            return Err(link_error(
                language_field(LinkComponent::Signature),
                LinkErrorKind::ReservedOperator {
                    operator: "kernel families".to_owned(),
                    reason: "requires distinct constant, variable, and hole families".to_owned(),
                },
            ));
        }
        let declarations = language_declarations(language)?;
        let vocabularies = language_vocabularies(language)?;
        Ok(Self {
            name: language.name.clone(),
            sorts,
            declarations,
            features: Vec::new(),
            vocabularies,
            open_sorts: vec![language.type_sort.clone(), language.term_sort.clone()],
            classifiers: IndexMap::from([(language.term_sort.clone(), language.type_sort.clone())]),
            allows_vacuous_binding: true,
            holes: false,
            constant_family,
            variable_family,
            category_sort: language.category_sort.clone(),
            syntax: SignatureSyntax::SExpression,
            feature_delimiter: FeatureDelimiter::Brackets,
            constructors: &[],
            functors: &[],
        })
    }

    fn extend_sorts(&self, signature: &SignatureIr) -> Result<(), LinkError> {
        for (index, sort) in signature.sorts.iter().enumerate() {
            if !self.sorts.contains(sort) && *sort != signature.category_sort {
                return Err(link_error(
                    grammar_indexed(LinkComponent::Sort, index),
                    LinkErrorKind::ForeignGrammarSort { name: sort.clone() },
                ));
            }
        }
        if !self.sorts.contains(&signature.category_sort) {
            return Err(link_error(
                grammar_field(LinkComponent::Sort),
                LinkErrorKind::UndeclaredSort {
                    name: signature.category_sort.clone(),
                },
            ));
        }
        for (index, sort) in signature.sorts.iter().enumerate() {
            if *sort != signature.category_sort && !self.sorts.contains(sort) {
                return Err(link_error(
                    grammar_indexed(LinkComponent::Sort, index),
                    LinkErrorKind::ForeignGrammarSort { name: sort.clone() },
                ));
            }
        }
        Ok(())
    }

    fn extend_declarations(&mut self, signature: &SignatureIr) -> Result<(), LinkError> {
        let targets = grammar_targets(signature)?;
        let primitives: Vec<String> = signature.primitives.clone();
        let mut feature_atoms = IndexMap::new();
        let mut operator_features: IndexMap<String, Vec<String>> = IndexMap::new();
        for (index, operator) in signature.operators.iter().enumerate() {
            let location = grammar_indexed(LinkComponent::Operator, index);
            if let Some(target) = &operator.erases_to {
                let feature = validate_feature(
                    operator,
                    target,
                    &targets,
                    signature.feature_delimiter,
                    location,
                )?;
                if primitives.contains(target) {
                    feature_atoms.insert(
                        operator.name.clone(),
                        FeaturedAtom {
                            target: target.clone(),
                            location,
                        },
                    );
                } else {
                    operator_features
                        .entry(target.clone())
                        .or_default()
                        .push(feature);
                }
            } else {
                if operator.binding.is_empty()
                    && operator.sort == signature.category_sort
                    && operator
                        .arg_sorts
                        .iter()
                        .all(|sort| *sort == signature.category_sort)
                {
                    let field = if operator.arg_sorts.is_empty() {
                        "primitives"
                    } else {
                        "constructors"
                    };
                    return Err(link_error(
                        location,
                        LinkErrorKind::ReservedOperator {
                            operator: operator.name.clone(),
                            reason: format!("belongs among the category {field}"),
                        },
                    ));
                }
                self.declarations.push(joint_operator(operator, location)?);
            }
        }
        for (index, constructor) in signature.constructors.iter().enumerate() {
            let location = grammar_indexed(LinkComponent::Constructor, index);
            validate_constructor(constructor, &signature.category_sort, location)?;
            self.declarations
                .push(constructor_operator(constructor, location)?);
        }
        self.features.extend(operator_features);
        let category = category_vocabulary(signature, feature_atoms)?;
        self.vocabularies
            .insert(signature.category_sort.clone(), category);
        Ok(())
    }

    fn extend_vocabularies(&mut self, signature: &SignatureIr) -> Result<(), LinkError> {
        for (index, (sort, names)) in signature.vocabularies.iter().enumerate() {
            if *sort == signature.category_sort {
                continue;
            }
            let location = grammar_indexed(LinkComponent::Vocabulary, index);
            require_declared_sort(&self.sorts, sort, location)?;
            self.vocabularies.insert(
                sort.clone(),
                vocabulary(sort, names.clone(), IndexMap::new(), location)?,
            );
        }
        Ok(())
    }

    fn extend_openness(&mut self, signature: &SignatureIr) -> Result<(), LinkError> {
        for (index, sort) in signature.open_sorts.iter().enumerate() {
            let location = grammar_indexed(LinkComponent::OpenSort, index);
            require_declared_sort(&self.sorts, sort, location)?;
            if !self.open_sorts.contains(sort) {
                self.open_sorts.push(sort.clone());
            }
        }
        for (index, (sort, classifier)) in signature.classifiers.iter().enumerate() {
            let location = grammar_indexed(LinkComponent::Classifier, index);
            require_declared_sort(&self.sorts, sort, location)?;
            require_declared_sort(&self.sorts, classifier, location)?;
            self.classifiers.insert(sort.clone(), classifier.clone());
        }
        Ok(())
    }

    fn finish(mut self, store_nodes: u64) -> Result<LinkedGrammar, CompileError> {
        let declared_names = self
            .declarations
            .iter()
            .map(|operator| operator.name.clone())
            .collect::<Vec<_>>();
        self.add_featured_operators()
            .map_err(|source| CompileError::Link { source })?;
        self.validate_declarations()
            .map_err(|source| CompileError::Link { source })?;
        self.add_derived_operators()
            .map_err(|source| CompileError::Link { source })?;
        let ordered = self
            .code_order()
            .map_err(|source| CompileError::Link { source })?;
        self.build_linked(&ordered, declared_names, store_nodes)
    }
}

fn language_declarations(language: &LanguageIr) -> Result<Vec<JointOperator>, LinkError> {
    let mut operators = Vec::new();
    for (index, constructor) in language.constructors.iter().enumerate() {
        let location = language_indexed(LinkComponent::Constructor, index);
        validate_constructor(constructor, &language.type_sort, location)?;
        operators.push(constructor_operator(constructor, location)?);
    }
    let church = language.binders.style == BinderStyle::Church;
    let abstraction_arguments = if church {
        vec![language.type_sort.clone(), language.term_sort.clone()]
    } else {
        vec![language.term_sort.clone()]
    };
    operators.push(JointOperator {
        name: language.abstraction.clone(),
        sort: language.term_sort.clone(),
        arguments: abstraction_arguments,
        binding: vec![JointBinding {
            arguments: vec![usize::from(church)],
            sort: language.term_sort.clone(),
            annotation: church.then_some(0),
        }],
        family: language.binder_family.clone(),
        payload: PayloadKind::None,
        erases_to: None,
        location: language_indexed(LinkComponent::Operator, operators.len()),
    });
    operators.push(JointOperator {
        name: language.application.clone(),
        sort: language.term_sort.clone(),
        arguments: vec![language.term_sort.clone(), language.term_sort.clone()],
        binding: Vec::new(),
        family: "application".to_owned(),
        payload: PayloadKind::None,
        erases_to: None,
        location: language_indexed(LinkComponent::Operator, operators.len()),
    });
    Ok(operators)
}

fn language_vocabularies(
    language: &LanguageIr,
) -> Result<IndexMap<String, JointVocabulary>, LinkError> {
    let type_names = language
        .atomic_types
        .iter()
        .map(|atomic| atomic.name.clone())
        .collect();
    let term_names = language
        .constants
        .iter()
        .map(|constant| constant.name.clone())
        .collect();
    Ok(IndexMap::from([
        (
            language.type_sort.clone(),
            vocabulary(
                &language.type_sort,
                type_names,
                IndexMap::new(),
                language_indexed(LinkComponent::Vocabulary, 0),
            )?,
        ),
        (
            language.term_sort.clone(),
            vocabulary(
                &language.term_sort,
                term_names,
                IndexMap::new(),
                language_indexed(LinkComponent::Vocabulary, 1),
            )?,
        ),
    ]))
}

fn grammar_targets(signature: &SignatureIr) -> Result<IndexMap<String, JointOperator>, LinkError> {
    let mut targets = IndexMap::new();
    for (index, primitive) in signature.primitives.iter().enumerate() {
        let location = grammar_indexed(LinkComponent::Primitive, index);
        targets.insert(
            primitive.clone(),
            JointOperator {
                name: primitive.clone(),
                sort: signature.category_sort.clone(),
                arguments: Vec::new(),
                binding: Vec::new(),
                family: "primitive".to_owned(),
                payload: PayloadKind::None,
                erases_to: None,
                location,
            },
        );
    }
    for (index, operator) in signature.operators.iter().enumerate() {
        if operator.erases_to.is_none() {
            targets.insert(
                operator.name.clone(),
                joint_operator(operator, grammar_indexed(LinkComponent::Operator, index))?,
            );
        }
    }
    for (index, constructor) in signature.constructors.iter().enumerate() {
        let location = grammar_indexed(LinkComponent::Constructor, index);
        validate_constructor(constructor, &signature.category_sort, location)?;
        targets.insert(
            constructor.name.clone(),
            constructor_operator(constructor, location)?,
        );
    }
    Ok(targets)
}

fn validate_feature(
    operator: &OperatorIr,
    target: &str,
    targets: &IndexMap<String, JointOperator>,
    delimiter: FeatureDelimiter,
    location: LinkLocation,
) -> Result<String, LinkError> {
    let feature = match delimiter {
        FeatureDelimiter::Brackets => operator
            .name
            .strip_prefix(&format!("{target}["))
            .and_then(|rest| rest.strip_suffix(']'))
            .filter(|value| !value.is_empty()),
        FeatureDelimiter::Colon => operator
            .name
            .strip_prefix(&format!("{target}:"))
            .filter(|value| !value.is_empty() && !value.contains(':')),
    }
    .ok_or_else(|| {
        feature_error(
            operator,
            target,
            "is not target plus one feature in the declared syntax",
            location,
        )
    })?;
    let bare = targets
        .get(target)
        .ok_or_else(|| feature_error(operator, target, "names an undeclared target", location))?;
    if operator.sort != bare.sort {
        return Err(feature_error(
            operator,
            target,
            "has a result sort different from its target",
            location,
        ));
    }
    if !operator.arg_sorts.is_empty()
        || !operator.binding.is_empty()
        || operator.family != "primitive"
    {
        return Err(feature_error(
            operator,
            target,
            "must take its complete shape from the target",
            location,
        ));
    }
    Ok(feature.to_owned())
}

fn feature_error(
    operator: &OperatorIr,
    target: &str,
    reason: &str,
    location: LinkLocation,
) -> LinkError {
    link_error(
        location,
        LinkErrorKind::InvalidFeature {
            operator: operator.name.clone(),
            target: target.to_owned(),
            reason: reason.to_owned(),
        },
    )
}

fn category_vocabulary(
    signature: &SignatureIr,
    feature_atoms: IndexMap<String, FeaturedAtom>,
) -> Result<JointVocabulary, LinkError> {
    let mut names = Vec::new();
    for (index, name) in signature.primitives.iter().enumerate() {
        push_category_constant(
            &mut names,
            &signature.category_sort,
            name,
            grammar_indexed(LinkComponent::Primitive, index),
        )?;
    }
    for (name, feature) in &feature_atoms {
        push_category_constant(&mut names, &signature.category_sort, name, feature.location)?;
    }
    let vocabulary_location = signature
        .vocabularies
        .iter()
        .position(|(sort, _)| sort == &signature.category_sort)
        .map_or(grammar_field(LinkComponent::Vocabulary), |index| {
            grammar_indexed(LinkComponent::Vocabulary, index)
        });
    if let Some(explicit) = signature.vocabularies.get(&signature.category_sort) {
        for name in explicit {
            push_category_constant(
                &mut names,
                &signature.category_sort,
                name,
                vocabulary_location,
            )?;
        }
    }
    let cores = feature_atoms
        .into_iter()
        .map(|(name, feature)| (name, feature.target))
        .collect();
    vocabulary(&signature.category_sort, names, cores, vocabulary_location)
}

fn push_category_constant(
    names: &mut Vec<String>,
    sort: &str,
    name: &str,
    location: LinkLocation,
) -> Result<(), LinkError> {
    if names.iter().any(|held| held == name) {
        return Err(link_error(
            location,
            LinkErrorKind::DuplicateConstant {
                sort: sort.to_owned(),
                name: name.to_owned(),
            },
        ));
    }
    names.push(name.to_owned());
    Ok(())
}

fn vocabulary(
    sort: &str,
    names: Vec<String>,
    cores: IndexMap<String, String>,
    location: LinkLocation,
) -> Result<JointVocabulary, LinkError> {
    let mut seen = Vec::new();
    for name in &names {
        if seen.contains(name) {
            return Err(link_error(
                location,
                LinkErrorKind::DuplicateConstant {
                    sort: sort.to_owned(),
                    name: name.clone(),
                },
            ));
        }
        seen.push(name.clone());
    }
    for (source, target) in &cores {
        if !names.contains(source) || !names.contains(target) || source == target {
            return Err(link_error(
                location,
                LinkErrorKind::InvalidFeature {
                    operator: source.clone(),
                    target: target.clone(),
                    reason: "does not form a distinct direct vocabulary core".to_owned(),
                },
            ));
        }
    }
    Ok(JointVocabulary {
        names,
        cores,
        location,
    })
}

fn joint_operator(
    operator: &OperatorIr,
    location: LinkLocation,
) -> Result<JointOperator, LinkError> {
    let binding = operator
        .binding
        .iter()
        .map(|site| {
            let arguments = site
                .arguments
                .iter()
                .copied()
                .map(|position| {
                    usize::try_from(position).map_err(|_| {
                        binding_error(
                            &operator.name,
                            "has an argument position outside this platform's index range",
                            location,
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let annotation = site
                .annotation
                .map(usize::try_from)
                .transpose()
                .map_err(|_| {
                    binding_error(
                        &operator.name,
                        "has an annotation position outside this platform's index range",
                        location,
                    )
                })?;
            Ok(JointBinding {
                arguments,
                sort: site.sort.clone(),
                annotation,
            })
        })
        .collect::<Result<Vec<_>, LinkError>>()?;
    Ok(JointOperator {
        name: operator.name.clone(),
        sort: operator.sort.clone(),
        arguments: operator.arg_sorts.clone(),
        binding,
        family: operator.family.clone(),
        payload: PayloadKind::None,
        erases_to: operator.erases_to.clone(),
        location,
    })
}

fn constructor_operator(
    constructor: &ConstructorIr,
    location: LinkLocation,
) -> Result<JointOperator, LinkError> {
    let Some((sort, arguments)) = constructor.kind.split_last() else {
        return Err(link_error(
            location,
            LinkErrorKind::InvalidConstructorKind {
                constructor: constructor.name.clone(),
                expected: String::new(),
                received: Vec::new(),
            },
        ));
    };
    Ok(JointOperator {
        name: constructor.name.clone(),
        sort: sort.clone(),
        arguments: arguments.to_vec(),
        binding: Vec::new(),
        family: "constructor".to_owned(),
        payload: PayloadKind::None,
        erases_to: None,
        location,
    })
}

fn validate_constructor(
    constructor: &ConstructorIr,
    sort: &str,
    location: LinkLocation,
) -> Result<(), LinkError> {
    if constructor.kind.len() < 2 || constructor.kind.iter().any(|held| held != sort) {
        return Err(link_error(
            location,
            LinkErrorKind::InvalidConstructorKind {
                constructor: constructor.name.clone(),
                expected: sort.to_owned(),
                received: constructor.kind.clone(),
            },
        ));
    }
    let arity = constructor.kind.len() - 1;
    match &constructor.notation {
        ConstructorNotationIr::Prefix(notation) if notation.spelling.is_empty() => {
            notation_error(constructor, "has an empty prefix spelling", location)
        }
        ConstructorNotationIr::Infix(notation)
            if notation.spelling.is_empty() || notation.precedence == 0 || arity != 2 =>
        {
            notation_error(
                constructor,
                "needs a nonempty, positive-precedence binary infix notation",
                location,
            )
        }
        ConstructorNotationIr::Mixfix(notation)
            if notation.precedence == 0
                || notation.parts.len() != arity + 1
                || notation.parts.first().is_none_or(String::is_empty) =>
        {
            notation_error(
                constructor,
                "has a mixfix notation incompatible with its arity",
                location,
            )
        }
        _ => Ok(()),
    }
}

fn notation_error(
    constructor: &ConstructorIr,
    reason: &str,
    location: LinkLocation,
) -> Result<(), LinkError> {
    Err(link_error(
        location,
        LinkErrorKind::InvalidConstructorNotation {
            constructor: constructor.name.clone(),
            reason: reason.to_owned(),
        },
    ))
}

impl JointSignature<'_> {
    fn add_featured_operators(&mut self) -> Result<(), LinkError> {
        let mut generated = Vec::new();
        for (target, features) in &self.features {
            let bare = self
                .declarations
                .iter()
                .find(|operator| operator.name == *target)
                .ok_or_else(|| {
                    link_error(
                        grammar_field(LinkComponent::Operator),
                        LinkErrorKind::InvalidFeature {
                            operator: target.clone(),
                            target: target.clone(),
                            reason: "names an undeclared operator target".to_owned(),
                        },
                    )
                })?;
            for feature in features {
                let mut operator = bare.clone();
                operator.name = self.feature_delimiter.name(target, feature);
                operator.erases_to = Some(target.clone());
                operator.location = grammar_field(LinkComponent::Operator);
                generated.push(operator);
            }
        }
        self.declarations.extend(generated);
        Ok(())
    }

    fn validate_declarations(&self) -> Result<(), LinkError> {
        let mut names = Vec::new();
        for operator in &self.declarations {
            validate_operator_name(operator, &self.constant_family, &self.variable_family)?;
            if names.contains(&operator.name) {
                return Err(link_error(
                    operator.location,
                    LinkErrorKind::DuplicateOperator {
                        name: operator.name.clone(),
                    },
                ));
            }
            names.push(operator.name.clone());
            validate_operator_shape(operator, &self.sorts, &self.classifiers)?;
        }
        for operator in &self.declarations {
            let Some(target_name) = &operator.erases_to else {
                continue;
            };
            let target = self
                .declarations
                .iter()
                .find(|candidate| candidate.name == *target_name)
                .ok_or_else(|| {
                    link_error(
                        operator.location,
                        LinkErrorKind::InvalidFeature {
                            operator: operator.name.clone(),
                            target: target_name.clone(),
                            reason: "names an undeclared erasure target".to_owned(),
                        },
                    )
                })?;
            if target.erases_to.is_some()
                || target.sort != operator.sort
                || target.arguments != operator.arguments
            {
                return Err(link_error(
                    operator.location,
                    LinkErrorKind::InvalidFeature {
                        operator: operator.name.clone(),
                        target: target_name.clone(),
                        reason: "does not preserve an idempotent operator shape".to_owned(),
                    },
                ));
            }
        }
        Ok(())
    }

    fn add_derived_operators(&mut self) -> Result<(), LinkError> {
        let bound = self
            .declarations
            .iter()
            .flat_map(|operator| operator.binding.iter().map(|site| site.sort.clone()))
            .collect::<Vec<_>>();
        let location = grammar_field(LinkComponent::Signature);
        for sort in &self.sorts {
            if self.vocabularies.contains_key(sort) {
                self.declarations.push(derived_operator(
                    format!("#const[{sort}]"),
                    sort,
                    &self.constant_family,
                    PayloadKind::Constant,
                    location,
                ));
            }
            if bound.contains(sort) {
                self.declarations.push(derived_operator(
                    format!("#bvar[{sort}]"),
                    sort,
                    &self.variable_family,
                    PayloadKind::Index,
                    location,
                ));
            }
            if self.open_sorts.contains(sort) || bound.contains(sort) {
                self.declarations.push(derived_operator(
                    format!("#fvar[{sort}]"),
                    sort,
                    &self.variable_family,
                    PayloadKind::Free,
                    location,
                ));
            }
            if self.holes {
                self.declarations.push(derived_operator(
                    format!("#hole[{sort}]"),
                    sort,
                    "hole",
                    PayloadKind::None,
                    location,
                ));
            }
        }
        let mut names = Vec::new();
        for operator in &self.declarations {
            if names.contains(&operator.name) {
                return Err(link_error(
                    operator.location,
                    LinkErrorKind::DuplicateOperator {
                        name: operator.name.clone(),
                    },
                ));
            }
            names.push(operator.name.clone());
        }
        Ok(())
    }

    fn code_order(&self) -> Result<Vec<JointOperator>, LinkError> {
        if self.sorts.len() > usize::from(u16::MAX) + 1 {
            return Err(inventory_overflow("sorts", self.sorts.len()));
        }
        if u32::try_from(self.declarations.len()).is_err() {
            return Err(inventory_overflow("operators", self.declarations.len()));
        }
        let mut families = vec![self.constant_family.clone(), self.variable_family.clone()];
        for operator in &self.declarations {
            if operator.family != "hole" && !families.contains(&operator.family) {
                families.push(operator.family.clone());
            }
        }
        families.push("hole".to_owned());
        let mut indexed = self
            .declarations
            .iter()
            .cloned()
            .enumerate()
            .collect::<Vec<_>>();
        indexed.sort_by_key(|(index, operator)| {
            (
                self.sorts.iter().position(|sort| *sort == operator.sort),
                families
                    .iter()
                    .position(|family| *family == operator.family),
                operator.arguments.len(),
                *index,
            )
        });
        Ok(indexed.into_iter().map(|(_, operator)| operator).collect())
    }

    fn build_linked(
        self,
        ordered: &[JointOperator],
        declared_names: Vec<String>,
        store_nodes: u64,
    ) -> Result<LinkedGrammar, CompileError> {
        let sort_ids =
            dense_sort_ids(&self.sorts).map_err(|source| CompileError::Link { source })?;
        let operator_ids =
            dense_operator_ids(ordered).map_err(|source| CompileError::Link { source })?;
        let signature = self
            .core_signature(ordered, &sort_ids, &operator_ids)
            .map_err(|source| CompileError::Link { source })?;
        let category_syntax = self
            .link_category_syntax(&operator_ids, &sort_ids)
            .map_err(|source| CompileError::Link { source })?;
        let fingerprint = self.fingerprint(ordered, &operator_ids)?;
        let vocabularies = self
            .linked_vocabularies(&sort_ids)
            .map_err(|source| CompileError::Link { source })?;
        let operators = linked_operators(ordered, &sort_ids, &operator_ids);
        let open_sorts = self
            .sorts
            .iter()
            .filter(|sort| self.open_sorts.contains(sort))
            .map(|sort| sort_ids[sort])
            .collect();
        let classifiers = self
            .sorts
            .iter()
            .filter_map(|sort| {
                self.classifiers
                    .get(sort)
                    .map(|classifier| LinkedClassifier {
                        sort: sort_ids[sort],
                        classifier: sort_ids[classifier],
                    })
            })
            .collect();
        let features = self
            .features
            .iter()
            .map(|(name, values)| LinkedOperatorFeatures {
                name: name.clone(),
                operator: operator_ids[name],
                values: values.clone(),
            })
            .collect();
        let declarations = declared_names
            .into_iter()
            .map(|name| LinkedOperatorDeclaration {
                operator: operator_ids[&name],
                name,
            })
            .collect::<Vec<_>>();
        let counts = LinkedGrammarCounts {
            sorts: self.sorts.len(),
            declarations: declarations.len(),
            operators: ordered.len(),
            vocabulary_constants: vocabularies
                .iter()
                .map(|vocabulary| vocabulary.names.len())
                .sum(),
            category_constructors: category_syntax.constructors.len(),
        };
        let capacity = usize::try_from(store_nodes).map_err(|_| CompileError::Link {
            source: link_error(
                language_field(LinkComponent::StoreCapacity),
                LinkErrorKind::InventoryOverflow {
                    inventory: "store nodes",
                    size: usize::MAX,
                },
            ),
        })?;
        let store =
            TermStore::new(signature, capacity).map_err(|source| CompileError::Store { source })?;
        Ok(LinkedGrammar {
            ir: LinkedGrammarIr {
                name: self.name,
                sorts: self.sorts,
                operators,
                declarations,
                vocabularies,
                open_sorts,
                classifiers,
                features,
                holes: self.holes,
                allows_vacuous_binding: self.allows_vacuous_binding,
                constant_family: self.constant_family,
                variable_family: self.variable_family,
                category_syntax,
                fingerprint,
                counts,
            },
            store,
            declaration_fingerprint: None,
        })
    }
}

fn linked_operators(
    ordered: &[JointOperator],
    sort_ids: &HashMap<String, SortId>,
    operator_ids: &HashMap<String, OperatorId>,
) -> Vec<LinkedOperator> {
    ordered
        .iter()
        .map(|operator| LinkedOperator {
            name: operator.name.clone(),
            operator: operator_ids[&operator.name],
            sort: sort_ids[&operator.sort],
            arguments: operator
                .arguments
                .iter()
                .map(|sort| sort_ids[sort])
                .collect(),
            family: operator.family.clone(),
            payload: linked_payload(operator.payload),
            erased: operator
                .erases_to
                .as_ref()
                .map_or(operator_ids[&operator.name], |target| operator_ids[target]),
            binding: operator
                .binding
                .iter()
                .map(|site| LinkedBindingSite {
                    arguments: site.arguments.clone(),
                    sort: sort_ids[&site.sort],
                    annotation: site.annotation,
                })
                .collect(),
        })
        .collect()
}

const fn linked_payload(payload: PayloadKind) -> LinkedPayloadKind {
    match payload {
        PayloadKind::None => LinkedPayloadKind::None,
        PayloadKind::Constant => LinkedPayloadKind::Constant,
        PayloadKind::Index => LinkedPayloadKind::Index,
        PayloadKind::Free => LinkedPayloadKind::Free,
    }
}

fn validate_operator_name(
    operator: &JointOperator,
    constant_family: &str,
    variable_family: &str,
) -> Result<(), LinkError> {
    let reason = if operator.name.starts_with('#') {
        Some("takes a name beginning with the kernel prefix '#'".to_owned())
    } else if [constant_family, variable_family, "hole"].contains(&operator.family.as_str()) {
        Some(format!(
            "declares the kernel-owned family {:?}",
            operator.family
        ))
    } else {
        None
    };
    if let Some(reason) = reason {
        return Err(link_error(
            operator.location,
            LinkErrorKind::ReservedOperator {
                operator: operator.name.clone(),
                reason,
            },
        ));
    }
    Ok(())
}

fn validate_operator_shape(
    operator: &JointOperator,
    sorts: &[String],
    classifiers: &IndexMap<String, String>,
) -> Result<(), LinkError> {
    require_declared_sort(sorts, &operator.sort, operator.location)?;
    for sort in &operator.arguments {
        require_declared_sort(sorts, sort, operator.location)?;
    }
    for site in &operator.binding {
        require_declared_sort(sorts, &site.sort, operator.location)?;
        if site.arguments.is_empty()
            || site.arguments.windows(2).any(|pair| pair[0] >= pair[1])
            || site
                .arguments
                .iter()
                .any(|position| *position >= operator.arguments.len())
        {
            return Err(binding_error(
                &operator.name,
                "has empty, unordered, or out-of-range scope positions",
                operator.location,
            ));
        }
        if let Some(annotation) = site.annotation {
            if annotation >= operator.arguments.len() || site.arguments.contains(&annotation) {
                return Err(binding_error(
                    &operator.name,
                    "has an out-of-range or scoped annotation position",
                    operator.location,
                ));
            }
            let Some(classifier) = classifiers.get(&site.sort) else {
                return Err(binding_error(
                    &operator.name,
                    "annotates a sort without a classifier",
                    operator.location,
                ));
            };
            if operator.arguments[annotation] != *classifier {
                return Err(binding_error(
                    &operator.name,
                    "puts an annotation at an argument of the wrong classifier sort",
                    operator.location,
                ));
            }
        }
    }
    Ok(())
}

fn derived_operator(
    name: String,
    sort: &str,
    family: &str,
    payload: PayloadKind,
    location: LinkLocation,
) -> JointOperator {
    JointOperator {
        name,
        sort: sort.to_owned(),
        arguments: Vec::new(),
        binding: Vec::new(),
        family: family.to_owned(),
        payload,
        erases_to: None,
        location,
    }
}

fn dense_sort_ids(sorts: &[String]) -> Result<HashMap<String, SortId>, LinkError> {
    sorts
        .iter()
        .enumerate()
        .map(|(index, sort)| {
            let code =
                u16::try_from(index).map_err(|_| inventory_overflow("sorts", sorts.len()))?;
            Ok((sort.clone(), code))
        })
        .collect()
}

fn dense_operator_ids(
    operators: &[JointOperator],
) -> Result<HashMap<String, OperatorId>, LinkError> {
    operators
        .iter()
        .enumerate()
        .map(|(index, operator)| {
            let code = u32::try_from(index)
                .map_err(|_| inventory_overflow("operators", operators.len()))?;
            Ok((operator.name.clone(), code))
        })
        .collect()
}

impl JointSignature<'_> {
    fn core_signature(
        &self,
        ordered: &[JointOperator],
        sort_ids: &HashMap<String, SortId>,
        operator_ids: &HashMap<String, OperatorId>,
    ) -> Result<CoreSignatureSpec, LinkError> {
        let mut vocabularies = vec![None; self.sorts.len()];
        let mut constant_cores = vec![None; self.sorts.len()];
        for (sort, vocabulary) in &self.vocabularies {
            let sort_id = usize::from(sort_ids[sort]);
            vocabularies[sort_id] = Some(vocabulary.names.len());
            constant_cores[sort_id] = Some(vocabulary_core_ids(vocabulary)?);
        }
        let mut constants = vec![None; self.sorts.len()];
        let mut bound = vec![None; self.sorts.len()];
        let mut free = vec![None; self.sorts.len()];
        let mut holes = vec![None; self.sorts.len()];
        let operators = ordered
            .iter()
            .map(|operator| {
                let code = operator_ids[&operator.name];
                let sort = sort_ids[&operator.sort];
                match operator.payload {
                    PayloadKind::Constant => constants[usize::from(sort)] = Some(code),
                    PayloadKind::Index => bound[usize::from(sort)] = Some(code),
                    PayloadKind::Free => free[usize::from(sort)] = Some(code),
                    PayloadKind::None if operator.name.starts_with("#hole[") => {
                        holes[usize::from(sort)] = Some(code);
                    }
                    PayloadKind::None => {}
                }
                core_operator(operator, sort_ids, operator_ids)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CoreSignatureSpec {
            name: self.name.clone(),
            sorts: self.sorts.clone(),
            operators,
            vocabularies,
            constant_cores,
            constants,
            bound,
            free,
            holes,
            allows_vacuous_binding: self.allows_vacuous_binding,
        })
    }

    fn linked_vocabularies(
        &self,
        sort_ids: &HashMap<String, SortId>,
    ) -> Result<Vec<LinkedVocabulary>, LinkError> {
        self.sorts
            .iter()
            .filter_map(|sort| {
                self.vocabularies.get(sort).map(|vocabulary| {
                    Ok(LinkedVocabulary {
                        sort: sort_ids[sort],
                        names: vocabulary.names.clone(),
                        cores: vocabulary_core_ids(vocabulary)?,
                    })
                })
            })
            .collect()
    }
}

fn core_operator(
    operator: &JointOperator,
    sort_ids: &HashMap<String, SortId>,
    operator_ids: &HashMap<String, OperatorId>,
) -> Result<CoreOperatorSpec, LinkError> {
    let binding = operator
        .binding
        .iter()
        .map(|site| CoreBindingSite {
            arguments: site.arguments.clone(),
            sort: sort_ids[&site.sort],
        })
        .collect();
    let shifts = (0..operator.arguments.len())
        .map(|position| {
            let mut row = Vec::new();
            for sort in sort_ids.keys() {
                let amount = operator
                    .binding
                    .iter()
                    .filter(|site| site.sort == *sort && site.arguments.contains(&position))
                    .count();
                if amount != 0 {
                    let amount = u32::try_from(amount).map_err(|_| {
                        binding_error(
                            &operator.name,
                            "binds too many variables of one sort at one argument",
                            operator.location,
                        )
                    })?;
                    row.push((sort_ids[sort], amount));
                }
            }
            row.sort_by_key(|(sort, _)| *sort);
            Ok(row)
        })
        .collect::<Result<Vec<_>, LinkError>>()?;
    Ok(CoreOperatorSpec {
        name: operator.name.clone(),
        sort: sort_ids[&operator.sort],
        arguments: operator
            .arguments
            .iter()
            .map(|sort| sort_ids[sort])
            .collect(),
        payload: operator.payload,
        erased: operator
            .erases_to
            .as_ref()
            .map_or(operator_ids[&operator.name], |target| operator_ids[target]),
        binding,
        shifts,
    })
}

fn vocabulary_core_ids(vocabulary: &JointVocabulary) -> Result<Vec<usize>, LinkError> {
    vocabulary
        .names
        .iter()
        .map(|name| {
            let target = vocabulary.cores.get(name).unwrap_or(name);
            vocabulary
                .names
                .iter()
                .position(|candidate| candidate == target)
                .ok_or_else(|| {
                    link_error(
                        vocabulary.location,
                        LinkErrorKind::InvalidFeature {
                            operator: name.clone(),
                            target: target.clone(),
                            reason: "names a core outside its vocabulary".to_owned(),
                        },
                    )
                })
        })
        .collect()
}

impl JointSignature<'_> {
    fn link_category_syntax(
        &self,
        operator_ids: &HashMap<String, OperatorId>,
        sort_ids: &HashMap<String, SortId>,
    ) -> Result<LinkedCategorySyntax, LinkError> {
        let kind = match self.syntax {
            SignatureSyntax::Declared => CategorySyntaxKind::Declared,
            SignatureSyntax::SExpression => CategorySyntaxKind::SExpression,
        };
        if kind == CategorySyntaxKind::Declared && self.constructors.is_empty() {
            return Err(link_error(
                grammar_field(LinkComponent::CategorySyntax),
                LinkErrorKind::InvalidConstructorNotation {
                    constructor: "category syntax".to_owned(),
                    reason: "requires at least one declared constructor".to_owned(),
                },
            ));
        }
        let constructors = self
            .constructors
            .iter()
            .map(|constructor| LinkedCategoryConstructor {
                name: constructor.name.clone(),
                operator: operator_ids[&constructor.name],
                notation: linked_notation(&constructor.notation),
            })
            .collect();
        let functors = self.link_functors(operator_ids)?;
        Ok(LinkedCategorySyntax {
            kind,
            feature_delimiter: self.feature_delimiter,
            category_sort: sort_ids[&self.category_sort],
            constructors,
            functors,
        })
    }

    fn link_functors(
        &self,
        operator_ids: &HashMap<String, OperatorId>,
    ) -> Result<Vec<LinkedFunctor>, LinkError> {
        let mut views = IndexMap::new();
        for (index, view) in self.functors.iter().enumerate() {
            let location = grammar_indexed(LinkComponent::CategorySyntax, index);
            if views.insert(view.name.clone(), view).is_some() {
                return Err(functor_error(view, "is declared more than once", location));
            }
        }
        self.functors
            .iter()
            .enumerate()
            .map(|(index, view)| {
                let location = grammar_indexed(LinkComponent::CategorySyntax, index);
                let constructor = self
                    .constructors
                    .iter()
                    .find(|constructor| constructor.name == view.constructor)
                    .ok_or_else(|| {
                        functor_error(view, "names an undeclared constructor", location)
                    })?;
                if constructor.kind.len() != 3 {
                    return Err(functor_error(
                        view,
                        "requires a binary constructor",
                        location,
                    ));
                }
                let result = usize::try_from(view.result).map_err(|_| {
                    functor_error(view, "has an out-of-range result slot", location)
                })?;
                let argument = usize::try_from(view.argument).map_err(|_| {
                    functor_error(view, "has an out-of-range argument slot", location)
                })?;
                if result >= 2 || argument >= 2 || result == argument {
                    return Err(functor_error(
                        view,
                        "requires distinct slots zero and one",
                        location,
                    ));
                }
                let opposite = views
                    .get(&view.opposite)
                    .ok_or_else(|| functor_error(view, "has no declared opposite", location))?;
                validate_opposite(view, opposite, location)?;
                Ok(LinkedFunctor {
                    name: view.name.clone(),
                    constructor: operator_ids[&view.constructor],
                    opposite: view.opposite.clone(),
                    argument_side: linked_side(view.argument_side),
                    result,
                    argument,
                })
            })
            .collect()
    }

    fn fingerprint(
        &self,
        ordered: &[JointOperator],
        operator_ids: &HashMap<String, OperatorId>,
    ) -> Result<String, CompileError> {
        let mut rows = Vec::new();
        for operator in ordered {
            let code = u64::from(operator_ids[&operator.name]) + 1;
            rows.push(json!([
                code,
                operator.name,
                operator.arguments.len(),
                operator.sort,
                operator.erases_to.as_ref().unwrap_or(&operator.name)
            ]));
        }
        rows.extend(
            self.sorts
                .iter()
                .enumerate()
                .map(|(index, sort)| json!(["sort", index, sort])),
        );
        for operator in ordered {
            rows.extend(
                operator
                    .arguments
                    .iter()
                    .enumerate()
                    .map(|(index, sort)| json!(["argument", operator.name, index, sort])),
            );
        }
        Self::fingerprint_operator_rows(ordered, &mut rows);
        self.fingerprint_signature_rows(&mut rows)?;
        let bytes = canonical::value_bytes(&rows).map_err(|source| CompileError::Encode {
            declaration: DeclarationKind::Grammar,
            source,
        })?;
        Ok(sha256_hex(&bytes))
    }

    fn fingerprint_operator_rows(ordered: &[JointOperator], rows: &mut Vec<Value>) {
        for operator in ordered {
            rows.push(json!([
                "family",
                operator.name,
                operator.family,
                payload_name(operator.payload)
            ]));
        }
        for operator in ordered {
            for (index, site) in operator.binding.iter().enumerate() {
                let scope = site
                    .arguments
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                rows.push(json!([
                    operator.name,
                    index,
                    scope,
                    site.sort,
                    site.annotation
                ]));
            }
        }
    }

    fn fingerprint_signature_rows(&self, rows: &mut Vec<Value>) -> Result<(), CompileError> {
        for sort in &self.sorts {
            if let Some(vocabulary) = self.vocabularies.get(sort) {
                let written = canonical::value_bytes(&vocabulary.names).map_err(|source| {
                    CompileError::Encode {
                        declaration: DeclarationKind::Grammar,
                        source,
                    }
                })?;
                rows.push(json!(["vocabulary", sort, sha256_hex(&written)]));
                for name in &vocabulary.names {
                    if let Some(core) = vocabulary.cores.get(name) {
                        rows.push(json!(["constant-core", sort, name, core]));
                    }
                }
            }
        }
        rows.extend(
            self.sorts
                .iter()
                .filter(|sort| self.open_sorts.contains(sort))
                .map(|sort| json!(["open", sort])),
        );
        rows.extend(self.sorts.iter().filter_map(|sort| {
            self.classifiers
                .get(sort)
                .map(|classifier| json!(["classifier", sort, classifier]))
        }));
        for (primitive, features) in &self.features {
            rows.extend(
                features
                    .iter()
                    .enumerate()
                    .map(|(index, feature)| json!(["feature", primitive, index, feature])),
            );
        }
        rows.push(json!(["holes", usize::from(self.holes)]));
        rows.push(json!([
            "binding-policy",
            "vacuous",
            usize::from(self.allows_vacuous_binding)
        ]));
        Ok(())
    }
}

fn linked_notation(notation: &ConstructorNotationIr) -> LinkedConstructorNotation {
    match notation {
        ConstructorNotationIr::Prefix(notation) => LinkedConstructorNotation::Prefix {
            spelling: notation.spelling.clone(),
        },
        ConstructorNotationIr::Infix(notation) => LinkedConstructorNotation::Infix {
            spelling: notation.spelling.clone(),
            precedence: notation.precedence,
            associativity: match notation.associativity {
                ConstructorAssociativity::Left => LinkedAssociativity::Left,
                ConstructorAssociativity::Right => LinkedAssociativity::Right,
                ConstructorAssociativity::None => LinkedAssociativity::None,
            },
        },
        ConstructorNotationIr::Mixfix(notation) => LinkedConstructorNotation::Mixfix {
            parts: notation.parts.clone(),
            precedence: notation.precedence,
        },
    }
}

const fn linked_side(side: ArgumentSide) -> LinkedArgumentSide {
    match side {
        ArgumentSide::Left => LinkedArgumentSide::Left,
        ArgumentSide::Right => LinkedArgumentSide::Right,
        ArgumentSide::Either => LinkedArgumentSide::Either,
    }
}

fn validate_opposite(
    view: &FunctorIr,
    opposite: &FunctorIr,
    location: LinkLocation,
) -> Result<(), LinkError> {
    if opposite.opposite != view.name {
        return Err(functor_error(
            view,
            "does not have a mutual opposite",
            location,
        ));
    }
    if (view.result, view.argument) != (opposite.result, opposite.argument) {
        return Err(functor_error(
            view,
            "disagrees with its opposite on child slots",
            location,
        ));
    }
    let sides_match = matches!(
        (view.argument_side, opposite.argument_side),
        (ArgumentSide::Right, ArgumentSide::Left)
            | (ArgumentSide::Left, ArgumentSide::Right)
            | (ArgumentSide::Either, ArgumentSide::Either)
    );
    if !sides_match {
        return Err(functor_error(
            view,
            "has an incompatible opposite side",
            location,
        ));
    }
    if view.argument_side == ArgumentSide::Either && view.constructor != opposite.constructor {
        return Err(functor_error(
            view,
            "uses another constructor for a self-opposite view",
            location,
        ));
    }
    Ok(())
}

fn functor_error(view: &FunctorIr, reason: &str, location: LinkLocation) -> LinkError {
    link_error(
        location,
        LinkErrorKind::InvalidFunctor {
            name: view.name.clone(),
            reason: reason.to_owned(),
        },
    )
}

const fn payload_name(payload: PayloadKind) -> &'static str {
    match payload {
        PayloadKind::None => "none",
        PayloadKind::Constant => "constant",
        PayloadKind::Index => "index",
        PayloadKind::Free => "free",
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

fn push_sort(sorts: &mut Vec<String>, sort: &str, location: LinkLocation) -> Result<(), LinkError> {
    if sorts.iter().any(|held| held == sort) {
        return Err(link_error(
            location,
            LinkErrorKind::DuplicateSort {
                name: sort.to_owned(),
            },
        ));
    }
    sorts.push(sort.to_owned());
    Ok(())
}

fn require_declared_sort(
    sorts: &[String],
    sort: &str,
    location: LinkLocation,
) -> Result<(), LinkError> {
    if sorts.iter().any(|held| held == sort) {
        Ok(())
    } else {
        Err(link_error(
            location,
            LinkErrorKind::UndeclaredSort {
                name: sort.to_owned(),
            },
        ))
    }
}

fn binding_error(operator: &str, reason: &str, location: LinkLocation) -> LinkError {
    link_error(
        location,
        LinkErrorKind::InvalidBinding {
            operator: operator.to_owned(),
            reason: reason.to_owned(),
        },
    )
}

const fn inventory_overflow(inventory: &'static str, size: usize) -> LinkError {
    link_error(
        grammar_field(LinkComponent::Signature),
        LinkErrorKind::InventoryOverflow { inventory, size },
    )
}

const fn link_error(location: LinkLocation, kind: LinkErrorKind) -> LinkError {
    LinkError::new(location, kind)
}

const fn language_field(component: LinkComponent) -> LinkLocation {
    LinkLocation::field(DeclarationKind::Language, component)
}

const fn language_indexed(component: LinkComponent, index: usize) -> LinkLocation {
    LinkLocation::indexed(DeclarationKind::Language, component, index)
}

const fn grammar_field(component: LinkComponent) -> LinkLocation {
    LinkLocation::field(DeclarationKind::Grammar, component)
}

const fn grammar_indexed(component: LinkComponent, index: usize) -> LinkLocation {
    LinkLocation::indexed(DeclarationKind::Grammar, component, index)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FiniteF64(f64);

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
struct Flag(bool);

impl Serialize for FiniteF64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for FiniteF64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FiniteVisitor;

        impl de::Visitor<'_> for FiniteVisitor {
            type Value = FiniteF64;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a finite JSON number")
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value.is_finite() {
                    Ok(FiniteF64(value))
                } else {
                    Err(E::custom("a declaration float must be finite"))
                }
            }
        }

        deserializer.deserialize_f64(FiniteVisitor)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use proptest::prelude::*;
    use serde_json::json;

    use super::{
        ApplicationRuleIr, ApplicationRuleKind, BinderStyle, CompositionRestriction,
        ConstructorNotationIr, FeatureDelimiter, GrammarIr, HeadPolicy, LanguageIr,
        MixfixConstructorNotationIr, MixfixStyle, OpaqueRuleIr, OperatorIr,
        PrefixConstructorNotationIr, PrefixStyle, RuleEntryIr, RuleIr,
    };
    use crate::{
        CategorySyntaxKind, CompileError, DeclarationEnvelope, DeclarationKind, GrammarCompiler,
        LinkComponent, LinkDeclarations, LinkErrorKind, LinkedConstructorNotation, LinkedGrammar,
        LinkedPayloadKind, LinkedRuleKind, PreparedGrammarIr, RulePlanErrorKind, RulePlanLocation,
        RulePlanSource, canonical,
    };

    const LANGUAGE: &[u8] = br#"{"abstraction":"lam","application":"eval","atomic_types":[{"kind":"type","name":"e"}],"binder_family":"binder","binders":{"style":"curry"},"bounds":{"Cat":{"depth":4,"free_ranks":0,"layout":"complete-tree","nodes":null,"order":"df"},"Tm":{"depth":8,"free_ranks":0,"layout":"preorder-arity","nodes":24,"order":"df"},"Ty":{"depth":4,"free_ranks":0,"layout":"complete-tree","nodes":null,"order":"df"}},"category_sort":"Cat","constant_family":"constant","constants":[],"constructors":[{"kind":["Ty","Ty","Ty"],"name":"->","notation":{"associativity":"right","precedence":1,"spelling":"\u2192","style":"infix"}}],"function_type":"->","leaf_lexicon":null,"logic":{"domains":{},"equality":true,"functions":true,"intensional":false,"order":"first","postulates":[]},"name":"unicode-\ud83d\ude00","store_nodes":1000000,"term_sort":"Tm","type_sort":"Ty","variable_family":"variable"}"#;

    const GRAMMAR: &[u8] = br#"{"budget":{"beam":8,"beta_fuel":1000,"closure_nodes":2048,"closure_rounds":2,"depth":5,"mass":1.0,"max_category_arity":null,"max_category_size":null,"threads":1,"unary_rounds":1},"capacity_evidence":null,"formalism":"declared","goal":{"categories":["S"],"kind":"root","root_weights":null},"interpretations":[{"name":"category","reading":{"kind":"category"}},{"name":"string","reading":{"kind":"string"}},{"name":"dependency","reading":{"kind":"dependency"}},{"name":"meaning","reading":{"kind":"meaning"}}],"lexicon":{"entries":[{"category":"S","form":"a","meaning":null,"weight":0.0}],"kind":"inline"},"matching":{"kind":"identity"},"max_hypotheses":0,"name":"tiny","normal_form":{"kind":"none"},"rules":{"name":"none","rules":[]},"signature":{"allows_vacuous_binding":true,"category_sort":"Cat","classifiers":{},"conjunctions":null,"constructors":[],"feature_delimiter":"brackets","functors":[],"holes":false,"name":"tiny","open_sorts":[],"operators":[],"primitives":["S"],"sorts":["Cat"],"syntax":"declared","vocabularies":{}}}"#;
    const PROPOSITIONAL: &[u8] = include_bytes!("../tests/fixtures/propositional-language.json");
    const CCGBANK: &[u8] = include_bytes!("../tests/fixtures/ccgbank-grammar.json");
    const BASE_OPERATOR_ORDER: [&str; 11] = [
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
    ];

    fn packaged_declarations() -> Result<(LanguageIr, GrammarIr), serde_json::Error> {
        Ok((
            serde_json::from_slice(PROPOSITIONAL)?,
            serde_json::from_slice(CCGBANK)?,
        ))
    }

    fn prepare_through_boundary(
        language: &LanguageIr,
        grammar: &GrammarIr,
    ) -> Result<PreparedGrammarIr, Box<dyn Error>> {
        let language_bytes = canonical::language_bytes(language)?;
        let grammar_bytes = canonical::grammar_bytes(grammar)?;
        Ok(GrammarCompiler.prepare(
            DeclarationEnvelope::new(&language_bytes),
            DeclarationEnvelope::new(&grammar_bytes),
        )?)
    }

    fn link_through_boundary(
        language: &LanguageIr,
        grammar: &GrammarIr,
    ) -> Result<LinkedGrammar, Box<dyn Error>> {
        let prepared = prepare_through_boundary(language, grammar)?;
        Ok(GrammarCompiler.link(&prepared)?)
    }

    fn grammar_with(
        field: &str,
        declaration: &serde_json::Value,
    ) -> Result<(LanguageIr, GrammarIr), Box<dyn Error>> {
        let (language, grammar) = packaged_declarations()?;
        let mut document = serde_json::to_value(&grammar)?;
        document[field] = declaration.clone();
        Ok((language, serde_json::from_value(document)?))
    }

    fn plan_with(field: &str, declaration: &serde_json::Value) -> Result<(), Box<dyn Error>> {
        let (language, grammar) = grammar_with(field, declaration)?;
        let prepared = prepare_through_boundary(&language, &grammar)?;
        let linked = GrammarCompiler.link(&prepared)?;
        GrammarCompiler.plan_rules(&prepared, &linked)?;
        Ok(())
    }

    #[test]
    fn an_installed_normal_form_matching_and_goal_compile_as_extensions()
    -> Result<(), Box<dyn Error>> {
        plan_with(
            "normal_form",
            &json!({"kind": "third-normal-form", "bound": 2}),
        )?;
        plan_with(
            "matching",
            &json!({"kind": "third-matching", "modes": ["a"]}),
        )?;
        plan_with("goal", &json!({"kind": "third-goal", "categories": ["S"]}))?;
        plan_with(
            "normal_form",
            &json!({"kind": "all-of", "normal_forms": [{"kind": "third-normal-form"}]}),
        )?;
        Ok(())
    }

    #[test]
    fn a_malformed_builtin_normal_form_matching_or_goal_is_refused() -> Result<(), Box<dyn Error>> {
        for (field, declaration, family) in [
            ("normal_form", json!({"kind": "eisner"}), "normal form"),
            (
                "matching",
                json!({"kind": "modal", "modes": ["a"]}),
                "matching",
            ),
            ("goal", json!({"kind": "root"}), "goal"),
        ] {
            let Err(error) = plan_with(field, &declaration) else {
                return Err(format!("a malformed built-in {family} was planned").into());
            };
            let Some(CompileError::RulePlan { source }) = error.downcast_ref::<CompileError>()
            else {
                return Err(format!("the {family} refusal was {error}").into());
            };
            assert!(matches!(
                source.kind(),
                RulePlanErrorKind::MalformedBuiltinComponent { family: refused, .. }
                    if *refused == family
            ));
        }
        Ok(())
    }

    fn operator<'a>(linked: &'a LinkedGrammar, name: &str) -> Option<&'a super::LinkedOperator> {
        linked
            .ir()
            .operators()
            .iter()
            .find(|operator| operator.name() == name)
    }

    fn operator_names(linked: &LinkedGrammar) -> Vec<&str> {
        linked
            .ir()
            .operators()
            .iter()
            .map(super::LinkedOperator::name)
            .collect()
    }

    fn assert_duplicate_constant_at(
        error: CompileError,
        name: &str,
        component: LinkComponent,
        index: usize,
    ) -> Result<(), Box<dyn Error>> {
        let CompileError::Link { source } = error else {
            return Err(format!("expected a signature-linking refusal, got {error}").into());
        };
        assert!(matches!(
            source.kind(),
            LinkErrorKind::DuplicateConstant { sort, name: repeated }
                if sort == "Cat" && repeated == name
        ));
        assert_eq!(source.location().declaration(), DeclarationKind::Grammar);
        assert_eq!(source.location().component(), component);
        assert_eq!(source.location().index(), Some(index));
        Ok(())
    }

    #[test]
    fn language_ir_preserves_python_canonical_bytes() -> Result<(), Box<dyn Error>> {
        let declaration: LanguageIr = serde_json::from_slice(LANGUAGE)?;

        assert_eq!(declaration.name(), "unicode-😀");
        assert_eq!(canonical::language_bytes(&declaration)?, LANGUAGE);
        Ok(())
    }

    #[test]
    fn grammar_ir_preserves_python_canonical_bytes() -> Result<(), Box<dyn Error>> {
        let declaration: GrammarIr = serde_json::from_slice(GRAMMAR)?;

        assert_eq!(declaration.name(), "tiny");
        assert_eq!(declaration.declared_rule_families(), 0);
        assert_eq!(canonical::grammar_bytes(&declaration)?, GRAMMAR);
        Ok(())
    }

    #[test]
    fn typed_decoder_refuses_unknown_fields() -> Result<(), Box<dyn Error>> {
        let mut document = LANGUAGE.to_vec();
        let closing = document.pop();
        assert_eq!(closing, Some(b'}'));
        document.extend_from_slice(br#","other":true}"#);

        let Err(error) = serde_json::from_slice::<LanguageIr>(&document) else {
            return Err("an unknown boundary field was accepted".into());
        };
        assert!(
            error.to_string().contains("unknown field"),
            "unexpected decoder error: {error}"
        );
        Ok(())
    }

    #[test]
    fn linking_locates_a_duplicate_language_sort() -> Result<(), Box<dyn Error>> {
        let (mut language, grammar) = packaged_declarations()?;
        language.category_sort.clone_from(&language.term_sort);
        let Err(error) = language.link_declarations(&grammar) else {
            return Err("a duplicate language sort was linked".into());
        };

        assert!(matches!(
            error,
            CompileError::Link { source }
                if matches!(source.kind(), LinkErrorKind::DuplicateSort { name } if name == "Tm")
        ));
        Ok(())
    }

    #[test]
    fn linking_refuses_a_cross_declaration_operator_conflict() -> Result<(), Box<dyn Error>> {
        let (language, mut grammar) = packaged_declarations()?;
        grammar.signature.constructors[0]
            .name
            .clone_from(&language.application);
        let Err(error) = language.link_declarations(&grammar) else {
            return Err("a grammar constructor shadowed a language operator".into());
        };

        assert!(matches!(
            error,
            CompileError::Link { source }
                if matches!(source.kind(), LinkErrorKind::DuplicateOperator { name } if name == "eval")
        ));
        Ok(())
    }

    #[test]
    fn linking_refuses_a_constructor_of_the_wrong_kind() -> Result<(), Box<dyn Error>> {
        let (language, mut grammar) = packaged_declarations()?;
        grammar.signature.constructors[0].kind[1] = "Ty".to_owned();
        let Err(error) = language.link_declarations(&grammar) else {
            return Err("a non-category constructor was linked as category syntax".into());
        };

        assert!(matches!(
            error,
            CompileError::Link { source }
                if matches!(source.kind(), LinkErrorKind::InvalidConstructorKind { constructor, .. } if constructor == "/")
        ));
        Ok(())
    }

    #[test]
    fn operator_features_copy_and_erase_the_target_shape() -> Result<(), Box<dyn Error>> {
        let (language, mut grammar) = packaged_declarations()?;
        grammar.signature.operators.push(OperatorIr {
            name: "/[marked]".to_owned(),
            sort: "Cat".to_owned(),
            arg_sorts: Vec::new(),
            binding: Vec::new(),
            family: "primitive".to_owned(),
            erases_to: Some("/".to_owned()),
        });
        let linked = link_through_boundary(&language, &grammar)?;
        let slash = linked
            .ir()
            .operators()
            .iter()
            .position(|operator| operator.name() == "/")
            .ok_or("slash must have a dense code")?;
        let marked = linked
            .ir()
            .operators()
            .iter()
            .position(|operator| operator.name() == "/[marked]")
            .ok_or("featured slash must have a dense code")?;
        let operator = &linked.store().signature().operators[marked];

        assert_eq!(operator.arguments, [2, 2]);
        assert_eq!(usize::try_from(operator.erased)?, slash);
        assert_eq!(linked.ir().features().len(), 1);
        assert_eq!(linked.ir().features()[0].name(), "/");
        assert_eq!(
            usize::try_from(linked.ir().features()[0].operator())?,
            slash
        );
        assert_eq!(linked.ir().features()[0].values(), ["marked"]);
        assert_eq!(
            operator_names(&linked),
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
                "/[marked]",
            ]
        );
        assert_eq!(
            linked.ir().fingerprint(),
            "928cb290104985ec3f9703c60b23320d0d71559eab8b6a3884836d03f0e339b3"
        );
        Ok(())
    }

    #[test]
    fn bracket_features_admit_composite_suffixes() -> Result<(), Box<dyn Error>> {
        let (language, mut grammar) = packaged_declarations()?;
        grammar.signature.operators.push(OperatorIr {
            name: "S[adj][conj]".to_owned(),
            sort: "Cat".to_owned(),
            arg_sorts: Vec::new(),
            binding: Vec::new(),
            family: "primitive".to_owned(),
            erases_to: Some("S".to_owned()),
        });

        let linked = link_through_boundary(&language, &grammar)?;
        let categories = linked
            .ir()
            .vocabularies()
            .iter()
            .find(|vocabulary| vocabulary.sort() == linked.ir().category_syntax().category_sort())
            .ok_or("category vocabulary must be linked")?;
        let bare = categories
            .names()
            .iter()
            .position(|name| name == "S")
            .ok_or("bare category must be linked")?;
        let composite = categories
            .names()
            .iter()
            .position(|name| name == "S[adj][conj]")
            .ok_or("composite feature must be linked")?;

        assert_eq!(categories.cores()[composite], bare);
        Ok(())
    }

    #[test]
    fn bracket_features_refuse_empty_or_unclosed_suffixes() -> Result<(), Box<dyn Error>> {
        for name in ["S[]", "S[adj"] {
            let (language, mut grammar) = packaged_declarations()?;
            grammar.signature.operators.push(OperatorIr {
                name: name.to_owned(),
                sort: "Cat".to_owned(),
                arg_sorts: Vec::new(),
                binding: Vec::new(),
                family: "primitive".to_owned(),
                erases_to: Some("S".to_owned()),
            });
            let Err(error) = language.link_declarations(&grammar) else {
                return Err(format!("malformed feature {name:?} was linked").into());
            };

            assert!(matches!(
                error,
                CompileError::Link { source }
                    if matches!(source.kind(), LinkErrorKind::InvalidFeature { operator, target, .. }
                        if operator == name && target == "S")
            ));
        }
        Ok(())
    }

    #[test]
    fn colon_features_generate_and_erase_the_target_shape() -> Result<(), Box<dyn Error>> {
        let (language, mut grammar) = packaged_declarations()?;
        grammar.signature.feature_delimiter = FeatureDelimiter::Colon;
        for operator in &mut grammar.signature.operators {
            let Some(target) = &operator.erases_to else {
                continue;
            };
            let feature = operator
                .name
                .strip_prefix(&format!("{target}["))
                .and_then(|rest| rest.strip_suffix(']'))
                .ok_or("packaged feature must use brackets")?;
            operator.name = format!("{target}:{feature}");
        }
        grammar.signature.operators.push(OperatorIr {
            name: "/:marked".to_owned(),
            sort: "Cat".to_owned(),
            arg_sorts: Vec::new(),
            binding: Vec::new(),
            family: "primitive".to_owned(),
            erases_to: Some("/".to_owned()),
        });

        let linked = link_through_boundary(&language, &grammar)?;
        let slash = operator(&linked, "/").ok_or("slash must be linked")?;
        let marked = operator(&linked, "/:marked").ok_or("feature must be linked")?;

        assert_eq!(marked.arguments(), slash.arguments());
        assert_eq!(marked.erased(), slash.operator());
        let categories = linked
            .ir()
            .vocabularies()
            .iter()
            .find(|vocabulary| vocabulary.sort() == linked.ir().category_syntax().category_sort())
            .ok_or("category vocabulary must be linked")?;
        assert!(categories.names().iter().any(|name| name == "S:dcl"));
        assert!(!categories.names().iter().any(|name| name == "S[dcl]"));
        Ok(())
    }

    #[test]
    fn curry_and_church_binders_preserve_annotation_metadata() -> Result<(), Box<dyn Error>> {
        let (mut language, grammar) = packaged_declarations()?;
        let curry = link_through_boundary(&language, &grammar)?;
        let curry_lam = operator(&curry, "lam").ok_or("curry abstraction must be linked")?;
        assert_eq!(curry_lam.arguments(), [0]);
        assert_eq!(curry_lam.binding()[0].arguments(), [0]);
        assert_eq!(curry_lam.binding()[0].sort(), 0);
        assert_eq!(curry_lam.binding()[0].annotation(), None);

        language.binders.style = BinderStyle::Church;
        let church = link_through_boundary(&language, &grammar)?;
        let church_lam = operator(&church, "lam").ok_or("church abstraction must be linked")?;
        assert_eq!(church_lam.arguments(), [1, 0]);
        assert_eq!(church_lam.binding()[0].arguments(), [1]);
        assert_eq!(church_lam.binding()[0].sort(), 0);
        assert_eq!(church_lam.binding()[0].annotation(), Some(0));
        assert_eq!(operator_names(&curry), BASE_OPERATOR_ORDER);
        assert_eq!(operator_names(&church), BASE_OPERATOR_ORDER);
        assert_eq!(
            church.ir().fingerprint(),
            "d8047d86ed97f9b525cb50559731f56359e8e3cac4103beb34baaf5aeb22061e"
        );
        Ok(())
    }

    #[test]
    fn custom_kernel_families_survive_linking_and_digesting() -> Result<(), Box<dyn Error>> {
        let (mut language, grammar) = packaged_declarations()?;
        language.constant_family = "literal".to_owned();
        language.variable_family = "atom".to_owned();
        let linked = link_through_boundary(&language, &grammar)?;

        assert_eq!(linked.ir().constant_family(), "literal");
        assert_eq!(linked.ir().variable_family(), "atom");
        assert_eq!(
            operator(&linked, "#const[Tm]")
                .ok_or("constant operator must be linked")?
                .family(),
            "literal"
        );
        assert_eq!(
            operator(&linked, "#fvar[Ty]")
                .ok_or("free-variable operator must be linked")?
                .family(),
            "atom"
        );
        assert_eq!(operator_names(&linked), BASE_OPERATOR_ORDER);
        assert_eq!(
            linked.ir().fingerprint(),
            "a317b1bd8a1c37d147e67582d37c05d9ffb7388cd13cd12c2326e093478e357d"
        );
        Ok(())
    }

    #[test]
    fn open_sorts_classifiers_and_holes_remain_typed_metadata() -> Result<(), Box<dyn Error>> {
        let (language, mut grammar) = packaged_declarations()?;
        grammar.signature.open_sorts.push("Cat".to_owned());
        grammar
            .signature
            .classifiers
            .insert("Cat".to_owned(), "Ty".to_owned());
        grammar.signature.holes = true;
        grammar.signature.allows_vacuous_binding = false;
        let linked = link_through_boundary(&language, &grammar)?;

        assert_eq!(linked.ir().open_sorts(), [0, 1, 2]);
        assert_eq!(linked.ir().classifiers().len(), 2);
        assert_eq!(linked.ir().classifiers()[0].sort(), 0);
        assert_eq!(linked.ir().classifiers()[0].classifier(), 1);
        assert_eq!(linked.ir().classifiers()[1].sort(), 2);
        assert_eq!(linked.ir().classifiers()[1].classifier(), 1);
        assert!(linked.ir().has_holes());
        assert!(!linked.ir().allows_vacuous_binding());
        assert!(!linked.store().signature().allows_vacuous_binding);
        assert_eq!(linked.ir().counts().operators(), 15);
        assert!(linked.ir().operators().iter().any(|operator| {
            operator.name() == "#hole[Cat]" && operator.payload() == LinkedPayloadKind::None
        }));
        assert_eq!(
            operator_names(&linked),
            [
                "#const[Tm]",
                "#bvar[Tm]",
                "#fvar[Tm]",
                "lam",
                "eval",
                "#hole[Tm]",
                "#const[Ty]",
                "#fvar[Ty]",
                "->",
                "#hole[Ty]",
                "#const[Cat]",
                "#fvar[Cat]",
                "/",
                "\\",
                "#hole[Cat]",
            ]
        );
        assert_eq!(
            linked.ir().fingerprint(),
            "3ba27802555d792ac8a038c47c28a4166c8e63cd6759598ec26f07e0d3de4843"
        );
        Ok(())
    }

    #[test]
    fn both_syntax_modes_link_without_changing_the_signature() -> Result<(), Box<dyn Error>> {
        let (language, mut grammar) = packaged_declarations()?;
        let declared = link_through_boundary(&language, &grammar)?;
        grammar.signature.syntax = super::SignatureSyntax::SExpression;
        let s_expression = link_through_boundary(&language, &grammar)?;

        assert_eq!(
            declared.ir().category_syntax().kind(),
            CategorySyntaxKind::Declared
        );
        assert_eq!(
            s_expression.ir().category_syntax().kind(),
            CategorySyntaxKind::SExpression
        );
        assert_eq!(declared.ir().fingerprint(), s_expression.ir().fingerprint());
        assert_eq!(
            declared.store().signature(),
            s_expression.store().signature()
        );
        assert_eq!(operator_names(&declared), BASE_OPERATOR_ORDER);
        assert_eq!(operator_names(&s_expression), BASE_OPERATOR_ORDER);
        Ok(())
    }

    #[test]
    fn every_constructor_notation_links_in_declaration_order() -> Result<(), Box<dyn Error>> {
        let (language, mut grammar) = packaged_declarations()?;
        grammar.signature.constructors[1].notation =
            ConstructorNotationIr::Mixfix(MixfixConstructorNotationIr {
                style: MixfixStyle::Mixfix,
                parts: vec!["[".to_owned(), "|".to_owned(), "]".to_owned()],
                precedence: 11,
            });
        grammar.signature.constructors.push(super::ConstructorIr {
            name: "box".to_owned(),
            kind: vec!["Cat".to_owned(), "Cat".to_owned()],
            notation: ConstructorNotationIr::Prefix(PrefixConstructorNotationIr {
                style: PrefixStyle::Prefix,
                spelling: "box".to_owned(),
            }),
        });
        let linked = link_through_boundary(&language, &grammar)?;
        let constructors = linked.ir().category_syntax().constructors();

        assert_eq!(constructors.len(), 3);
        assert!(matches!(
            constructors[0].notation(),
            LinkedConstructorNotation::Infix { spelling, .. } if spelling == "/"
        ));
        assert!(matches!(
            constructors[1].notation(),
            LinkedConstructorNotation::Mixfix { parts, precedence: 11 }
                if parts == &["[", "|", "]"]
        ));
        assert!(matches!(
            constructors[2].notation(),
            LinkedConstructorNotation::Prefix { spelling } if spelling == "box"
        ));
        assert_eq!(
            operator_names(&linked),
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
                "box",
                "/",
                "\\",
            ]
        );
        assert_eq!(
            linked.ir().fingerprint(),
            "6567a8fda684e9765a19a6a319cc0d1de76d2cbcf6188104820bc69eb72f7c07"
        );
        Ok(())
    }

    #[test]
    fn duplicate_category_primitives_are_refused() -> Result<(), Box<dyn Error>> {
        let (language, mut grammar) = packaged_declarations()?;
        grammar.signature.primitives.push("S".to_owned());
        let prepared = prepare_through_boundary(&language, &grammar)?;
        let Err(error) = GrammarCompiler.link(&prepared) else {
            return Err("a duplicate category primitive was linked".into());
        };
        assert!(matches!(
            error,
            CompileError::Link { source }
                if matches!(source.kind(), LinkErrorKind::DuplicateConstant { name, .. } if name == "S")
        ));
        Ok(())
    }

    #[test]
    fn primitive_featured_collision_keeps_the_operator_location() -> Result<(), Box<dyn Error>> {
        let (language, mut grammar) = packaged_declarations()?;
        grammar.signature.primitives.push("S[dcl]".to_owned());
        let prepared = prepare_through_boundary(&language, &grammar)?;
        let Err(error) = GrammarCompiler.link(&prepared) else {
            return Err("a primitive-featured category collision was linked".into());
        };

        assert_duplicate_constant_at(error, "S[dcl]", LinkComponent::Operator, 0)?;
        Ok(())
    }

    #[test]
    fn primitive_explicit_collision_uses_the_vocabulary_row_location() -> Result<(), Box<dyn Error>>
    {
        let (language, mut grammar) = packaged_declarations()?;
        grammar
            .signature
            .vocabularies
            .insert("Ty".to_owned(), vec!["marker".to_owned()]);
        grammar
            .signature
            .vocabularies
            .insert("Cat".to_owned(), vec!["S".to_owned()]);
        let prepared = prepare_through_boundary(&language, &grammar)?;
        let Err(error) = GrammarCompiler.link(&prepared) else {
            return Err("a primitive-explicit category collision was linked".into());
        };

        assert_duplicate_constant_at(error, "S", LinkComponent::Vocabulary, 0)?;
        Ok(())
    }

    #[test]
    fn featured_explicit_collision_uses_the_vocabulary_row_location() -> Result<(), Box<dyn Error>>
    {
        let (language, mut grammar) = packaged_declarations()?;
        grammar
            .signature
            .vocabularies
            .insert("Ty".to_owned(), vec!["marker".to_owned()]);
        grammar
            .signature
            .vocabularies
            .insert("Cat".to_owned(), vec!["S[dcl]".to_owned()]);
        let prepared = prepare_through_boundary(&language, &grammar)?;
        let Err(error) = GrammarCompiler.link(&prepared) else {
            return Err("a featured-explicit category collision was linked".into());
        };

        assert_duplicate_constant_at(error, "S[dcl]", LinkComponent::Vocabulary, 0)?;
        Ok(())
    }

    #[test]
    fn failed_foreign_extension_does_not_contaminate_the_next_link() -> Result<(), Box<dyn Error>> {
        let (language, grammar) = packaged_declarations()?;
        let mut foreign = grammar.clone();
        foreign.signature.sorts.push("Alien".to_owned());
        let Err(error) = language.link_declarations(&foreign) else {
            return Err("a grammar introduced a foreign sort".into());
        };
        assert!(matches!(
            error,
            CompileError::Link { source }
                if matches!(source.kind(), LinkErrorKind::ForeignGrammarSort { name } if name == "Alien")
        ));

        let first = language.link_declarations(&grammar)?;
        let second = language.link_declarations(&grammar)?;
        assert_eq!(first.ir(), second.ir());
        assert_eq!(first.store().signature(), second.store().signature());
        Ok(())
    }

    #[test]
    fn opaque_extension_rules_traverse_the_same_dense_plan() -> Result<(), Box<dyn Error>> {
        let (language, mut grammar) = packaged_declarations()?;
        let position = grammar
            .rules
            .rules
            .iter()
            .position(|entry| !matches!(entry, RuleEntryIr::Rule(_)))
            .ok_or("the packaged grammar declares no combinator set")?;
        grammar.rules.rules.insert(
            position,
            RuleEntryIr::Rule(RuleIr::Opaque(OpaqueRuleIr {
                head_policy: HeadPolicy::PassThrough,
                restriction: None,
                matching: None,
                introduces: None,
                consumes: None,
                rule: "modal-application".to_owned(),
                extension: indexmap::IndexMap::from([
                    (
                        "name".to_owned(),
                        serde_json::from_value(json!("modal-application"))?,
                    ),
                    (
                        "premises".to_owned(),
                        serde_json::from_value(json!([{"pattern": "term", "name": "X"}]))?,
                    ),
                    (
                        "conclusion".to_owned(),
                        serde_json::from_value(json!({"pattern": "term", "name": "X"}))?,
                    ),
                    ("targets".to_owned(), serde_json::from_value(json!(null))?),
                    (
                        "normal_form_role".to_owned(),
                        serde_json::from_value(json!({
                            "kind": "other",
                            "direction": null,
                            "principal_slot": null,
                            "order": null,
                            "crossed": false,
                        }))?,
                    ),
                    ("mode".to_owned(), serde_json::from_value(json!("box"))?),
                    (
                        "runtime".to_owned(),
                        serde_json::from_value(json!({"kind": "pattern", "head_slot": 0}))?,
                    ),
                    (
                        "theory".to_owned(),
                        serde_json::from_value(json!({
                            "kind": "schemata",
                            "name": "modal-application",
                            "forward_complete": true,
                            "schemata": [{
                                "premises": [{"pattern": "term", "name": "X"}],
                                "conclusion": {"pattern": "term", "name": "X"},
                            }],
                        }))?,
                    ),
                ]),
            })),
        );
        let prepared = prepare_through_boundary(&language, &grammar)?;
        let linked = GrammarCompiler.link(&prepared)?;
        let plan = GrammarCompiler.plan_rules(&prepared, &linked)?;

        assert_eq!(plan.rules().len(), 37);
        assert_eq!(plan.explicit_rules(), 21);
        assert_eq!(plan.rules()[20].identifier(), 20);
        assert_eq!(plan.rules()[20].tag(), "modal-application");
        assert_eq!(
            plan.rules()[20].location().source(),
            RulePlanSource::Explicit
        );
        assert_eq!(plan.rules()[20].location().index(), 20);
        assert!(matches!(
            plan.rules()[20].kind(),
            LinkedRuleKind::Extension { .. }
        ));
        assert_eq!(plan.plugin_tags()[4], "modal-application");

        let RuleEntryIr::Rule(RuleIr::Opaque(rule)) = &mut grammar.rules.rules[20] else {
            return Err("the late-rule fixture lost its opaque declaration".into());
        };
        rule.extension.insert(
            "theory".to_owned(),
            serde_json::from_value(json!({
                "kind": "schemata",
                "name": "modal-application",
                "forward_complete": true,
                "schemata": [{
                    "premises": [{"pattern": "term", "name": "X"}],
                    "conclusion": {"pattern": "term", "name": "Y"},
                }],
            }))?,
        );
        let prepared = prepare_through_boundary(&language, &grammar)?;
        let linked = GrammarCompiler.link(&prepared)?;
        let Err(CompileError::RulePlan { source }) = GrammarCompiler.plan_rules(&prepared, &linked)
        else {
            return Err("a late runtime relation diverged from its proof schema".into());
        };
        assert!(matches!(
            source.kind(),
            RulePlanErrorKind::InvalidPatternShape { .. }
        ));
        assert!(
            source
                .to_string()
                .contains("identical to its runtime relation")
        );
        Ok(())
    }

    #[test]
    fn rule_planning_locates_an_unknown_functor_reference() -> Result<(), Box<dyn Error>> {
        let (language, mut grammar) = packaged_declarations()?;
        let index = grammar.rules.rules.len();
        grammar
            .rules
            .rules
            .push(RuleEntryIr::Rule(RuleIr::Application(ApplicationRuleIr {
                restriction: None,
                matching: None,
                introduces: None,
                consumes: None,
                rule: ApplicationRuleKind::Value,
                view: "missing".to_owned(),
            })));
        let prepared = prepare_through_boundary(&language, &grammar)?;
        let linked = GrammarCompiler.link(&prepared)?;
        let Err(CompileError::RulePlan { source }) = GrammarCompiler.plan_rules(&prepared, &linked)
        else {
            return Err("an unknown functor reference was planned".into());
        };

        let location = source.location().ok_or("the refusal lost its source row")?;
        assert_eq!(location.source(), RulePlanSource::Explicit);
        assert_eq!(location.index(), index);
        assert_eq!(location.member(), None);
        assert!(matches!(
            source.kind(),
            RulePlanErrorKind::UnknownFunctor { name } if name == "missing"
        ));
        Ok(())
    }

    #[test]
    fn rule_planning_refuses_generated_composition_relations_that_coincide()
    -> Result<(), Box<dyn Error>> {
        let (language, mut grammar) = packaged_declarations()?;
        let Some(RuleEntryIr::Combinators(combinators)) = grammar
            .rules
            .rules
            .iter_mut()
            .find(|entry| matches!(entry, RuleEntryIr::Combinators(_)))
        else {
            return Err("the packaged grammar declares no combinator set".into());
        };
        combinators.direction = CompositionRestriction::Unrestricted;
        let prepared = prepare_through_boundary(&language, &grammar)?;
        let linked = GrammarCompiler.link(&prepared)?;
        let Err(CompileError::RulePlan { source }) = GrammarCompiler.plan_rules(&prepared, &linked)
        else {
            return Err("coinciding generated composition relations were planned".into());
        };

        assert_eq!(
            source.location().map(RulePlanLocation::source),
            Some(RulePlanSource::Combinator)
        );
        let RulePlanErrorKind::CoincidingRules { pairs } = source.kind() else {
            return Err(format!("expected a coincidence refusal, got {source}").into());
        };
        assert_eq!(pairs.len(), 4);
        assert!(pairs.iter().any(|pair| {
            pair.contains("'composition:/:1:harmonic:unrestricted'")
                && pair.contains("'composition:/:1:crossed:unrestricted'")
                && pair.ends_with("'crossed'")
        }));
        Ok(())
    }

    #[test]
    fn rule_planning_refuses_a_linked_signature_from_another_pair() -> Result<(), Box<dyn Error>> {
        let (language, grammar) = packaged_declarations()?;
        let first = prepare_through_boundary(&language, &grammar)?;
        let linked = GrammarCompiler.link(&first)?;
        let mut other = grammar;
        other.name.push_str("-other");
        let second = prepare_through_boundary(&language, &other)?;
        let Err(CompileError::RulePlan { source }) = GrammarCompiler.plan_rules(&second, &linked)
        else {
            return Err("a foreign linked signature was accepted by rule planning".into());
        };

        assert_eq!(source.location(), None);
        assert!(matches!(
            source.kind(),
            RulePlanErrorKind::SignatureFingerprint { expected, received }
                if expected != received
        ));
        Ok(())
    }

    proptest! {
        #[test]
        fn arbitrary_duplicate_operator_names_are_refused(name in "[a-z]{1,12}") {
            let Ok((mut language, grammar)) = packaged_declarations() else {
                return Err(TestCaseError::fail("packaged declarations must decode"));
            };
            language.abstraction.clone_from(&name);
            language.application.clone_from(&name);
            let result = language.link_declarations(&grammar);

            prop_assert!(
                matches!(
                    result,
                    Err(CompileError::Link { source })
                        if matches!(source.kind(), LinkErrorKind::DuplicateOperator { name: repeated } if repeated == &name)
                ),
                "duplicate operator name was accepted"
            );
        }

        #[test]
        fn linked_fingerprints_are_stable_for_declared_constants(name in "x[a-z]{1,12}") {
            let Ok((language, mut grammar)) = packaged_declarations() else {
                return Err(TestCaseError::fail("packaged declarations must decode"));
            };
            grammar
                .signature
                .vocabularies
                .entry("Cat".to_owned())
                .or_default()
                .push(name);
            let first = language.link_declarations(&grammar);
            let second = language.link_declarations(&grammar);
            prop_assert!(first.is_ok(), "first link failed: {first:?}");
            prop_assert!(second.is_ok(), "second link failed: {second:?}");
            if let (Ok(first), Ok(second)) = (first, second) {
                prop_assert_eq!(first.ir(), second.ir());
                prop_assert_eq!(first.store().signature(), second.store().signature());
            }
        }
    }
}
