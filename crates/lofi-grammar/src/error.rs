//! Typed refusals from grammar declaration preparation and signature linking.

use std::{error::Error, fmt};

use lofi_chart::term::TermError;

/// The declaration document involved in a compilation refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeclarationKind {
    /// A higher-order language declaration.
    Language,
    /// A categorial grammar declaration.
    Grammar,
}

impl fmt::Display for DeclarationKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Language => formatter.write_str("language"),
            Self::Grammar => formatter.write_str("grammar"),
        }
    }
}

/// A declaration component at which signature linking was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkComponent {
    /// The document-level signature name.
    Name,
    /// A sort declaration.
    Sort,
    /// A category primitive declaration.
    Primitive,
    /// A term operator declaration.
    Operator,
    /// A constructor declaration.
    Constructor,
    /// A vocabulary declaration.
    Vocabulary,
    /// An open-sort declaration.
    OpenSort,
    /// A classifier declaration.
    Classifier,
    /// A category-syntax declaration.
    CategorySyntax,
    /// The linked signature as a whole.
    Signature,
    /// The kernel-derived constant-operator family.
    ConstantFamily,
    /// The kernel-derived variable-operator family.
    VariableFamily,
    /// The native term-store capacity.
    StoreCapacity,
}

impl fmt::Display for LinkComponent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Name => "name",
            Self::Sort => "sort",
            Self::Primitive => "primitive",
            Self::Operator => "operator",
            Self::Constructor => "constructor",
            Self::Vocabulary => "vocabulary",
            Self::OpenSort => "open sort",
            Self::Classifier => "classifier",
            Self::CategorySyntax => "category syntax",
            Self::Signature => "signature",
            Self::ConstantFamily => "constant family",
            Self::VariableFamily => "variable family",
            Self::StoreCapacity => "store capacity",
        })
    }
}

/// The exact declaration position at which signature linking was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinkLocation {
    declaration: DeclarationKind,
    component: LinkComponent,
    index: Option<usize>,
}

impl LinkLocation {
    /// Locate a scalar or keyed declaration component.
    #[must_use]
    pub const fn field(declaration: DeclarationKind, component: LinkComponent) -> Self {
        Self {
            declaration,
            component,
            index: None,
        }
    }

    /// Locate one element of an ordered declaration component.
    #[must_use]
    pub const fn indexed(
        declaration: DeclarationKind,
        component: LinkComponent,
        index: usize,
    ) -> Self {
        Self {
            declaration,
            component,
            index: Some(index),
        }
    }

    /// Return the declaration document containing the refusal.
    #[must_use]
    pub const fn declaration(self) -> DeclarationKind {
        self.declaration
    }

    /// Return the kind of declaration component containing the refusal.
    #[must_use]
    pub const fn component(self) -> LinkComponent {
        self.component
    }

    /// Return the position in an ordered component, when one applies.
    #[must_use]
    pub const fn index(self) -> Option<usize> {
        self.index
    }
}

impl fmt::Display for LinkLocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} {}", self.declaration, self.component)?;
        if let Some(index) = self.index {
            write!(formatter, "[{index}]")?;
        }
        Ok(())
    }
}

/// The semantic invariant that prevented two prepared declarations from linking.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkErrorKind {
    /// A sort name occurred more than once.
    DuplicateSort {
        /// The repeated sort name.
        name: String,
    },
    /// An operator name occurred more than once.
    DuplicateOperator {
        /// The repeated operator name.
        name: String,
    },
    /// A vocabulary constant occurred more than once.
    DuplicateConstant {
        /// The vocabulary sort.
        sort: String,
        /// The repeated constant name.
        name: String,
    },
    /// A declaration named a sort outside the joint language signature.
    UndeclaredSort {
        /// The unknown sort name.
        name: String,
    },
    /// The grammar attempted to introduce a sort outside the language signature.
    ForeignGrammarSort {
        /// The foreign sort name.
        name: String,
    },
    /// A constructor kind was not an endofunctor over its owning sort.
    InvalidConstructorKind {
        /// The constructor name.
        constructor: String,
        /// The required sort at every kind position.
        expected: String,
        /// The received kind in argument-then-result order.
        received: Vec<String>,
    },
    /// A constructor notation was incompatible with its arity.
    InvalidConstructorNotation {
        /// The constructor name.
        constructor: String,
        /// The reason the notation was refused.
        reason: String,
    },
    /// A functor view was incompatible with the linked constructors.
    InvalidFunctor {
        /// The functor-view name.
        name: String,
        /// The reason the view was refused.
        reason: String,
    },
    /// A featured operator did not have the target-derived form.
    InvalidFeature {
        /// The featured operator name.
        operator: String,
        /// The declared erasure target.
        target: String,
        /// The reason the feature declaration was refused.
        reason: String,
    },
    /// A declared operator used a kernel-owned name or family.
    ReservedOperator {
        /// The operator name.
        operator: String,
        /// The reserved declaration detail.
        reason: String,
    },
    /// A binding site was incompatible with its operator shape.
    InvalidBinding {
        /// The operator name.
        operator: String,
        /// The binding-shape detail.
        reason: String,
    },
    /// The linked inventory cannot be represented by the core identifier type.
    InventoryOverflow {
        /// The overflowing inventory, such as `sorts`, `operators`, or `store nodes`.
        inventory: &'static str,
        /// The received inventory size.
        size: usize,
    },
}

impl fmt::Display for LinkErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSort { name } => write!(formatter, "sort {name:?} is repeated"),
            Self::DuplicateOperator { name } => {
                write!(formatter, "operator {name:?} is declared more than once")
            }
            Self::DuplicateConstant { sort, name } => {
                write!(formatter, "vocabulary {sort:?} repeats constant {name:?}")
            }
            Self::UndeclaredSort { name } => write!(formatter, "sort {name:?} is undeclared"),
            Self::ForeignGrammarSort { name } => {
                write!(
                    formatter,
                    "grammar sort {name:?} is outside the language signature"
                )
            }
            Self::InvalidConstructorKind {
                constructor,
                expected,
                received,
            } => write!(
                formatter,
                "constructor {constructor:?} has kind {received:?}, not an endofunctor over {expected:?}"
            ),
            Self::InvalidConstructorNotation {
                constructor,
                reason,
            } => write!(formatter, "constructor {constructor:?} {reason}"),
            Self::InvalidFunctor { name, reason } => {
                write!(formatter, "functor view {name:?} {reason}")
            }
            Self::InvalidFeature {
                operator,
                target,
                reason,
            } => write!(
                formatter,
                "featured operator {operator:?} targeting {target:?} {reason}"
            ),
            Self::ReservedOperator { operator, reason } => {
                write!(formatter, "operator {operator:?} {reason}")
            }
            Self::InvalidBinding { operator, reason } => {
                write!(formatter, "binding of operator {operator:?} {reason}")
            }
            Self::InventoryOverflow { inventory, size } => {
                write!(
                    formatter,
                    "{inventory} inventory of size {size} exceeds its dense code"
                )
            }
        }
    }
}

/// A located refusal from the semantic signature-linking boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkError {
    location: LinkLocation,
    kind: LinkErrorKind,
}

impl LinkError {
    /// Build a refusal at one declaration location.
    #[must_use]
    pub const fn new(location: LinkLocation, kind: LinkErrorKind) -> Self {
        Self { location, kind }
    }

    /// Return the precise declaration location.
    #[must_use]
    pub const fn location(&self) -> LinkLocation {
        self.location
    }

    /// Return the semantic refusal kind.
    #[must_use]
    pub const fn kind(&self) -> &LinkErrorKind {
        &self.kind
    }
}

impl fmt::Display for LinkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.location, self.kind)
    }
}

impl Error for LinkError {}

/// The declaration family that produced one planned rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RulePlanSource {
    /// A row in the explicit `rules` sequence.
    Explicit,
    /// A row expanded from a combinator set.
    Combinator,
    /// A row expanded from a coercion set.
    Coercion,
}

impl fmt::Display for RulePlanSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Explicit => "explicit rule",
            Self::Combinator => "combinator set",
            Self::Coercion => "coercion set",
        })
    }
}

/// The exact source row and optional expansion member of one planned rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RulePlanLocation {
    source: RulePlanSource,
    index: usize,
    member: Option<usize>,
}

impl RulePlanLocation {
    /// Locate an explicit rule declaration.
    #[must_use]
    pub const fn explicit(index: usize) -> Self {
        Self {
            source: RulePlanSource::Explicit,
            index,
            member: None,
        }
    }

    /// Locate one member expanded from a combinator set.
    #[must_use]
    pub const fn combinator(index: usize, member: usize) -> Self {
        Self {
            source: RulePlanSource::Combinator,
            index,
            member: Some(member),
        }
    }

    /// Locate one member expanded from a coercion set.
    #[must_use]
    pub const fn coercion(index: usize, member: usize) -> Self {
        Self {
            source: RulePlanSource::Coercion,
            index,
            member: Some(member),
        }
    }

    /// Return the declaration family containing the row.
    #[must_use]
    pub const fn source(self) -> RulePlanSource {
        self.source
    }

    /// Return the declaration row index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }

    /// Return the generated member index, when this row was expanded.
    #[must_use]
    pub const fn member(self) -> Option<usize> {
        self.member
    }
}

impl fmt::Display for RulePlanLocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}[{}]", self.source, self.index)?;
        if let Some(member) = self.member {
            write!(formatter, " member[{member}]")?;
        }
        Ok(())
    }
}

/// The invariant that prevented native rule planning.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RulePlanErrorKind {
    /// The supplied linked signature came from different declarations.
    SignatureFingerprint {
        /// The fingerprint required by the prepared declaration pair.
        expected: String,
        /// The supplied linked signature fingerprint.
        received: String,
    },
    /// A rule names no linked functor view.
    UnknownFunctor {
        /// The unknown view name.
        name: String,
    },
    /// A fixed pattern names no linked operator.
    UnknownOperator {
        /// The unknown operator name.
        name: String,
    },
    /// A constant pattern names no linked sort.
    UnknownSort {
        /// The unknown sort name.
        name: String,
    },
    /// A constant pattern names no constant of its selected sort.
    UnknownConstant {
        /// The linked sort name.
        sort: String,
        /// The unknown constant name.
        name: String,
    },
    /// A deterministic rule pattern does not inhabit its required signature slot.
    InvalidPatternShape {
        /// The precise rejected shape invariant.
        reason: String,
    },
    /// A built-in tag reached the opaque extension representation.
    MalformedBuiltin {
        /// The built-in rule tag.
        tag: String,
    },
    /// A normal-form, matching or goal declaration names a built-in kind in a
    /// shape that kind does not accept.
    MalformedBuiltinComponent {
        /// The component family.
        family: &'static str,
        /// The built-in kind.
        kind: String,
    },
    /// A coercion entry did not have one source and one target.
    MalformedCoercion {
        /// The rejected entry text.
        entry: String,
    },
    /// Generated rules declare equal relations over categories.
    CoincidingRules {
        /// Every coinciding pair, in expansion order.
        pairs: Vec<String>,
    },
    /// A rule restriction names premises the rule does not have.
    UnenforceableRestriction {
        /// Why the restriction cannot be enforced.
        reason: String,
    },
    /// The expanded inventory exceeds the dense identifier type.
    InventoryOverflow {
        /// The number of rules requested.
        size: usize,
    },
}

impl fmt::Display for RulePlanErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SignatureFingerprint { expected, received } => write!(
                formatter,
                "linked signature fingerprint {received:?} does not match prepared fingerprint {expected:?}"
            ),
            Self::UnknownFunctor { name } => write!(formatter, "unknown functor view {name:?}"),
            Self::UnknownOperator { name } => write!(formatter, "unknown operator {name:?}"),
            Self::UnknownSort { name } => write!(formatter, "unknown sort {name:?}"),
            Self::UnknownConstant { sort, name } => {
                write!(formatter, "unknown constant {name:?} of sort {sort:?}")
            }
            Self::InvalidPatternShape { reason } => formatter.write_str(reason),
            Self::MalformedBuiltinComponent { family, kind } => {
                write!(
                    formatter,
                    "built-in {family} {kind:?} has an invalid declaration shape"
                )
            }
            Self::MalformedBuiltin { tag } => {
                write!(
                    formatter,
                    "built-in rule {tag:?} has an invalid declaration shape"
                )
            }
            Self::MalformedCoercion { entry } => write!(
                formatter,
                "coercion entry {entry:?} must have the form 'source => target'"
            ),
            Self::CoincidingRules { pairs } => write!(
                formatter,
                "declares the same relation twice: {}, which does not change the relation over categories",
                pairs.join("; ")
            ),
            Self::UnenforceableRestriction { reason } => formatter.write_str(reason),
            Self::InventoryOverflow { size } => {
                write!(
                    formatter,
                    "rule inventory of size {size} exceeds its dense identifier"
                )
            }
        }
    }
}

/// A located refusal from native rule planning.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RulePlanError {
    location: Option<RulePlanLocation>,
    kind: RulePlanErrorKind,
}

impl RulePlanError {
    /// Build a refusal attached to one source row.
    #[must_use]
    pub const fn located(location: RulePlanLocation, kind: RulePlanErrorKind) -> Self {
        Self {
            location: Some(location),
            kind,
        }
    }

    /// Build a refusal concerning the plan as a whole.
    #[must_use]
    pub const fn global(kind: RulePlanErrorKind) -> Self {
        Self {
            location: None,
            kind,
        }
    }

    /// Return the source row, when one caused the refusal.
    #[must_use]
    pub const fn location(&self) -> Option<RulePlanLocation> {
        self.location
    }

    /// Return the refused invariant.
    #[must_use]
    pub const fn kind(&self) -> &RulePlanErrorKind {
        &self.kind
    }
}

impl fmt::Display for RulePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(location) = self.location {
            write!(formatter, "{location}: ")?;
        }
        self.kind.fmt(formatter)
    }
}

impl Error for RulePlanError {}

/// A typed refusal from a staged grammar compilation.
#[derive(Debug)]
pub enum CompileError {
    /// Canonical JSON could not be decoded into the current typed schema.
    Decode {
        /// The document that could not be decoded.
        declaration: DeclarationKind,
        /// The typed JSON decoding failure.
        source: serde_json::Error,
    },
    /// A document decoded but did not use the one canonical byte form.
    NonCanonical {
        /// The document that used another byte form.
        declaration: DeclarationKind,
    },
    /// A typed document could not be written back to canonical JSON.
    Encode {
        /// The document that could not be encoded.
        declaration: DeclarationKind,
        /// The typed JSON encoding failure.
        source: serde_json::Error,
    },
    /// The two prepared declarations could not form one semantic signature.
    Link {
        /// The located semantic refusal.
        source: LinkError,
    },
    /// The expanded rule inventory could not link to the native signature.
    RulePlan {
        /// The located rule-planning refusal.
        source: RulePlanError,
    },
    /// The linked signature could not initialize the native term store.
    Store {
        /// The core term-store refusal.
        source: TermError,
    },
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode {
                declaration,
                source,
            } => write!(
                formatter,
                "cannot decode {declaration} declaration: {source}"
            ),
            Self::NonCanonical { declaration } => {
                write!(formatter, "{declaration} declaration is not canonical JSON")
            }
            Self::Encode {
                declaration,
                source,
            } => write!(
                formatter,
                "cannot encode {declaration} declaration: {source}"
            ),
            Self::Link { source } => write!(formatter, "cannot link declarations at {source}"),
            Self::RulePlan { source } => write!(formatter, "cannot plan rules at {source}"),
            Self::Store { source } => write!(formatter, "cannot initialize linked store: {source}"),
        }
    }
}

impl Error for CompileError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Decode { source, .. } | Self::Encode { source, .. } => Some(source),
            Self::Link { source } => Some(source),
            Self::RulePlan { source } => Some(source),
            Self::Store { source } => Some(source),
            Self::NonCanonical { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CompileError, DeclarationKind};

    #[test]
    fn a_noncanonical_refusal_names_its_declaration() {
        let error = CompileError::NonCanonical {
            declaration: DeclarationKind::Language,
        };

        assert_eq!(
            error.to_string(),
            "language declaration is not canonical JSON"
        );
    }
}
