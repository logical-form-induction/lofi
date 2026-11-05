//! Native compilation of LoFI grammar declarations.
//!
//! This crate owns the pure-Rust declaration boundary. It decodes the
//! canonical language and grammar documents
//! into typed intermediate representations, links their joint signature and
//! core term store, and emits one deterministic rule program.

pub mod canonical;
mod compiler;
mod error;
mod ir;
mod rules;

#[cfg(test)]
pub(crate) use compiler::PreparedGrammarIr;
pub use compiler::{CompiledGrammar, DeclarationEnvelope, GrammarCompiler};
pub use error::{
    CompileError, DeclarationKind, LinkComponent, LinkError, LinkErrorKind, LinkLocation,
    RulePlanError, RulePlanErrorKind, RulePlanLocation, RulePlanSource,
};
pub use ir::{
    CanonicalGrammarIr, CanonicalLanguageIr, CategorySyntaxKind, GrammarIr, LanguageIr,
    LinkedArgumentSide, LinkedAssociativity, LinkedBindingSite, LinkedCategoryConstructor,
    LinkedCategorySyntax, LinkedClassifier, LinkedConstructorNotation, LinkedFunctor,
    LinkedGrammar, LinkedGrammarCounts, LinkedGrammarIr, LinkedOperator, LinkedOperatorDeclaration,
    LinkedOperatorFeatures, LinkedPayloadKind, LinkedVocabulary,
};
pub use rules::{
    LinkedAbsorptionRecipe, LinkedClausalAdjunctRecipe, LinkedCompositionDirection,
    LinkedCoordinationRecipe, LinkedNormalFormRole, LinkedOperatorPattern, LinkedPattern,
    LinkedPatternRecipe, LinkedPayloadPattern, LinkedRule, LinkedRuleKind, LinkedRulePlan,
    LinkedRuleRestriction, LinkedTheory, LinkedTheoryConstructor,
};

trait LinkDeclarations {
    fn link_declarations(&self, grammar: &GrammarIr) -> Result<LinkedGrammar, CompileError>;
}
