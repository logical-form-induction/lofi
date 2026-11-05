//! Rule programs bound to the term scopes their premises come from.
//!
//! A [`ScopedRuleProgram`] pairs a compiled [`RuleProgram`] with, per rule, the
//! store and term scope its conclusions are built in. A scope is an opaque
//! owner of term stores, such as one term signature: premises from two scopes
//! never meet in one application, and conclusions are interned in the store
//! their scope holds for the premises' generation.

use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::chart::{ChartError, ThreadPool};
use crate::reachability::{ReachabilityRule, ReachabilityTemplate};
use crate::rule::{
    RuleApplication, RuleCategoryMeta, RuleCategoryQuery, RuleError, RuleFragmentApplication,
    RuleJoinKey, RuleProgram,
};
use crate::term::{SortId, Term, TermStore};

/// An owner of the term stores a family of exact terms is interned in.
pub trait Scope: fmt::Debug + Send + Sync {
    /// Return the live store this scope holds for one term's generation.
    fn store_for(&self, term: &Term) -> Option<TermStore>;
}

/// An error raised by a caller-supplied callback during drawing.
pub type HostError = Box<dyn Error + Send + Sync>;

/// A refusal at the scoped drawing boundary.
#[derive(Debug)]
pub enum DrawError<O> {
    /// A supplied value is invalid.
    Invalid(String),
    /// An internal invariant does not hold.
    Internal(String),
    /// The dense-id chart refused an operation.
    Chart(ChartError),
    /// A rule kernel refused an application in one scope.
    Rule {
        /// The kernel's refusal.
        error: RuleError,
        /// The scope the application ran in.
        scope: Arc<O>,
    },
    /// A caller-supplied callback failed.
    Host(HostError),
}

impl<O> DrawError<O> {
    /// Build an invalid-value refusal.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    /// Build an internal-invariant refusal.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
}

impl<O> fmt::Display for DrawError<O> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) | Self::Internal(message) => formatter.write_str(message),
            Self::Chart(error) => write!(formatter, "{error}"),
            Self::Rule { error, .. } => write!(formatter, "{error}"),
            Self::Host(error) => write!(formatter, "{error}"),
        }
    }
}

/// A result at the scoped drawing boundary.
pub type DrawResult<O, T> = Result<T, DrawError<O>>;

/// One exact term with the scope it belongs to.
pub type ScopedTerm<O> = (Term, Arc<O>);

/// Conclusions of one rule, with the scope they were interned in.
pub type ScopedOutputs<O> = Vec<(u16, Vec<Term>, Arc<O>)>;

/// One candidate-rule application job over exact scoped premises.
pub type ScopedRuleJob<O> = (Vec<u16>, Vec<ScopedTerm<O>>);

/// One fragment application job over exact scoped premises.
pub type ScopedFragmentJob<O> = (Vec<(u16, RuleJoinKey)>, Vec<ScopedTerm<O>>);

/// Return the opaque identity of one scope.
#[must_use]
pub fn scope_id<O>(scope: &Arc<O>) -> usize {
    Arc::as_ptr(scope).addr()
}

/// Where one compiled rule interns its conclusions.
#[derive(Debug)]
pub struct RuleBoundary<O> {
    /// The scope of the rule's representative function, if it has one.
    pub representative_scope: Option<Arc<O>>,
    /// The store the rule constructs conclusions in, if it owns one.
    pub store: Option<TermStore>,
    /// The scope of that store.
    pub scope: Option<Arc<O>>,
}

impl<O> Clone for RuleBoundary<O> {
    fn clone(&self) -> Self {
        Self {
            representative_scope: self.representative_scope.clone(),
            store: self.store.clone(),
            scope: self.scope.clone(),
        }
    }
}

/// Resolve the store and scope one application interns its conclusions in.
///
/// A rule that owns a store uses it; a store-free rule uses the store its
/// first premise's scope holds.
///
/// # Errors
///
/// Returns [`DrawError::Invalid`] when a representative function belongs to
/// another scope, a store-free rule has no premise, or the premise's store is
/// no longer live.
pub fn application_store<O: Scope>(
    boundary: &RuleBoundary<O>,
    inputs: &[ScopedTerm<O>],
) -> DrawResult<O, (TermStore, Arc<O>)> {
    let same_representatives = |scope: &Arc<O>| {
        boundary
            .representative_scope
            .as_ref()
            .is_none_or(|representative| Arc::ptr_eq(scope, representative))
    };
    if let (Some(store), Some(scope)) = (&boundary.store, &boundary.scope) {
        if !same_representatives(scope) {
            return Err(DrawError::invalid(
                "a representative kernel belongs to another term signature",
            ));
        }
        return Ok((store.clone(), scope.clone()));
    }
    let (term, scope) = inputs.first().ok_or_else(|| {
        DrawError::invalid("a store-free native rule requires at least one premise")
    })?;
    let store = scope
        .store_for(term)
        .ok_or_else(|| DrawError::invalid("the premise term's native store is unavailable"))?;
    if !same_representatives(scope) {
        return Err(DrawError::invalid(
            "a representative kernel belongs to another term signature",
        ));
    }
    Ok((store, scope.clone()))
}

/// The reachability templates shared by every program over one grammar.
pub type TemplateCache = Arc<Mutex<Vec<Arc<ReachabilityTemplate>>>>;

/// One ordered rule inventory bound to the scopes its rules intern in.
#[derive(Debug)]
pub struct ScopedRuleProgram<O> {
    rules: RuleProgram,
    boundaries: Vec<RuleBoundary<O>>,
    templates: TemplateCache,
}

impl<O: Scope> ScopedRuleProgram<O> {
    /// Bind a compiled rule program to one boundary per rule.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Invalid`] when the boundary count differs from the
    /// rule count.
    pub fn new(
        rules: RuleProgram,
        boundaries: Vec<RuleBoundary<O>>,
        templates: TemplateCache,
    ) -> DrawResult<O, Self> {
        if boundaries.len() != rules.len() {
            return Err(DrawError::invalid(format!(
                "{} rule boundaries were supplied for {} native rules",
                boundaries.len(),
                rules.len()
            )));
        }
        Ok(Self {
            rules,
            boundaries,
            templates,
        })
    }

    /// Return the number of rules.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rules.len()
    }

    /// Report whether the program has no rules.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rules.len() == 0
    }

    /// Return the reachability template for one rule inventory, compiling it once.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Invalid`] when the template cannot be compiled and
    /// [`DrawError::Internal`] when its cache lock is poisoned.
    pub fn reachability_template(
        &self,
        rules: &[ReachabilityRule],
        default_sort: SortId,
    ) -> DrawResult<O, Arc<ReachabilityTemplate>> {
        let mut templates = self.templates.lock().map_err(|_| {
            DrawError::internal("the rule-program reachability template lock is poisoned")
        })?;
        if let Some(template) = templates
            .iter()
            .find(|template| template.matches(rules, default_sort))
        {
            return Ok(Arc::clone(template));
        }
        let template = Arc::new(
            ReachabilityTemplate::new(rules, default_sort)
                .map_err(|error| DrawError::invalid(error.to_string()))?,
        );
        templates.push(Arc::clone(&template));
        drop(templates);
        Ok(template)
    }

    fn check_shapes(&self, shapes: &[(usize, bool)]) -> DrawResult<O, ()> {
        if shapes.len() == self.rules.len() {
            Ok(())
        } else {
            Err(DrawError::invalid(format!(
                "{} rule shapes were supplied for {} native rules",
                shapes.len(),
                self.rules.len()
            )))
        }
    }

    /// Split binary rules into those drawn without and with join keys.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Invalid`] when the shapes do not cover the program
    /// or a rule position exceeds the chart id width.
    pub fn binary_partition(
        &self,
        shapes: &[(usize, bool)],
    ) -> DrawResult<O, (Vec<u16>, Vec<u16>)> {
        self.check_shapes(shapes)?;
        let mut unjoined = Vec::new();
        let mut joined = Vec::new();
        for (position, (arity, complete)) in shapes.iter().enumerate() {
            if *arity != 2 {
                continue;
            }
            let position = u16::try_from(position).map_err(|_| {
                DrawError::invalid("a native rule position exceeds the chart id width")
            })?;
            if *complete && self.rules.uses_join_keys(position).unwrap_or(false) {
                joined.push(position);
            } else {
                unjoined.push(position);
            }
        }
        Ok((unjoined, joined))
    }

    /// Compile candidate-index metadata for exact scoped categories.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Invalid`] when the shapes do not cover the program
    /// and [`DrawError::Rule`] when a rule refuses a category.
    pub fn category_meta_batch(
        &self,
        shapes: &[(usize, bool)],
        terms: &[ScopedTerm<O>],
        workers: Option<&ThreadPool>,
    ) -> DrawResult<O, Vec<RuleCategoryMeta>> {
        self.check_shapes(shapes)?;
        let queries = terms
            .iter()
            .map(|(term, scope)| RuleCategoryQuery {
                term: term.clone(),
                scope: scope_id(scope),
            })
            .collect::<Vec<_>>();

        self.rules
            .category_meta_batch(shapes, &queries, workers)
            .into_iter()
            .zip(terms)
            .map(|(result, (_, scope))| {
                result.map_err(|error| DrawError::Rule {
                    error,
                    scope: scope.clone(),
                })
            })
            .collect()
    }

    fn boundary(&self, position: u16) -> DrawResult<O, &RuleBoundary<O>> {
        self.boundaries.get(usize::from(position)).ok_or_else(|| {
            DrawError::invalid(format!(
                "rule id {position} is outside a program of {} rules",
                self.rules.len()
            ))
        })
    }

    fn scoped_store(
        &self,
        position: u16,
        terms: &[ScopedTerm<O>],
    ) -> DrawResult<O, (TermStore, Arc<O>)> {
        let (store, scope) = application_store(self.boundary(position)?, terms)?;
        if terms.iter().any(|(_, held)| !Arc::ptr_eq(&scope, held)) {
            return Err(DrawError::invalid(
                "native rule premises belong to different term signatures",
            ));
        }
        Ok((store, scope))
    }

    /// Apply candidate rules to one exact premise tuple.
    ///
    /// # Errors
    ///
    /// Returns a refusal for the reasons [`Self::apply_batches`] gives.
    pub fn apply(
        &self,
        candidates: &[u16],
        terms: &[ScopedTerm<O>],
    ) -> DrawResult<O, ScopedOutputs<O>> {
        let jobs = vec![(candidates.to_vec(), terms.to_vec())];
        self.apply_batches(&jobs, None)
            .map(|mut rows| rows.pop().unwrap_or_default())
    }

    /// Apply each job's candidate rules to its exact premise tuple.
    ///
    /// # Errors
    ///
    /// Returns [`DrawError::Invalid`] when a rule is unknown or premises mix
    /// scopes, and [`DrawError::Rule`] when a rule refuses its premises.
    pub fn apply_batches(
        &self,
        jobs: &[ScopedRuleJob<O>],
        workers: Option<&ThreadPool>,
    ) -> DrawResult<O, Vec<ScopedOutputs<O>>> {
        let application_count = jobs.iter().map(|(candidates, _)| candidates.len()).sum();
        let mut applications = Vec::with_capacity(application_count);
        let mut scopes = Vec::with_capacity(application_count);
        let mut row_lengths = Vec::with_capacity(jobs.len());

        for (candidates, terms) in jobs {
            let inputs = terms
                .iter()
                .map(|(term, _)| term.clone())
                .collect::<Vec<_>>();
            row_lengths.push(candidates.len());
            for position in candidates.iter().copied() {
                let (store, scope) = self.scoped_store(position, terms)?;
                applications.push(RuleApplication {
                    rule: position,
                    scope: scope_id(&scope),
                    store,
                    inputs: inputs.clone(),
                });
                scopes.push(scope);
            }
        }

        let mut results = self
            .rules
            .apply_batch(&applications, workers)
            .into_iter()
            .zip(scopes);
        let mut rows = Vec::with_capacity(row_lengths.len());
        for length in row_lengths {
            let row = results
                .by_ref()
                .take(length)
                .map(|(result, scope)| match result {
                    Ok((rule, outputs)) => Ok((rule, outputs, scope)),
                    Err(error) => Err(DrawError::Rule { error, scope }),
                })
                .collect::<DrawResult<O, Vec<_>>>()?;
            rows.push(row);
        }
        Ok(rows)
    }

    /// Apply each job's rule fragments, merging the outputs of one rule.
    ///
    /// # Errors
    ///
    /// Returns a refusal for the reasons [`Self::apply_batches`] gives.
    pub fn apply_fragment_batches(
        &self,
        jobs: &[ScopedFragmentJob<O>],
        workers: Option<&ThreadPool>,
    ) -> DrawResult<O, Vec<ScopedOutputs<O>>> {
        let application_count = jobs.iter().map(|(fragments, _)| fragments.len()).sum();
        let mut applications = Vec::with_capacity(application_count);
        let mut scopes = Vec::with_capacity(application_count);
        let mut row_lengths = Vec::with_capacity(jobs.len());

        for (fragments, terms) in jobs {
            let inputs = terms
                .iter()
                .map(|(term, _)| term.clone())
                .collect::<Vec<_>>();
            row_lengths.push(fragments.len());
            for (position, fragment) in fragments {
                let (store, scope) = self.scoped_store(*position, terms)?;
                applications.push(RuleFragmentApplication {
                    rule: *position,
                    scope: scope_id(&scope),
                    store,
                    inputs: inputs.clone(),
                    fragment: fragment.clone(),
                });
                scopes.push(scope);
            }
        }

        let mut results = self
            .rules
            .apply_fragment_batch(&applications, workers)
            .into_iter()
            .zip(scopes);
        let mut rows = Vec::with_capacity(row_lengths.len());
        for length in row_lengths {
            let fragments = results
                .by_ref()
                .take(length)
                .map(|(result, scope)| match result {
                    Ok((rule, outputs)) => Ok((rule, outputs, scope)),
                    Err(error) => Err(DrawError::Rule { error, scope }),
                })
                .collect::<DrawResult<O, Vec<_>>>()?;
            let mut row: ScopedOutputs<O> = Vec::new();
            for (rule, outputs, scope) in fragments {
                if let Some((_, held, _)) = row.iter_mut().find(|(held, _, _)| *held == rule) {
                    for output in outputs {
                        if !held.contains(&output) {
                            held.push(output);
                        }
                    }
                } else {
                    row.push((rule, outputs, scope));
                }
            }
            rows.push(row);
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::{ProgramRule, RepresentativeKernel, RuleKernel};
    use crate::term::{OperatorSpec, PayloadKind, SignatureSpec, TermError};

    #[derive(Debug)]
    struct Owner(TermStore);

    impl Scope for Owner {
        fn store_for(&self, _term: &Term) -> Option<TermStore> {
            Some(self.0.clone())
        }
    }

    fn store() -> Result<TermStore, TermError> {
        TermStore::new(
            SignatureSpec {
                name: "program".to_owned(),
                sorts: vec!["Cat".to_owned()],
                operators: vec![OperatorSpec {
                    name: "NP".to_owned(),
                    sort: 0,
                    arguments: vec![],
                    payload: PayloadKind::None,
                    erased: 0,
                    binding: vec![],
                    shifts: vec![],
                }],
                vocabularies: vec![None],
                constant_cores: vec![None],
                constants: vec![None],
                bound: vec![None],
                free: vec![None],
                holes: vec![None],
                allows_vacuous_binding: true,
            },
            16,
        )
    }

    fn empty_program(rules: usize) -> RuleProgram {
        RuleProgram::new(
            (0..rules)
                .map(|_| ProgramRule {
                    kernel: RuleKernel::Empty,
                    matching: 0,
                    representatives: RepresentativeKernel::Identity,
                    constants: Vec::new(),
                })
                .collect(),
        )
    }

    fn boundary(scope: Option<Arc<Owner>>) -> RuleBoundary<Owner> {
        RuleBoundary {
            representative_scope: None,
            store: None,
            scope,
        }
    }

    #[test]
    fn a_boundary_count_must_match_the_rule_count() {
        let refused = ScopedRuleProgram::<Owner>::new(
            empty_program(2),
            vec![boundary(None)],
            TemplateCache::default(),
        );
        assert!(matches!(refused, Err(DrawError::Invalid(_))));
    }

    #[test]
    fn a_store_free_rule_interns_in_its_first_premise_scope() -> Result<(), Box<dyn Error>> {
        let store = store()?;
        let scope = Arc::new(Owner(store.clone()));
        let np = store.intern(0, Vec::new(), None)?;
        let (_, resolved) = application_store(&boundary(None), &[(np, Arc::clone(&scope))])
            .map_err(|e| e.to_string())?;
        assert!(Arc::ptr_eq(&resolved, &scope));
        assert!(matches!(
            application_store::<Owner>(&boundary(None), &[]),
            Err(DrawError::Invalid(_))
        ));
        Ok(())
    }

    #[test]
    fn premises_from_two_scopes_are_refused() -> Result<(), Box<dyn Error>> {
        let store = store()?;
        let first = Arc::new(Owner(store.clone()));
        let second = Arc::new(Owner(store.clone()));
        let np = store.intern(0, Vec::new(), None)?;
        let program = ScopedRuleProgram::new(
            empty_program(1),
            vec![boundary(None)],
            TemplateCache::default(),
        )
        .map_err(|e| e.to_string())?;
        let refused = program.apply(&[0], &[(np.clone(), first), (np, second)]);
        assert!(matches!(refused, Err(DrawError::Invalid(_))));
        Ok(())
    }

    #[test]
    fn binary_partition_requires_one_shape_per_rule() -> Result<(), Box<dyn Error>> {
        let program = ScopedRuleProgram::<Owner>::new(
            empty_program(2),
            vec![boundary(None), boundary(None)],
            TemplateCache::default(),
        )
        .map_err(|e| e.to_string())?;
        assert!(matches!(
            program.binary_partition(&[(2, true)]),
            Err(DrawError::Invalid(_))
        ));
        let (unjoined, joined) = program
            .binary_partition(&[(2, false), (1, true)])
            .map_err(|e| e.to_string())?;
        assert_eq!(unjoined, vec![0]);
        assert_eq!(joined, Vec::<u16>::new());
        Ok(())
    }
}
