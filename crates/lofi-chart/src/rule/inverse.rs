//! Checking and residual directions of the binary-functor rule kernels.
//!
//! Forward application enumerates the conclusions a premise tuple licenses.
//! The directions here run the same relations the other way: they name the
//! orientation and spine that license one step, decide whether a premise tuple
//! and conclusion are related, and recover a missing premise from the
//! conclusion and the premises that remain. Each walks the spine exactly as
//! forward application does, so the directions cannot disagree about which
//! steps a declaration licenses.

use super::{
    CompositionRule, Frame, RaisingRule, RepresentativeKernel, RepresentativeTable, RestrictedRule,
    RuleError, RuleKernel, SubstitutionRule, at_path, push_unique, rebuild, register_tree,
    walk_frames,
};
use crate::term::{Term, TermStore};

/// The physical orientation and container spine that license one composition step.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompositionWitness {
    /// Premise slot of the principal functor.
    pub principal_slot: usize,
    /// Premise slot of the container whose spine the conclusion carries.
    pub container_slot: usize,
    /// Constructors of the carried spine frames, outermost first.
    pub spine: Vec<String>,
}

impl CompositionWitness {
    /// Return the composition order of the step.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.spine.len()
    }
}

fn register_all<'a>(
    kernel: &RepresentativeKernel,
    table: &mut RepresentativeTable,
    terms: impl IntoIterator<Item = &'a Term>,
) {
    for term in terms {
        register_tree(kernel, term, table);
    }
}

fn single_hole(known: &[Option<Term>], arity: usize) -> Option<usize> {
    if known.len() != arity {
        return None;
    }
    let mut holes = known
        .iter()
        .enumerate()
        .filter_map(|(at, value)| value.is_none().then_some(at));
    let hole = holes.next()?;
    holes.next().is_none().then_some(hole)
}

fn frame_parts(rule_functors: &super::FunctorSpec, frame: &Frame) -> Result<Term, RuleError> {
    at_path(&frame.term, &rule_functors.argument_path)
}

impl CompositionRule {
    fn peel(&self, term: &Term) -> Result<Vec<(Vec<Frame>, Term)>, RuleError> {
        let mut peeled = Vec::new();
        let mut current = term.clone();
        let mut frames = Vec::new();
        loop {
            if self.admitted_order(frames.len()) && self.licensed(&frames) {
                peeled.push((frames.clone(), current.clone()));
            }
            if self.order == Some(frames.len()) || !self.functors.is_functor(&current) {
                return Ok(peeled);
            }
            frames.push(Frame {
                term: current.clone(),
                slot: *self.functors.result_path.first().unwrap_or(&0),
            });
            current = at_path(&current, &self.functors.result_path)?;
        }
    }

    fn principal_parts(&self, term: &Term) -> Result<Option<(Term, Term)>, RuleError> {
        if !self.functors.is_functor(term) || term.operator_name() != self.slash {
            return Ok(None);
        }
        Ok(Some((
            at_path(term, &self.functors.result_path)?,
            at_path(term, &self.functors.argument_path)?,
        )))
    }

    /// Return the witnesses of the step a premise pair and conclusion make.
    ///
    /// Each witness names the premise slots of one physical orientation and
    /// the container spine the conclusion carries. A pair the rule does not
    /// relate to the conclusion has none.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a declared term path or term construction is
    /// invalid.
    pub fn witnesses(
        &self,
        store: &TermStore,
        kernel: &RepresentativeKernel,
        table: &mut RepresentativeTable,
        inputs: &[Term],
        output: &Term,
    ) -> Result<Vec<CompositionWitness>, RuleError> {
        if inputs.len() != 2 {
            return Ok(Vec::new());
        }
        register_all(kernel, table, inputs.iter().chain([output]));
        let mut witnesses = Vec::new();
        for (principal_slot, container_slot) in self.functors.orientations(&self.slash) {
            let Some((result, argument)) = self.principal_parts(&inputs[principal_slot])? else {
                continue;
            };
            for (frames, foot) in self.peel(&inputs[container_slot])? {
                if !table.equivalent(&argument, &foot)? {
                    continue;
                }
                let candidate =
                    rebuild(store, &frames, &self.functors.result_path, result.clone())?;
                register_tree(kernel, &candidate, table);
                if !table.equivalent(&candidate, output)? {
                    continue;
                }
                let witness = CompositionWitness {
                    principal_slot,
                    container_slot,
                    spine: frames
                        .iter()
                        .map(|frame| frame.term.operator_name().to_owned())
                        .collect(),
                };
                if !witnesses.contains(&witness) {
                    witnesses.push(witness);
                }
            }
        }
        Ok(witnesses)
    }

    /// Return the order at which the rule draws a conclusion from a premise pair.
    ///
    /// # Errors
    ///
    /// Returns [`RuleError::InconsistentOrder`] when witnesses of one step
    /// carry spines of different lengths, and a typed error when a declared
    /// term path or term construction is invalid.
    pub fn applied_order(
        &self,
        store: &TermStore,
        kernel: &RepresentativeKernel,
        table: &mut RepresentativeTable,
        inputs: &[Term],
        output: &Term,
    ) -> Result<Option<usize>, RuleError> {
        let mut orders = self
            .witnesses(store, kernel, table, inputs, output)?
            .iter()
            .map(CompositionWitness::order)
            .collect::<Vec<_>>();
        orders.sort_unstable();
        orders.dedup();
        match orders.as_slice() {
            [] => Ok(None),
            [order] => Ok(Some(*order)),
            _ => Err(RuleError::InconsistentOrder(orders)),
        }
    }

    fn residuate(
        &self,
        store: &TermStore,
        table: &RepresentativeTable,
        output: &Term,
        known: &[Option<Term>],
        missing: usize,
    ) -> Result<Vec<Term>, RuleError> {
        let mut residuals = Vec::new();
        for (principal_slot, container_slot) in self.functors.orientations(&self.slash) {
            if missing == principal_slot
                && let Some(container) = &known[container_slot]
            {
                for (frames, foot) in self.peel(container)? {
                    let Some((carried, result)) =
                        walk_frames(&self.functors, output, frames.len())?
                    else {
                        continue;
                    };
                    if self.same_frames(&carried, &frames)? {
                        push_unique(
                            &mut residuals,
                            self.functors.make(store, &self.slash, result, foot)?,
                        );
                    }
                }
            } else if missing == container_slot
                && let Some(principal) = &known[principal_slot]
                && let Some((result, argument)) = self.principal_parts(principal)?
            {
                for (frames, foot) in self.peel(output)? {
                    if table.equivalent(&result, &foot)? {
                        push_unique(
                            &mut residuals,
                            rebuild(store, &frames, &self.functors.result_path, argument.clone())?,
                        );
                    }
                }
            }
        }
        Ok(residuals)
    }

    fn same_frames(&self, left: &[Frame], right: &[Frame]) -> Result<bool, RuleError> {
        for (left, right) in left.iter().zip(right) {
            if left.term.operator_name() != right.term.operator_name()
                || !frame_parts(&self.functors, left)?
                    .is_identical(&frame_parts(&self.functors, right)?)
            {
                return Ok(false);
            }
        }
        Ok(left.len() == right.len())
    }
}

impl SubstitutionRule {
    fn parallel(
        &self,
        table: &RepresentativeTable,
        left: &[Frame],
        right: &[Frame],
    ) -> Result<bool, RuleError> {
        for (left, right) in left.iter().zip(right) {
            if left.term.operator_name() != right.term.operator_name()
                || !table.equivalent(
                    &frame_parts(&self.functors, left)?,
                    &frame_parts(&self.functors, right)?,
                )?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn residuate(
        &self,
        store: &TermStore,
        table: &RepresentativeTable,
        output: &Term,
        known: &[Option<Term>],
        missing: usize,
    ) -> Result<Vec<Term>, RuleError> {
        let orientations = self.functors.orientations(&self.slash);
        if orientations.is_empty() {
            return Err(RuleError::InvalidSlash(self.slash.clone()));
        }
        if orientations
            .iter()
            .any(|(_, container_slot)| *container_slot != missing)
        {
            return Ok(Vec::new());
        }
        let required = if self.crossed {
            self.functors
                .opposite(&self.slash)
                .ok_or_else(|| RuleError::InvalidSlash(self.slash.clone()))?
        } else {
            &self.slash
        };
        let mut residuals = Vec::new();
        for (functor_slot, _) in orientations {
            let Some(functor) = &known[functor_slot] else {
                continue;
            };
            for order in self.orders(functor, output)? {
                let Some((functor_frames, functor_foot)) = self.walk(functor, order)? else {
                    continue;
                };
                let Some((output_frames, output_foot)) = self.walk(output, order)? else {
                    continue;
                };
                if !self.parallel(table, &functor_frames, &output_frames)?
                    || output_frames
                        .last()
                        .is_none_or(|frame| frame.term.operator_name() != required)
                    || !self.functors.is_functor(&functor_foot)
                    || functor_foot.operator_name() != self.slash
                {
                    continue;
                }
                let result = at_path(&functor_foot, &self.functors.result_path)?;
                if !table.equivalent(&result, &output_foot)? {
                    continue;
                }
                let argument = at_path(&functor_foot, &self.functors.argument_path)?;
                push_unique(
                    &mut residuals,
                    rebuild(store, &output_frames, &self.functors.result_path, argument)?,
                );
            }
        }
        Ok(residuals)
    }
}

impl RaisingRule {
    fn raised_input(
        &self,
        table: &RepresentativeTable,
        output: &Term,
    ) -> Result<Option<Term>, RuleError> {
        if !self.functors.is_functor(output) || output.operator_name() != self.slash {
            return Ok(None);
        }
        let opposite = self
            .functors
            .opposite(&self.slash)
            .ok_or_else(|| RuleError::InvalidSlash(self.slash.clone()))?;
        let target = at_path(output, &self.functors.result_path)?;
        let inner = at_path(output, &self.functors.argument_path)?;
        if !self.functors.is_functor(&inner) || inner.operator_name() != opposite {
            return Ok(None);
        }
        if !table.equivalent(&target, &at_path(&inner, &self.functors.result_path)?)? {
            return Ok(None);
        }
        if let Some(targets) = &self.targets {
            let mut declared = false;
            for held in targets {
                if table.equivalent(&target, held)? {
                    declared = true;
                    break;
                }
            }
            if !declared {
                return Ok(None);
            }
        }
        let input = at_path(&inner, &self.functors.argument_path)?;
        if !self.raises(table, &input)? {
            return Ok(None);
        }
        Ok(Some(input))
    }
}

impl RuleKernel {
    fn constants(&self) -> Vec<&Term> {
        match self {
            Self::Raising(RaisingRule {
                targets, arguments, ..
            }) => targets.iter().chain(arguments).flatten().collect(),
            Self::Restricted(rule) => {
                let mut constants = rule.rule.constants();
                constants.extend(rule.restriction.constants());
                constants
            }
            _ => Vec::new(),
        }
    }

    /// Decide whether the rule relates a premise tuple to a conclusion.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a declared term path or term construction is
    /// invalid.
    pub fn admits(
        &self,
        store: &TermStore,
        kernel: &RepresentativeKernel,
        table: &mut RepresentativeTable,
        inputs: &[Term],
        output: &Term,
    ) -> Result<bool, RuleError> {
        register_all(
            kernel,
            table,
            inputs.iter().chain([output]).chain(self.constants()),
        );
        if let Self::Restricted(rule) = self {
            if !rule.restriction.admits_all(table, inputs)? {
                return Ok(false);
            }
            return rule.rule.admits(store, kernel, table, inputs, output);
        }
        if let Self::Raising(rule) = self {
            if inputs.len() != 1 {
                return Ok(false);
            }
            return rule
                .raised_input(table, output)?
                .map_or(Ok(false), |input| table.equivalent(&inputs[0], &input));
        }
        for candidate in self.apply(store, table, inputs)? {
            register_tree(kernel, &candidate, table);
            if table.equivalent(&candidate, output)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Recover the fillers of the one missing premise of a known step.
    ///
    /// `known` holds each premise in rule order, with `None` at the hole. A
    /// tuple with no hole or more than one has no finite residual, and so has
    /// none.
    ///
    /// # Errors
    ///
    /// Returns [`RuleError::UnsupportedInverse`] for a kernel that declares no
    /// residual, and a typed error when a declared term path or term
    /// construction is invalid.
    pub fn residuate(
        &self,
        store: &TermStore,
        kernel: &RepresentativeKernel,
        table: &mut RepresentativeTable,
        output: &Term,
        known: &[Option<Term>],
    ) -> Result<Vec<Term>, RuleError> {
        let arity = match self {
            Self::Composition(_) | Self::Substitution(_) => 2,
            Self::Raising(_) => 1,
            Self::Restricted(rule) => return rule.residuate(store, kernel, table, output, known),
            Self::Empty | Self::Pattern(_) | Self::Absorption(_) | Self::CoordinationPhase(_) => {
                return Err(RuleError::UnsupportedInverse("residuation"));
            }
        };
        let Some(missing) = single_hole(known, arity) else {
            return Ok(Vec::new());
        };
        register_all(
            kernel,
            table,
            known
                .iter()
                .flatten()
                .chain([output])
                .chain(self.constants()),
        );
        match self {
            Self::Composition(rule) => rule.residuate(store, table, output, known, missing),
            Self::Substitution(rule) => rule.residuate(store, table, output, known, missing),
            Self::Raising(rule) => Ok(rule.raised_input(table, output)?.into_iter().collect()),
            Self::Empty
            | Self::Pattern(_)
            | Self::Absorption(_)
            | Self::CoordinationPhase(_)
            | Self::Restricted(_) => Err(RuleError::UnsupportedInverse("residuation")),
        }
    }
}

impl RestrictedRule {
    /// Recover the fillers of the missing premise that the restriction admits.
    fn residuate(
        &self,
        store: &TermStore,
        kernel: &RepresentativeKernel,
        table: &mut RepresentativeTable,
        output: &Term,
        known: &[Option<Term>],
    ) -> Result<Vec<Term>, RuleError> {
        register_all(
            kernel,
            table,
            known.iter().flatten().chain(self.restriction.constants()),
        );
        for (slot, term) in known.iter().enumerate() {
            if let Some(term) = term
                && !self.restriction.admits_slot(table, term, slot)?
            {
                return Ok(Vec::new());
            }
        }
        let Some(missing) = known.iter().position(Option::is_none) else {
            return Ok(Vec::new());
        };
        let mut admitted = Vec::new();
        for filler in self.rule.residuate(store, kernel, table, output, known)? {
            register_tree(kernel, &filler, table);
            if self.restriction.admits_slot(table, &filler, missing)? {
                admitted.push(filler);
            }
        }
        Ok(admitted)
    }
}

#[cfg(test)]
mod tests {
    use super::{CompositionWitness, single_hole};
    use crate::term::{OperatorSpec, PayloadKind, SignatureSpec, TermStore};

    #[test]
    fn a_witness_order_is_its_spine_length() {
        let witness = CompositionWitness {
            principal_slot: 1,
            container_slot: 0,
            spine: vec!["fwd".to_owned(); 3],
        };

        assert_eq!(witness.order(), 3, "three carried frames make order three");
    }

    #[test]
    fn exactly_one_hole_is_residuable() -> Result<(), crate::term::TermError> {
        let store = TermStore::new(
            SignatureSpec {
                name: "holes".to_owned(),
                sorts: vec!["Cat".to_owned()],
                operators: vec![OperatorSpec {
                    name: "a".to_owned(),
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
        )?;
        let a = store.intern(0, vec![], None)?;

        assert_eq!(
            single_hole(&[None, Some(a.clone())], 2),
            Some(0),
            "one hole"
        );
        assert_eq!(single_hole(&[None, None], 2), None, "two holes");
        assert_eq!(single_hole(&[Some(a.clone()), Some(a)], 2), None, "no hole");
        assert_eq!(single_hole(&[None], 2), None, "a short tuple");
        Ok(())
    }
}
