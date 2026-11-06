# Explanation

These pages explain why LoFI is built the way it is: what each grammatical and
modeling family assumes, what a comparison between two declarations licenses,
and how the implementation keeps those assumptions separate. They are organized
in four groups. Each page stands alone, but the foundations are assumed by the
rest, and the experimental-design pages read best after the induction pages.

## Foundations

1. [Categories, semantic types, and logical forms](logical-forms-as-types.md).
   Term formation, semantic typing, and the interpretation of derivations.
2. [Grammars as deductive systems](derivations-as-proofs.md). How a signature,
   a rule inventory, a normal form and a goal compile into one deductive system.
3. [Charts and approximation](charts-and-approximation.md). Chart parsing over
   spans, semiring choice, pruning, and which approximations are measured.
4. [Term-layout capabilities](term-layout-capabilities.md). Why a term layout's
   capability is not an arity ceiling, and where a rank limit applies.

## Induction

5. [Induction objectives](induction-objectives.md). What each loss family
   assumes about the data.
6. [Inference as evidence for logical forms](inference-as-evidence.md). How
   entailment labels, provers and model checking constrain lexical meanings.
7. [Judgment models](judgment-models.md). How response models link elicited
   judgments to the quantities a grammar predicts.
8. [Probing through published sites](probing-through-sites.md). Why probes
   reach a model only through the sites it publishes.

## Experimental design

9. [Swappable grammatical assumptions](swappable-assumptions.md). Why every
   choice is a declared, discoverable family, and what makes two comparable.
10. [Source and learner declarations](synthetic-designs.md). The grammar that
    generates data and the grammar a learner assumes.
11. [Experiments and commitments](experiments-and-commitments.md).
    Preregistration, freezes, stop rules, corrections and contraction.

## Architecture

12. [Documents are the declaration language](documents-are-the-dsl.md).
    Composition, provenance, point identity and the boundary fingerprint.
13. [The layer order](layering.md). The ordered layers and the import contracts
    that enforce them.
14. [Boundaries](boundaries.md). Where declared values enter, where they are
    validated, and how interior code relies on established invariants.
