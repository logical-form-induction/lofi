# Swappable grammatical assumptions

To compare a grammar with only application against one that also permits
composition, change the [declared rule inventory](../how-to/grammars/choose-combinatory-rules.md).
Both inventories supply rules through the same interface, so the data,
parser, and training procedure can be held fixed.
The [normal form](derivations-as-proofs.md#normal-forms) must still be compatible
with the chosen rules, and both grammars must fit the parser's declared bounds.

## Every choice is a declared value

A LoFI run never selects behavior by a flag inside an implementation. Each
choice a practitioner might vary is a field of the [run document](documents-are-the-dsl.md#the-document-is-the-program) whose value is
a declaration, and each declaration is one variant of a family. The formalism,
each combinatory rule, the matching relation, the [normal form](derivations-as-proofs.md#normal-forms), the goal, the
[semiring](charts-and-approximation.md#what-a-semiring-decides), the [cell pruner](charts-and-approximation.md#pruning), the token and [term encoders](../how-to/models/configure-term-encoders-and-decoders.md#choose-an-encoder), the [chart parser](../how-to/models/configure-a-chart-parser.md), each
loss term, the prover, the fuzzy logic, the [sweep design](../how-to/experiments/choose-a-sweep-design.md) and the
[multiple comparison correction](experiments-and-commitments.md#corrections) are all families of this kind.

A family's root is a tagged union. A variant is selected by its discriminator,
usually `kind` (`formalism` for grammars), and validated against that
variant's own declaration class, which forbids unknown fields. The slot in the
run document is typed by the root, so any variant of the family fills it and
nothing outside the family does. Variants share a configuration interface
while implementing different computations.

Here *family* means a set of interchangeable declarations, such as the
available loss functions. A *tagged union* represents a choice among them:
the tag identifies the variant, and that variant determines which additional
fields are valid. Thus selecting a loss also selects the settings that can
be supplied for it. The [configuration reference](../reference/configuration/index.md)
lists those settings; the [entry point catalog](../reference/entry-points.md)
lists the packaged families and variants.

The reason for this design is the research question LoFI is built for: whether
a particular grammatical assumption is the one the data support. That question
can be asked only when the alternative assumptions are values of the same type,
built by the same machinery, under otherwise identical conditions. A choice
hard-coded in an implementation cannot be crossed in an experiment.

## How a family is discovered

Families and their variants are registered through package entry points in
[`pyproject.toml`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml), never through a central Python registry or an import list.

- The `lofi.families` group names each component family and its root
  declaration class, for instance `loss = "lofi.spec.components.loss:LossSpec"`.
- Each family has its own group, `lofi.<family>`, whose entries name the
  variants. An entry names a plugin object that pairs the variant's
  declaration class with the builder of its runtime value
  ([`ComponentPlugin`](../reference/registry.md#lofi.registry.ComponentPlugin)). Declaration families that
  are not model components, such as `lofi.rule`, `lofi.matching`,
  `lofi.normal_form`, `lofi.semiring` and `lofi.correction`, follow the same
  pattern with their own plugin classes.

[`load_plugins`](../reference/spec.md) loads the family roots first, then the
declaration families, then every component family's variants, each group in a
stable order. Loading a root before its variants lets another distribution
contribute both a new family and its members without editing LoFI. Discovery
refuses a family registered twice, a variant tag registered twice, an entry
that names something other than a plugin, and a plugin whose declaration class
is not the class the document selected. An unknown tag is refused with the
list of installed tags.

Because the schemas of installed variants are part of the run specification's
schema, installing a plugin changes what a document may say. The boundary
fingerprint described in [Documents are the declaration
language](documents-are-the-dsl.md) records the schema of every model the run
specification reaches with every installed variant, so a record written under
one set of plugins is distinguishable from one written under another.

The [plugin guide](../how-to/extending/plugins.md) describes how to register a
variant.

## What makes two choices comparable

Two points compare one choice when their resolved documents differ only in the
declarations that express it. Composition makes that condition checkable: a
resolved scientific document and seed can be compared field by field, and
the [provenance](documents-are-the-dsl.md#provenance) record names the source of every leaf. The truncated point
hash is a compact identifier, subject to collisions. An
experiment's sweep expresses the choice as an axis, and every other section is
shared by construction.

A controlled comparison also requires stating the estimand and the sources
of variation. LoFI’s source comparison workflow imposes four further policies.
Their use of capacity matching and complexity charging follows that workflow's
selection criterion; other scientific comparisons may use different criteria.

1. **The consumers accept both choices.** A component that serves only some
   deduction profiles declares a [`ProfileBound`](../reference/formalism.md#lofi.formalism.ProfileBound), and a grammar outside it is
   refused when the grammar is compiled (see [Grammars as deductive
   systems](derivations-as-proofs.md)). A neural family that allocates
   parameters by operator rank declares an [`ArityBound`](../reference/nn.md#lofi.nn.arity.ArityBound), and a grammar that
   reaches an undeclared rank is refused (see [Term layout
   capabilities](term-layout-capabilities.md)). A comparison never silently
   runs one arm under a degraded consumer.
2. **Capacity is matched.** Two learner grammars with different category
   inventories can induce models of different size. The source comparison
   reducer refuses candidates whose measured [`parameter_count`](../reference/run.md#lofi.run.parameter_count) differs within
   a comparison group, and it compares the source against capacity-matched
   twins. Capacity is measured from the built model and recorded in the
   point's metrics.
3. **The approximation is shared.** A cell beam on one arm and an exhaustive
   chart on the other, or two different [residual mass](charts-and-approximation.md#the-residual-mass-budget) bounds for term
   decoding, compare different approximations as well as different choices.
   A grammar's budget is part of its declaration, so two grammars with
   different budgets differ in their approximation too. [Charts and
   approximation](charts-and-approximation.md) describes what each
   approximation costs.
4. **The selection criterion charges for complexity.** The source comparison
   reducer scores each candidate by held-out fit minus a description length
   charge. Language inclusion leaves this score to be measured: the familiar
   inequality for nested statistical models concerns global training optima
   under a common objective. [Source and learner
   declarations](synthetic-designs.md) describes that comparison.

For statistical families $\mathcal{P}_0\subseteq\mathcal{P}_1$ and a common
training log likelihood $\ell_{\mathrm{train}}$,

$$
\sup_{p\in\mathcal{P}_0}\ell_{\mathrm{train}}(p)
\leq\sup_{p\in\mathcal{P}_1}\ell_{\mathrm{train}}(p).
$$

Applying this inequality to an actual fit requires assessing optimizer
convergence and any regularization; held-out likelihood must be evaluated on
held-out data. Effective capacity also depends on how a model uses its
parameters, beyond their count. With seed $j$ shared across arms $a$ and $b$,
the paired difference is $d_j=s_{a,j}-s_{b,j}$ and its mean is
$\overline d=m^{-1}\sum_{j=1}^{m}d_j$. Pairing makes the comparison unit
explicit; seeds from different data generation procedures are not necessarily
matched observations.

A comparison supports a claim about its declared alternatives under its
declared conditions. Generalizing to further alternatives or data distributions
requires further comparisons. The packaged variants serve as evaluation
fixtures for this machinery; practitioners can declare additional alternatives.
