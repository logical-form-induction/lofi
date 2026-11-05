# Probing through published sites

This page explains why a probe reaches a model only through the sites the
model's components publish. It describes what a site is, what publishing one
guarantees and what it leaves unchecked, why role inventories are shared across
component kinds, and what a probe result does and does not license.

## A site is a published promise

A site names one value a component computes: the component it belongs to, the
role the value plays (`CHART_CELL`, `SPLIT_DISTRIBUTION`, `NODE_ENCODING`,
`COMBINATOR_LOGITS`, and so on), and a key that picks one point among those the
component publishes for that role, such as a layer index. Each site carries a
specification: the name of every axis, outermost first; what the numbers mean
(`real`, `log-probability`, `probability`, `integer` or `mask`); whether a
gradient flows through it; and whether it may be observed, patched or both.

A component's `SiteMap` is immutable and belongs to one component. The
alternative, letting a probe find a tensor by walking module attributes or by
matching a substring of a module path, makes every probe a dependency on the
implementation it inspects: a renamed attribute, a fused layer or a reordered
axis silently changes what is measured. A site map turns the same dependency
into an interface. The component states what it exposes, under names a probe
can be written against, and a change to the computation either keeps that
promise or changes the map, where the change is visible.

The same discipline covers models LoFI did not build. An external model's sites
are declared over full module paths, and a path the model does not have is
refused when a region opens rather than matched approximately
([Instrument an external Torch model](../how-to/probes/external-model-site-maps.md)).

## What publishing guarantees

Every operation on a site is checked at the region that performs it. Opening a
region refuses a site the component does not publish and an access it does not
permit: an observation of a patch-only site, a patch of an observe-only one, a
gradient at a site declared non-differentiable. A probe declared in a run
document is checked earlier still. The held-out passes are run once before
training to find which published points they reach, and a probe naming a role
or site no reached component publishes is refused before any training step, so
a run never trains to completion only to find its probe unanswerable.

Patching is scoped. A replacement holds for one pass inside one region, and
leaving the region restores every component, after a completed pass, an
exception or a refusal, so a probe leaves the trained model as it found it.
Observations are detached copies; a gradient is taken only through a
`differentiate` region over sites checked as `GradientSites`.

Some guarantees are conditional. Whether each reading matches its declared
axes, numeric class and fill is checked only when `LOFI_CHECK=1` is set; with
it unset, a component that publishes a wrong specification is trusted. A site
specification describes the value, not its meaning: `SPLIT_DISTRIBUTION`
promises a log-probability over splits for each span, not that the distribution
corresponds to anything a linguist would call a constituent.

## Shared role inventories

A role means the same thing wherever it is published. Each component kind
(token encoder, subword pooler, term encoder, term decoder, controller, parser)
has a declared inventory: the roles it must publish, structures of roles it
publishes together or not at all, and mutually exclusive alternatives. A
parser always publishes `CHART_CELL`; it publishes the inside score, outside
score and split distribution together if at all. A term encoder's recurrence
roles come in one of three sets, for a vanilla recurrence, a GRU and an LSTM.
The inventory is checked when a site map is built.

This is what lets a diagnostic be written once. `span-length-bias` declares
that it requires `CONSTITUENT_MARGINAL` and reads that role's `item` and `span`
axes, and it runs on any parser that publishes the role, whatever its chart
family, semiring or encoder. A diagnostic whose roles a pass does not reach is
simply not selected, and one that refuses a pass's readings is counted as a
refusal, so the absence of a measurement is itself recorded.
[Run the packaged diagnostics](../how-to/probes/run-diagnostics.md) lists the
catalog and the roles each requires.

External models, parameter-only views and components with no shared inventory
publish a custom map. A custom map is still owned by one component and still
immutable, but it carries no shared role inventory, so a diagnostic that runs
over it relies on the declaration's own account of its roles.

## Probes are only as rich as their passes

A probe sees what a held-out pass reaches. The diagnostics, intervention and
classifier probes need only published points; the circuit probe needs a pass
that also publishes a computation graph and the endpoints to interrogate it,
and the causal and sparse-autoencoder probes need a causal population of
aligned clean and counterfactual inputs. A pass that publishes none of these
makes those probes unanswerable, and they are refused rather than run over
something weaker. The same reasoning governs tokenization: a causal intervention
on a span of a pretrained encoder's input is meaningful only where the span maps
exactly onto the encoder's rows, so alignment between source tokens and model
rows is computed exactly (`tokenizer_alignment`). A sentence some checkpoint
cannot align whole, because a word has no row, a row crosses two words or the
sentence overflows the window, is excluded rather than approximated, and the
exclusion is counted under that checkpoint and reason beside the pairs the
alignment policy refuses, so the reported population accounts for every
candidate sentence. The population itself is declared rather than assembled by
hand: a probe document's `counterfactuals` section generates it under the run's
own grammar and adds only the substitution settings and the candidate file.

## What a probe result licenses

A probe result is a measurement of a trained point at its published sites over
declared inputs. An intervention's effect on the log partition says how much
the computation downstream of a site depends on the value there, on those
inputs, under that replacement. It does not say that the site represents a
grammatical category, and two interventions that differ in their replacement
(zero, mean, resampled) measure different things. Inferential rows carry their
null distribution, their raw and corrected p-values and the dependence regime
of their correction family, so a claim drawn from them can be read against the
family it was tested in. Comparing two points through a probe is a comparison
of the same promised values; comparing a probe result with a gold analysis
requires the alignment, control task or null the probe declares.

[Probe a trained point](../how-to/probes/run-probes.md) declares and executes
probes.

## References

Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., and Garriga-Alonso,
A. (2023). Towards automated circuit discovery for mechanistic
interpretability. *Advances in Neural Information Processing Systems*, 36.

Geiger, A., Wu, Z., Potts, C., Icard, T., and Goodman, N. (2024). Finding
alignments between interpretable causal variables and distributed neural
representations. *Proceedings of the Third Conference on Causal Learning and
Reasoning*, PMLR 236, 160–187.

Hanna, M., Pezzelle, S., and Belinkov, Y. (2024). Have faith in faithfulness:
Going beyond circuit overlap when finding model mechanisms. *First Conference
on Language Modeling*.

Hewitt, J., and Liang, P. (2019). Designing and interpreting probes with
control tasks. *Proceedings of EMNLP-IJCNLP 2019*, 2733–2743.
