# Probing through published sites

This page explains why a probe reaches a model only through the sites the
model's components publish. It describes what a site is, what publishing one
guarantees and what it leaves unchecked, why role inventories are shared across
component kinds, and what a probe result does and does not license.

## Observations and interventions

For input $x$ and fixed trained parameters $\boldsymbol{\theta}$, write
$\mathbf{H}_s(x)$ for the tensor published at site $s$. Its declared
axes specify a product shape $n_1\times\cdots\times n_k$ and its mask
$\mathbf{M}_s$ specifies live entries. A probability dtype
requires entries in $[0,1]$ and a log probability dtype requires entries at
most zero when numeric checks are enabled. These checks do not enforce
normalization. Normalization belongs to the particular role: a split PMF
sums to one over alternatives, whereas constituent marginals need not sum
to one over spans.

An observation returns $\operatorname{stopgrad}(\mathbf{H}_s(x))$.
An intervention replaces that value by $\mathbf{H}'_s$ during the forward
computation. If $o(x)$ denotes a scalar downstream outcome, write
$o(x;s\leftarrow\mathbf{H}'_s)$ for the result of that replacement.
This notation describes an intervention on the implemented computation;
it does not by itself identify an intervention on a linguistic variable.

The packaged intervention probe measures an absolute change. For pass $b$,
let $J_b$ contain coordinates live under the clean reading’s mask at
outcome sites reached in both passes.
Its reported value over $m$ passes is

$$
\widehat\delta=\frac1m\sum_{b=1}^{m}
\frac1{|J_b|}\sum_{j\in J_b}|o'_{b,j}-o_{b,j}|.
$$

The two-stage average gives each pass equal weight; it is not necessarily
the average over all coordinates pooled across passes. Empty outcome sets
are refused. Zero, mean, and resampled replacements define different
measurements, even at the same site.

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
gradient at a site declared nondifferentiable. A probe declared in a run
document is checked earlier still. The held-out passes are run once before
training to find which published points they reach, and a probe naming a role
or site no reached component publishes is refused before any training step, so
a run never trains to completion only to find its probe unanswerable.

Patching is scoped. A replacement holds for one pass inside one region, and
leaving the region restores every component, after a completed pass, an
exception or a refusal, so a probe leaves the trained model as it found it.
Observations are detached readings; a gradient is taken only through a
`differentiate` region over sites checked as `GradientSites`.

Some guarantees are conditional. Whether each reading matches its declared
axes, numeric class and fill is checked only when `LOFI_CHECK=1` is set; with
it unset, a component that publishes a wrong specification is trusted. A site
specification describes the value, not its meaning: `SPLIT_DISTRIBUTION`
promises a log probability over splits for each span, not that the distribution
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
and the causal and sparse autoencoder probes need a causal population of
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

Classifier probes fit a prediction function on representations and evaluate
it on held-out labels. Their selectivity is
$s=a_{\mathrm{task}}-a_{\mathrm{control}}$, where both accuracies use
the same declared probe capacity and their respective training tasks
\citep{hewitt-liang2019}. High task accuracy alone can reflect a probe's
ability to memorize regularities rather than information used by the model.

For distributed interchange interventions, an orthogonal matrix
$\mathbf{R}\in\mathbb{R}^{h\times h}$ and coordinate mask
$\mathbf{M}\in\mathbb{R}^{h\times h}$ define, in column vector notation,

$$
\mathbf{h}'=\mathbf{R}
\left[(\mathbf{I}-\mathbf{M})\mathbf{R}^{\mathsf T}\mathbf{h}
+\mathbf{M}\mathbf{R}^{\mathsf T}\mathbf{h}_{\mathrm{source}}\right],
\qquad\mathbf{R}^{\mathsf T}\mathbf{R}=\mathbf{I}.
$$

A binary diagonal mask exchanges a subspace; a soft diagonal mask blends it
during optimization. Interchange accuracy compares the resulting outputs
with the declared high-level counterfactual outcomes on held-out aligned
pairs \citep{geiger-etal2024}. It is relative to that counterfactual population
and variable alignment, not a universal equivalence of two models.

Circuit discovery searches for subsets of a published computation graph
whose retained or removed components preserve a declared behavior
\citep{conmy-etal2023}. With mean clean score $s_{\mathrm{clean}}$, corrupt
score $s_{\mathrm{corrupt}}$, and circuit score $s_C$, the normalized recovery is

$$
r_C=\frac{s_C-s_{\mathrm{corrupt}}}
{s_{\mathrm{clean}}-s_{\mathrm{corrupt}}}.
$$

Equal endpoints leave recovery undefined and are refused. Recovery need not
lie in $[0,1]$. Sufficiency of retained components, effects of removing them,
and contributions of individual components answer different questions;
circuit overlap alone does not establish behavioral faithfulness
\citep{hanna-etal2024}.

For the packaged sparse autoencoder, stack $n$ site vectors as rows of
$\mathbf{X}\in\mathbb{R}^{n\times d}$. With hidden width $h$, its activations
and reconstructions are

$$
\mathbf{Z}=\operatorname{ReLU}(\mathbf{X}\mathbf{W}_e
+\mathbf{1}\mathbf{b}_e^{\mathsf T}),\qquad
\widehat{\mathbf{X}}=\mathbf{Z}\mathbf{W}_d
+\mathbf{1}\mathbf{b}_d^{\mathsf T},
$$

where $\mathbf{W}_e\in\mathbb{R}^{d\times h}$ and
$\mathbf{W}_d\in\mathbb{R}^{h\times d}$. Training minimizes

$$
\ell_{\mathrm{sae}}=\frac{\|\widehat{\mathbf{X}}-\mathbf{X}\|_F^2}{nd}
+\frac{\lambda}{nh}\sum_{i=1}^{n}\sum_{j=1}^{h}z_{ij}.
$$

Decoder rows are normalized after each update. Encoder and decoder weights
are separately trainable; the decoder is initialized from the encoder's
transpose, but they remain untied. Sparse autoencoder feature analysis is
background for this probe \citep{huben-etal2024}; the equation states LoFI's
particular objective and normalization. Reported activation sparsity is the
fraction of exactly zero activations. Reconstruction error or sparsity alone
does not identify a linguistic variable; the held-out alignment and
intervention measurements supply separate evidence.

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

\bibliography
