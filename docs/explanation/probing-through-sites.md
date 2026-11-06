# Probing through published sites

## What can be learned from an internal value

A [probe](../how-to/probes/run-probes.md) examines a trained model's internal
computation through the [sites it publishes](#a-site-is-a-published-promise).
For instance, a parser can expose the vector it computes for “sees Mary,” a
[span of the sentence](charts-and-approximation.md#reusing-partial-analyses)
“John sees Mary.” A classifier trained on such vectors can test whether they
predict grammatical categories. Replacing a vector and running the remaining
computation tests how the replacement changes the parser's output.

[`Site`](../reference/nn.md#lofi.nn.sites.Site) identifies a value, and
[`SiteMap`](../reference/nn.md#lofi.nn.sites.SiteMap)
declares its permitted uses; [`instrument`](../reference/probe.md#lofi.probe.instrument)
opens access to them during a computation. Their implementations are in
[`lofi.nn.sites`](https://github.com/logical-form-induction/lofi/blob/main/lofi/nn/sites.py)
and [`lofi.probe._instrument`](https://github.com/logical-form-induction/lofi/blob/main/lofi/probe/_instrument.py).

To connect category prediction with the parser's use of that information,
[Classifier controls and interchange
interventions](#what-a-probe-result-licenses) assess memorization and the effects
of replacements under their respective assumptions.

## Observations and interventions

A [tensor](../reference/nn.md#lofi.nn.sites.SiteSpec) is a multidimensional numerical array. Its **axes** say what
the dimensions index, such as sentence, span, and feature. A [mask](../reference/nn.md#lofi.nn.sites.SiteSpec) marks
which positions correspond to real observations when arrays also contain
padding. These declarations let a probe distinguish, for instance, the
features of one span from the same feature across many spans.

For input $x$ and fixed trained parameters $\boldsymbol{\theta}$, write
$\mathbf{H}_s(x)$ for the tensor published at site $s$. Its declared
axes specify a product shape $n_1\times\cdots\times n_k$ and its mask
$\mathbf{M}_s$ specifies live entries. A probability dtype
requires entries in $[0,1]$ and a log probability dtype requires entries at
most zero when numeric checks are enabled. Normalization is a separate
requirement of the particular role: a split PMF
sums to one over alternatives, whereas [constituent marginals](charts-and-approximation.md#inside-outside-and-marginals) need not sum
to one over spans.

An observation returns $\operatorname{stopgrad}(\mathbf{H}_s(x))$.
An intervention replaces that value by $\mathbf{H}'_s$ during the forward
computation. If $o(x)$ denotes a scalar downstream outcome, write
$o(x;s\leftarrow\mathbf{H}'_s)$ for the result of that replacement.
This notation specifies the computational replacement. Interpreting it as an
intervention on a linguistic variable requires an alignment between that
variable and the replaced value.

Replacing a span vector with zeros is one possible intervention. It may
change the parser's output, but zero may also be an unusual value that
ordinary inputs never produce. Replacing it with a mean or with a value
from another example asks a different question. Interpretation thus depends
on the replacement and on the population of examples, as well as the site.

The packaged intervention probe measures an absolute change. For pass $b$,
let $J_b$ contain coordinates live under the clean reading’s mask at
outcome sites reached in both passes.
Its reported value over $m$ passes is

$$
\widehat\delta=\frac1m\sum_{b=1}^{m}
\frac1{|J_b|}\sum_{j\in J_b}|o'_{b,j}-o_{b,j}|.
$$

The two-stage average gives each pass equal weight. Pooling coordinates
across passes would instead weight passes by their number of live coordinates. Empty outcome sets
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

A component's [`SiteMap`](../reference/nn.md#lofi.nn.sites.SiteMap) is immutable and belongs to one component. The
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
[`differentiate`](../reference/probe.md#lofi.probe.differentiate) region over sites checked as [`GradientSites`](../reference/probe.md#lofi.probe.GradientSites).

Some guarantees are conditional. Whether each reading matches its declared
axes, numeric class and fill is checked only when [`LOFI_CHECK=1`](../how-to/extending/checks.md) is set; with
it unset, a component that publishes a wrong specification is trusted. A site
specification describes a computational value: `SPLIT_DISTRIBUTION` promises
log probabilities over splits for each span. Connecting those splits to
linguistic constituents requires evaluation against a linguistic analysis.

## Shared role inventories

A role means the same thing wherever it is published. Each component kind
(token encoder, subword pooler, [term encoder](../how-to/models/configure-term-encoders-and-decoders.md#choose-an-encoder), [term decoder](../how-to/models/configure-term-encoders-and-decoders.md#choose-decoders), [controller](../how-to/models/configure-a-controller.md#scores-and-action-probabilities), parser)
has a declared inventory: the roles it must publish, structures of roles it
publishes together or not at all, and mutually exclusive alternatives. A
parser always publishes `CHART_CELL`; it publishes the inside score, outside
score and split distribution together if at all. A term encoder's recurrence
roles come in one of three sets, for a vanilla recurrence, a GRU and an LSTM.
The inventory is checked when a site map is built.

This is what lets a diagnostic be written once. [`span-length-bias`](../how-to/probes/run-diagnostics.md) declares
that it requires `CONSTITUENT_MARGINAL` and reads that role's `item` and `span`
axes, and it runs on any parser that publishes the role, whatever its chart
family, [semiring](charts-and-approximation.md#what-a-semiring-decides) or encoder. A diagnostic whose roles a pass does not reach is
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
classifier probes need only published points; the [circuit probe](../how-to/probes/run-probes.md#the-probe-kinds) needs a pass
that also publishes a computation graph and the endpoints to interrogate it,
and the causal and [sparse autoencoder probes](../how-to/probes/run-probes.md#the-probe-kinds) need a causal population of
aligned clean and [counterfactual inputs](../how-to/probes/run-probes.md#declare-a-counterfactual-population). A pass that publishes none of these
makes those probes unanswerable, and they are refused. The same reasoning governs tokenization: a causal intervention
on a span of a pretrained encoder's input is meaningful only where the span maps
exactly onto the encoder's rows, so alignment between source tokens and model
rows is computed exactly ([`tokenizer_alignment`](../reference/probe.md#lofi.probe.tokenizer_alignment)). A sentence some checkpoint
cannot align whole, because a word has no row, a row crosses two words or the
sentence overflows the window, is excluded, and the
exclusion is counted under that checkpoint and reason beside the pairs the
alignment policy refuses, so the reported population accounts for every
candidate sentence. A probe document's [`counterfactuals` section](../how-to/probes/run-probes.md#declare-a-counterfactual-population) generates it under the run's
own grammar and adds only the substitution settings and the candidate file.

## What a probe result licenses

[Classifier probes](../how-to/probes/run-probes.md#the-probe-kinds) fit a prediction function on representations and evaluate
it on held-out labels. Their selectivity is
$s=a_{\mathrm{task}}-a_{\mathrm{control}}$, where both accuracies use
the same declared probe capacity and their respective training tasks
\citep{hewitt-liang2019}. The control task tests how much accuracy the probe
can achieve by memorizing regularities, a possible source of high task accuracy.

An interchange intervention can replace selected directions within a
representation instead of the whole vector. One way to express this is to
rotate coordinates, take selected coordinates from a source example, leave
the others from the original example, and rotate back. The matrix expression
below specifies the exchange; held-out counterfactual comparisons test the
proposed linguistic interpretation of the selected directions.

For [distributed interchange interventions](../how-to/probes/run-probes.md#the-probe-kinds), an orthogonal matrix
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
pairs \citep{geiger-etal2024}. The resulting comparison of models is scoped to
that counterfactual population and variable alignment.

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
and contributions of individual components answer different questions.
Evaluate behavioral faithfulness through circuit outputs; overlap measures
structural agreement \citep{hanna-etal2024}.

For chosen scores $s_{\mathrm{clean}}=4$, $s_{\mathrm{corrupt}}=2$, and
$s_C=5$, recovery is $(5-2)/(4-2)=1.5$. A value above one is thus possible:
the circuit's score exceeds the clean reference on this measurement. The
ratio measures recovery relative to the clean–corrupt score difference.

Sparse autoencoding asks whether a representation can be approximated using
relatively few active learned features. Identifying these features with
grammatical categories or meanings additionally requires linguistic evidence.
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
$\mathbf{W}_d\in\mathbb{R}^{h\times d}$ are the weight matrices,
$\mathbf{b}_e\in\mathbb{R}^{h}$ and $\mathbf{b}_d\in\mathbb{R}^{d}$
are biases, and $\mathbf{1}\in\mathbb{R}^{n}$ is the all-ones vector.
Writing $z_{ij}$ for entry $(i,j)$ of $\mathbf Z$, training minimizes

$$
\ell_{\mathrm{sae}}=\frac{\|\widehat{\mathbf{X}}-\mathbf{X}\|_F^2}{nd}
+\frac{\lambda}{nh}\sum_{i=1}^{n}\sum_{j=1}^{h}z_{ij}.
$$

The nonnegative weight $\lambda$ controls the penalty on mean activation.
Decoder rows are normalized after each update. Encoder and decoder weights
are separately trainable; the decoder is initialized from the encoder's
transpose, but they remain untied. Sparse autoencoder feature analysis is
background for this probe \citep{huben-etal2024}; the equation states LoFI's
particular objective and normalization. Reported activation sparsity is the
fraction of exactly zero activations. Held-out alignment and intervention
measurements supply evidence for assessing a linguistic interpretation of the features.

A probe result is a measurement of a trained point at its published sites over
declared inputs. An intervention's effect on the log partition says how much
the computation downstream of a site depends on the value there, on those
inputs, under that replacement. A grammatical-category interpretation also
requires alignment with that category. Changing the replacement (zero, mean,
resampled) changes the intervention being measured. Inferential rows carry their
null distribution, their [raw and corrected p-values](experiments-and-commitments.md#corrections) and the dependence regime
of their correction family, so a claim drawn from them can be read against the
family it was tested in. Comparing two points through a probe is a comparison
of the same promised values; comparing a probe result with a gold analysis
requires the alignment, control task or null the probe declares.

[Probe a trained point](../how-to/probes/run-probes.md) declares and executes
probes.

## References

\bibliography
