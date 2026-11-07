# Configure readouts

A readout under [`model.readouts`](../../reference/spec.md#lofi.spec.ReadoutSpec) declares which terms a run reads off the
chart: at every span it observes one [representation space](configure-term-encoders-and-decoders.md#declare-a-space) through that space's
decoders, and publishes either the decoder's symbol marginal rows, the mass of a
declared target set, or weighted complete terms under a decoding strategy.
Losses and evaluators then name a readout, or the output it publishes, as
their input. This guide declares each of the three routes, chooses a decoding
strategy, and shows how a readout is refused. A last section declares the span
decoders under `model.span_decoders`, which reconstruct words rather than terms.

The examples compose
[`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml).
Its `defaults` select `model.readouts: types`, the packaged fragment
[`lofi/configs/model/readouts/types.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/model/readouts/types.yaml), which declares two readouts:

```yaml
types:
  kind: terms
  space: categories
  reads: inside
  targets: grammar-goal
  output: sentences
category-terms:
  kind: terms
  space: categories
  reads: inside
  decoding: {strategy: exhaustive, residual_mass: 0.01, cap: 64}
```

The document then refines `category-terms` with its own
`decoding: {strategy: exhaustive, residual_mass: null, cap: 64}`. The setup the examples share:

```python
from pathlib import Path

from lofi.spec import ConfigValueError, RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("docs/examples/synthetic-parser.yaml")
```

## Read the composed readouts

```python
run = codec(document)
for name, readout in run.model.readouts.items():
    print(
        name,
        readout.space,
        readout.reads,
        readout.targets,
        readout.output,
        readout.decoding,
    )
    print(
        "  requires", sorted(readout.required_chart_outputs()), readout.estimate_kind()
    )
```

```text
category-terms categories inside None None ExhaustiveDecodingSpec(strategy='exhaustive', residual_mass=None, cap=64, clamp=None, exact_residual=1e-06)
  requires ['cells'] enumerated
types categories inside grammar-goal sentences None
  requires ['alpha', 'cells', 'mu'] None
```

The one packaged readout kind is `terms`, registered under the
`lofi.readout` [entry point group](../../reference/entry-points.md#registered-groups-and-names). Its fields:

| Field | What it says |
| --- | --- |
| `space` | The [representation space](configure-term-encoders-and-decoders.md#declare-a-space) observed, by its role under [`model.spaces`](../../reference/spec.md#lofi.spec.SpaceSpec). |
| `reads` | Which chart value is decoded: `inside` reads the chart's cells, and `span` reads its span vectors. |
| [`targets`](../../reference/configuration/model.md#lofi.spec.readouts.TermsReadoutSpec), `output` | A target set supplied by the grammar and the name its mass is published under. `grammar-goal` is the grammar's goal categories. |
| `mass` | How a target's mass is computed: `factorized` (the default) multiplies the symbol probabilities at the slots the target fills, and `coupled` takes the decoder's weight for the whole target, each slot conditioned on the target's symbols above it. |
| `decoding` | A strategy for decoding complete terms, or `null` for none. |
| `meanings` | How complete [logical forms](../../explanation/logical-forms-as-types.md#semantic-typing) are obtained at internal spans: `decoded` (the default) decodes one at each span, and `proof-term` pushes lexical forms through the parser's semantic proof actions, under a `weighting` from the `lofi.meaning_weighting` group (`distinct` or `multiplicity`) and a required `residual_mass` that bounds each derived phrase cell ([Bound proof term phrase cells](#bound-proof-term-phrase-cells)). |

`required_chart_outputs` is what the chart must publish for the readout to
run. The `types` readout's target mass needs the [split posteriors](../../explanation/charts-and-approximation.md#inside-outside-and-marginals) `alpha` and
`mu` as well as the cells, which is why a chart without them is refused
([Configure a chart parser](configure-a-chart-parser.md) shows the refusal).

For a target set $A$ of complete terms and cell vector $\mathbf{h}_s$, a
coupled target score sums the decoder weights
$u_s(A)=\sum_{t\in A}w(t\mid\mathbf{h}_s)$, excluding terms outside the
published rows' support. The factorized route instead uses

$$
u_s^{\mathrm{fact}}(A)=\sum_{t\in A}\prod_{i\in J(t)}p_i(o_i(t)\mid\mathbf{h}_s),
$$

where $J(t)$ lists the slots the target fills and $o_i(t)$ is its operator
symbol at slot $i$ (payload alternatives are summed in these rows). Multiplication of marginal rows is an approximation to a joint
term distribution; it need not equal teacher-forced decoding. The target
readout then multiplies the member score by the chart's constituent factor
$\mu_s$ and publishes the binary pair $(m_s,1-m_s)$, with
$m_s=\min(1,\mu_su_s(A))$. Masks track cells for which both factors are
defined. Factorized target events can overlap; the clamp caps their
constituent-weighted sum at one when reporting the mass.

## Choose a route

Choose a route by the question the objective needs answered. [Marginal rows](../../explanation/induction-objectives.md#coherence)
ask about individual symbol choices. Target mass asks how much total weight
belongs to a specified set of expressions. Complete-term decoding asks which
explicit expressions receive weight. In a hypothetical distribution, two
permitted sentence categories with probabilities 0.4 and 0.3 have target mass
0.7 together; reading only the most probable category would miss part of it.

A readout publishes one of three things:

1. **Marginal rows.** With no [`targets`](../../reference/configuration/model.md#lofi.spec.readouts.TermsReadoutSpec) and no `decoding`, the readout publishes
   the decoder's per-symbol marginal rows, which an objective such as
   coherence reads.
2. **Target mass.** With `targets` and an `output`, it publishes the mass of
   the target set under that output name. The document's grammaticality loss
   reads it as `readout: sentences`.
3. **Complete terms.** With a `decoding` strategy, it publishes weighted
   complete terms and their [residual mass](../../explanation/charts-and-approximation.md#the-residual-mass-budget). The document's coherence loss reads
   `category-terms` this way.

Target scoring is a separate readout route, so a readout cannot declare both a
target and a decoding strategy, and an output needs a target source. A
`proof-term` meaning route builds on decoded lexical forms, so it needs a
decoding strategy:

```python
for overrides in (
    ('model.readouts.types.decoding={"strategy": "greedy"}',),
    ("model.readouts.category-terms.output=sentences",),
    ('model.readouts.types.meanings={"kind": "proof-term", "residual_mass": 0.99}',),
):
    try:
        codec(document, overrides=overrides)
    except ConfigValueError as refused:
        print(refused)
```

```text
model.readouts.types was refused: target mass is computed from marginal rows, not decoded terms
model.readouts.category-terms was refused: a target-mass output needs a target source
model.readouts.types was refused: proof-term meanings require a lexical decoding strategy
```
## Choose a decoding strategy

A [decoding strategy](../../reference/configuration/model.md) determines how the readout inspects alternatives. A beam
retains a limited collection of promising partial candidates; sampling draws
candidates according to its declared sampling procedure; exhaustive decoding
explores its bounded representation subject to the cap. To assess how much
of the distribution the returned terms cover, inspect the reported residual
mass under the decoder's declared support.

`decoding.strategy` selects one of four strategies. The packaged fragments
under [`lofi/configs/model/readouts/forms/decoding/`](https://github.com/logical-form-induction/lofi/tree/main/lofi/configs/model/readouts/forms/decoding/) declare `beam` (width 16),
`exhaustive` (cap 64) and `sample` (rescore tolerance `1e-5`) for a readout
named `forms`:

| `strategy` | Decodes | Fields |
| --- | --- | --- |
| [`greedy`](../../reference/spec.md#lofi.spec.GreedyDecodingSpec) | The stepwise argmax. | [`clamp`](../../reference/spec.md#lofi.spec.GreedyDecodingSpec). |
| [`beam`](../../reference/spec.md#lofi.spec.BeamDecodingSpec) | A bounded beam. | `width`, `clamp`. |
| [`exhaustive`](../../reference/spec.md#lofi.spec.ExhaustiveDecodingSpec) | Every term, up to `cap` terms and under a [residual mass](../../explanation/charts-and-approximation.md#the-residual-mass-budget) bound. | `cap`, `residual_mass`, `clamp`, `exact_residual` (default `1e-6`: an untruncated enumeration whose residual falls at or under it is reported exact). |
| [`sample`](../../reference/spec.md#lofi.spec.SampleDecodingSpec) | Repeated draws, rescored. | `draws` (default 1), `rescore_tolerance`, `restriction`, `clamp`. |

`clamp` is a term in the language's surface syntax that fixes the decoder's
choice wherever the term has a node. A sampling `restriction` is `full` (the
default, every legal choice), `top-k` with a positive `k`, or `nucleus` with a
mass `p` in `(0, 1]`, each at a positive `temperature` (default `1.0`). A
draw runs the restriction installed under its kind in the
`lofi.sampling_restriction` group.

The readout runs the strategy installed under its name in the
`lofi.decoding_strategy` group. A strategy states whether it enumerates terms
under a residual mass bound, which `exhaustive` does and which paired
coherence requires, and whether it rescores its terms under teacher forcing,
which `sample` does and which refuses a decoder without teacher-forced rows.

Replace the exhaustive decoding of `category-terms` with a beam:

```python
beam = codec(
    document,
    overrides=(
        'model.readouts.category-terms.decoding={"strategy": "beam", "width": 16}',
    ),
)
print(beam.model.readouts["category-terms"].decoding)
```

```text
BeamDecodingSpec(strategy='beam', width=16, clamp=None)
```

Or with eight nucleus samples:

```python
sample = codec(
    document,
    overrides=(
        'model.readouts.category-terms.decoding={"strategy": "sample", "draws": 8,'
        ' "rescore_tolerance": 1e-5, "restriction": {"kind": "nucleus", "p": 0.9}}',
    ),
)
print(sample.model.readouts["category-terms"].decoding)
print(sample.model.readouts["category-terms"].estimate_kind())
```

```text
SampleDecodingSpec(strategy='sample', restriction=NucleusSamplingSpec(temperature=1.0, kind='nucleus', p=0.9), draws=8, rescore_tolerance=1e-05, clamp=None)
sampled
```

`estimate_kind` reports whether a readout's complete terms are `sampled` or
`enumerated`. A pairwise inference grounding reads it: its premise and
hypothesis readouts must both decode complete terms, under one kind of
estimate, or the grounding is refused
([Reward with a theorem prover](../training/reward-with-a-theorem-prover.md)).

## Bound an exhaustive decoding

Exhaustive decoding is the one strategy that truncates, so it is the one that
draws on the run's [approximation bound](configure-a-chart-parser.md#bound-the-approximation).
A `residual_mass` on the decoding overrides the run level
[`approximation.residual_mass`](../../reference/spec.md#lofi.spec.ApproximationSpec), and a decoding without one takes the run level
bound. The packaged `types` fragment declares an override of `0.01` on
`category-terms`; the document sets that override to `null`, so the composed
`category-terms` takes the document's `approximation.residual_mass: 0.75`. A
bound of one is refused:

```python
try:
    codec(
        document,
        overrides=(
            'model.readouts.category-terms.decoding={"strategy": "exhaustive",'
            ' "cap": 64, "residual_mass": 1.0}',
        ),
    )
except ConfigValueError as refused:
    print(refused)
```

```text
model.readouts.category-terms.decoding.residual_mass was refused: an exhaustive residual mass lies in [0, 1), not at 1.0
```

## Bound proof term phrase cells

Retaining most of each word's mass can still leave less mass for a phrase.
As an illustrative product, retaining 0.9 at each of two lexical factors gives
0.81 before any additional filtering. More factors compound the omission.
The route thus records a separate bound for derived phrase cells.
A [residual mass bound](../../explanation/charts-and-approximation.md#the-residual-mass-budget) quantifies omitted alternatives. Its effect on a downstream
loss or evaluation score depends on how that quantity weights those alternatives.

On the `proof-term` meaning route only the words of a sentence are decoded.
Each word's frontier is checked against the readout's own bound, the decoding's
`residual_mass` or else the run level [`approximation.residual_mass`](../../reference/spec.md#lofi.spec.ApproximationSpec), as above.
A phrase cell is instead derived from its children's frontiers, and it keeps
only the mass they retain, so its residual compounds the omissions of every
word beneath it: three words that each leave a residual of 0.9 retain at most
0.001 of their joint mass before the phrase's own frontier cap and its
non-semantic actions remove more. A bound that suits a word would refuse
almost every phrase, so the route declares a separate `residual_mass` for its
derived cells, and a document declares both:

```python
proof_terms = codec(
    document,
    overrides=(
        'model.readouts.category-terms.meanings={"kind": "proof-term",'
        ' "residual_mass": 0.999}',
    ),
)
print(proof_terms.approximation.residual_mass)
print(proof_terms.model.readouts["category-terms"].meanings)
```

```text
0.75
ProofTermMeaningSpec(kind='proof-term', weighting='distinct', residual_mass=0.999)
```

Here every word of `category-terms` is held to the run's 0.75, and every
phrase cell to 0.999. A derived cell whose frontier is truncated past its
bound stops the run with a refusal naming the span it was derived at, in the
form `proof-term span (i, j): measured residual mass r exceeds the declared
bound b`. The derived bound has no run level default, so a proof term
route without one is refused:

```python
from lofi.spec import MissingSettingError

try:
    codec(
        document,
        overrides=('model.readouts.category-terms.meanings={"kind": "proof-term"}',),
    )
except MissingSettingError as refused:
    print(refused)
```

```text
model.readouts.category-terms.meanings.residual_mass is required and no source set it
```

## Decode the words of a span

A span decoder under `model.span_decoders` reconstructs the words a chart span
covers. The one packaged kind, `rnn-span-decoder` in the `lofi.span_decoder`
group, declares a recurrent `cell` (from the `lofi.recurrent_cell` group:
`gru` by default, `lstm`, `elman-tanh` or `elman-relu`), `layers`, an optional
`state_width`, a feedforward `head`, and the `end` token that terminates a
span (default `<eos>`). The S1 experiment's
[`experiments/s1/s1-common-run.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/s1/s1-common-run.yaml)
declares one under the role `reconstruction`:

```yaml
span_decoders:
  reconstruction: {kind: rnn-span-decoder}
```

The `span-reconstruction` and `outside-reconstruction` loss terms name it by
that role in their `decoder` field
([Combine losses](../training/combine-losses.md)).

## See also

- [Configure term encoders and decoders](configure-term-encoders-and-decoders.md)
  for the spaces a readout observes.
- [Combine losses](../training/combine-losses.md) for the objectives that read
  readouts.
- The [model configuration reference](../../reference/configuration/model.md)
  for every readout and decoding field.
