# Train on judgments

An inference loss's `response` declares what the observed responses to its
pairs are. The `label` response reads each pair's inference label; the
`judgment` response reads an ordinal judgment per pair through a declared
instrument, a table giving the probability of each response level under each
inference label. This guide trains the L1 point against judgments, shows how a
mismatched declaration is refused, and describes the predicted judgment source
that turns a trained point's predictions into a judgment dataset.

## Declare the instrument

[`docs/examples/judgments/judgment-response.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/judgments/judgment-response.yaml)
is an overlay on
[`experiments/l1/l1-run.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/l1/l1-run.yaml),
the run the [fifth tutorial](../../tutorials/05-induce-logical-forms.md)
trains:

```yaml
approximation: {residual_mass: 0.95}
loss:
  terms:
    - objective:
        kind: inference
        signal: likelihood
        response:
          kind: judgment
          probabilities:
            entails: [0.05, 0.15, 0.8]
            neutral: [0.15, 0.7, 0.15]
            contradicts: [0.8, 0.15, 0.05]
        auxiliary_weight: 0.0
data:
  datasets:
    train:
      dataset:
        judgments: [2, 1, 0, 2, 1, 0]
```

| Line | What it says |
| --- | --- |
| `response: {kind: judgment, probabilities: …}` | One row per inference label, each a probability distribution over the same number of levels (here three). The labels must be inference labels, and at training time they must be exactly the labels the grounding produces. |
| `signal: likelihood` | The loss is the negative log probability of each observed level; `reward` pays that probability as a policy-gradient reward instead. |
| `judgments: [2, 1, 0, 2, 1, 0]` | One zero-based response level per pair of the synthetic inference dataset, in pair order. |
| `approximation: {residual_mass: 0.95}` | Under the judgment response the L1 run's term decoding measures more unexplored mass than the run's declared `0.9` bound and is refused, so the overlay raises the bound. A judgment response asks the grounding both entailment questions of each pair, where a label response may ask only the observed one. |

For each pair, the response computes the probability of every level by
summing the instrument's rows under the grounded label distribution, then
scores the observed level. A level outside the instrument's width is refused
when the batch is scored.

Write $Z$ for the latent inference label and $Y$ for the response level.
The instrument is a row-stochastic matrix $\mathbf{A}$ with
$a_{zk}=\mathbb{P}(Y=k\mid Z=z)$ and $\sum_k a_{zk}=1$.
Given grounded label PMF $p_Z(z\mid x)$ for pair $x$, it induces

$$
p_Y(k\mid x)=\sum_z a_{zk}p_Z(z\mid x),\qquad
\ell_{\mathrm{response}}=-\frac1n\sum_{i=1}^{n}\log p_Y(y_i\mid x_i).
$$

Here $n$ counts observed responses whose grounding is decided; abstained
pairs do not contribute to this average. The instrument is fixed by the
declaration. Different label PMFs that map
to the same response PMF cannot be distinguished by these responses alone.
The reward route uses the observed response probability as its credit; it
is not the same objective as negative log likelihood. See
[Induction objectives](../../explanation/induction-objectives.md) for the
estimator and gradient conventions, and [Judgment models](../../explanation/judgment-models.md)
for fitted response models.

## Train it

Apply the overlay with `--extra`:

```bash
uv run lofi run --config experiments/l1/l1-run.yaml \
  --extra docs/examples/judgments/judgment-response.yaml \
  --root runs/judgments --seed 0 --experiment-seed 108
```

```text
490b45f5edbf4a48 completed runs/judgments/runs/490b45f5edbf4a48
```

Read back what the point was trained under, and its inference loss at each
step:

```python
import json
from pathlib import Path

from lofi.run import read_metrics

[directory] = Path("runs/judgments").glob("runs/[0-9a-f]*")
resolved = json.loads((directory / "config.resolved.json").read_text())
term = resolved["loss"]["terms"][0]["objective"]
print(term["response"]["kind"], term["signal"])
print(resolved["data"]["datasets"]["train"]["dataset"]["judgments"])

recorded = read_metrics(directory)
for epoch in recorded.fitting.epochs:
    for step in epoch.steps:
        print(
            round(step.metrics["loss/inference"], 4),
            step.metrics["loss/inference/questions"],
        )
```

```text
judgment likelihood
[2, 1, 0, 2, 1, 0]
0.7999 8.0
2.3398 8.0
```

The `loss/inference` family carries the same prover counters as label
training (calls, memo hits, abstentions, residual), so the two responses can be
compared on one run document by changing only `response` and `judgments`.

## How a declaration is refused

A dataset that declares `judgments` must give exactly one level per pair, each
nonnegative. The refusal is raised when the document is composed, before
anything is built:

```bash
uv run lofi run --config experiments/l1/l1-run.yaml \
  --extra docs/examples/judgments/judgment-response.yaml \
  --set 'data.datasets.train.dataset.judgments=[2,1]' \
  --root runs/judgments --experiment-seed 108
```

```text
lofi run: data.datasets.dataset was refused: an inference run with 6 pairs declares 2 judgment responses
```

The command exits with status `2`. An instrument whose rows differ in width,
have fewer than two levels, or do not each sum to one, and an instrument that
names an unknown inference label, are refused the same way. A `judgment`
response requires the `judgments` training batch, so a dataset that declares no
`judgments` cannot supply it; in a trainer with several steps, the `judgment`
step (`{kind: judgment, name: …, dataset: …}`) is the one that consumes
inference pairs together with their judgments.

## Judgments predicted by a trained point

The `lofi.judgment_source` entry-point group registers one source,
`predicted`, which turns a trained inference point's predictions over
constructed bead items into a judgment dataset. Its document,
[`PredictedJudgmentSourceSpec`](../../reference/configuration/training.md),
names the producer, the items and the response space:

| Field | What it declares |
| --- | --- |
| `kind` | `predicted`. |
| `producer_run`, `producer_point` | The run and point whose predictions are read. |
| `item_set` | The [judgment-item construction](construct-judgment-items.md) document the items are built from, relative to this document. |
| `inputs` | The rendered element each input role the point publishes reads; an inference point publishes `premise` and `hypothesis`. |
| `factors` | The bead expression each factor every row records is read by over the item's attributes. |
| `question` | The question the judgments answer. |
| `labels` | At least two distinct response labels, in order. |
| `participant` | The participant every predicted row is recorded under; `lofi-prediction` by default. |

`lofi.run.execute_predicted_judgments(document, predictor)` validates such a
document once, constructs its items and returns the `PredictedJudgmentSource`
the trained-point predictor's predictions over them make. Graph-seeded pair
items, for instance, are constructed by the `entailment-graph` lexicon
generator described there. No run or experiment document selects a judgment
source, so this source is reached from Python only.

Fitting a response model to judgments, rather than training a parser on them,
is covered in [Fit a judgment model](fit-a-judgment-model.md).
