# Probe a trained point

A run document's `probes` section declares named probes, each built by the
plugin its `kind` names in the `lofi.probe` entry point group and each reaching
the model only through the sites its components publish. This guide declares a
probe in a run, reads the rows it writes, probes a completed run again with
`lofi probe`, and lists what each probe kind declares and requires. The
[explanation of probing through sites](../../explanation/probing-through-sites.md)
says why probes are confined to sites.

## Declare a probe in a run

[`docs/examples/probes/probed-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/probes/probed-parser.yaml)
adds one probe to the packaged synthetic parser:

```yaml
defaults:
  - run/synthetic-parser
probes:
  diag:
    kind: diagnostics
```

The key, `diag`, is the probe's role: every row it writes carries it. Train the
point:

```bash
uv run lofi run --config docs/examples/probes/probed-parser.yaml \
  --root runs/probes --seed 0 --experiment-seed 5
```

```text
d976a68761a68207 completed runs/probes/runs/d976a68761a68207
```

The probes are composed against the prepared model before it is fitted. The
held-out batches are run once to find which published points they reach, and a
probe naming a role or site that no component publishes where a held-out pass
reaches it is refused then, before any training step: the run records a
preparation failure, and the command exits with status `1`. After training,
the probes run over the trained parser's held-out batches, and their rows are
written to the run directory's `probes.json` and sent to the run's declared
metric sinks as one table.

## Read the rows

`read_probe_rows` returns one `ProbeRow` per measurement:

```python
from pathlib import Path

from lofi.run.probing import PROBES_NAME, read_probe_rows

[directory] = Path("runs/probes").glob("runs/[0-9a-f]*")
for row in read_probe_rows(directory / PROBES_NAME):
    if not row.measure.startswith("finite-readings/nonfinite"):
        print(row.probe, row.measure, row.site, row.value, row.count)
```

```text
diag finite-readings/live parser/CHART_CELL/cells 24.0 1
diag finite-readings/live parser/CONSTITUENT_MARGINAL/mu 6.0 1
diag finite-readings/live parser/IDENTITY_SCORE/identity 4.0 1
diag finite-readings/live parser/INSIDE_SCORE/inside 6.0 1
diag finite-readings/live parser/LOG_PARTITION/log_z 1.0 1
diag finite-readings/live parser/OUTSIDE_CELL/outside_cells 24.0 1
diag finite-readings/live parser/OUTSIDE_SCORE/outside 6.0 1
diag finite-readings/live parser/RECEIVER_EDGE/receiver 32.0 1
diag finite-readings/live parser/SPAN_REPRESENTATION/spans 48.0 1
diag finite-readings/live parser/SPLIT_DISTRIBUTION/alpha 4.0 1
diag finite-readings/log-zero parser/SPLIT_DISTRIBUTION/alpha 0.0 1
diag span-length-bias/normalized-span-length/uniform parser/CONSTITUENT_MARGINAL/mu 0.6666666865348816 1
diag span-length-bias/normalized-span-length/weighted parser/CONSTITUENT_MARGINAL/mu 0.6666666865348816 1
diag span-length-bias/spans parser/CONSTITUENT_MARGINAL/mu 2.0 1
diag cutpoint-entropy/refused None 1.0 1
diag split-posterior-kl/refused None 1.0 1
```

Each row carries:

| Field | What it holds |
| --- | --- |
| `probe`, `kind` | The declared role and the probe's kind. |
| `measure` | The measurement, prefixed by the diagnostic or intervention that took it. |
| `site` | The site read, as `<component>/<ROLE>/<key>`, or `None` for a measurement over no one site. |
| `value`, `count` | The measured value and how many passes or items it was taken over. |
| `null`, `raw_p`, `adjusted_p`, `rejected`, `family`, `dependence` | An inferential row's null distribution, raw and corrected p-values, decision, correction family and dependence regime; `None` otherwise. |

The rows show every site the trained parser publishes, under the component
`parser`. A diagnostic that refuses a pass's readings is counted rather than
dropped, under `<diagnostic>/refused`.
[Run the packaged diagnostics](run-diagnostics.md) covers the diagnostics
catalog.

## Probe a completed run again

`lofi probe` restores a completed run's release checkpoint and executes the
probes of a document composed over the run's recorded configuration. The
document may change only `probes`, `counterfactuals` and `sinks`, since every
other section is what the checkpoint was trained under.
[`docs/examples/probes/intervention.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/probes/intervention.yaml)
zeroes the parser's receiver edges:

```yaml
probes:
  patch:
    kind: intervention
    interventions:
      - site: {component: parser, role: RECEIVER_EDGE, key: [receiver]}
        intervention: {kind: zero}
```

```bash
uv run lofi probe --run runs/probes/runs/d976a68761a68207 \
  --out runs/probes/patched.json --extra docs/examples/probes/intervention.yaml
```

```text
37 runs/probes/patched.json
```

The command prints the number of rows and the file it created. The run's
declared probes run again beside the new one, and the run directory is left as
it was:

```python
for row in read_probe_rows(Path("runs/probes/patched.json")):
    if row.probe == "patch":
        print(
            row.probe, row.kind, row.measure, row.site, round(row.value, 4), row.count
        )
```

```text
patch intervention zero/effect parser/RECEIVER_EDGE/receiver 0.075 1
```

An intervention probe runs each held-out pass once unpatched and once per
declared intervention, and reports the mean absolute change each caused at the
points of its `outcome` role, `LOG_PARTITION` by default.

`lofi probe` refuses, with status `2` and nothing written, a document that
changes another section, a probe the trained point cannot answer, and an
existing output file:

```bash
uv run lofi probe --run runs/probes/runs/d976a68761a68207 \
  --out runs/probes/again.json --set trainer.epochs=3
uv run lofi probe --run runs/probes/runs/d976a68761a68207 \
  --out runs/probes/again.json --set 'probes.diag.roles=["ATTENTION_WEIGHTS"]'
uv run lofi probe --run runs/probes/runs/d976a68761a68207 \
  --out runs/probes/patched.json
```

```text
lofi probe: probing a trained run may change only ['counterfactuals', 'probes', 'sinks'], and the composed document changes ['trainer']
lofi probe: no component publishes ATTENTION_WEIGHTS permitting OBSERVE; the trained point publishes ['CHART_CELL', 'CONSTITUENT_MARGINAL', 'IDENTITY_SCORE', 'INSIDE_SCORE', 'LOG_PARTITION', 'OUTSIDE_CELL', 'OUTSIDE_SCORE', 'RECEIVER_EDGE', 'SPAN_REPRESENTATION', 'SPLIT_DISTRIBUTION']
lofi probe: the probe rows file runs/probes/patched.json exists
```

A run that did not complete, a directory that holds no trained run, and a
document that declares no probe are refused the same way.

## Name a site

A probe names a site by `{component, role, key}` (`ProbeSiteDecl`): the
component's name (`parser` above), a member of `lofi.nn.sites.Role`, and the
site's key within that role. The rows above spell every published site as
`<component>/<ROLE>/<key>`. A site is refused if it is not published, or if it
does not permit the access the probe needs: observation to read it, patching to
intervene on it.

## The probe kinds

| `kind` | Declares | Requires of a held-out pass |
| --- | --- | --- |
| `diagnostics` | `roles`: role names to read, or `null` for every published role. | Observable points of the roles. |
| `intervention` | `interventions`: at least one `{site, intervention}`, no site twice; `outcome`: the role compared. The packaged interventions in the `lofi.intervention` group are `zero`, `mean`, `noise` (`sigma`, `seed`), `shuffle` (`seed`) and `resample` (`seed`). | Patchable intervention sites and observable outcome points. |
| `classifier` | `role` (default `CHART_CELL`), `labels`, `held_out_fraction`, `split_seed`, `hidden`, `epochs`, `learning_rate`, `seed`, `control_seed`, `block_fractions`. Reports selectivity against a control task, online codelength, and accuracy after LEACE erasure. | One vector per span at the role's points, and labeled spans in at least two held-out items. |
| `circuit` | `acdc_trials` (required), `attribution_steps`, `acdc_threshold`, `null_replicates`, `seed`, `completeness_draws`, `bootstrap_replicates`, `bootstrap_level`, `sensitivity_threshold`, `include_zero`, `fdr_alpha`, `dependence`. Attributes edges with EAP-IG, prunes them with budgeted ACDC, and scores faithfulness against matched random circuits. | A computation graph and the clean and counterfactual endpoints to interrogate it. |
| `causal` | `sites` (`{layer, position, site}`), `searches` (search arms of distinct kinds, below), `joint_search` (the kind of the arm the joint condition and controls fit, default `fixed-rank`), `objective`, `budget_hours` (required), and the annealing, schedule, bracketing, bucketing, null and correction settings. Runs each declared DAS search arm with interchange intervention accuracy. | A causal population: aligned clean and counterfactual inputs. |
| `sae` | `site`, `width`, `sparsity_penalty`, `epochs`, `learning_rate`, `seed`. Fits a sparse autoencoder to one real-valued site. | A causal population. |
| `external` | `model` (an `ExternalModelSpec` with a `checkpoint` source), `backend`, `artifacts` (the cache `root` and `offline` flag), `stimulus_lengths`. Runs split posterior diagnostics over a released DIORA checkpoint; the packaged fragment `probes/diora/naacl-2019` declares the NAACL 2019 release. | Supplies its own passes. |

A causal probe's `searches` name their kinds and default to both arms below. A `fixed-rank` arm declares `rank` (default `1`), the number of
leading rotated coordinates it swaps. A `boundless` arm declares the initial
interval `boundary_lower` and `boundary_upper` (defaults `0.0` and `1.0`) and
`boundary_margin` (how far an initial boundary share is held from 0 and 1
before its logit is taken, default `1.0e-6`). Both kinds are listed in the
`lofi.das_search` group, so an installed package adds a search by subclassing
`DASSearchDecl`, giving it a `strategy(context)` method that returns the search
under the shared schedule, null, objective and annealing, and adding an entry
there.

A causal probe may also declare an `arm`: the node vocabulary representation
the probed point is (`representation`, one of `derivation-variables`,
`trained-transformer-sae-latents`, `dense-linear-probe` and
`random-transformer-sae-control`) and the declared search arm whose sweep it
records (`search`). For every held-out pass the probe then persists that sweep's
per-cell accuracies, their nulls and the evaluated pair identities as the
record `pass-<n>` under its role in `probe-records.json`, beside `probes.json`
in the run directory (`lofi probe` writes it beside its rows file). An
experiment's `node-vocabulary` reducer compares four such records.

Within the causal declaration the field reads:

```yaml
arm: {representation: dense-linear-probe, search: fixed-rank}
```

The held-out passes `lofi run` and `lofi probe` build over a trained parser
publish no computation graph and no causal population, so over the packaged
parsers the `circuit`, `causal` and `sae` kinds are refused when composed:

```text
lofi probe: probe 'circuit' requires a computation graph, but held-out pass 0 publishes none
```

Those three kinds are reached from Python, through the functions they are
built on: `receiver_circuit` draws a chart's receiver edge circuit
(`ReceiverCircuit`), `attribute_edges` attributes each edge by EAP-IG at a set
of `GradientSites`, and `acdc` prunes under a `PruningBudget`; `lofi.probe`
exports all of them. [Instrument an external Torch model](external-model-site-maps.md)
covers sites declared over a third-party model.

## Declare a counterfactual population

A probe document's `counterfactuals` section declares the population of
interchange intervention pairs a causal measurement reads. The pairs are derived
under the run's own grammar, so the section declares no rules, lexicon or goal,
only what the substitution adds, and every field but `filter` is required:

| Field | What it declares |
| --- | --- |
| `sentences` | The file of candidate sentences, one per line, tokens separated by whitespace. |
| `unary_depth` | How many rounds of unary closure a derivation of one string may take. |
| `readings` | How many derivations of one string are read before reading stops; at least two. |
| `render` | How a category is written as the decision at its cell: `declared`, the grammar's declared category syntax, or `s-expression`. |
| `pieces`, `specials`, `contiguous` | The alignment policy a pair's varied span must satisfy under every tokenizer: which pieces of a word stand for it (`first`, `last` or `all`), whether the span keeps the rows no word produced (`exclude` or `include`), and whether its rows must be contiguous. |
| `seed` | The seed the order of one string's pairs is drawn under. |
| `filter` | Optional. A score filter reported over the accepted pairs in three arms, below; absent, the population is reported unfiltered only. |

[`docs/examples/probes/counterfactuals.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/probes/counterfactuals.yaml)
declares one over
[`docs/examples/probes/candidates.txt`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/probes/candidates.txt):

```yaml
counterfactuals:
  sentences: docs/examples/probes/candidates.txt
  unary_depth: 0
  readings: 16
  render: declared
  pieces: all
  specials: exclude
  contiguous: true
  seed: 0
```

A `filter` reads one declared score per candidate sentence and keeps a pair
when the score of the sentence it was generated from reaches an inclusive
minimum. It reports three arms together over the same ordered pairs: the
`primary` arm at `primary`, one `sweep` arm per threshold in `sweep`, and the
`unfiltered` arm keeping every pair. Its fields are:

| Field | What it declares |
| --- | --- |
| `scores` | A JSON Lines file with one object `{"tokens": [...], "score": number}` per candidate sentence, larger meaning more acceptable. It scores exactly the candidate sentences; a missing, extra or repeated sentence, or a score that is not a finite number, is refused. |
| `source` | Where the scores came from: `name`, `public` and `fit_signature`, the grammatical signature the scores were fitted under or `null`. |
| `primary` | The primary arm's inclusive minimum. |
| `sweep` | At least five distinct inclusive minima, the primary one among them. |

The primary arm requires a `public` source with no `fit_signature`: scores
fitted under one signature are not independent of the signatures a probe
compares, so such a source is refused.

```yaml
counterfactuals:
  # the fields above, then
  filter:
    scores: data/candidate-scores.jsonl
    source: {name: public-ratings, public: true, fit_signature: null}
    primary: 0.5
    sweep: [0.1, 0.3, 0.5, 0.7, 0.9]
```

`lofi.run.probing.counterfactual_generator` builds the declared generator from a
trained point and its document. It parses each candidate sentence with the
thread count the grammar's `budget` declares (`threads`, default 1). Each pair records the grammar's fingerprint, and
the digest of the candidate sentences (`candidates_digest`) as its dataset, so a
pair's identity does not depend on where the file lies.

## Campaign scripts

The modules under `scripts/probe/` run probing measurements over the trained
point a completed run directory holds, restored through
`lofi.run.hub.restore_trained_point`. Each reads a JSON declaration in which
every field is required.

### Alignment coverage

`scripts/probe/alignment.py` measures the exact tokenizer alignment of a
point's declared counterfactual population under released checkpoints. It reads
the `counterfactuals` section of the document `--extra` and `--set` compose over
the run's recorded configuration, exactly as `lofi probe` composes one, and a
declaration naming the checkpoints:

| Field | What it declares |
| --- | --- |
| `checkpoints` | The released checkpoints, in evaluation order, each `{checkpoint, source}`, where `checkpoint` is a hub source `{kind: hub, repository, revision}` and `source` is a local directory holding the release or `null` to fetch it through the declared artifacts cache. |
| `join` | The string a sentence's words are joined by before tokenization. |
| `window_rows` | The row bound a sentence must fit whole, or `null` for each tokenizer's published bound. |
| `artifacts` | The cache a checkpoint with no `source` is fetched into: an absolute `root` and `offline`, whether an absent release is refused rather than fetched. |

```bash
uv run python -m scripts.probe.alignment --run RUN \
  --extra docs/examples/probes/counterfactuals.yaml \
  --declaration alignment.json --out coverage.json
```

Every sentence is aligned under every checkpoint. A sentence some checkpoint
cannot align whole, because a word has no row (`zero-piece-word`), a row
crosses two words (`crossing-piece`) or the sentence overflows the window
(`context-overflow`), yields no candidate: it is excluded, and the report counts
it under each such checkpoint and reason. The report records the grammar
fingerprint, the seed, the policy, the checkpoint identities, the candidate
digest, the sentence and excluded-sentence counts, the exclusions by checkpoint
and reason, the candidate and accepted pair counts, the aligned fraction, the
refusals by checkpoint and reason, the accepted pair identities, and under
`filter` either `null` or the declared filter's source and its arms over the
accepted pairs, each with its `kind`, `threshold`, `count`, `dropped` and `kept`
pair identities. The script refuses a document that declares no
`counterfactuals` section and an existing report.

### Attribution calibration

`scripts/probe/acdc_calibration.py` compares EAP-IG with budgeted ACDC over the
receiver edges of the chart scorer a declared component role holds. Each cell's
clean and counterfactual sentences, which must have the same number of words,
are pooled through the point's own token encoder and subword pooler
(`lofi.run.probing.pooled_leaves`).
[`docs/examples/probes/calibration.json`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/probes/calibration.json)
declares one cell:

```json
{
  "scorer": "parser",
  "grid": [
    {
      "name": "three-words",
      "clean": ["word-0", "word-2", "word-4"],
      "counterfactual": ["word-4", "word-6", "word-0"]
    }
  ],
  "steps": 4,
  "threshold": {"kind": "relative", "fraction": 0.01},
  "maximum_trials": 8,
  "divergence": "mean-absolute",
  "metric": "log-partition",
  "budget_seconds": 600.0,
  "price_warmup": 1,
  "price_repeats": 3
}
```

```bash
uv run python -m scripts.probe.acdc_calibration \
  --run runs/probes/runs/d976a68761a68207 \
  --declaration docs/examples/probes/calibration.json \
  --out runs/probes/calibration
```

The `threshold` names its kind. A `relative` threshold prunes an edge whose
divergence increment is strictly below `fraction` times the cell's
full corruption divergence, the divergence of the counterfactual output from
the clean one, so one fraction prunes alike in cells whose divergences differ in
scale; `0.01` prunes an increment below 1% of it. An `absolute` threshold prunes
an increment strictly below a fixed `bound`, in the units of the divergence.
The `circuit` probe's `acdc_threshold` takes the same two kinds and defaults to
`{"kind": "absolute", "bound": 0.01}`. Both kinds are listed in the
`lofi.acdc_threshold` group, so an installed package adds a threshold by
subclassing `ACDCThresholdDecl` and adding an entry there.

The script writes `manifest.json`, recording the declaration, the trained
point, the environment, the wall time and the outcome, and `rows.jsonl`, one
row per cell. Each cell's `acdc` record holds the declared `threshold` and the
`cutoff` it gave that cell. Each estimator's circuit records its faithfulness both as a
normalized share and as the raw divergences of its circuit-only and fully
corrupted outputs from the full model's:

```python
import json

for line in Path("runs/probes/calibration/rows.jsonl").read_text().splitlines():
    row = json.loads(line)
    for estimator in ("eap_ig", "acdc"):
        record = row[estimator]
        print(
            row["cell"],
            estimator,
            round(record["faithfulness"], 4),
            round(record["circuit_divergence"], 4),
            round(record["corrupted_divergence"], 4),
        )
```

```text
three-words eap_ig 1.0 0.0 0.014
three-words acdc 1.0 0.0 0.014
```

The share is one minus the circuit divergence over the corrupted divergence, and
is `null` where patching every edge leaves the output unchanged.
