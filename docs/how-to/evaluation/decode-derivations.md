# Decode derivations

The `decode` evaluator reads the term beams a trained point publishes through
a [term readout](../models/configure-readouts.md#choose-a-route) and, when it declares an `astar` block, parses each held-out
sentence from its per-token beams with supertag-factored [A*](../../explanation/charts-and-approximation.md#a-widening) search and
selects one derivation by Viterbi or minimum Bayes risk (MBR) decoding. This
guide declares the evaluator, chooses each part of the search, and shows how an
inconsistent choice is refused.

## Declare the evaluator

The packaged run document
[`lofi/configs/run/synthetic-decode-supertag.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/run/synthetic-decode-supertag.yaml)
extends the synthetic parser with one decode evaluator under the role
`decoded-types`:

```yaml
defaults:
  - synthetic-parser
evaluators:
  decoded-types:
    kind: decode
    interval: {replicates: 50, level: 0.95}
    readout: category-terms
    output: lexical-types
    astar: {strategy: {kind: binary}, root_mode: {kind: goal}, heuristic: {kind: supertag-outside}}
    decode:
      kind: mbr
      utility: {kind: constituent}
      candidates: 16
    conventions:
      punctuation: keep
      unary_chains: keep
      trivial_spans: keep
      minimum_length: 0
      aggregation: sentence
      coverage: zero
      self_arcs: allow
      gold_derivation: canonical
```

Reading the declaration field by field:

| Field | What it says |
| --- | --- |
| `readout` | The [term readout](../models/configure-readouts.md#choose-a-route) whose complete terms the evaluator reads. It must name a readout of kind `terms` that the model declares. |
| `output` | `lexical-types` reads category beams, one per token; `logical-forms` reads [logical form](../../explanation/logical-forms-as-types.md#semantic-typing) beams. |
| `astar` | The [A*](../../explanation/charts-and-approximation.md#a-widening) search over the lexical beams. Omit it to report the beams alone. |
| `decode` | How one derivation is chosen from the A* chart: `viterbi`, or `mbr` with a utility and a candidate count. |
| `interval`, `conventions` | The [bootstrap interval](evaluate-a-run.md#set-the-interval) and the eight scoring conventions every held-out evaluator declares, as in [Evaluate a trained point](evaluate-a-run.md). |
## Choose the search

The search combines candidate word categories into a complete derivation
licensed by the grammar.
[A*](../../explanation/charts-and-approximation.md#a-widening) prioritizes partial analyses using their accumulated score and a bound on
what completion can achieve. Its guarantees depend on those bounds and on the search conditions. A
locally promising category can still lead to a low-scoring or incomplete analysis.

A narrow lexical beam can omit a category needed for any complete root.
Widening admits more categories before trying the same grammar's rules again.

Each part of `astar` is a tagged declaration whose `kind` names an entry in
its own [entry point group](../../reference/entry-points.md#registered-groups-and-names), so a plugin can add a variant to any of them:

| Field | Group | Packaged kinds | Default |
| --- | --- | --- | --- |
| `strategy` | `lofi.astar_strategy` | `linear` probes the global supertag score levels from best to worst; `binary` binary-searches the ordered levels. | `linear` |
| `root_mode` | `lofi.astar_root_mode` | `goal` finishes only at spanning items the grammar's goal accepts; `free` finishes at every spanning item. | `goal` |
| `heuristic` | `lofi.astar_heuristic` | `supertag-outside` bounds a candidate by the best tag at every other position; `zero` uses the neutral bound everywhere. | `supertag-outside` |
| `proposer` | `lofi.astar_proposer` | `supertag-beam` proposes, for rules that only check a conclusion, the categories decoded anywhere in the sentence's beams. | none |

The run's `parse` section supplies the [semiring](../../explanation/charts-and-approximation.md#what-a-semiring-decides), retention, reachability and
[pruning](../../explanation/charts-and-approximation.md#pruning) policy. A* requires `parse.retain: forest` and a semiring with
log probability weights. MBR also requires [posterior marginals](../../explanation/charts-and-approximation.md#inside-outside-and-marginals), supplied by
`parse.semiring: {kind: log-inside}`. A Viterbi decision may use that semiring
or `parse.semiring: {kind: viterbi}`.

The sibling document
[`synthetic-decode-zero.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/run/synthetic-decode-zero.yaml)
changes three choices of the first and inherits the rest:

```yaml
defaults:
  - synthetic-decode-supertag
evaluators:
  decoded-types:
    astar:
      strategy: {kind: linear}
      heuristic: {kind: zero}
    decode: {kind: viterbi}
```

Composing both documents and reading the settings each evaluator is built
from:

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))

for name in ("synthetic-decode-supertag", "synthetic-decode-zero"):
    run = codec(Path(f"lofi/configs/run/{name}.yaml"))
    decoder = run.evaluators["decoded-types"]
    settings = decoder.settings()
    astar = settings.astar
    print(name)
    print("  metrics:", decoder.metrics())
    print(
        "  astar:",
        astar.strategy.name,
        astar.root_mode.name,
        run.parse.build().semiring.name,
        astar.heuristic.name,
        None if astar.proposer is None else astar.proposer.name,
    )
    decode = settings.decode.conventions
    print("  decode:", decode["decode_method"], decode["decode_candidates"])
```

```text
synthetic-decode-supertag
  metrics: ('decode-candidates', 'decode-parsed', 'decode-refused', 'decode-attempts', 'decode-widenings')
  astar: binary goal log-inside supertag-outside None
  decode: mbr 16
synthetic-decode-zero
  metrics: ('decode-candidates', 'decode-parsed', 'decode-refused', 'decode-attempts', 'decode-widenings')
  astar: linear goal log-inside zero None
  decode: viterbi 1
```

A Viterbi decode compares one candidate, so its candidate count is `1`.

## Choose the decision rule

Viterbi asks which single derivation has the highest score. [Minimum Bayes risk](#choose-the-decision-rule)
asks which candidate has the best expected utility under the available
posterior. These can differ: several lower-probability trees may agree on
constituents that the most probable individual tree lacks. A constituent
utility rewards that agreement. The distribution used for this comparison is
the posterior supplied by the admitted search results under their declared bounds.

`decode.kind: viterbi` selects the highest-scoring derivation among the roots
the search finished at. `decode.kind: mbr` unfolds up to `candidates`
derivations from those roots, ranks them by score, and keeps the one with the
highest utility. The packaged utility, `constituent` in the `lofi.mbr_utility`
group, scores a candidate by the posterior marginal mass of its constituents,
read from the [outside pass](../../explanation/charts-and-approximation.md#inside-outside-and-marginals) of the [A*](../../explanation/charts-and-approximation.md#a-widening) chart under the declared root mode.
`candidates` must be positive. Decision rules are the `lofi.decode_method`
plugin family, so an installed package adds one with a [`DecodeSpec`](../../reference/configuration/evaluation.md#lofi.spec.evaluators.decode.DecodeSpec) variant
and an entry under that group.

A* scores each lexical entry by the weight its beam gives it and every rule
application by the multiplicative identity, so the search ranks derivations by
their lexical scores alone.
## What each row reports

Read search effort and parsing success alongside the chosen tree's score.
A wider beam can make a root available at additional cost; an unresolved item
can contribute differently under the declared failure conventions. The
reported means summarize held-out sentences, which are the resampling units
for their uncertainty procedure.

The evaluator reports `decode-candidates` alone when it declares no `astar`
block, and five metrics when it does. Each value is the mean over held-out
sentences, resampled by sentence for its interval:

| Metric | Per sentence |
| --- | --- |
| `decode-candidates` | The number of terms in the beams read: the whole-sentence beam without `astar`, one beam per token with it. |
| `decode-parsed` | `1` when the search reached a root, else `0`. |
| `decode-refused` | `1` when it did not, else `0`. |
| `decode-attempts` | The number of chart parses the strategy ran, one per global score level it tried. |
| `decode-widenings` | The number of lexical candidates added to the beams after the first parse was seeded. |

`linear` parses at each lower score level in turn and stops at the first that
reaches a root, so it returns the first successful level. `binary` parses the
seed beams, then searches the ordered levels for the narrowest that succeeds.

Every row's convention vector extends the declared conventions with the
readout, the output, each search choice by name, the decision rule, its utility
and its candidate count, so two decode rows with different searches never share
a convention hash.

## How a declaration is refused

The declaration is checked as a whole when the document is composed. Each
change below breaks one constraint of the run or its `decoded-types`
declaration. Validate the whole run so the evaluator can check the parse it
will consume:

```python
from copy import deepcopy
import json

from lofi.spec import RunSpec

run = codec(Path("lofi/configs/run/synthetic-decode-supertag.yaml"))
declared = json.loads(run.model_dump_json())

for section, field, value in (
    ("parse", "semiring", {"kind": "viterbi"}),
    ("decoded-types", "decode", None),
    ("decoded-types", "output", "logical-forms"),
):
    document = deepcopy(declared)
    target = (
        document["parse"] if section == "parse" else document["evaluators"][section]
    )
    target[field] = value
    try:
        RunSpec.model_validate(document)
    except Exception as refused:
        print(str(refused).splitlines()[-1][:200])
```

```text
  <root>: evaluators.decoded-types was refused: the decode needs posterior marginals, which parse.semiring 'viterbi' does not supply [validator_error]
  evaluators.decoded-types: an A* decode declares both astar and decode, or neither [validator_error]
  evaluators.decoded-types: logical-form beams do not enter supertag-factored A* [validator_error]
```

A dotted override such as `--set evaluators.decoded-types.decode=null` cannot
make these changes. Several evaluator kinds declare `astar`, `decode` and
`output`, each with its own type, so the override's text is refused with an
[`AmbiguousSettingError`](../../reference/spec.md#lofi.spec.AmbiguousSettingError) because it cannot be read until the kind is selected.
Change these fields in the document itself.

An MBR decode needs [posterior marginals](../../explanation/charts-and-approximation.md#inside-outside-and-marginals), which the `viterbi` [semiring](../../explanation/charts-and-approximation.md#what-a-semiring-decides) does not
supply. `astar` and `decode` are declared together or not at all. [Logical form](../../explanation/logical-forms-as-types.md#semantic-typing)
beams do not enter supertag-factored [A*](../../explanation/charts-and-approximation.md#a-widening), so a `logical-forms` decode reports
its beams only. When the evaluator is built, a declared `astar` block is also
refused unless the trained grammar's profile is lexicalized.

The [charts and approximation](../../explanation/charts-and-approximation.md)
page explains what the semiring and the search bound in a chart.
