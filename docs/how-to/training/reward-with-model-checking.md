# Reward with model checking

A `model-checking` pair scorer evaluates each inference pair's premise and
hypothesis logical forms in a declared set of finite structures, under a
declared fuzzy logic, and scores the pair by how true the hypothesis is among
the structures in which the premise holds. It takes the place of the prover
scorer in a `pairwise` grounding, and an `inference` loss term trains on its
scores exactly as it does on proof verdicts. A model-checking score is
recorded as truth in the declared structures, never as a proof verdict. This
guide selects a structure source, declares the logic and the decision policy,
and shows how each is refused.

## Select a model-checking grounding

Three packaged fragments under `lofi/configs/model/grounding/` extend
`prover-enumerated.yaml` and replace its scorer:

| Fragment | Structure source |
| --- | --- |
| `model-checking-sampled` | A seeded draw of finite structures over declared carriers. |
| `model-checking-learned` | Relaxed finite structures fitted to the language's declared postulates during training. |
| `model-checking-given` | One read-only intended structure built from a knowledge base. |

Select one on top of a likelihood-trained run.
[`experiments/l1/l1-run.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/l1/l1-run.yaml)
selects `prover-enumerated` and trains
`{kind: inference, signal: likelihood, response: {kind: label}}`, so a group
override swaps the scorer and leaves the rest of the run as it is:

```bash
uv run lofi show-config \
  --config experiments/l1/l1-run.yaml \
  --set model/grounding=model-checking-sampled
```

The sampled and learned fragments draw their seeds from
`${lofi.seed:model-structures}`, a sub-seed of the run seed. The command line
installs a run context for that resolver. Composing from Python needs one
installed explicitly with `run_context`:

```python
from pathlib import Path

from lofi.spec import (
    RunContext,
    RunSpecCodec,
    config_search_path,
    load_plugins,
    run_context,
)

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("experiments/l1/l1-run.yaml")

with run_context(RunContext(data_root=Path.cwd(), seed=11)):
    for option in ("model-checking-sampled", "model-checking-learned", "model-checking-given"):
        run = codec(document, overrides=(f"model/grounding={option}",))
        scorer = run.model.grounding.scorer
        print(option, scorer.kind, scorer.source.kind)

print(scorer.logic)
print(scorer.source.knowledge_base)
print(scorer.truth_threshold, scorer.abstain_band, scorer.premise_floor, scorer.unsatisfied_premise)
```

```text
model-checking-sampled model-checking sampled
model-checking-learned model-checking learned
model-checking-given model-checking given
FuzzyLogicSpec(tnorm='product', implication='residuum', quantifier='extreme', equality='crisp', margin=0.0)
WikidataKnowledgeBaseSpec(kind='wikidata', snapshot='lofi.resource:data/knowledge_base/wikidata-mini.jsonl', mapping='lofi.resource:data/knowledge_base/wikidata-mini.mapping.json', world='closed', unmapped='undecided')
0.5 0.0 0.5 undecided
```

Without the context, composition refuses the seed with
"`${lofi.seed:...}` needs a run context; no data root or seed is in scope".

## Declare the scorer

The `model-checking-sampled` fragment declares every field the scorer takes:

```yaml
defaults: [prover-enumerated]
scorer:
  kind: model-checking
  source:
    kind: sampled
    structures: 16
    carriers: {e: 2, v: 1}
    density: 0.5
    full_function_space_cap: 64
    henkin_pool: 8
    seed: "${lofi.seed:model-structures}"
    resample: once
    atomic_sampling: uniform
    denotation_streams: independent
    function_sampling: table
    truth_sampling: bernoulli
  unsatisfied_premise: undecided
  premise_floor: 0.5
  abstain_band: 0.0
  truth_threshold: 0.5
  logic: {tnorm: product, implication: residuum, quantifier: extreme, equality: crisp, margin: 0.0}
```

For each pair the scorer visits every structure the source supplies. A
structure counts when the premise, conjoined with the language's theory,
reaches `premise_floor`; the score is the conjunction mass of premise and
hypothesis over the counted structures divided by the premise mass over them.

| Field | What it says |
| --- | --- |
| `truth_threshold` | The truth value in `[0, 1]` at which a hypothesis counts as holding, and the point a score is decided against. |
| `abstain_band` | A score within this distance of `truth_threshold` is left undecided. `0.0` decides every score. The band may not cross either end of the unit interval. |
| `premise_floor` | The truth value in `[0, 1]` a premise must reach for a structure to count. |
| `unsatisfied_premise` | When no structure counts: `undecided` leaves the pair undecided; `zero` decides it with score `0.0`. |

A pair whose forms cannot be evaluated in a structure is left undecided.

## Choose a structure source

The `lofi.structure_source` entry-point group registers the three sources.
All three bound their higher-order domains the same way: a function type whose
full space has at most `full_function_space_cap` members is enumerated in
full, and a larger one is represented by a pool of `henkin_pool` functions.

| `kind` | Fields |
| --- | --- |
| `sampled` | `structures` drawn over `carriers` (one positive size per atomic type) at denotation `density`, in `[0, 1]`, under `seed`. `resample` is `once` or `per-step`. `atomic_sampling`, `denotation_streams`, `function_sampling`, and `truth_sampling` select independently registered sampling capabilities, shown below. |
| `learned` | `structures` over `carriers`, fitted under `seed` from denotation logits drawn at the positive `initialization_scale` (the standard deviation of their normal start). `denotations` is `free`, in which each atomic denotation has its own learned logits, or `from-vocabulary`. The source publishes an auxiliary satisfiability term, which the `inference` objective's `auxiliary_weight` adds to the loss. A fitted structure scores pairs only once its theory holds at the scorer's `truth_threshold`, the floor a sampled or given structure must also meet; the evidence counts the structures read. |
| `given` | A `knowledge_base` that defines one intended structure, with `full_function_space_cap` and `henkin_pool`. |

The sampled source resolves its choices through four entry-point groups:

| Field | Group | Packaged choice |
| --- | --- | --- |
| `atomic_sampling` | `lofi.atomic_sampling` | `uniform` draws uniformly from the atomic carrier. |
| `denotation_streams` | `lofi.denotation_streams` | `independent` derives a stream from the seed, step, structure position, and constant identifier. |
| `function_sampling` | `lofi.function_sampling` | `table` draws one result for each domain member. |
| `truth_sampling` | `lofi.truth_sampling` | `bernoulli` draws true with probability `density`. |

An installed entry supplies the corresponding `AtomicSampling`,
`DenotationStreams`, `FunctionSampling`, or `TruthSampling` capability from
`lofi.lf.models`. These choices determine the distribution of sampled
structures. The scorer estimates conditional satisfaction over those draws;
it does not assume a uniform distribution over all possible structures.


The `given` source reads a knowledge base from the `lofi.knowledge_base`
group. `wikidata` is the one registered kind. The packaged fragment points at
the small fixture shipped under `lofi/configs/data/knowledge_base/`:

```yaml
knowledge_base:
  kind: wikidata
  snapshot: lofi.resource:data/knowledge_base/wikidata-mini.jsonl
  mapping: lofi.resource:data/knowledge_base/wikidata-mini.mapping.json
  world: closed
  unmapped: undecided
```

`world: closed` reads an absent fact as false; `world: open` reads it as
unknown and leaves its question undecided. `unmapped` selects an entry in
`lofi.unmapped_constant` implementing `UnmappedConstantPolicy`. Its `resolve`
method receives the constant identifier, language, and finite frame, and returns
a denotation or `None`. The packaged `undecided` policy returns `None`, leaving
an unmapped constant unknown. The [configuration
guide](../setup/configuration.md) describes the snapshot and mapping files and
how a project replaces them with its own.

## Declare the fuzzy logic

`logic` fixes every model-theoretic choice the evaluator makes:

| Field | Values |
| --- | --- |
| `tnorm` | A t-norm registered under `lofi.fuzzy_logic`: `product`, `goedel` or `lukasiewicz`. |
| `implication` | An implication registered under `lofi.fuzzy_implication`: `residuum`, the t-norm's residual implication, or `material`. |
| `quantifier` | A quantifier registered under `lofi.fuzzy_quantifier`: `tnorm`, which folds a universal's instances with the t-norm and an existential's with its disjunction, or `extreme`, which takes their minimum or maximum. |
| `equality` | An equality registered under `lofi.fuzzy_equality`: `crisp` or `margin`. |
| `margin` | A finite, nonnegative margin. Under `equality: margin`, two real values within it of each other are equal. |

There is one logic declaration per scorer. A learned source fits its
structures under the same declaration rather than a second one of its own.

## How a declaration is refused

Range and enumeration checks run at composition:

```python
def refusal(overrides):
    try:
        codec(document, overrides=overrides)
    except Exception as refused:
        print(type(refused).__name__, "|", str(refused).splitlines()[-1])


sampled = ("model/grounding=model-checking-sampled",)
with run_context(RunContext(data_root=Path.cwd(), seed=11)):
    refusal((*sampled, "model.grounding.scorer.abstain_band=0.6"))
    refusal((*sampled, "model.grounding.scorer.logic.margin=-0.1"))
    refusal((*sampled, "model.grounding.scorer.logic.implication=kleene"))
    refusal((*sampled, "model.grounding.scorer.source.resample=always"))
```

```text
ConfigValueError | model.grounding.scorer was refused: the abstention band crosses a unit-interval boundary
ConfigValueError | model.grounding.scorer.logic.margin was refused: an equality margin is nonnegative, not -0.1
ConfigValueError | model.grounding.scorer.logic.implication was refused: unknown implication 'kleene'; installed implications are material, residuum
SettingParseError | model.grounding.scorer.source.resample expects one of ['once', 'per-step']; got 'always'
```

The `tnorm`, `implication`, `quantifier` and `equality` names are checked
against their installed families at composition, so a plugin that registers
another operation under one of these groups is selectable by its name.

The grounding-level checks of [Reward with a theorem
prover](reward-with-a-theorem-prover.md) apply unchanged, because the scorer
sits inside the same `pairwise` grounding: a `reward` signal requires a
declared `reward`, sampled forms require a moving baseline, and a grounding
requires an `inference` term and an inference-pair dataset.

[Inference as evidence](../../explanation/inference-as-evidence.md) explains
how truth in a set of structures differs from entailment as evidence about a
logical form. The declaration classes are under [Model
declarations](../../reference/configuration/model.md), and the structures and
evaluator under [Logical forms](../../reference/lf.md).
