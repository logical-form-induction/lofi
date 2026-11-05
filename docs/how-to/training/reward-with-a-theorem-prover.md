# Reward with a theorem prover

A `pairwise` grounding with a `prover` scorer sends each inference pair's
premise and hypothesis logical forms to the language's declared theorem
prover, and an `inference` loss term turns the verdicts into a training
signal. Under `signal: likelihood` the verdicts score the observed label;
under `signal: reward` a declared payoff converts each decided verdict into a
reward, with a moving control variate when forms are sampled.
This guide declares the prover, the scorer, the reward and the loss term, and
shows how each mismatch is refused.

## Start from a prover-reward run

[`experiments/l4/l4-run.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/l4/l4-run.yaml)
trains lexical logical forms from synthetic inference pairs under a prover
reward. Its grounding comes from one packaged fragment and one override:

```yaml
defaults:
  - data.datasets.train.dataset: synthetic-inference
  - data.datasets.train.featurizer: pair
  - model.grounding: prover-reward
model:
  grounding:
    scorer:
      budget: {calls_per_step: 4096, wall_per_step: 30.0, cpu_seconds: null}
loss:
  terms:
    - objective:
        kind: inference
        signal: reward
        response: {kind: label}
        auxiliary_weight: 0.0
```

`model/grounding/prover-reward.yaml` extends `prover-enumerated.yaml` with a
control variate, a reward and a countermodel search:

```yaml
defaults: [prover-enumerated]
control_variate: {kind: moving, decay: 0.9}
reward:
  proved: 1.0
  refuted: 0.0
  contradiction: negated-hypothesis
  abstention: exclude
scorer:
  countermodels: {method: exhaustive, carriers: {e: 1, v: 1}, structures: 64, seed: 0, full_function_space_cap: 64}
```

Compose the run and read the grounding back:

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("experiments/l4/l4-run.yaml")

run = codec(document)
grounding = run.model.grounding
print(grounding.kind, grounding.premise, grounding.hypothesis, grounding.composition)
print(grounding.scorer.kind, grounding.scorer.budget)
print(grounding.scorer.countermodels)
print(grounding.reward)
print(grounding.control_variate)
print(run.loss.terms[0].objective)
print(run.data.datasets["train"].dataset.source.language.logic.prover)
```

```text
pairwise forms forms MarginalCompositionSpec(rule='marginal')
prover ProofBudgetSpec(calls_per_step=4096, wall_per_step=30.0, cpu_seconds=None)
CountermodelSpec(method='exhaustive', carriers={'e': 1, 'v': 1}, structures=64, seed=0, full_function_space_cap=64)
RewardSpec(proved=1.0, refuted=0.0, contradiction='negated-hypothesis', abstention='exclude')
MovingControlVariateSpec(kind='moving', decay=0.9, initial=0.0)
InferenceLossSpec(kind='inference', signal='reward', response=LabelResponseSpec(kind='label'), auxiliary_weight=0.0)
TruthTableProverSpec(timeout=5.0, workers=1, batch_size=64, cache_size=None, memory_mb=None, cpu_seconds=None, seed=None, polymorphism='refuse', certificates=(), kind='truth-table', atoms=12)
```

The last line is the prover itself. It belongs to the logical-form language,
under `logic.prover`, not to the grounding. `l4-run.yaml` takes the learner's
language from the data (`model.language: {from: data}`), so this is the
language that declares the prover the scorer calls.

## Declare the prover

A language declares its prover under `logic`. The packaged
`language/propositional.yaml` declares
`prover: {kind: truth-table, timeout: 30.0}`, and
`language/_shared/higher-order-logic.yaml` declares
`prover: {kind: tptp, engine: vampire, dialect: thf, timeout: 10.0, polymorphism: monomorphize}`.
The `lofi.prover` entry-point group registers three families:

| `kind` | Decides entailment by | Own fields |
| --- | --- | --- |
| `truth-table` | Exhaustive propositional evaluation. | `atoms` (default `12`), the most atoms a problem may hold. |
| `smt` | A long-lived SMT solver process. | `engine` (default `z3`; `cvc5` is also registered under `lofi.smt_engine`), `executable`, `logic` (default `ALL`), `options`, `prefix`, `arguments`, `terminate_grace` (default `0.2` seconds). |
| `tptp` | A sandboxed TPTP prover. | `engine` (default `eprover`; `vampire` and `leo3` are also registered under `lofi.tptp_engine`), `executable`, `dialect` (`fof`, `tff` or `thf`; default `fof`), `prefix`, `options`, `problem_file` (default `problem.p`). |

`executable: null` finds the engine on the search path. `prefix` holds the
arguments that lead the command. Left unset, it is the registered engine's own
prefix: `--auto --proof-object=1 --tstp-format` for E, `--mode casc` for
Vampire, none for Leo-III, `-in -smt2` for Z3 and
`--lang=smt2 --incremental --produce-models -` for cvc5. A declared prefix
replaces it, so `prefix: []` drops it. The TPTP `options` and the SMT
`arguments` follow the prefix and the resource options. A TPTP call writes its
problem to `problem_file` in a fresh sandbox directory and passes that name
last. A stopped SMT worker has `terminate_grace` seconds to exit after the
terminate signal before it is killed. Each TPTP dialect serves its own fragment:
`fof` and `tff` first-order logic, `thf` higher-order logic, which a registered
engine may narrow but never widen. Every family shares the
proof-service fields: `timeout` per call (default `30.0` seconds), `workers`,
`batch_size`, `cache_size`, `memory_mb`, `cpu_seconds`, `seed`,
`certificates`, and `polymorphism`, which says how a translation treats a
polymorphic constant: `refuse` (default), `monomorphize` or `erase`.

The declaration classes validate their own fields:

```python
from lofi.spec.provers import SmtProverSpec, TptpProverSpec, TruthTableProverSpec

print(SmtProverSpec(engine="cvc5", logic="UF").settings().engine)
print(TptpProverSpec(engine="vampire", dialect="thf", polymorphism="monomorphize"))
try:
    TruthTableProverSpec(atoms=0)
except Exception as refused:
    print(type(refused).__name__, str(refused).splitlines()[-1])
```

```text
cvc5
TptpProverSpec(timeout=30.0, workers=1, batch_size=64, cache_size=None, memory_mb=None, cpu_seconds=None, seed=None, polymorphism='monomorphize', certificates=(), kind='tptp', engine='vampire', executable=None, dialect='thf', prefix=None, options=(), problem_file='problem.p')
ValidationError   <root>: a truth-table atom cap is positive [axiom_failed]
```

The SMT and TPTP families run external programs, so the engine must be
installed where the point trains; `experiments/l2/test_l2.py` skips its prover
test when Vampire is absent.

## Declare the scorer and its budget

`model/grounding/prover-enumerated.yaml` is the base every prover grounding
extends:

```yaml
kind: pairwise
premise: forms
hypothesis: forms
composition: {rule: marginal}
scorer:
  kind: prover
  budget: {calls_per_step: 512, wall_per_step: 30.0, cpu_seconds: null}
```

| Field | What it says |
| --- | --- |
| `premise`, `hypothesis` | The term readouts that supply each side's logical forms. Both must read complete terms under one estimate regime. |
| `composition` | How verdicts over decoded form pairs combine into one pair score: `marginal` takes the expectation over the decided pairs, `best` keeps the heaviest decided pair, and `tempered` takes the expectation at a declared temperature. |
| `scorer.budget` | `calls_per_step`, the most uncached prover calls in one step; `wall_per_step`, in seconds; `cpu_seconds`, cumulative for this use site. `null` is no limit. |
| `scorer.countermodels` | An optional crisp search over finite frames that refutes a pair before the prover is called. `method` is `random` or `exhaustive`, with `carriers`, `structures`, `seed` and `full_function_space_cap`. |

The scorer normalizes both forms, refutes any pair for which the countermodel
search finds a frame, and dispatches only the remaining pairs to the prover.
The prover scorer carries no learned state, so it is not checkpointed.

Two further pair scorers search for proofs themselves and verify each one
exactly. `model/grounding/guided-proof.yaml` (`kind: guided-proof`) learns a
ranking over declared signed-tableau steps with `steps`, `depth`, `top_k`,
`instantiations` and `hidden`; `model/grounding/neural-proof.yaml`
(`kind: neural-proof`) runs a differentiable backward chainer over declared
`templates` with `depth`, `width` and a `similarity` kernel. The kernel's
`kind` names an entry of the `lofi.similarity_kernel` group, `cosine` or `rbf`
when packaged, read at a positive `temperature`. Both scorers take a
`verification` budget of the same shape as `budget`.

## Declare the reward and the control variate

`reward` maps a decided label probability to a payoff:

| Field | What it says |
| --- | --- |
| `proved` | The payment at probability one. It must exceed `refuted`. |
| `refuted` | The payment at probability zero. |
| `contradiction` | `negated-hypothesis` proves contradiction as entailment of the negated hypothesis; `joint-inconsistency` as inconsistency of the two forms together. |
| `abstention` | `exclude`, the only value: an abstention pays nothing and is left out of the objective's mean. |

A decided probability *p* pays `refuted + (proved - refuted) * p`.

`control_variate: {kind: moving, decay, initial}` keeps an exponential moving
mean of decided rewards, which the reward objective subtracts from each
payment. `decay` lies in `[0, 1)`, and `initial` (default `0.0`) is the value
before the first observation. `moving` is the one control variate registered
under `lofi.control_variate`.

## Pair the loss term with the grounding

The `inference` objective reads the grounding:

| Field | What it says |
| --- | --- |
| `signal` | `likelihood` scores the observed label under the grounded distribution; `reward` trains on the declared payoff. |
| `response` | `{kind: label}` supervises categorical inference labels; `{kind: judgment, probabilities}` supervises an ordinal response instrument conditional on the labels. |
| `auxiliary_weight` | A nonnegative weight on the grounding's auxiliary term, added to the loss where the grounding publishes one (default `0.0`). |

[`experiments/l1/l1-run.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/l1/l1-run.yaml)
is the likelihood counterpart: it selects `prover-enumerated`, declares no
reward, and trains `{kind: inference, signal: likelihood, response: {kind: label}}`.

## How a mismatch is refused

The run checks the grounding against its readouts and its loss when the
document is composed. `l4-run.yaml` decodes its `forms` readout by sampling,
so it must keep its control variate; a reward signal needs a reward; and the
payoffs must be ordered:

```python
def refusal(overrides):
    try:
        codec(document, overrides=overrides)
    except Exception as refused:
        print(type(refused).__name__, "|", str(refused).splitlines()[-1])


refusal(("model.grounding.control_variate=null",))
refusal(("model.grounding.reward=null",))
refusal(("model.grounding.reward.proved=0.0",))
```

```text
ConfigDocumentError | experiments/l4/l4-run.yaml cannot be composed: pairwise grounding over sampled forms requires a control variate
ConfigDocumentError | experiments/l4/l4-run.yaml cannot be composed: model.grounding requires a declared reward spec
ConfigValueError | model.grounding.reward was refused: a proved payoff is greater than a refuted payoff, not 0.0 <= 0.0
```

The converse checks hold as well. Forms decoded exhaustively, as in
`l1-run.yaml`, refuse a control variate; a likelihood signal refuses a
declared reward; a grounding with no `inference` term is refused with
"model.grounding requires an inference objective"; and a grounding over a
dataset that yields no inference pairs is refused with "model.grounding
requires a selected inference-pair dataset".

[Inference as evidence](../../explanation/inference-as-evidence.md) explains
what a proof verdict licenses about a lexical logical form. The declaration
classes are under [Model declarations](../../reference/configuration/model.md),
the payoff and control variate under [Rewards](../../reference/rewards.md), and the
proof service under [Logical forms](../../reference/lf.md).
