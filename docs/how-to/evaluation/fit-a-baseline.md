# Fit an external baseline

A baseline fit document declares one external comparison system under
`system`, whose `kind` names a plugin in the `lofi.baseline_system`
[entry point group](../../reference/entry-points.md#registered-groups-and-names). An experiment over a baseline fit document fits the system
once per point and records its measurements as report rows, in the same
results layout as the rows of LoFI points. This guide declares the two packaged systems, Bisk and
Hockenmaier's CCG induction and the LM-plus-prover pipelines, runs them, and
reads what they record.
## Declare a fit

A [baseline](../../reference/run.md) is a specified system against which a claim about LoFI can be tested.
Interpret the score difference in light of each system's data split, grammar
assumptions, training budget, and scoring conventions. The common row format
lets an experiment collect these scores for comparison.

[`experiments/s1/bisk-hockenmaier-fit.json`](https://github.com/logical-form-induction/lofi/blob/main/experiments/s1/bisk-hockenmaier-fit.json)
is the fit the CCGbank campaign runs:

```json
{
  "system": {
    "kind": "bisk-hockenmaier",
    "ccgbank_source": "data/bisk-hockenmaier/ccgbank",
    "dependency_gold_source": "data/bisk-hockenmaier/evaluation.conll",
    "reference_source": "data/bisk-hockenmaier/reference.json",
    "reference_tolerance": 0.02,
    "nominal_category": "N",
    "sentence_category": "S",
    "coordination_category": "conj",
    "gold_sentence_atom": "S",
    "gold_coordination_atom": "conj",
    "nominal_tags": ["NN", "NNS", "NNP", "NNPS"],
    "verbal_tags": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
    "coordination_tags": ["CC"],
    "other_tags": ["JJ", "JJR", "JJS", "RB", ".", ",", ":"],
    "punctuation_tags": [".", ",", ":"],
    "headed_or_modifier_arguments": ["N", "S"],
    "sentence_arguments": ["N"],
    "em_max_iterations": 50,
    "convergence_tolerance": 1e-6
  }
}
```

A fit document holds `system` and nothing else.
[`BaselineFitCodec`](../../reference/run.md#lofi.run.BaselineFitCodec) composes it as [`RunSpecCodec`](../../reference/spec.md#lofi.spec.RunSpecCodec)
composes a run document, resolving `defaults` and applying overrides, and
returns the validated [`BaselineFitSpec`](../../reference/run.md#lofi.run.BaselineFitSpec):

```python
from pathlib import Path

from lofi.run import BaselineFitCodec

codec = BaselineFitCodec()
for document in (
    "experiments/s1/bisk-hockenmaier-fit.json",
    "experiments/l1/lm-prover-fit.json",
):
    fit = codec(Path(document))
    print(fit.system.kind, fit.system.metrics())
```

```text
bisk-hockenmaier ('supertagging-length-10', 'dependencies-length-10', 'supertagging-length-20', 'dependencies-length-20', 'supertagging-all-lengths', 'dependencies-all-lengths')
lm-prover ('logical-form-normal-form', 'many-to-one-accuracy', 'random-relabeling-floor', 'inference-accuracy', 'inference-coverage', 'prover-calls', 'prover-memo-hit-rate', 'prover-cpu-seconds', 'prover-abstention-rate')
```

`metrics()` returns the rows a fit promises, in row order; an experiment's
`rows` name them under the evaluator role `baseline`. A treatment the system
does not admit is refused at composition. Moving the evaluation section into
the training sections:

```python
try:
    codec(
        Path("experiments/s1/bisk-hockenmaier-fit.json"),
        overrides=("system.evaluation_section=2",),
    )
except Exception as refused:
    print(type(refused).__name__, "|", str(refused)[:200])
```

```text
ConfigValueError | system was refused: baseline held-out sections occur in training: [2]
```

A fit is refused when its system declares no metric, an unnamed metric or a
repeated one.

## The Bisk-Hockenmaier system

`kind: bisk-hockenmaier` reruns the CCG induction system of \citet{bisk-hockenmaier2012} on CCGbank, since no prediction file was released for it,
and scores supertagging and unlabeled directed attachment on the evaluation
section. Its declaration states every treatment the published description
fixes:

| Fields | What they declare |
| --- | --- |
| `ccgbank_source`, `dependency_gold_source` | The CCGbank AUTO files and the CoNLL-2008 dependency gold, each resolved against the experiment document's directory. |
| `reference_source`, `reference_tolerance` | The published Figure 4(a) scores, and how far a rerun score may lie from each. |
| `training_sections`, `training_max_length`, `development_section`, `evaluation_section` | The sections read and the length bound on training sentences; defaults 2 to 21, 10, 0 and 23. |
| `evaluation_length_bands`, `length_excludes_punctuation` | The length bands scored, increasing and ending in one unbounded band (default `[10, 20, null]`), and whether length counts punctuation. |
| `grammar` | The grammar the system induces a [lexicon](../grammars/declare-a-lexicon.md#list-entries-inline) over, a complete [grammar declaration](../grammars/choose-a-formalism.md); by default the packaged [`lofi/configs/system/grammar/bisk-hockenmaier.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/system/grammar/bisk-hockenmaier.yaml), described below. |
| `language` | The registered or inline language the grammar is compiled within (default `propositional`); the baseline reads no meanings, but the language bounds the category store. |
| `nominal_category`, `sentence_category`, `coordination_category` | The grammar primitives that play the nominal, sentence and coordination roles. With the goal category, they must be exactly the grammar's primitives, and the coordination category must be a conjunction marker. |
| `gold_sentence_atom`, `gold_coordination_atom` | The two distinct gold corpus atoms (`S` and `conj` in CCGbank) whose feature-erased occurrences score as the sentence and coordination categories; every other gold atom scores as the nominal category. |
| `nominal_tags`, `verbal_tags`, `coordination_tags`, `other_tags`, `punctuation_tags`, `gerund_tag` | The part of speech classes that seed the lexicon. |
| `headed_or_modifier_arguments`, `sentence_arguments`, `max_arity`, `second_order_modifiers`, `category_induction_steps` | How categories are induced from the seed lexicon. |
| `em_max_iterations`, `convergence_tolerance`, `probability_floor` | The EM horizon, its convergence test and the probability floor (default $\exp(-6)$). |
| `parse`, `threads` | The chart parse each sentence's packed forest is built with (default `{retain: forest, reachability: roots}`), and how many chart cells of one width it may fill concurrently (default 1, positive). The thread count does not change a score, so it is not one of the system's conventions. |

The remaining fields, such as `k: 100` and
`dependency_scoring: conll-2008-unlabeled-directed-attachment`, admit only the
published value and default to it, so a declaration states them only to make
them explicit.

The declaration is checked as a whole: the part of speech classes must be
disjoint and contain the punctuation tags and the gerund tag; the development
and evaluation sections must differ and lie outside training; the grammar's
goal must be one root category; every [type raising](../../explanation/derivations-as-proofs.md#categories-productions-and-lexical-axioms) rule must declare finite
targets and arguments; and the grammar budget's `threads` must stay
1, since the parse takes its thread count from the system's own `threads`.
### Factorization and estimation

The [factorization](#factorization-and-estimation) assigns a local weight to the choices at each derivation
node, such as its category, head, and direction. A complete derivation combines
these local weights. When derivations are unobserved, estimation accounts for alternative
analyses of each sentence. The displayed factors assign weights to these choices at one node.

The implementation associates each derivation node with categorical events.
For parent category $A$, head category $H$, sister category $S$, direction
$d\in\{\mathrm{left},\mathrm{right}\}$, and part of speech tag $t$, its local
weight is

$$
w_{\boldsymbol{\theta}}(v)=
\begin{cases}
p_{\boldsymbol{\theta}}(\mathrm{lexical}\mid A)
 p_{\boldsymbol{\theta}}(t\mid A,\mathrm{lexical}),&v\text{ lexical},\\
p_{\boldsymbol{\theta}}(\mathrm{unary}\mid A)
 p_{\boldsymbol{\theta}}(H\mid A,\mathrm{unary}),&v\text{ unary},\\
p_{\boldsymbol{\theta}}(d\mid A)
 p_{\boldsymbol{\theta}}(H\mid A,d)
 p_{\boldsymbol{\theta}}(S\mid A,H,d),&v\text{ binary}.
\end{cases}
$$

The lexical outcome is the observed word's POS tag. The factors belong
to separate categorical families, each normalized within its conditioning
context. A derivation $T$ has weight
$w_{\boldsymbol{\theta}}(T)=\prod_{v\in T}w_{\boldsymbol{\theta}}(v)$.
For sentence $x$, let $\mathcal{F}_x$ be the set of complete derivations
represented by its retained packed forest and
$\mathcal{K}_x\subseteq\mathcal{F}_x$ its current top-$k$ derivations. The
conditional PMF on either nonempty set $\mathcal{A}$ is

$$
q_{\boldsymbol{\theta}}(T\mid x,\mathcal{A})=
\frac{w_{\boldsymbol{\theta}}(T)}
{\sum_{T'\in\mathcal{A}}w_{\boldsymbol{\theta}}(T')}.
$$

Write $n_e(T)$ for the number of occurrences of event $e$ in $T$. The
expectation step computes both restricted and full forest expected counts:

$$
a_e=\sum_x\mathbb{E}_{T\sim q_{\boldsymbol{\theta}}(\cdot\mid x,\mathcal{K}_x)}[n_e(T)],
\qquad
b_e=\sum_x\mathbb{E}_{T\sim q_{\boldsymbol{\theta}}(\cdot\mid x,\mathcal{F}_x)}[n_e(T)].
$$

Training sentences with no complete derivation under the induced [lexicon](../grammars/declare-a-lexicon.md#list-entries-inline)
are excluded from these sums; if none remain, fitting is refused. Event
support is fixed from the complete training forests. Full forest counts
come from inside–outside rule marginals. For context $c$,
let $E_c$ be its event support, $a_c=\sum_{e\in E_c}a_e$, and
$b_c=\sum_{e\in E_c}b_e$. The update combines these estimates as

$$
\lambda_c=\frac{a_c}{a_c+b_c},\qquad
m_e=\lambda_c\frac{a_e}{a_c}+(1-\lambda_c)\frac{b_e}{b_c}.
$$

A component with zero total contributes zero, and $\lambda_c=0$ when both
totals vanish. For a supported context with $|E_c|\varepsilon\leq1$, the
probability floor update is the constrained categorical maximization

$$
\underset{\{p_e:e\in E_c\}}{\operatorname{maximize}}\quad
\sum_{e\in E_c}m_e\log p_e
\quad\text{subject to}\quad
\sum_{e\in E_c}p_e=1,\qquad p_e\geq\varepsilon.
$$

The implementation fixes outcomes below the floor at $\varepsilon$ and
redistributes the remaining mass proportionally among the others. A context
with no expected counts receives a uniform row. An infeasible floor is
refused. Initialization is uniform over each context's supported outcomes.
Iteration stops when

$$
\max_e\left|p_e^{(r+1)}-p_e^{(r)}\right|\leq\tau
$$

or `em_max_iterations` is reached; $\tau$ is `convergence_tolerance`.
The criterion checks parameter convergence under the implemented full-forest/top-$k$
mixture and floor. Global optimality and monotonicity of an unrestricted
observed data likelihood require separate justification. The published induction system
provides the methodological reference \citep{bisk-hockenmaier2012}; these
equations specify the procedure implemented here.

### The grammar

The packaged grammar declares the published system's choices:

| Part | Packaged value |
| --- | --- |
| `signature` | Primitives `N`, `S`, `conj` and `TOP`; the slash constructors `/` and `\` with their [functor views](../grammars/declare-a-signature.md#declare-functor-views); a unary marking constructor `@conj`; and a conjunction inventory with marker `conj`. |
| `rules.combinators` | Forward and backward application, harmonic and crossed composition of order 1 under Steedman's direction restriction, and [type raising](../../explanation/derivations-as-proofs.md#categories-productions-and-lexical-axioms) of the argument atoms, $X\Rightarrow T/(T\backslash X)$ and $X\Rightarrow T\backslash(T/X)$ for $X,T\in\{\mathrm{N},\mathrm{S}\}$, declared by `raising_arguments` and `raising_targets`. |
| `rules.rules` | The binarized coordination rule as the two phases $\mathrm{conj}\;X\Rightarrow C(X)$ and $X\;C(X)\Rightarrow X$, where $C(X)$ is the intermediate category written `@conj(X)` in the declaration, and the unary rules $\mathrm{N}\Rightarrow\mathrm{TOP}$ and $\mathrm{S}\Rightarrow\mathrm{TOP}$. |
| `goal`, `matching` | The root `TOP`, matched by identity. |
| `normal_form` | `hockenmaier-bisk` with `order_bound: 1`, `raised_application: true` and `coordination_raising: true`. |
| `budget` | One unary step per cell and categories of at most five arguments. |
| `lexicon` | A three-entry witness the compile checks the goal against; the system induces its own [lexicon](../grammars/declare-a-lexicon.md#list-entries-inline) from part of speech tags. |

The category set the support parse explores stays finite only if type raising
is bounded. Every type raising declaration must state finite targets and
finite arguments, and a grammar that raises to or from every category is
refused. Finite targets alone bound only the result `T`: a declaration that
raises every category `X` still raises composed categories, composes the
results, and raises those again. The packaged grammar thus raises only
the argument atoms, which a fit document may change through
`raising_arguments`. Coordination covers every category: the
packaged grammar's conjunction inventory declares `marking: '@[conj]'`, the
unary constructor whose application to any category, raised, composed or
`TOP` included, is its marked form. A grammar that leaves `marking` out
coordinates only its declared pairs. A sentence holding a verb other than the gerund tag derives
`TOP` only from the sentence category; any other sentence derives it by every
goal rule the grammar declares.

A fit document changes any part of the grammar by selecting the packaged
grammar as a configuration group and overriding a leaf, as
[`experiments/s1/bisk-hockenmaier-reduced-fit.json`](https://github.com/logical-form-induction/lofi/blob/main/experiments/s1/bisk-hockenmaier-reduced-fit.json)
does to bound categories at three arguments (an excerpt):

```json
{
  "defaults": [{"system.grammar": "bisk-hockenmaier"}],
  "system": {
    "kind": "bisk-hockenmaier",
    "max_arity": 1,
    "grammar": {"budget": {"max_category_arity": 3}}
  }
}
```

The same selection reaches the grammar from the command line, as the overrides
`system/grammar=bisk-hockenmaier` and, say,
`system.grammar.normal_form.raised_application=false`. The system's conventions
record the digests of the grammar and language declarations, so a changed
grammar changes the convention hash.

The system reports two metrics per length band, supertagging accuracy and
directed attachment accuracy over nonpunctuation tokens, in band order, as
`metrics()` printed above.

Before it returns, the fit compares each dependency score with the published
figure in the artifact `reference_source` names, whose format is that of the
packaged
[`lofi/configs/eval/bisk_hockenmaier_2012.json`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/eval/bisk_hockenmaier_2012.json).
A score further from its published value than `reference_tolerance` fails the
fit, naming every score outside the gate, so a rerun that does not reproduce
the published system reports no rows. The reference artifact is itself validated: it must cite Figure 4(a), the
WSJ10 training band and the declared evaluation section, and list exactly the
three dependency metrics.

## The LM-plus-prover system

This baseline starts from recorded model generations. LoFI checks whether each
generated expression [parses](../../explanation/logical-forms-as-types.md#semantic-typing), has an admissible type, and supports the requested
inference under the prover. Each evaluation reuses the recorded generations and computes scores from
these checks, retaining invalid expressions and undecided proofs as outcomes.

`kind: lm-prover` replays recorded language model formalizations through
LoFI's own logical form parser, type checker and prover, which compute every
reported measurement from the stored generations.

| Field | What it declares |
| --- | --- |
| `data_source`, `generations_source`, `prompt_source` | The evaluation items, the recorded generations and the prompt they were generated from. |
| `model` | The identity of the model that generated them. |
| `pipeline` | `{kind: linc, samples: n}` samples `n` independent formalizations and votes on the prover's verdicts; `{kind: logic-lm, refinements: n}` takes one formalization and up to `n` solver feedback refinements. |
| `language` | The [logical form language](../grammars/declare-a-logical-form-language.md#use-a-registered-language) the generations are parsed and type-checked in, with its prover. |
| `budget`, `normalization`, `normalization_fuel` | The proof budget, the normalization applied before comparison, and its positive reduction budget. |
| `measure`, `alignment`, `scoring` | The logical form measure, the alignment between predicted and gold forms, and the eight scoring conventions of [Evaluate a trained point](evaluate-a-run.md). |

The alignment must fit on `dev` and score on `test`. The system reports
`logical-form-<measure>` and `<alignment>-accuracy`, named by the declared
measure and alignment kind, followed by the seven fixed metrics `metrics()`
printed above.

## Run the fit

A baseline fit runs as the points of an experiment whose `config` names the
fit document with `kind: baseline-fit`.
[`experiments/s1/bisk-hockenmaier.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/s1/bisk-hockenmaier.yaml)
fits the system under four seeds and names its rows under the evaluator role
`baseline`:

```yaml
role: known-grammar-baseline
freeze_id: freeze/1
config: {kind: baseline-fit, path: bisk-hockenmaier-fit.json}
launch: {kind: serial}
sweep:
  grid: []
  seeds: [0, 1, 2, 3]
  expected_points: 4
seed_policy: {root: 114}
rows:
  - {evaluator: baseline, metric: supertagging-length-10}
  - {evaluator: baseline, metric: dependencies-length-10}
  # ... one row per declared metric
primary: {evaluator: baseline, metric: dependencies-all-lengths}
```

The rest of the document, its [preregistration](../../explanation/experiments-and-commitments.md#preregistration), [contraction](../../explanation/experiments-and-commitments.md#contraction), stop rules,
decision point and evaluation plan, is declared as for any experiment; see
[Run and price a sweep](../experiments/sweeps.md). The fit's source paths are
resolved against the experiment document's directory. The production document
expects licensed CCGbank AUTO files and the dependency reference under
`experiments/s1/data/bisk-hockenmaier/`;
[`experiments/s1/bisk-hockenmaier-reduced.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/s1/bisk-hockenmaier-reduced.yaml) runs the same fit on a
one-sentence repository fixture. The LM-plus-prover experiments are
[`experiments/l1/lm-prover.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/l1/lm-prover.yaml) (LINC; \citealt{olausson-etal2023}) and
[`experiments/l1/logic-lm.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/l1/logic-lm.yaml) (Logic-LM; \citealt{pan-etal2023}), each with a
reduced twin. These declarations instantiate the method families; reproducing
the papers’ results also requires their data and experimental conditions.

Launch the experiment from the checkout its freeze tags, like any other. The
following is a template command:

```text
uv run lofi launch \
  --experiment experiments/s1/bisk-hockenmaier-reduced.yaml \
  --freeze freezes/1/FREEZE.json --out launch.json
```

A baseline fit point is fitted in the launching process, by the serial
executor. A point whose completed result is cached is served from its
directory, and an infrastructure failure is retried within the attempts the
point is given, as for run points.

## Read what a fit records

A completed fit writes two files in its point directory. `rows.json` holds one
row per metric, and the fit is refused if the system emits metrics other than
the ones it declares, or in another order. Each row carries:

| Columns | What they hold |
| --- | --- |
| `freeze_id`, `config_hash`, `point_id` | The freeze, the digest of the resolved fit document, and the point. |
| `task` | Always `baseline`, the role under which an experiment names baseline rows. |
| `family`, `system`, `metric`, `arm` | The comparison family, the system, the metric and the arm, such as `re-running`. |
| `cites`, `reason` | The citation the figure comes from and why the system was rerun or replayed. |
| `value`, `n`, `unit` | The measurement, the units it counts and their kind, such as `accuracy`. |
| `seed`, `convention_hash`, `conventions` | The run seed, and the system's declared conventions with the digest of every source file, and their hash. |

The approximation columns state no enumeration budget. `baseline-fit.json`
records the validated fit, the SHA-256 digest of each source it read, the seed
derivation, the installed package versions and the source revision.

## Compare with LoFI points

A baseline experiment runs under a freeze and writes its results directory in
the layout [Run and price a sweep](../experiments/sweeps.md) describes, so its
rows are ingested against the freeze as a run experiment's are. The L1
campaign runs its neural experiment and its two LM-plus-prover experiments
through one freeze and ingests the three result roots, so the [logical form](../../explanation/logical-forms-as-types.md#semantic-typing) and
inference measurements of LINC and Logic-LM sit beside those of the trained
points. [Reduce and report results](../experiments/reduce-and-report-results.md)
declares the tables an experiment renders from its rows.

## References

\bibliography
