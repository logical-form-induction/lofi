# Inference as evidence for logical forms

This page explains how entailment labels constrain the logical forms a learner
assigns to words. It describes how a label is decided from proof questions,
what a prover, a countermodel search and fuzzy model checking each contribute
to that decision, how a decision becomes a training signal, and what inference
labels cannot determine.

## Syntax, consequence, and models

Let $L$ be the declared logic, $\Gamma$ its background theory, and
$\varphi,\psi$ the closed premise and hypothesis formulas. A proof judgment
$\Gamma,\varphi\vdash_L\psi$ asserts derivability in $L$.
Semantic consequence relative to a class of structures $\mathcal{K}$ is

$$
\Gamma,\varphi\models_{\mathcal{K}}\psi
\quad\Longleftrightarrow\quad
\text{for every }\mathcal{M}\in\mathcal{K},\quad
\mathcal{M}\models\Gamma\cup\{\varphi\}
\ \Longrightarrow\ \mathcal{M}\models\psi.
$$

Soundness connects a calculus's derivations to its chosen semantics;
completeness is a separate property. A bounded prover run is not a complete
decision procedure merely because its underlying calculus is complete.
The logical-form signature and the prover's supported translation determine
the fragment being queried; unsupported binders or types are not silently
erased. For typed lambda syntax and typing judgments, see
[Categories, semantic types, and logical forms](logical-forms-as-types.md).

In many-sorted first-order semantics, each sort $s$ has a nonempty domain
$D_s$. A constant denotes a member of its domain, a function symbol denotes
a function of its declared argument sorts, and an $n$-place predicate denotes
a relation on the corresponding product of domains. With assignment $g$,
the classical quantifier clauses are

$$
\begin{aligned}
\mathcal{M},g\models\forall x{:}s\,\chi
&\iff \text{for all }a\in D_s,\ \mathcal{M},g[x\mapsto a]\models\chi,\\
\mathcal{M},g\models\exists x{:}s\,\chi
&\iff \text{for some }a\in D_s,\ \mathcal{M},g[x\mapsto a]\models\chi.
\end{aligned}
$$

Higher-order translation additionally requires the backend's semantics for
function types. Claims about a first-order fragment do not establish
completeness for arbitrary higher-order formulas. TPTP distinguishes these
logical input languages and their result statuses \citep{sutcliffe2017}.

## A label is a pair of proof questions

An inference pair has one premise and one hypothesis, each a string, and a
label from the declared labeling. The `three-way` labeling distinguishes
`entails`, `contradicts` and `neutral`; `rte` distinguishes `entails` from
`non-entails`; `graded` reads an entailment degree on the probability scale.

Under inference-only supervision, a learner does not see the logical forms
of the pair. It assigns forms to the two
strings through its grammar and lexicon, and the grounding asks two questions
of the declared logic: whether the premise's form proves the hypothesis's (the
positive question) and whether it proves the hypothesis's negation (the
negative question). The two verdicts determine a cell:

$$
q_+:\Gamma,\varphi\vdash_L\psi,\qquad
q_-:\Gamma,\varphi\vdash_L\neg\psi.
$$

| Premise proves hypothesis | Premise proves its negation | Cell |
| --- | --- | --- |
| yes | no | entails |
| no | yes | contradicts |
| no | no | neutral |
| yes | yes | degenerate |

Here “no” means a decided refutation, not failure to find a proof.

In classical logic, both questions can hold when $\Gamma\cup\{\varphi\}$
is inconsistent: explosion then derives every formula and its negation.
This is a statement about the premise together with the theory and the
declared logic, not a theorem about every possible consequence relation.
Which label
each cell supports is the logical-form language's relation-cell semantics, a
label map that the labeling then coarsens into its own vocabulary. The
classical map counts a degenerate cell toward both entailment and
contradiction, as classical consequence does; the strict map lets it support
no label. The map matters, because it decides whether a learner may satisfy an
entailment label by assigning the premise a contradictory form. A
`joint-inconsistency` contradiction adds a second negative question, whether
the hypothesis proves the premise's negation.

During training under the observed-label query, only the questions the gold
label needs are asked: the positive question unless the label is
`contradicts`, the negative one unless it is `entails`.

For graded scores $e$ and $c$ of the positive and negative questions, the
grounding constructs cell weights
$\mathbf{v}=(e(1-c),(1-e)c,(1-e)(1-c),ec)$ in the table's order.
This product construction is an implementation choice for combining degrees,
not a theorem that proof questions are independent random events. Each label
receives the sum of its compatible cell weights, then these sums are
normalized across labels for that form pair. With `joint-inconsistency`,
the two negative-question degrees are combined by their minimum. A question
omitted by the observed-label query is assigned degree $1/2$; a question
actually asked but undecided instead prevents that pair from contributing.

## Deciding a question

A verdict is `Proved`, `Refuted`, `Graded` with a degree, or `Abstain` with a
cause (`timeout`, `budget-spent`, `out-of-fragment`, `gave-up` or
`search-bound`). An abstention leaves both truth values open. Three kinds of
decision procedure are available.

**Provers.** The `truth-table` prover decides the quantifier-free,
variable-free fragment by enumerating truth assignments, up to a declared atom
cap, and returns a proof or the first falsifying assignment as a countermodel;
equality, quantifiers, abstractions and free variables put a formula outside
its fragment. The `tptp` prover translates to first-order or higher-order TPTP
and calls E, Vampire or Leo-III, mapping the SZS status to a verdict. The `smt`
prover calls Z3 or cvc5 incrementally. A proof service applies per-step call,
wall-clock and CPU budgets, and abstains when one is spent. Only decided
verdicts are memoized.

These backends implement different calculi and fragments
\citep{schulz2013,kovacs-voronkov2013,steen-benzmueller2018,demoura-bjorner2008,barbosa-etal2022}.
An external solver status is accepted through the declared adapter; a
`Proved` result does not itself assert that LoFI independently checked a
proof certificate.

**Countermodel search.** Before a prover is called, a bounded search over
complete finite structures with declared carrier sizes may find a structure in
which the theory and the premise hold and the hypothesis fails. Such a
countermodel is a certified refutation. The search is exhaustive only when its
`structures` bound reaches the number of structures over the declared carriers;
otherwise it is a partial search. In either case, failing to find a
countermodel proves nothing about structures outside those carriers. A found
countermodel satisfies
$\mathcal{M}\models\Gamma\cup\{\varphi\}$ and
$\mathcal{M}\not\models\psi$; this single structure refutes the corresponding
universal semantic-consequence claim.

For higher-order model evaluation, a full function domain is
$D_{\sigma\to\tau}=D_\tau^{D_\sigma}$. LoFI enumerates full function
spaces only within its cap; some sampled or given evaluation routes instead
use finite pools. A pooled evaluation is not quantification over the full
function space. The countermodel route rejects required non-full domains
before treating a result as a certified countermodel.

**Fuzzy model checking.** A model-checking scorer evaluates the pair in a set
of structures under a declared [fuzzy logic](../reference/configuration/model.md):
a t-norm (`product`, `goedel` or `lukasiewicz`) with its t-conorm, an
implication (`material` or the t-norm's `residuum`), a quantifier reading
(`extreme`, the minimum and maximum, or folded through the t-norm) and an
equality (`crisp` or with a margin). Its score is the t-norm of premise and
hypothesis summed over structures in which the premise holds above a floor,
divided by the premise's summed value, an estimate of conditional
satisfaction. Structures are given by a knowledge base (an intended model),
sampled, or learned. A model-checking score is not certified: it reports how
well the forms fit the structures, not whether the entailment holds in every
model.

### Truth degrees and fuzzy operations

For truth degrees $a,b\in[0,1]$, the packaged conjunctions and residua are
the standard product, Gödel, and Łukasiewicz operations \citep{hajek1998}:

| Family | $a\otimes b$ | $a\Rightarrow b$ |
| --- | --- | --- |
| Product | $ab$ | $1$ if $a\leq b$; otherwise $b/a$ |
| Gödel | $\min(a,b)$ | $1$ if $a\leq b$; otherwise $b$ |
| Łukasiewicz | $\max(0,a+b-1)$ | $\min(1,1-a+b)$ |

LoFI separately chooses standard negation $\neg a=1-a$, dual disjunction
$a\oplus b=1-[(1-a)\otimes(1-b)]$, and either the displayed residuum or
material implication $(1-a)\oplus b$. Standard negation need not equal
$a\Rightarrow0$, so these configurable combinations should not all be
identified with one axiomatized fuzzy logic.

For a nonempty finite domain $D$, `extreme` uses
$\|\forall x\,\chi\|=\min_{a\in D}\|\chi[a/x]\|$ and
$\|\exists x\,\chi\|=\max_{a\in D}\|\chi[a/x]\|$.
The folded option uses iterated $\otimes$ and $\oplus$ instead; with product
conjunction, the universal value is a product, not a minimum.
`crisp` equality is the indicator of equality of model values. `margin`
equality remains binary: for floats it is
$\mathbf{1}\{|a-b|\leq\varepsilon\}$, and for other values it uses crisp
equality. It is not a smooth similarity kernel.

For consulted structures $\mathcal{M}_1,\ldots,\mathcal{M}_m$, let
$a_j$ be the conjunction of the premise and all theory truth degrees,
$b_j=\|\psi\|_{\mathcal{M}_j}$, and
$J=\{j:a_j\geq\tau_{\mathrm{prem}}\}$. The implemented score is

$$
s(\varphi,\psi)=\frac{\sum_{j\in J}a_j\otimes b_j}{\sum_{j\in J}a_j}
\quad\text{when }\sum_{j\in J}a_j>0.
$$

With crisp truth this is the empirical fraction of premise-satisfying
structures that satisfy the hypothesis. With graded truth it is the stated
ratio of truth aggregates, not automatically a conditional probability under
a generative model of structures. A zero denominator is undecided or scored
zero according to `unsatisfied_premise`. A positive `abstain_band` excludes
scores within that distance of `truth_threshold`.

The two routes give evidence of different strength. A sound positive prover
verdict supports consequence in the backend's semantics. A model-checking score is a claim about the
structures consulted, and a learner can satisfy it with forms that happen to
agree on those structures.

## From a decision to a gradient

The grounding composes the verdicts into a distribution over labels for each
pair, and abstained mass goes to a residual rather than to a negative verdict.
The [`inference`](induction-objectives.md#inference) loss then scores that
distribution against the observed label, either as a likelihood or as an
expected payoff. For observed label $y$, reward is
$c_{\mathrm{refuted}}+(c_{\mathrm{proved}}-c_{\mathrm{refuted}})p(y)$.
Overlapping label support can prevent $p(y)$ from being one even when a
question is proved. Abstained responses are excluded from the mean.

For one alignment, write $a_{uv}=p_{\boldsymbol{\theta}}(u\mid w^{\mathrm{prem}})
p_{\boldsymbol{\theta}}(v\mid w^{\mathrm{hyp}})$ for a decoded form pair's
weight and $A_y$ for pairs whose evidence decides label $y$. If $c_y(u,v)$
is their normalized per-pair label-support score defined above, marginal composition computes

$$
\widetilde p(y)=\sum_{(u,v)\in A_y}
\frac{a_{uv}}{\sum_{(u',v')\in A_y}a_{u'v'}}c_y(u,v).
$$

The `tempered` option normalizes $a_{uv}^{1/\tau}$ for temperature $\tau>0$;
`best` selects a highest-weight decided pair. Multiple alignments are then
combined by the declared composition: marginal uses their arithmetic mean,
best their maximum, and tempered their power mean with exponent $1/\tau$.
The resulting categorical label scores are normalized over labels, including
under the observed-label query. This normalization matters when
the relation map allows overlapping label support. The observed-label
training query may decide only the requested label. These are conditional
computations over available decisions; the residual reports evidence not
scored rather than inventing a negative verdict for it.

Prover and countermodel verdicts are constants: no gradient flows through a
proof. For a fixed enumerated support, gradients differentiate the finite
weighted sum through decoder probabilities for marginal and tempered
composition. The `best` route instead retains a straight-through derivative
through the selected pair's log weight; this is not the ordinary derivative
of its hard selected forward score. None of these is a reparameterized
derivative through a discrete sampled formula. For a sampled form pair
$U\sim p_{\boldsymbol{\theta}}$ and credit $c(U)$, the score-function identity is
\citep{williams1992}

$$
\nabla_{\boldsymbol{\theta}}\mathbb{E}[c(U)]
=\mathbb{E}\left[(c(U)-b)\nabla_{\boldsymbol{\theta}}
\log p_{\boldsymbol{\theta}}(U)\right].
$$

The baseline $b$ must not depend on the current sampled outcome, conditional
on the sampling context, for its expected contribution to vanish. LoFI uses
the declared moving-average control variate for its nondifferentiable sampled
route. A differentiable credit adds
$\mathbb{E}[\nabla_{\boldsymbol{\theta}}c_{\boldsymbol{\theta}}(U)]$.
Truncation, decision masks, and renormalization change the objective whose
gradient is estimated; the identity alone does not remove that bias.

In particular, LoFI samples restricted choices, deduplicates completed
terms, and retains their decoder log weights. Its sampled surrogate uses
detached normalized composition shares $\pi_k$ over decided candidates,
not equal-weight independent Monte Carlo samples. With score
$\widehat s=\sum_k\pi_k c_k>0$, its log-score contribution is

$$
\sum_k\frac{\pi_k(c_k-b)}{\widehat s}
\nabla_{\boldsymbol{\theta}}\log a_k,
$$

plus the direct scorer contribution on the mixed route. This weighted,
deduplicated, truncated surrogate is not generally unbiased for the full
decoder expectation.

For the moving control variate, let $b_t$ be the current baseline, decay
$\delta\in[0,1)$, and $r_{t,1},\ldots,r_{t,n}$ the decided finite response
payments of the current batch. The update is

$$
b_{t+1}=\delta b_t+(1-\delta)\frac1n\sum_{i=1}^{n}r_{t,i}.
$$

An empty batch leaves the baseline unchanged. Grounding uses $b_t$ before
this update; payments are detached. For an observed-label affine payoff
$c_{\mathrm{refuted}}+(c_{\mathrm{proved}}-c_{\mathrm{refuted}})p$, the
probability-scale baseline is
$(b_t-c_{\mathrm{refuted}})/(c_{\mathrm{proved}}-c_{\mathrm{refuted}})$.
Payments are the response's probabilities or transformed credits, not
necessarily individual sampled zero-or-one proof outcomes. This conversion
distinguishes the payoff scale from the probability scale; it does not remove
the estimator limitations described above.

## What inference labels identify

Entailment labels constrain lexical forms only through the relations among the
sentences they are attached to. Two lexicons whose forms differ by a renaming
of non-logical constants that preserves every labeled relation are
indistinguishable to an inference objective. The L1 experiment thus
reports, beside exact normal-form match, the random-relabeling floor, recovery
relative to the logical vocabulary's fiber, and the fiber's size, so that a
failure to recover a meaning can be told apart from a relabeling the labels
cannot resolve.

The labels also inherit the limits of the logic. A pair the prover abstains on
contributes nothing; a fragment the declared prover cannot decide is invisible
to the objective; and a degenerate cell under the classical map rewards an
inconsistent premise. What an inference-trained lexicon shows is that its
forms reproduce the labeled relations under the declared logic and label map.
It does not show that the forms are the intended meanings. To set up a
grounding, see [Reward with a theorem
prover](../how-to/training/reward-with-a-theorem-prover.md) and [Reward with
model checking](../how-to/training/reward-with-model-checking.md).

Formally, if a permutation $\pi$ of nonlogical vocabulary preserves every
queried relation, then a lexicon $\Lambda$ and its renamed version
$\pi\Lambda$ have the same inference observations. Recovery can then be
identified only up to the equivalence class
$[\Lambda]=\{\Lambda':\Lambda'\text{ agrees on all observed relations}\}$.
Exact recovery of an intended representative requires additional evidence.

## References

\bibliography
