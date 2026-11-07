# Induction objectives

An induction objective assigns a scalar cost to the outputs a model publishes.
This page defines the packaged losses, their averaging units and gradients,
and the assumptions needed to interpret them as evidence about a grammar.
It covers label likelihoods, compositional consistency, reconstruction,
simplicity, supervision, regularization, and combinations of these terms.

## What training is asked to improve

A [loss](../how-to/training/combine-losses.md) is a numerical cost that training tries to reduce. An induction
objective specifies which evidence should make one proposed analysis less
costly than another. Different objectives can favor different analyses of
the same sentences, so choosing the objective is part of the scientific model.

For instance, [reconstruction](#reconstruction) rewards representations that
help predict the input words, while [inference](inference-as-evidence.md)
rewards meanings that support observed relations between sentences. A model
can preserve enough information to predict words without assigning the
intended meanings. The choice of evidence determines what the objective can
distinguish.

A [decoder](../how-to/models/configure-term-encoders-and-decoders.md) turns a learned numerical representation into choices over
terms; a [readout](../how-to/models/configure-readouts.md) extracts a specified quantity, such as target probability
mass, from the model. An objective can use only the quantities the model
exposes. A [mask](../reference/nn.md#ports) indicates which positions are real observations rather
than padding. A gradient describes the cost's local sensitivity to parameter
changes; the [optimizer](../how-to/training/configure-a-trainer.md) uses it
to choose an update. The corresponding implementations are documented under
[losses](../reference/nn.md#losses) and [readouts](../reference/nn.md#readouts).

## Objectives read published outputs

Write $\boldsymbol{\theta}$ for the trainable parameters, $w_i$ for input
item $i$, and $\mathbf{h}_{i,s}$ for its representation at span $s$.
A decoder supplies complete term weights
$w_{\boldsymbol{\theta}}(t\mid\mathbf{h}_{i,s})$, obtained from locally
normalized choices. Admitted terms can have total weight below one when
decoding encounters dead ends or rejects completed terms. Below, $p$ and $q$
denote the normalized distributions supplied to a comparison; target mass
[readouts](../how-to/models/configure-readouts.md#choose-a-route) retain their declared scoring convention. A root target mass is
computed from neural outputs; the symbolic parser has its own partition function.

Each objective declares its required outputs: cells, marginals, readouts,
labels, or constraints. A missing output is an error. In the equations below,
index sets contain only the observations the implementation scores, after
its masks and selection rules. An empty set makes the term inapplicable;
the epoch mean excludes that batch for the term. Logs are natural
logs unless a symbol cost declaration explicitly uses bits.

## Grammaticality

For a positive observation and no label noise, the cost is $-\log m$.
Assigning probability $0.9$ to the observed outcome costs about $0.105$;
assigning $0.1$ costs about $2.303$. These chosen numbers illustrate why
training favors more probability on the observed label. If every observation
is positive, however, assigning probability 1 to every string also reduces
this cost. The objective needs other evidence to distinguish acceptance from
indiscriminate acceptance.

Let $I$ contain items with observed binary labels $y_i\in\{0,1\}$ and
let $m_i\in[0,1]$ be the root target mass. Under symmetric label flip
probability $\eta\in[0,1/2)$,

$$
\widetilde m_i=(1-\eta)m_i+\eta(1-m_i),\qquad
\ell_{\mathrm{gram}}=-\frac1{|I|}\sum_{i\in I}
\left[y_i\log\widetilde m_i+(1-y_i)\log(1-\widetilde m_i)\right].
$$

The implementation selects the log probability of the observed outcome,
avoiding undefined products such as $0\log0$. With `noise: 0`, a zero
probability for an observed label gives infinite cost. At $\eta=1/2$ the
label is uninformative; above $1/2$ it is anticorrelated. LoFI restricts the declared rate to below $1/2$.

Positive-only labels can minimize this term by assigning mass one to every
string. They thus require additional constraints if the intended question is
whether the learner distinguishes grammatical from ungrammatical strings.
A declared [judgment response](judgment-models.md#linking-families-state-how-a-prediction-enters) replaces the binary observation model with its
response likelihood; see [Judgment models](judgment-models.md).

## Coherence

Consider a span such as *sees Mary*. One route decodes a distribution of
terms from the representation of that whole span. Another combines the
representations of *sees* and *Mary* under a candidate action, then decodes
the result. Coherence compares those distributions. It asks whether two
neural routes agree; the formulas retain the distinction between this
comparison and applying a symbolic rule to already decoded terms.

Binary coherence compares the parent decoder PMF $p_{i,s}$ with a candidate
PMF $q_{i,s,k,a}$ obtained by (i) merging the vectors of the daughters at
split $k$ and (ii) applying the decoder for action $a$. It does not first
decode two symbolic terms and then apply a symbolic grammar rule to them.

Let $\mu_{i,s}$ be the [constituent marginal](charts-and-approximation.md#inside-outside-and-marginals), $\pi_{i,s,k}$ the conditional
split probability, and $\rho_{i,s,k,a}$ the softmax probability of action
$a$. At step $j$, a warmup of $j_0>0$ steps gives
$\tau_j=\min(1,j/j_0)$ and $v_{i,s}=1-\tau_j+\tau_j\mu_{i,s}$.
Without warmup, $\tau_j=1$. With optional item weight $b_i$ (one by default),
the binary contribution is

$$
\ell_{\mathrm{coh,bin}}=\frac1{|I_{\mathrm{bin}}|}
\sum_{i,s,k,a} b_i v_{i,s}\pi_{i,s,k}\rho_{i,s,k,a}
\operatorname{d}(p_{i,s},q_{i,s,k,a}).
$$

The sum includes valid splits and actions with positive effective weight;
$I_{\mathrm{bin}}$ contains the distinct participating items, so the
denominator counts items while the numerator sums their weighted span contributions. Gradients flow through
the participating distributions and weights. Unary coherence is computed
separately and added, with its own participating-item denominator. For exhausted frontiers, let $e_{\boldsymbol{\theta}}$ be the [term encoder](../how-to/models/configure-term-encoders-and-decoders.md#choose-an-encoder).
For unary action $a$, let $A_a$ be its domain among decoded premise terms and
$R_a(t)$ its finite nonempty set of conclusions. The reference is the
premise PMF conditioned on $A_a$, denoted $p_a$. The candidate is the mixture

$$
q_a(u)=\sum_{t\in A_a}p_a(t)\frac1{|R_a(t)|}
\sum_{v\in R_a(t)}p_{\boldsymbol{\theta}}(u\mid e_{\boldsymbol{\theta}}(v)).
$$

The unary contribution averages $b_i v_{i,s}\rho_{i,s,a}
\operatorname{d}(p_a,q_a)$ over participating items, summing cells and actions;
the action softmax includes only actions with a nonempty decoded domain and
finite logits. Truncated execution retains subprobability weights and residuals, then
conditions each comparison on the shared enumerated support.
Declared
cross-space consistency relations add their own comparison terms.

For reference PMF $p$ and candidate PMF $q$ on a common support, the options are

$$
\begin{aligned}
\operatorname{KL}(p\|q)&=\sum_t p(t)\log\frac{p(t)}{q(t)},\\
\operatorname{CE}(p,q)&=-\sum_t p(t)\log q(t),\\
\operatorname{d}_{\alpha}(p,q)&=
\frac{\sum_t[p(t)^\alpha+(\alpha-1)q(t)^\alpha
-\alpha p(t)q(t)^{\alpha-1}]}{\alpha(\alpha-1)},\quad\alpha>1.
\end{aligned}
$$

KL is the directed information divergence \citep{kullback-leibler1951}.
Zero reference mass contributes zero; positive reference mass with zero
candidate mass gives infinite KL or cross-entropy. The `fenchel-young` option
uses the displayed Bregman divergence generated by negative Tsallis entropy,
with KL at $\alpha=1$. It corresponds to a Fenchel–Young loss evaluated at
scores generated from the candidate distribution \citep{blondel-etal2020}. This distinction matters when
relating the implementation to $\alpha$-entmax \citep{peters-etal2019}.

The exact estimator requires exhausted decoding frontiers and [residual mass](charts-and-approximation.md#the-residual-mass-budget)
at most [`exact_residual`](../how-to/models/configure-readouts.md#bound-an-exhaustive-decoding). Exhaustion and remaining mass are checked separately, with the latter compared against the declared tolerance. The truncated
estimator compares PMFs conditioned on their enumerated common support and
checks omitted probability mass; bounding divergence error requires additional
conditions, as described in
[Charts and approximation](charts-and-approximation.md#the-residual-mass-budget).
Coherence can be small for an internally consistent but linguistically
incorrect assignment of terms. Identifying the grammar requires evidence
that distinguishes such internally consistent assignments.

## Reconstruction

Reconstruction asks a representation to retain enough information to predict
the words it represents. The averaging unit matters. For an illustrative
pair of spans with sequence lengths 2 and 4 (including their end tokens),
suppose their total costs are 2 and 12. Equal span weighting averages their
per-token costs, giving $(2/2+12/4)/2=2$. Equal token weighting instead gives
$(2+12)/(2+4)=7/3$. Neither calculation is interchangeable with the other:
the longer span receives more influence in the second.

For a selected span $s$, let $x_{s,1},\ldots,x_{s,n_s}$ be its words followed
by the end token, so $n_s$ includes that token. Teacher-forced reconstruction
has row cost

$$
c_s=-\sum_{j=1}^{n_s}\log p_{\boldsymbol{\theta}}
(x_{s,j}\mid x_{s,<j},\mathbf{h}_s).
$$

With $b_s=1$ for uniform weighting or
$b_s=\operatorname{stopgrad}(\mu_s)$ for marginal weighting,
[`span-reconstruction`](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/losses/span_reconstruction.py) uses

$$
\ell_{\mathrm{span}}=
\begin{cases}
\displaystyle\frac{\sum_s b_s c_s/n_s}{\sum_s b_s},
&\text{span normalization},\\[6pt]
\displaystyle\frac{\sum_s b_s c_s}{\sum_s b_s n_s},
&\text{token normalization}.
\end{cases}
$$

Only live spans within the declared length bound contribute.
$\operatorname{stopgrad}(x)$ has value $x$ and zero derivative through its
argument, keeping the marginal weights fixed along this gradient path. A zero total weight makes the objective inapplicable.

A detached marginal sets the contribution of its span at the current value.
During backpropagation, that weighting path holds the marginal fixed.

[`outside-reconstruction`](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/losses/outside_reconstruction.py) uses the outside vector $\mathbf{o}_{i,j}$ of
leaf $j$ to score the two-token sequence $(w_{i,j},\mathrm{EOS})$:

$$
\ell_{\mathrm{out}}=-\frac1{2|J|}\sum_{(i,j)\in J}
\left[\log p_{\boldsymbol{\theta}}(w_{i,j}\mid\mathbf{o}_{i,j})
+\log p_{\boldsymbol{\theta}}(\mathrm{EOS}\mid w_{i,j},\mathbf{o}_{i,j})\right].
$$

Here $J$ contains live, unpadded leaves. Predicting a word from its outside
representation follows DIORA \citep{drozdov-etal2019}; this particular
likelihood of the word and end token is LoFI's decoder objective. Reconstruction
rewards representations useful for word prediction. Whether the resulting
bracketing recovers the source grammar remains an empirical question.

## Inference

Let $I$ be the observed responses retained by the grounding and response
model. For a categorical label $y_i$, the likelihood cost is
$-\log p_{\boldsymbol{\theta}}(y_i\mid w_i^{\mathrm{prem}},w_i^{\mathrm{hyp}})$.
For payoff $u_i(y)=c_{\mathrm{refuted}}+
(c_{\mathrm{proved}}-c_{\mathrm{refuted}})\mathbf{1}\{y=y_i\}$, the reward cost is
$-\sum_y p_{\boldsymbol{\theta}}(y\mid w_i^{\mathrm{prem}},w_i^{\mathrm{hyp}})u_i(y)$.
This is affine in the probability of the observed label, with the two credits
set by `refuted` and `proved`. The objective averages these costs over $I$.
A judgment instrument marginalizes over latent labels to obtain the observed
response probability $r_i$: its likelihood cost is $-\log r_i$ and its
reward cost is $-r_i$.

Abstention is recorded separately from a negative verdict. Configured
auxiliary penalties add their weighted cost, and configured form credits
subtract their expected credit. These additions need not be likelihood terms.
[Inference as evidence for logical forms](inference-as-evidence.md) defines
the grounding, its gradients, and its identifiability limits.

## Simplicity

Suppose an additional constructor has declared cost 2. Including it adds 2
to a term's structural cost; assigning probability $1/4$ to that choice adds
$1/2$ to its expected contribution. This is illustrative arithmetic under
the declared cost map. A probabilistic or code-length interpretation requires
a distribution or coding scheme that justifies the assigned costs.

For scored cells $S$, slot set $J$, constructor alphabet $C$, slot PMF
$p_{s,j}(c)$, and declared nonnegative cost $\kappa(c)$,

$$
\ell_{\mathrm{simp}}=\frac1{|S|}\sum_{s\in S}\sum_{j\in J}\sum_{c\in C}
 p_{s,j}(c)\kappa(c).
$$

Padding is assigned zero cost. `uniform` charges a fixed amount per
constructor, `log-alphabet` charges the code's alphabet-based cost, and
`per-constructor` uses the declared map. The default scope includes width-one
cells only; `all-cells` also prices the interior representations.

This is an expected constructor cost regularizer, motivated by description
length \citep{rissanen1978}. Interpreting these costs as a prefix code requires a valid coding scheme.
A Gibbs interpretation
$p(t)\propto\exp[-\lambda\sum_{v\in t}\kappa(o_v)]$ additionally requires
a finite normalizing constant. A complete minimum description length objective
also specifies how the grammar and data are encoded.

## Supervision and constraint

Supervision supplies a particular target, such as a logical form or a
constituent span. This differs from stating a general condition on all
possible targets. For instance, scoring one supplied well-typed form singles
out that expression; scoring the set of well-typed forms would require summing
over its members.
The loss and the source of its targets jointly determine what evidence the
learner receives.

[`form-supervision`](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/losses/form_supervision.py) averages $-\log w_{\boldsymbol{\theta}}(t_i\mid
\mathbf{h}_{i,\mathrm{root}})$ over supplied gold forms. [`form-typing`](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/losses/form_typing.py) instead
scores each supplied term–label pair separately. With $y_i=1$ for a supplied
well-typed term and $m_i=w_{\boldsymbol{\theta}}(t_i\mid
\mathbf{h}_{i,\mathrm{root}})$, its cost is

$$
\ell_{\mathrm{type}}=-\frac1{|I|}\sum_{i\in I}
[y_i\log m_i+(1-y_i)\log(1-m_i)].
$$

This scores that particular supplied term or its complement under the
decoding process. The complement includes failed decoding paths and all other completed terms,
including other well-typed terms. Its averaging
unit is a supplied term–label pair, so one sentence can contribute several.

[`shaped-proof`](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/losses/shaped_proof.py) assigns a credit $u(t)=c_{o(t)}-\lambda|t|$, where $o(t)$ is
the decided outcome (ill-formed, well-formed, [type checking](logical-forms-as-types.md#semantic-typing), or proved).
For each item, it renormalizes decoded weights over decided candidates
$A_i$ and minimizes their negative expected credit:

$$
\ell_{\mathrm{proof}}=-\frac1{|I|}\sum_{i\in I}\sum_{t\in A_i}
\frac{p_i(t)}{\sum_{v\in A_i}p_i(v)}u(t).
$$

Items with no decided candidates are omitted. This conditional credit can
change when the decoding or proof budget changes.

For constraint instances $I$, [`pinning`](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/losses/pinning.py) minimizes
$-|I|^{-1}\sum_{i\in I}b_i\log m_i$, where $m_i$ is the target mass, its
complement for a negated target, or $1-\mu_{i,s}$ for a negative bracket.
Hard constraints use $b_i=1$; soft target constraints use their declared
penalty. A negative bracket with constituent probability one raises an error.

[`vector-anchor`](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/losses/vector_anchor.py) holds detached target vectors $\mathbf{v}_t$ and aggregates
$\|\mathbf{h}-\mathbf{v}_t\|_2^2$ or cosine distance over the target set,
using its minimum (`nearest`) or mean. Calling the result $d_i$, a positive
constraint costs $b_i d_i$ and a negative one costs
$b_i\max(0,\gamma-d_i)$ for margin $\gamma$. It averages over constraint
instances, keeping the detached anchors fixed.

Gold trees may determine which spans receive constraints through a declared
constraint source. The losses score the resulting constraints individually;
a supervised likelihood over complete trees would require a separate objective.
The supervision regime must name both the objective and the source that
selects its targets.

## Representation and entropy regularization

For live chart cells $S$ and target norm $a>0$, [`chart-scale`](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/losses/chart_scale.py) averages one of

$$
\left(\frac{\|\mathbf{h}_s\|_2^2}{a^2}-1\right)^2,\qquad
(\|\mathbf{h}_s\|_2-a)^2,\qquad
\bigl|\|\mathbf{h}_s\|_2-a\bigr|.
$$

These are respectively `squared-ratio`, `squared-distance`, and
`absolute-distance`; the first is a ratio of squared norms.

For a published derivation PMF $p_i$, its entropy is
$h(p_i)=-\sum_d p_i(d)\log p_i(d)$. [`derivation-entropy`](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/losses/derivation_entropy.py) averages this
quantity over live items to minimize entropy, or its negative to maximize
entropy. The parser must publish the entropy of its derivation PMF; a Viterbi
score supplies the highest-scoring derivation instead.

## Combining terms

If two component costs are $\ell_1$ and $\ell_2$, the choice
$\ell_1+2\ell_2$ gives the second twice its unweighted contribution.
The coefficients enter at their declared scale, and changing one changes the
relative training pressure of its term. The denominators within each component matter for the same reason.

At optimizer step $j$, let $A_j$ contain the terms applicable to the batch.
With nonnegative term weight $\lambda_k(j)$,

$$
\ell(\boldsymbol{\theta};j)=\sum_{k\in A_j}\lambda_k(j)\ell_k,
\qquad
\lambda_k(j)=\lambda_{k,0}+
\min(1,j/j_k)(\lambda_{k,1}-\lambda_{k,0})
$$

for a linear schedule of $j_k>0$ steps; an unscheduled weight is constant.
Weights enter the sum at their declared scale. Each term retains its own averaging unit, so
changing a weight or a denominator changes the optimization problem. Epoch
reports aggregate each term by its reported count; the reported total is
averaged over steps.

Interpreting a weighted sum of losses as a joint negative log likelihood
requires an explicit factorization and
normalization, including any dependence among observations and any powers
introduced by the weights. [Combine loss terms](../how-to/training/combine-losses.md)
connects these definitions to the [run document](documents-are-the-dsl.md#the-document-is-the-program).

## References

\bibliography
