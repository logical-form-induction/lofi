# Induction objectives

An induction objective assigns a scalar cost to the outputs a model publishes.
This page defines the packaged losses, their averaging units and gradients,
and the assumptions needed to interpret them as evidence about a grammar.
It covers label likelihoods, compositional consistency, reconstruction,
simplicity, supervision, regularization, and combinations of these terms.

## Objectives read published outputs

Write $\boldsymbol{\theta}$ for the trainable parameters, $w_i$ for input
item $i$, and $\mathbf{h}_{i,s}$ for its representation at span $s$.
A decoder supplies complete term weights
$w_{\boldsymbol{\theta}}(t\mid\mathbf{h}_{i,s})$, obtained from locally
normalized choices. Admitted terms can have total weight below one when
decoding encounters dead ends or rejects completed terms. Below, $p$ and $q$
denote the normalized distributions supplied to a comparison; target mass
readouts retain their declared scoring convention. A root target mass is
not the partition function of the symbolic parser.

Each objective declares its required outputs: cells, marginals, readouts,
labels, or constraints. A missing output is an error. In the equations below,
index sets contain only the observations the implementation scores, after
its masks and selection rules. An empty set makes the term inapplicable;
it does not introduce a zero observation into an epoch mean. Logs are natural
logs unless a symbol cost declaration explicitly uses bits.

## Grammaticality

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
label is uninformative; above $1/2$ it is anticorrelated, rather than
uninformative. LoFI restricts the declared rate to below $1/2$.

Positive-only labels can minimize this term by assigning mass one to every
string. They thus require additional constraints if the intended question is
whether the learner distinguishes grammatical from ungrammatical strings.
A declared judgment response replaces the binary observation model with its
response likelihood; see [Judgment models](judgment-models.md).

## Coherence

Binary coherence compares the parent decoder PMF $p_{i,s}$ with a candidate
PMF $q_{i,s,k,a}$ obtained by (i) merging the vectors of the daughters at
split $k$ and (ii) applying the decoder for action $a$. It does not first
decode two symbolic terms and then apply a symbolic grammar rule to them.

Let $\mu_{i,s}$ be the constituent marginal, $\pi_{i,s,k}$ the conditional
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
$I_{\mathrm{bin}}$ contains the distinct participating items. The denominator
is not the number of spans or the sum of their weights. Gradients flow through
the participating distributions and weights. Unary coherence is computed
separately and added, with its own participating-item denominator. For exhausted frontiers, let $e_{\boldsymbol{\theta}}$ be the term encoder.
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
scores generated from the candidate distribution, rather than an arbitrary
raw score vector \citep{blondel-etal2020}. This distinction matters when
relating the implementation to $\alpha$-entmax \citep{peters-etal2019}.

The exact estimator requires exhausted decoding frontiers and residual mass
at most `exact_residual`; exhaustion alone does not establish normalization. The truncated
estimator compares PMFs conditioned on their enumerated common support and
checks omitted probability mass, not a bound on divergence error; see
[Charts and approximation](charts-and-approximation.md#the-residual-mass-budget).
Coherence can be small for an internally consistent but linguistically
incorrect assignment of terms. It does not identify a grammar by itself.

## Reconstruction

For a selected span $s$, let $x_{s,1},\ldots,x_{s,n_s}$ be its words followed
by the end token, so $n_s$ includes that token. Teacher-forced reconstruction
has row cost

$$
c_s=-\sum_{j=1}^{n_s}\log p_{\boldsymbol{\theta}}
(x_{s,j}\mid x_{s,<j},\mathbf{h}_s).
$$

With $b_s=1$ for uniform weighting or
$b_s=\operatorname{stopgrad}(\mu_s)$ for marginal weighting,
`span-reconstruction` uses

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
argument: marginal weighting does not train the parser by changing these
weights. A zero total weight makes the objective inapplicable.

`outside-reconstruction` uses the outside vector $\mathbf{o}_{i,j}$ of
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
length \citep{rissanen1978}. Arbitrary costs do not automatically define a
prefix code or a normalized prior. A Gibbs interpretation
$p(t)\propto\exp[-\lambda\sum_{v\in t}\kappa(o_v)]$ additionally requires
a finite normalizing constant. The term is not, by itself, a complete minimum
description length objective for the grammar and data.

## Supervision and constraint

`form-supervision` averages $-\log w_{\boldsymbol{\theta}}(t_i\mid
\mathbf{h}_{i,\mathrm{root}})$ over supplied gold forms. `form-typing` instead
scores each supplied term–label pair separately. With $y_i=1$ for a supplied
well-typed term and $m_i=w_{\boldsymbol{\theta}}(t_i\mid
\mathbf{h}_{i,\mathrm{root}})$, its cost is

$$
\ell_{\mathrm{type}}=-\frac1{|I|}\sum_{i\in I}
[y_i\log m_i+(1-y_i)\log(1-m_i)].
$$

This scores that particular supplied term or its complement under the
decoding process. The complement includes failed decoding paths; it is
not the total probability of the set of all well-typed terms. Its averaging
unit is a supplied term–label pair, so one sentence can contribute several.

`shaped-proof` assigns a credit $u(t)=c_{o(t)}-\lambda|t|$, where $o(t)$ is
the decided outcome (ill-formed, well-formed, type checking, or proved).
For each item, it renormalizes decoded weights over decided candidates
$A_i$ and minimizes their negative expected credit:

$$
\ell_{\mathrm{proof}}=-\frac1{|I|}\sum_{i\in I}\sum_{t\in A_i}
\frac{p_i(t)}{\sum_{v\in A_i}p_i(v)}u(t).
$$

Items with no decided candidates are omitted. This conditional credit can
change when the decoding or proof budget changes.

For constraint instances $I$, `pinning` minimizes
$-|I|^{-1}\sum_{i\in I}b_i\log m_i$, where $m_i$ is the target mass, its
complement for a negated target, or $1-\mu_{i,s}$ for a negative bracket.
Hard constraints use $b_i=1$; soft target constraints use their declared
penalty. A negative bracket with constituent probability one raises an error.

`vector-anchor` holds detached target vectors $\mathbf{v}_t$ and aggregates
$\|\mathbf{h}-\mathbf{v}_t\|_2^2$ or cosine distance over the target set,
using its minimum (`nearest`) or mean. Calling the result $d_i$, a positive
constraint costs $b_i d_i$ and a negative one costs
$b_i\max(0,\gamma-d_i)$ for margin $\gamma$. It averages over constraint
instances, not target vectors, and does not update the detached anchors.

Gold trees may determine which spans receive constraints through a declared
constraint source. These losses do not themselves contain a supervised tree
likelihood. The supervision regime must thus name both the objective
and the source that selects its targets.

## Representation and entropy regularization

For live chart cells $S$ and target norm $a>0$, `chart-scale` averages one of

$$
\left(\frac{\|\mathbf{h}_s\|_2^2}{a^2}-1\right)^2,\qquad
(\|\mathbf{h}_s\|_2-a)^2,\qquad
\bigl|\|\mathbf{h}_s\|_2-a\bigr|.
$$

These are respectively `squared-ratio`, `squared-distance`, and
`absolute-distance`; the first is a ratio of squared norms.

For a published derivation PMF $p_i$, its entropy is
$h(p_i)=-\sum_d p_i(d)\log p_i(d)$. `derivation-entropy` averages this
quantity over live items to minimize entropy, or its negative to maximize
entropy. It requires a parser that publishes that quantity and does not
reinterpret a Viterbi score as an entropy.

## Combining terms

At optimizer step $j$, let $A_j$ contain the terms applicable to the batch.
With nonnegative term weight $\lambda_k(j)$,

$$
\ell(\boldsymbol{\theta};j)=\sum_{k\in A_j}\lambda_k(j)\ell_k,
\qquad
\lambda_k(j)=\lambda_{k,0}+
\min(1,j/j_k)(\lambda_{k,1}-\lambda_{k,0})
$$

for a linear schedule of $j_k>0$ steps; an unscheduled weight is constant.
Weights are not normalized. Each term retains its own averaging unit, so
changing a weight or a denominator changes the optimization problem. Epoch
reports aggregate each term by its reported count; the reported total is
averaged over steps.

A weighted sum of losses is not automatically a joint negative log
likelihood. Such an interpretation requires an explicit factorization and
normalization, including any dependence among observations and any powers
introduced by the weights. [Combine loss terms](../how-to/training/combine-losses.md)
connects these definitions to the run document.

## References

\bibliography
