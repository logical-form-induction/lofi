# Judgment models

A response model links elicited judgments to quantities predicted by a grammar
or language model. This page defines the predictor, response families,
linking functions, identification constraints, posterior objectives, and
calibration checks. These choices determine what a judgment comparison can
say about the grammar supplying its [readouts](../reference/judgment.md#lofi.judgment.LinkTerm).

## From responses to a measurement model

Two participants may rate the same sentence differently even if the
sentence receives one prediction from a grammar. They may use different
parts of the rating scale, differ in sensitivity to a predictor, or vary
from response to response. A judgment model describes these sources of
variation while relating the grammar's prediction to the observed response.

A [latent scale](#a-judgment-is-a-noisy-reading-of-a-latent-scale) is an unobserved numerical predictor of ratings. For ordered ratings, [cutpoints](../how-to/judgments/fit-a-judgment-model.md) separate the
response categories on that scale. A participant intercept can represent
a tendency to rate higher or lower; a participant slope can represent
different sensitivity to a predictor. The model locates the boundaries between
rating categories through its cutpoints, allowing unequal spacing.

## A judgment is a noisy reading of a latent scale

For response row $i$, participant $j(i)$, and item $k(i)$, a typical
crossed effects predictor has the form

$$
\eta_i=\mathbf{x}_i^{\mathsf T}\boldsymbol{\beta}
+\mathbf{z}_i^{\mathsf T}\mathbf{b}_{j(i)}
+\mathbf{u}_i^{\mathsf T}\mathbf{c}_{k(i)}+\sum_r g_r(m_{i,r}).
$$

Here $\mathbf{x}_i$ contains fixed effect covariates,
$\mathbf{z}_i$ and $\mathbf{u}_i$ select random effects, and $g_r$ is the
link for [readout](../reference/judgment.md#lofi.judgment.LinkTerm) $m_{i,r}$. A declaration need not include every term in
this schematic predictor. Gaussian blocks use
$\mathbf{b}_j\sim\mathcal{N}(\mathbf{0},\boldsymbol{\Sigma}_b)$ and similarly
for item effects; diagonal or correlated blocks and their priors are declared
separately. The covariance assumptions are part of the model.

For ordered response $Y_i\in\{1,\ldots,m\}$, discrimination $a_i>0$,
cutpoints $\kappa_{i,1}<\cdots<\kappa_{i,m-1}$, and
$\sigma(x)=(1+\exp(-x))^{-1}$, the cumulative logit model is
\citep{mccullagh1980,liddell-kruschke2018}

$$
\begin{aligned}
\mathbb{P}(Y_i\leq k\mid\eta_i,\boldsymbol{\kappa}_i,a_i)
 &=\sigma(a_i(\kappa_{i,k}-\eta_i)),\\
p_{Y_i}(k\mid\eta_i,\boldsymbol{\kappa}_i,a_i)
 &=\sigma(a_i(\kappa_{i,k}-\eta_i))
 -\sigma(a_i(\kappa_{i,k-1}-\eta_i)).
\end{aligned}
$$

For an arithmetic illustration with three categories, choose cutpoints
$-1$ and $1$, discrimination 1, and predictor 0. The cumulative
probabilities are $\sigma(-1)\approx0.269$ and
$\sigma(1)\approx0.731$, so the category probabilities are approximately
$0.269$, $0.462$, and $0.269$. Raising the predictor shifts probability
toward the higher categories.

The endpoint conventions are $\kappa_{i,0}=-\infty$ and
$\kappa_{i,m}=+\infty$. Ordered cutpoints are constructed as
$\kappa_{i,k}=b_i+\sum_{r=1}^{k}\operatorname{softplus}(\delta_{i,r})$,
before any declared centering or rescaling. Here $b_i$ is a scalar base,
and the unconstrained parameters $\delta_{i,r}$ become positive increments
through
$\operatorname{softplus}(x)=\log(1+\exp x)$. Participant offsets to the
increments thus preserve ordering; participant intercepts shift $\eta_i$.
They act on the response thresholds and predictor, respectively.

A `link` term carries a theoretical readout onto the scale; `factor` and
`interaction` terms account for design variables. Random intercepts and
slopes account for participant or item variation. Evidence for a readout is
conditional on the other terms in the predictor.

## Linking families state how a prediction enters

| Family | Mathematical link |
| --- | --- |
| [`linear`](../reference/judgment.md#lofi.judgment.LinearLinking) | $g(x)=\beta x$ |
| [`logit`](../reference/judgment.md#lofi.judgment.LogitLinking) | $g(m)=\beta[\log m-\log(1-m)]$ for a mass $m$ stored with its complement |
| [`probability`](../reference/judgment.md#lofi.judgment.ProbabilityLinking) | $g(m)=m$ |
| [`morcela`](../reference/judgment.md#lofi.judgment.MorcelaLinking) | $g(\ell,u,n)=\beta_0(\ell-\beta_1u+\gamma)/n^\delta$ |

In the last row, $\ell$ is sentence log probability, $u$ is unigram log
probability, and $n>0$ is token count. The original MORCELA score fixes
$\delta=1$ \citep{tjuatja-etal2025}; LoFI adds an overall slope and an
optional fitted exponent. SLOR sets $\beta_1=1$, $\gamma=0$, and $\delta=1$
\citep{pauls-klein2012,lau-etal2017}. Raw log probability sets
$\beta_1=\gamma=\delta=0$, before the overall slope. Neither $\ell$ nor
$u$ denotes a PMF.

A logit link has infinite endpoint values at masses zero and one; numerical
handling, including any clipping, must follow the declared [readout](../reference/judgment.md#lofi.judgment.LinkTerm). The same mass under `logit` and `probability` makes different claims
about how changes near an endpoint affect judgments.

## The latent scale has no units of its own

An identification problem occurs when different parameter values make the
same predictions. Imagine shifting both every cutpoint and every latent
prediction upward by the same amount: their differences stay the same, so
the response probabilities stay the same. More observations of those
responses cannot distinguish the two settings without additional constraints.
The general transformation below also includes a rescaling when
discrimination can change with it.

For the unconstrained ordinal likelihood, the transformation

$$
\eta_i'=c+d\eta_i,\qquad
\kappa_{i,k}'=c+d\kappa_{i,k},\qquad
 a_i'=a_i/d,\quad d>0
$$

leaves probabilities unchanged: $a_i'(\kappa_{i,k}'-\eta_i')=
a_i(\kappa_{i,k}-\eta_i)$. Rescaling latent values and cutpoints alone
does not preserve probabilities when discrimination is fixed. Priors may
break a likelihood invariance but then influence the parameter scale.

| Identification | Constraint |
| --- | --- |
| `fixed_discrimination` | Holds discrimination at a declared positive value. |
| `fixed_cutpoint_variance` | Holds cutpoint variance at a declared value. |
| [`identity_link_on_logit`](../reference/judgment.md#lofi.judgment.IdentityLinkOnLogit) | Holds link slopes at one, placing the [readout](../reference/judgment.md#lofi.judgment.LinkTerm)'s log odds directly on the latent scale. |

Location can still be shared between cutpoint bases and intercepts. A
posterior shift in one can be compensated by a shift in the other without
changing predictions. Parameter recovery thus requires examining the actual
constraints and priors alongside comparisons of posterior means with
generating values.

## Other response families

For forced choice without ties among $m$ alternatives with scores
$\eta_{i,1},\ldots,\eta_{i,m}$, the choice PMF is

$$
p_{Y_i}(k)=\frac{\exp(a_i\eta_{i,k})}{\sum_{j=1}^{m}\exp(a_i\eta_{i,j})}.
$$

The tied binary choice family applies a three-level cumulative model to the
difference of two scores, with the middle interval representing a tie.
Its three response categories thus describe the comparison between two alternatives.

For a continuous response rescaled to $y\in[0,1]$, the bounded family uses
location $\mu=\sigma(\eta)$ and scale $s>0$. With standard normal PDF
$f_0$ and CDF $F_0$, its PDF with respect to Lebesgue measure is

$$
f(y\mid\mu,s)=\frac{f_0((y-\mu)/s)}
{s[F_0((1-\mu)/s)-F_0(-\mu/s)]}\mathbf{1}_{[0,1]}(y).
$$

Probabilities are obtained by integrating this density over intervals;
individual points, including the permitted endpoints, have zero probability.
On an original scale
$x\in[a,b]$, the corresponding density includes the Jacobian $1/(b-a)$.
The fitted likelihood uses the normalized response scale.

## Conditioning on effects or integrating them out

For a participant already observed, fitted participant effects can inform
their next response. For a new participant, those individual effects are
unknown. Averaging predictions over plausible effects asks what a new
participant drawn from the modeled population might do. Setting every
effect to zero instead predicts one particular population-mean setting.
Because the response function is nonlinear, those two predictions generally
differ.

For population parameters $\boldsymbol{\theta}$ and local participant
effects $\mathbf{b}_j$, the conditional likelihood is
$\prod_j\prod_{i:j(i)=j}p(y_i\mid\boldsymbol{\theta},\mathbf{b}_j)$.
The population marginal likelihood integrates each participant's block
jointly:

$$
\prod_j\int\left[\prod_{i:j(i)=j}
p(y_i\mid\boldsymbol{\theta},\mathbf{b})\right]
f(\mathbf{b}\mid\boldsymbol{\theta})\,d\mathbf{b}.
$$

Integrating each response separately would discard the shared participant
effect. The [`conditional`](../reference/judgment.md#lofi.judgment.ConditionalLikelihood) objective scores responses given all drawn sites.
NUTS and the variational guides draw local sites and use this objective.
The [`truncated_marginal`](../reference/judgment.md#lofi.judgment.TruncatedMarginalLikelihood) objective is offered for MAP with cumulative ordered
responses and participant as the sole local grouping factor. It integrates
that participant's complete local block numerically.

The truncated evaluator integrates standard normal local coordinates over
$\|\mathbf{z}\|_2\leq r$. Its scalar route uses adaptive ODE integration;
its multivariate route uses positive quadrature weights. Both approximate the
unnormalized integral over the ball, retaining its Gaussian probability mass
as a factor in the likelihood.

For a new participant, population mean plug-in prediction
$p(y\mid\mathbf{b}=\mathbf{0})$ generally differs from
$\int p(y\mid\mathbf{b})f(\mathbf{b})\,d\mathbf{b}$. A known participant
can instead be predicted using their fitted effects. The declared [`plug_in`](../reference/judgment.md#lofi.judgment.FoldPredictions.plug_in)
or `marginal` convention applies consistently across folds; the new-participant
coverage threshold determines when a convention must be declared. Seen and
new participants are separate predictive regimes.

## Posterior objectives and calibration

The inference methods return different kinds of answers. Maximum a
posteriori estimation (MAP) seeks one parameter setting with high posterior
density. Posterior sampling produces draws intended to represent uncertainty
over settings. Variational inference fits a tractable distribution to
approximate that uncertainty. All require the likelihood, prior, and model
constraints to be specified; none is a shortcut around identification.

For likelihood $p(\mathbf{y}\mid\boldsymbol{\theta})$ and prior density
$f(\boldsymbol{\theta})$, the posterior density is proportional to their
product. Continuous responses replace the PMF likelihood by a product of
PDFs. MAP maximizes the log posterior density. A variational density
$g_{\boldsymbol{\lambda}}$ instead maximizes the evidence lower bound

$$
\mathbb{E}_{\boldsymbol{\Theta}\sim g_{\boldsymbol{\lambda}}}
\left[\log p(\mathbf{y}\mid\boldsymbol{\Theta})
+\log f(\boldsymbol{\Theta})-\log g_{\boldsymbol{\lambda}}(\boldsymbol{\Theta})\right].
$$

NUTS targets the posterior by Hamiltonian Monte Carlo
\citep{hoffman-gelman2014}. Gradient-based automatic variational inference
\citep{kucukelbir-etal2017} and the structured mixed model guides
\citep{goplerud-etal2025} use different approximations. Theoretical guarantees
for a particular Gaussian or binomial mixed model apply under that model's
assumptions; an ordinal program requires its own justification. Compiled ordinal programs
use Quivers AutoNormal or AutoStructured guides with Adam-based stochastic
variational inference. The separate Gaussian crossed model guide module
implements the coordinate ascent family associated with
\citealt{goplerud-etal2025}.

A posterior can remain predictive when individual parameters are weakly
identified. Simulation-based calibration checks inference under repeated
prior predictive simulation \citep{talts-etal2018}. Draw
$\boldsymbol{\Theta}\sim f$, then $Y\sim p(\cdot\mid\boldsymbol{\Theta})$,
and obtain $s$ independent exact posterior draws
$\boldsymbol{\Theta}^{(1)},\ldots,\boldsymbol{\Theta}^{(s)}$.
For a continuous scalar summary $h$, the rank

$$
r=\sum_{j=1}^{s}\mathbf{1}\{h(\boldsymbol{\Theta}^{(j)})<h(\boldsymbol{\Theta})\}
$$

is discrete uniform on $\{0,\ldots,s\}$ under that joint experiment.
Ties require randomized treatment; dependent Monte Carlo draws require
additional care. LoFI computes randomized ranks from NUTS draws and compares
NUTS, structured, and mean-field fits through interval coverage and cutpoint
recovery on the same simulated designs. Calibration is a property of the model, priors, design, and
inference together. Assessing the model against real judgments additionally
requires observed-data checks.

## What a fit licenses

Held-out log predictive density sums
$\log p(y_i\mid\mathcal{D}_{\mathrm{train}})$ over declared held-out units.
Leaving out a response differs from leaving out a participant, and the
uncertainty calculation must respect that unit \citep{vehtari-etal2017}.
Two grammars can be compared by supplying different [readouts](../reference/judgment.md#lofi.judgment.LinkTerm) to otherwise
identical response models over the same held-out observations. A comparison
that also changes linking, identification, or elicitation measures their
combined effect with the grammar.

[Fit a judgment model](../how-to/judgments/fit-a-judgment-model.md) declares
and runs a fit; [Train on judgments](../how-to/judgments/train-on-judgments.md)
uses a response instrument as a parser training signal.

## References

\bibliography
