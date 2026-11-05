# Judgment models

This page explains how a response model links elicited judgments to the
quantities a grammar or a language model predicts. It distinguishes the latent
scale from the readouts placed on it, says what a linking family, a scale
identification and a likelihood objective each assume, and records what a
fitted response model does and does not license about the grammar whose
predictions it reads.

## A judgment is a noisy reading of a latent scale

A judgment design holds one row per response: the participant who gave it, the
item it was given to, the response level, and the item's columns, which are
design factors (declared item attributes, each read by a bead expression) and
readouts (a predicted quantity such as an
item's score or a published root mass). A response model does not compare a
readout with a response directly. It places every row on a latent scale, as the
sum of the terms of its `predictor`, and observes the response through a
family that maps a latent value to a distribution over levels.

The cumulative family is the ordinal case: a latent value is compared with an
increasing set of cutpoints, and the probability of each level is the
logistic mass between two adjacent cutpoints. Its cutpoints are drawn as a base
and positive increments, so they are ordered by construction. Participants are
allowed to use the scale differently: a participant random intercept moves a
participant's latent values, and participant cutpoint offsets move where that
participant's level boundaries fall. The packaged synthetic designs are drawn
from exactly this structure, which is why they can serve as known truths for a
fit (see [Simulate a judgment design](../how-to/judgments/simulate-judgment-designs.md)).

The terms of the predictor separate what a theory predicts from what it does
not claim to predict. A `link` term carries a readout onto the latent scale; a
`factor` or `interaction` term absorbs effects of design factors; random
intercepts and slopes absorb variation over participants or items. A model
whose link term earns its place over the same model without it is evidence that
the readout tracks the judgments beyond what the other terms capture; a model
whose fixed and random terms already explain the judgments leaves the readout
nothing to explain.

## Linking families state how a prediction enters

A linking family is the assumption about how a readout becomes a latent value.
The `lofi.linking` group installs four:

| Family | Assumes |
| --- | --- |
| `linear` | The readout enters through a fitted slope. |
| `logit` | The readout is a published mass `m` stored with its complement, and its log-odds `log m − log(1 − m)` enter through a slope. |
| `probability` | The mass itself enters, with no slope. |
| `morcela` | The readout is a language model's sentence log probability `p`, corrected for unigram log probability `u` and token count `l` as `slope · (p − β·u + γ) / l^δ`; SLOR and raw log probability are nested settings of its parameters. |

The choice is substantive. The same root mass under `logit` and under
`probability` makes different claims about how certainty near one translates
into judgments, and two fits that differ only in their linking are a
comparison of those claims, not of the grammars that produced the mass.

## The latent scale has no units of its own

A latent scale observed only through ordinal responses is identified only up
to location and scale: shifting every latent value and every cutpoint, or
stretching both, leaves every response probability unchanged. A response model
must fix this, and `identification` declares how:

| Constraint | Fixes |
| --- | --- |
| `fixed_discrimination` | The discrimination at a declared value, so link slopes and cutpoints are read in its units. |
| `fixed_cutpoint_variance` | The variance of the cutpoints at a declared value; a fitted slope may then cross zero. |
| `identity_link_on_logit` | Every link slope at one, so a `logit` link's log-odds are the latent scale itself. |

Location is shared between the cutpoint base and any intercepts, and that
trade-off is visible in a fit: a posterior mean of the cutpoint base far from
its generating value can coexist with good predictions when intercepts move to
compensate. A comparison between fitted parameters is meaningful only between
fits that share an identification, and a comparison between fits under
different identifications is one of predictive performance alone.

## Conditioning on effects or integrating them out

The `conditional` objective scores each response given every drawn site,
participant effects included. NUTS and the variational guides draw every site,
so under them the conditional likelihood is the only one that changes the
posterior. The `truncated_marginal` objective integrates one group's local
block, such as the participant effects, over a bounded region by adaptive
quadrature, and so scores responses as a new participant from the population
would give them. It is offered only for MAP, where the population posterior is
optimized and the integral is the quantity of interest.

The two objectives answer different questions about held-out participants. A
participant seen in training can be predicted from their own fitted effects;
a new participant cannot, and is predicted either by `plug_in`, at the
population mean of the effects, or by `marginal`, integrating over them. Held-out
comparisons therefore report seen and new participants as separate regimes,
one row per participant, rather than pooling two predictive tasks into one
number. Once a fit declares which convention predicts new participants, that
convention predicts every new participant in every fold. The declared coverage
threshold only decides when a convention is required: a fold whose held-out
participants are mostly new, beyond the threshold, is refused unless one is
declared.

## What calibration checks

A posterior is useful only if the inference that produced it recovers the
parameters it reports. Simulation-based calibration tests this on the model's
own terms: parameters drawn from the priors generate a design's responses, and
the posterior fitted to those responses should place the truth at a uniformly
distributed rank. LoFI fits each replicate by NUTS and by the structured and
mean-field guides over the same compiled program, so the report also gives
each method's interval coverage of the truth, block by block, and reads each
method's recovery of the participants' cutpoints against a preregistered floor. Calibration is a
property of the response model, its priors, the design and the inference
together; it says nothing about whether the model is true of the judgments.

## What a fit licenses

A judgment fit is a statement about a response model over a design: its
posterior, its predictive performance on held-out rows, and its comparison with
other candidates over the same rows. It licenses a claim about a grammar only
through the readout the grammar supplies, and only relative to the other terms
of the predictor, the linking family and the identification. Two grammars are
compared through judgments when they supply the readout of the same link term in
otherwise identical fits; a difference in their predictive density under a
calibrated inference is then a difference in how well their predictions track
the judgments. A difference between fits that also change the linking, the
identification or the elicitation protocol is not attributable to the grammar.

[Fit a judgment model](../how-to/judgments/fit-a-judgment-model.md) declares
and runs a fit; [Train on judgments](../how-to/judgments/train-on-judgments.md)
uses an instrument in the other direction, as a training signal for a parser.

## References

Liddell, T. M., and Kruschke, J. K. (2018). Analyzing ordinal data with metric
models: What could possibly go wrong? *Journal of Experimental Social
Psychology*, 79, 328–348.

Talts, S., Betancourt, M., Simpson, D., Vehtari, A., and Gelman, A. (2018).
Validating Bayesian inference algorithms with simulation-based calibration.
arXiv:1804.06788.

Vehtari, A., Gelman, A., and Gabry, J. (2017). Practical Bayesian model
evaluation using leave-one-out cross-validation and WAIC. *Statistics and
Computing*, 27, 1413–1432.
