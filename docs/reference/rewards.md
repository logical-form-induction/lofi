# Rewards

A [reward objective](../explanation/induction-objectives.md#inference) assigns
credit to an outcome and minimizes negative expected credit. For instance,
a declared payoff can favor form pairs that support an observed [entailment
label](../explanation/inference-as-evidence.md#a-label-is-a-pair-of-proof-questions). Changing the credits changes that objective.

A [control variate](#control-variates) uses a baseline to reduce fluctuations
in a sampled gradient estimate. [From a decision to a gradient](../explanation/inference-as-evidence.md#from-a-decision-to-a-gradient)
explains when subtracting the baseline preserves the expected gradient and
why the weighted, truncated routes need additional qualifications.

This page renders [`lofi.reward`](#lofi.reward), the declared payoffs for proof-grounded
inference, and [`lofi.control_variate`](#lofi.control_variate), the [control variates](../explanation/inference-as-evidence.md#from-a-decision-to-a-gradient) that reduce the variance
of sampled policy gradients.

::: lofi.reward

## Control variates

::: lofi.control_variate
