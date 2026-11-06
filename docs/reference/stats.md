# Stats

This page renders `lofi.stats`: p-values, null models, dependence regimes and multiple comparison corrections.

The [experimental commitments](../explanation/experiments-and-commitments.md)
page defines paired randomization, equivalence hypotheses, adjusted test
values, familywise error, and false discovery rate. Each guarantee is
conditional on its stated sampling, null model, and dependence assumptions.

## Bootstrap intervals

For $n$ independent, identically distributed sampling units $x_1,\ldots,x_n$ and statistic
$\widehat\theta=s(x_1,\ldots,x_n)$, a nonparametric bootstrap draws $n$
indices independently and uniformly from $\{1,\ldots,n\}$, then recomputes
$s$ on those units \citep{efron1979}. Cluster resampling draws whole clusters
instead of individual observations. The resampling unit is part of the
estimand's uncertainty model.

Let $b$ bootstrap replicates be $\widehat\theta_1^*,\ldots,\widehat\theta_b^*$
and let $Q^*$ be their linearly interpolated empirical quantile function.
A percentile interval of nominal coverage $1-\alpha$ is
$[Q^*(\alpha/2),Q^*(1-\alpha/2)]$. The bias-corrected and accelerated (BCa)
interval adjusts these quantile levels \citep{efron1987}. Write $\Phi$ for
the standard normal CDF and $\widehat\theta_{(-i)}$ for the delete-one
jackknife estimate. LoFI computes

$$
\begin{aligned}
z_0&=\Phi^{-1}\!\left(\frac1b\sum_{j=1}^{b}
\mathbf{1}\{\widehat\theta_j^*<\widehat\theta\}\right),\\
\overline\theta_{(-\cdot)}&=\frac1n\sum_i\widehat\theta_{(-i)},\\
a&=\frac{\sum_i(\overline\theta_{(-\cdot)}-\widehat\theta_{(-i)})^3}
{6[\sum_i(\overline\theta_{(-\cdot)}-\widehat\theta_{(-i)})^2]^{3/2}},\\
q(u)&=\Phi\!\left(z_0+\frac{z_0+\Phi^{-1}(u)}
{1-a[z_0+\Phi^{-1}(u)]}\right).
\end{aligned}
$$

The endpoints are $Q^*(q(\alpha/2))$ and $Q^*(q(1-\alpha/2))$.
A constant jackknife sets $a=0$. An infinite bias correction, an invalid
adjustment denominator, or reversed quantile levels falls back to the
percentile interval. A nominal confidence level is not a finite-sample
coverage guarantee; dependence and the chosen resampling unit matter.

::: lofi.stats

## References

\bibliography
