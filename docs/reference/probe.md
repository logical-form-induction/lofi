# Probe

A [site](../explanation/probing-through-sites.md#a-site-is-a-published-promise)
is a named value that a model exposes for measurement or replacement.
[`Site`](nn.md#lofi.nn.sites.Site) identifies the value;
[`SiteMap`](nn.md#lofi.nn.sites.SiteMap) lists a component's sites;
[`SiteSpec`](nn.md#lofi.nn.sites.SiteSpec) declares their axes and permitted access.
A parser's [span](../explanation/charts-and-approximation.md#spans-and-the-sweep) vector is one such value.

[Observing that vector](../explanation/probing-through-sites.md#observations-and-interventions)
records it without changing the forward computation. Replacing it with zeros
and recomputing an output measures the effect of that replacement. A [classifier](../explanation/probing-through-sites.md#what-a-probe-result-licenses)
that predicts a label from the original vector instead measures what the
classifier can recover. [What a probe result licenses](../explanation/probing-through-sites.md#what-a-probe-result-licenses)
distinguishes these claims; [run a probe](../how-to/probes/run-probes.md)
connects the declarations to a trained point.

This page renders [`lofi.probe`](#lofi.probe): site map declarations and their lowering, the
instrumented and differentiable [regions](#lofi.probe.instrument) that observe, patch and take gradients
at published sites, the declared probe kinds and their settings, the
diagnostics catalog, [circuit discovery](../explanation/probing-through-sites.md#what-a-probe-result-licenses) (`receiver_circuit`, `attribute_edges`,
`acdc`), [causal abstraction](../explanation/probing-through-sites.md#what-a-probe-result-licenses) with exact [tokenizer alignment](../explanation/probing-through-sites.md#probes-are-only-as-rich-as-their-passes), and the [external
model wrapper](../how-to/probes/external-model-site-maps.md). [Probe a trained point](../how-to/probes/run-probes.md) shows
them in use.

::: lofi.probe
