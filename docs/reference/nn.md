# Neural components

A [token encoder](#tokens) supplies vectors for the words or subwords of an
input. A [parser](#parsers) combines information about its spans, and a
[readout](#readouts) exposes a quantity that a [loss](#losses) or [evaluator](eval.md) can
use. [Term encoders](#encoders) and [decoders](#decoders) connect vectors with
symbolic terms such as categories and logical forms.

For instance, a parser can publish a vector for the [span](../explanation/charts-and-approximation.md#spans-and-the-sweep) *John sleeps* and a
readout can assign weight to a [target category](../explanation/derivations-as-proofs.md#goals) from that vector. The
[model guides](../how-to/models/configure-readouts.md) specify the available
readouts. The [API reading guide](index.md#reading-a-python-api-entry)
explains arrays and shapes such as `(item, span, width)`.

This page renders [`lofi.nn`](#lofi.nn): encoders, decoders, controllers, parsers, losses,
and the sites inside a model that an observer may name. Importing it needs the
`nn` extra.

The model guides define [controller distributions](../how-to/models/configure-a-controller.md),
[neural chart composition](../how-to/models/configure-a-chart-parser.md),
[term representations and decoding](../how-to/models/configure-term-encoders-and-decoders.md),
[token representations](../how-to/models/choose-a-token-encoder.md), and
[readout measures](../how-to/models/configure-readouts.md).
[Induction objectives](../explanation/induction-objectives.md) specifies their
losses and gradient conventions.

::: lofi.nn

## Encoders

::: lofi.nn.encoders

## Codec

::: lofi.nn.codec

## Decoders

::: lofi.nn.decoders

## Parsers

::: lofi.nn.parsers

## Tokens

::: lofi.nn.tokens

## Controller

::: lofi.nn.controllers

## Sites

[Sites](../explanation/probing-through-sites.md#a-site-is-a-published-promise)
identify values exposed for observation or replacement.
[`Site`](#lofi.nn.sites.Site), [`SiteMap`](#lofi.nn.sites.SiteMap), and
[`SiteSpec`](#lofi.nn.sites.SiteSpec) record their identities and access contracts. The [probing layer](../explanation/probing-through-sites.md#a-site-is-a-published-promise) may import
this module as one of its three allowed neural dependencies.

::: lofi.nn.sites

## Sharing

::: lofi.nn.sharing

## Ports

[`ForwardMode`](#lofi.nn.ports.ForwardMode) distinguishes scoring supplied
terms (`OBSERVE`), drawing terms (`SAMPLE`), and summing over reachable terms
(`MARGINALIZE`). These requests need different capabilities from a [decoder](../how-to/models/configure-term-encoders-and-decoders.md).
[`ForwardOptions`](#lofi.nn.ports.ForwardOptions) specifies which options are
accepted in each mode.

::: lofi.nn.ports

## Transformer

::: lofi.nn.transformer

## Policy

[`DevicePolicy`](#lofi.nn.policy.DevicePolicy) records both where numerical
arrays live and their precision. Its [activation-checkpointing](../how-to/training/choose-a-device-policy.md) setting can
recompute wider [chart](../explanation/charts-and-approximation.md#reusing-partial-analyses) cells during the backward pass instead of retaining
them. This saves intermediate storage at the cost of repeated computation;
it differs from a [saved training checkpoint](checkpoints.md).

::: lofi.nn.policy

## Psi

For a [span](../explanation/charts-and-approximation.md#spans-and-the-sweep) with more than one possible split, a [controller](../how-to/models/configure-a-controller.md) can score several
actions at each split. [`RuleScores`](#lofi.nn.psi.RuleScores) keeps those
axes separate: `(item, span, split, action)`. Its [mask](#lofi.nn.psi.RuleScores.mask) identifies valid splits,
and its declared [null-action column](../how-to/models/configure-a-controller.md), when present, is excluded from split
weight. [Controller distributions](../how-to/models/configure-a-controller.md)
explains the action scores used by this calculation.

::: lofi.nn.psi

## Sampling

::: lofi.nn.sampling

## Arity

::: lofi.nn.arity

## Supervision

::: lofi.nn.supervision

## Losses

::: lofi.nn.loss

## Grounding

::: lofi.nn.grounding

## Readouts

::: lofi.nn.readouts
