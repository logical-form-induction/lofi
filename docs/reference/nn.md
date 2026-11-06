# Neural components

This page renders `lofi.nn`: encoders, decoders, controllers, parsers, losses,
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

Sites identify addressable points inside a model. The probing layer may import
this module as one of its three allowed neural dependencies.

::: lofi.nn.sites

## Sharing

::: lofi.nn.sharing

## Ports

::: lofi.nn.ports

## Transformer

::: lofi.nn.transformer

## Policy

::: lofi.nn.policy

## Psi

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
