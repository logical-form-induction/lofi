# Model declarations

To predict words from a [span representation](../../how-to/models/configure-a-chart-parser.md), a model needs a
[parser](#parsers) that supplies the representation and a [span decoder](#span-decoders)
that scores the words. To compare symbolic predictions, it instead needs the
appropriate [term decoders](#decoders) and [readouts](#readouts). The choice of
[objective](../../explanation/induction-objectives.md) thus determines which component outputs must be available.

The [readout guide](../../how-to/models/configure-readouts.md) connects these
outputs to declarations. [Term layouts](../../explanation/term-layout-capabilities.md)
and [deduction profiles](../../explanation/derivations-as-proofs.md#what-the-compiler-derives)
explain further compatibility requirements. The [neural API](../nn.md)
documents the built components.

This page renders the classes that validate a run document's `model` section, one section per component family.

## Tokens

::: lofi.spec.tokens.learned

::: lofi.spec.tokens.llm2vec

::: lofi.spec.tokens.pretrained

::: lofi.spec.tokens.static

## Poolers

When one source word occupies two tokenizer rows with vectors
$\mathbf{x}_1$ and $\mathbf{x}_2$, [mean pooling](#lofi.spec.poolers.mean.MeanSubwordPoolerSpec) uses
$(\mathbf{x}_1+\mathbf{x}_2)/2$, while [first pooling](#lofi.spec.poolers.first.FirstSubwordPoolerSpec) uses $\mathbf{x}_1$.
These are two summaries of the same aligned rows, with different information
available to the next component. The [token encoder guide](../../how-to/models/choose-a-token-encoder.md)
connects word alignment to the pooling declaration.

::: lofi.spec.poolers.attention

::: lofi.spec.poolers.first

::: lofi.spec.poolers.mean

## Encoders

::: lofi.spec.encoders.hrr

::: lofi.spec.encoders.linear

::: lofi.spec.encoders.rnn

::: lofi.spec.encoders.transformer

## Decoders

::: lofi.spec.decoders.diffusion

::: lofi.spec.decoders.hrr

::: lofi.spec.decoders.linear

::: lofi.spec.decoders.rnn

::: lofi.spec.decoders.transformer

## Parsers

::: lofi.spec.parsers.chart

::: lofi.spec.parsers.flat

::: lofi.spec.parsers.reduction

::: lofi.spec.parsers.span_crf

::: lofi.spec.parsers.split

::: lofi.spec.parsers.tagged

## Span functions

::: lofi.spec.span_functions

## Chart compositions

::: lofi.spec.chart_compositions

## Span decoders

::: lofi.spec.span_decoders.rnn

## Controllers

::: lofi.spec.controllers

## Readouts

::: lofi.spec.readouts

## Groundings

::: lofi.spec.groundings

## Fuzzy logics

For illustrative truth degrees $a=0.8$ and $b=0.5$, [product conjunction](../../explanation/inference-as-evidence.md#truth-degrees-and-fuzzy-operations) gives
$ab=0.4$, while [Gödel conjunction](../../explanation/inference-as-evidence.md#truth-degrees-and-fuzzy-operations) gives $\min(a,b)=0.5$. Selecting a family
changes the calculation performed on the same inputs. The
[truth-degree definitions](../../explanation/inference-as-evidence.md#truth-degrees-and-fuzzy-operations)
give the operations and the interpretation of their values as degrees of truth.

::: lofi.spec.fuzzy_logic

## Countermodel search

::: lofi.spec.countermodels

## Scorers

::: lofi.spec.scorers

## Structures

::: lofi.spec.structures

## Knowledge bases

::: lofi.spec.knowledge_bases

## Provers

::: lofi.spec.provers
