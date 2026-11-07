# Data

A [corpus reader](#corpora) reads observations from a source format. A
[featurizer](#featurizers) prepares the fields a model consumes, and
[batch partitioning](#batch-partitioning) groups observations for computation.
An inference pair, for instance, supplies two token sequences and a relation
label. A parsed-sentence observation supplies one sentence and its gold parse.

[Synthetic generation](#synthetic-generation) produces observations under a
declared source. [Source and learner declarations](../explanation/synthetic-designs.md)
explains how the same generated observations can be analyzed under different
learner grammars.

This page renders [`lofi.data`](#lofi.data): the corpus readers, cache, pipeline validation, batch partitioning, featurizers and synthetic generators.

::: lofi.data

## Corpora

::: lofi.data.corpora

### CCGBank

::: lofi.data.corpora.ccgbank

### Layers

::: lofi.data.corpora.layers

### MegaAcceptability

::: lofi.data.corpora.megaacceptability

### NLI

::: lofi.data.corpora.nli

### Entailment graphs

::: lofi.data.corpora.entailment_graph

### FraCaS

::: lofi.data.corpora.fracas

### MegaAttitude

::: lofi.data.corpora.megaattitude

### Parallel Meaning Bank

::: lofi.data.corpora.pmb

## Batch partitioning

::: lofi.data.sampler

## Corpus paths

::: lofi.data.paths

## Featurizers

::: lofi.data.featurizers

## Dependency structures

::: lofi.data.deps

## Synthetic generation

::: lofi.data.synthetic

## Synthetic training batches

::: lofi.data.synthetic.training
