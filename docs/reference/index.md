# Reference

The reference states what exists: the command line, the entry point groups a
plugin extends, the classes that validate each section of a configuration
document, and the public API of every package. API pages are rendered from the
NumPy-style docstrings in the source and are ordered by the
[layer order](../explanation/layering.md).

## Interfaces

- [Command line](cli.md) lists every `lofi` subcommand and its arguments.
- [Entry point groups](entry-points.md) lists each discovery group and the names
  LoFI registers in it.
- [Configuration documents](configuration/index.md) maps each document section
  onto the declaration classes that validate it.

## Packages

- [Terms](terms.md), [Judge](judge.md), [Lexicon](lexicon.md) and
  [Deduce](deduce.md) document the symbolic core.
- [Interpret](interpret.md), [Type checking](typecheck.md), and
  [Logical forms](lf.md) document the logical form layer, its finite models and
  its provers.
- [Formalism](formalism.md) documents deduction profiles and region protocols.
- [Grammars](grammars.md) documents the built-in grammar declarations.
- [Spec](spec.md) documents boundary values and codecs.
- [Registry](registry.md) documents component discovery.
- [Stats](stats.md) documents inference and multiple comparison procedures.
- [Neural components](nn.md), [Data](data.md), [Probe](probe.md) and
  [Judgment](judgment.md) document model components, inputs, and observations.
- [Train](train.md), [Rewards](rewards.md), [Eval](eval.md) and [Run](run.md)
  document training, proof-grounded payoffs, evaluation and experiment
  execution.
- [Checkpoints](checkpoints.md) specifies the on-disk layout, manifest fields,
  compatibility relation, and completed-step resume behavior.
- [Law suites](laws.md) document exported conformance properties.

A symbol is public when its package exports it. Underscore-prefixed modules are
private and are not included here.
