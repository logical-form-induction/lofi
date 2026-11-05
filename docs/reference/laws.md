# Law suites

This page renders the exported law modules, which check protocol implementations and package boundaries.
Install the `laws` extra for the property-based suites. See [Running the law
suites](../how-to/extending/laws.md).

Property-law modules depend on Hypothesis and remain outside runtime package
imports. `lofi.run` re-exports its source-tree laws, which do not use
Hypothesis.

## Law violation

Every suite raises one error, which `lofi` exports.

::: lofi.LawViolationError

## Terms

::: lofi.terms.laws

## Judge

::: lofi.judge.laws

## Lexicon

::: lofi.lexicon.laws

## Deduce

::: lofi.deduce.laws

## Interpret

::: lofi.interpret.laws

## Typecheck

::: lofi.typecheck.laws

## Data

::: lofi.data.laws

## Stats

::: lofi.stats.laws

## Eval

::: lofi.eval.laws

## Probe

::: lofi.probe.laws

## Train

::: lofi.train.laws

## Neural loss

::: lofi.nn.loss.laws

## Run

::: lofi.run.laws
