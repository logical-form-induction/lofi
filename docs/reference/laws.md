# Law suites

A law suite tests whether an implementation satisfies a contract. A
[term codec](terms.md#lofi.terms.TermCodec), for instance, should recover the
same valid term after encoding and decoding it when it promises that round trip.
Property-based tests generate inputs to search for violations. Their evidence
covers the inputs exercised by the finite suite; a universal claim requires
a proof over the full input domain.

Hypothesis is the Python library that generates these cases. The optional
`laws` installation extra supplies the testing dependencies; the
[law-suite guide](../how-to/extending/laws.md) shows how to run them when
adding or changing an implementation.

This page renders the exported law modules, which check [protocol implementations](index.md#reading-a-python-api-entry) and [package boundaries](../explanation/layering.md#narrower-contracts).
Install the `laws` extra for the property-based suites. See [Running the law
suites](../how-to/extending/laws.md).

Property law modules depend on Hypothesis and remain outside runtime package
imports. [`lofi.run`](run.md#lofi.run) reexports its source tree laws, which do not use
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
