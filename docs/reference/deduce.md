# Deduce

A [chart](../explanation/charts-and-approximation.md#spans-and-the-sweep) stores
analyses of portions of a sentence so larger analyses can reuse them. For a
two-word input, [`SpanIndex`](#lofi.deduce.SpanIndex) addresses the two one-word
spans and the [span](../explanation/charts-and-approximation.md#spans-and-the-sweep) covering both. Rules combine the available analyses within
those regions.

The [semiring](../explanation/charts-and-approximation.md#what-a-semiring-decides)
selects how alternatives contribute: a counting parse counts derivations,
whereas a Viterbi parse retains the largest derivation score. The
[semiring tutorial](../tutorials/07-charts-and-semirings.md) runs these
calculations through the API.

This page renders [`lofi.deduce`](#lofi.deduce): span indexing, semirings, charts, the deduction engine and outside scores.

::: lofi.deduce
