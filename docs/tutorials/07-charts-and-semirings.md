# Under the hood: charts and semirings

This chapter opens the chart that every parser in chapters 3 to 6 fills: it
indexes the spans of an input, builds the sweep that visits each span after its
parts, and runs that one sweep under two semirings, first counting the binary
trees over the input and then scoring them in log space. It needs only the
base install, since nothing here touches PyTorch.

## Spans and their IDs

A span is half-open: `(i, j)` covers the tokens from `i` up to but not including
`j`. `SpanIndex` computes spans for an input length and assigns their dense IDs.

```python
from lofi.deduce import SpanIndex

index = SpanIndex(5)

print(index.num_spans)
print(index.max_splits)
```

```text
15
4
```

A five-token input has fifteen spans of width one or more. Ids run in order of
increasing width and then increasing start, so the widest span, the one covering
the whole input, is the last:

```python
print(index.span_id(0, 5))
print(index.span_of(14))
```

```text
14
(0, 5)
```

A wider span always has a higher ID than its subspans. Sweeping in ID order
therefore processes every antecedent before the span that uses it.

A split point `k` divides `(i, j)` into `(i, k)` and `(k, j)`, both non-empty:

```python
print(list(index.splits(0, 3)))
```

```text
[1, 2]
```

## The sweep

`binary_sweep` yields each span with each two-part decomposition in this order.
The following function accumulates the sweep through the `Semiring` operations
`zero`, `one`, `plus`, and `times`:

```python
def total(index, semiring):
    """Combine every binary tree over the input, in the given semiring."""
    chart = {}

    for i in range(index.n):
        chart[i, i + 1] = semiring.one

    for cell, left, right in index.binary_sweep():
        combined = semiring.times(chart[left], chart[right])
        chart[cell] = semiring.plus(chart.get(cell, semiring.zero), combined)

    return chart[0, index.n]
```

Every width-one span starts at `one`. Every wider span sums the product of its
two parts over its split points.

## Counting

Hand it `Counting`, whose `plus` is addition and whose `times` is
multiplication over the non-negative integers, and the accumulated value at the
root is the number of binary trees over the input:

```python
from lofi.deduce import Counting, SpanIndex

print(total(SpanIndex(5), Counting()))
print(total(SpanIndex(6), Counting()))
```

```text
14
42
```

These are the corresponding Catalan numbers.

## Scoring

Now hand the same function `LogInside`, whose `plus` is a log-sum-exp and whose
`times` is addition, working in log space:

```python
from lofi.deduce import LogInside, SpanIndex

print(round(total(SpanIndex(5), LogInside()), 6))
```

```text
2.639057
```

This is the logarithm of fourteen. With equal tree weights, the inside score is
the logarithm of the count. Changing lexical or rule weights changes the inside
score without changing the counting result.

Recognition, Viterbi decoding, inside scoring, k-best extraction, and
expectation counting use the same traversal with different algebras. The
`parse.semiring` leaf of a run document, `log-inside` in every run of this
path, selects which algebra a run's chart is evaluated in.

## Try this

- Compare `total(SpanIndex(n), Counting())` with the Catalan number for a few
  more lengths, and note how quickly the count outgrows the lengths the
  tutorial corpora draw.
- Replace `semiring.one` at the width-one spans with a different log weight
  per token under `LogInside`, and check that the root score moves while the
  `Counting` total does not.

## Next

This chapter ends the path. The algebras that ship, and the protocol a new
one implements, are in the [deduce reference](../reference/deduce.md);
checking your own algebra against the semiring laws is
[Running the law suites](../how-to/extending/laws.md); and the decomposition
accessor is described in [The layer order](../explanation/layering.md).
