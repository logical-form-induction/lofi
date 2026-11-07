# Run the law suites

A law suite checks that an implementation of a LoFI protocol has the behavior
the protocol requires, over cases drawn with
[Hypothesis](https://hypothesis.readthedocs.io/). This guide finds the
installed suites, checks a [semiring](../../explanation/charts-and-approximation.md#what-a-semiring-decides) against its laws, runs one law alone, and
adds a conformance test to a plugin's own suite.

## Install and discover the suites

The suites need the `laws` extra:

```bash
uv pip install "lofi[laws] @ file://$PWD"
```

Each suite is a module, `lofi.<layer>.laws` for most layers and
[`lofi.nn.loss.laws`](../../reference/laws.md#lofi.nn.loss.laws) for the losses. No runtime module imports a suite, so the
base install does not need Hypothesis. [`lofi.run.laws`](../../reference/laws.md#lofi.run.laws), which checks
source tree boundaries, needs no Hypothesis and is reexported by [`lofi.run`](../../reference/run.md#lofi.run).
The `lofi.laws` [entry point group](../../reference/entry-points.md#registered-groups-and-names) names every installed suite under a
capability name, so a plugin discovers them without importing a runtime
package or keeping a module list:

```python
from importlib.metadata import entry_points

print(sorted(point.name for point in entry_points(group="lofi.laws")))
```

```text
['data', 'deduction', 'evaluation', 'interpretation', 'judgment', 'lexicon', 'neural-loss', 'probe', 'run', 'statistics', 'terms', 'training', 'typing']
```

`point.load()` imports the suite. The combinatory rule families publish their
cases through a second group, `lofi.rule_laws`, with one entry per family:
`composition`, `substitution`, `raising` and `pattern`. Each case declares a
small grammar and checks the rule the native grammar compiler builds from it,
so the laws hold of the rules a practitioner's grammar declaration gets.
## Check a semiring

A generated property test checks a stated relation over sampled inputs.
For associativity, it compares $(a+b)+c$ with $a+(b+c)$ for generated weights
$a,b,c$. The implementation supplies the operations, the fixture supplies
admissible inputs, and the equality predicate determines how the results are
compared. The [semiring explanation](../../explanation/charts-and-approximation.md#what-a-semiring-decides)
states why these laws matter when a chart combines alternatives.

A suite exports its individual laws and a fixture. Most suites also export a
tuple of their laws and an aggregate checker; a smaller suite may export only
one of the two. [`SemiringFixture`](../../reference/laws.md#lofi.deduce.laws.SemiringFixture) takes the algebra, a strategy for drawing
weights, an optional equality predicate and an example budget:

```python
import hypothesis.strategies as st

from lofi.deduce import Counting
from lofi.deduce.laws import SemiringFixture, check_semiring_laws

fixture = SemiringFixture(
    semiring=Counting(),
    values=st.integers(min_value=0, max_value=1000),
    max_examples=50,
)
print(check_semiring_laws(fixture))
```

```text
None
```

A check that passes returns `None`. A violation raises [`LawViolationError`](../../reference/laws.md#lofi.LawViolationError),
which `lofi` exports and every law suite raises, naming the law and the case.
An equality predicate that no semiring can satisfy shows the form:

```python
from lofi import LawViolationError

broken = SemiringFixture(
    semiring=Counting(),
    values=st.integers(min_value=0, max_value=1000),
    equal=lambda left, right: left == right + 1,
    max_examples=50,
)
try:
    check_semiring_laws(broken)
except LawViolationError as violation:
    print(f"{type(violation).__name__}: {violation}")
```

```text
LawViolationError: plus is not commutative on 0 and 0
```

A floating-point algebra takes a tolerant `equal`, since log space addition
is not associative to the last bit. `max_examples` sets the search budget.

## Run one law

The tuple names each law, and each law runs on the fixture alone:

```python
from lofi.deduce.laws import SEMIRING_LAWS, times_distributes_over_plus

print([law.__name__ for law in SEMIRING_LAWS])
times_distributes_over_plus(fixture)
```

```text
['plus_is_a_commutative_monoid', 'times_is_a_monoid', 'times_distributes_over_plus', 'zero_annihilates']
```

Some suites check a package implementation instead of one passed in.
[`SpanIndexFixture`](../../reference/laws.md#lofi.deduce.laws.SpanIndexFixture) sets the range of input lengths the span index is checked
over:

```python
from lofi.deduce.laws import SpanIndexFixture, check_span_index_laws

check_span_index_laws(SpanIndexFixture(max_n=16, max_examples=25))
```

## Add a conformance test

A failing generated case gives a concrete input on which the implementation
violates the checked property. A passing result is evidence over the generated cases and is limited by
the input range, equality criterion, and example budget. Keep those settings
visible when interpreting the result.
The [law API](../../reference/laws.md) exposes the fixtures and individual
checks used by a plugin's conformance test.

A plugin's test calls the aggregate checker under the `conformance` marker:

```python
import pytest


@pytest.mark.conformance
def test_my_semiring_satisfies_the_laws():
    check_semiring_laws(my_fixture)
```

The suites cover terms, rules, lexical sources, deduction, interpretation,
typing, data, statistics, evaluation, neural losses and parser outputs,
probes, training and run metric sinks. The [law suite
reference](../../reference/laws.md) lists every export, and the [plugin
guide](plugins.md) covers registering the implementation the laws check.
