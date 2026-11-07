# Boundaries

Validation checks a value before code relies on it. A negative count, for
instance, should be rejected where it is supplied, with an error naming that
count. Discovering it later while allocating an array would make the cause
harder to identify. LoFI performs these checks at the points where
[declarations become runtime values](documents-are-the-dsl.md#composition).

## Validation happens once, at a named boundary

A value is validated when it enters the system, and interior code then relies
on the invariant that validation established instead of checking it again.
Revalidating in an inner loop would cost time on every step and would also
blur which layer is responsible for a refusal. A refusal raised at the
boundary names the leaf of the document that caused it; the same failure
discovered deep inside a training step would name a tensor.

If $X$ is the input domain and $I:X\to\{0,1\}$ is a checked invariant,
validation admits values in $X_I=\{x\in X:I(x)=1\}$. An interior operation
$f:X_I\to Y_J$ may rely on $I$ and must preserve its promised output
invariant $J$, that is, $J(f(x))=1$ for admitted inputs. Proving this contract
requires an argument covering every admitted input. Schema validation,
finite tests, and conditional debug checks establish properties of the values
or executions they inspect and can expose violations.

An invariant is a condition that must continue to hold. Here $I(x)=1$ means
that input $x$ passes the condition, and $X_I$ collects the inputs that pass.
The notation $f:X_I\to Y_J$ says that the operation accepts those inputs and
promises outputs satisfying $J$. For instance, a range check establishes that
a probability lies between zero and one. A collection of probabilities also
needs a normalization check if later code relies on their sum being one.

## Where declared values enter

[`lofi.spec`](../reference/spec.md#lofi.spec) holds the run specification, component
wiring and their codecs. Each swappable component family has one tagged union
root, and a document's slots are typed by those roots, while each variant's
declaration class lives with its implementation's package and registers
through an [entry point](swappable-assumptions.md#how-a-family-is-discovered) (see [Swappable grammatical
assumptions](swappable-assumptions.md)). Composition validates the whole
document once, as a [`RunSpec`](../reference/spec.md#lofi.spec.RunSpec) (see [Documents are the declaration
language](documents-are-the-dsl.md)).

Typed boundary data uses [didactic](https://github.com/panproto/didactic) over
panproto. A didactic model is affordable at a run boundary but too costly to
construct in an inner loop, so it is never used in training steps, dataset
items, collate functions, chart cells, term constructors, loss calls, beam
steps or probe inner loops. A boundary model is converted into frozen interior
values and is not retained below its boundary. An import contract restricts
didactic and panproto to the boundary modules (see [The layer
order](layering.md)).

## Two kinds of check

**Field validation is unconditional.** Reported numbers are checked for
finiteness, counts and indices for nonnegativity, and identifiers for valid
names, whenever a value is constructed. An invalid type raises `TypeError` and
an invalid value raises `ValueError`. A layer defines a more specific error
when callers need more information about the failure, as
[`ProfileUnsupportedError`](../reference/formalism.md#lofi.formalism.ProfileUnsupportedError) and [`UnknownSettingError`](../reference/spec.md#lofi.spec.UnknownSettingError) do.

**Invariant checking is conditional** and off by default. Properties that are
expensive to establish, and that the code maintains by construction, are
checked only when the `LOFI_CHECK` debug checks are enabled. See [The debug
checks](../how-to/extending/checks.md).

## `assert` is never validation

Validation raises an exception explicitly. The test suite also runs under
`python -O`, which removes `assert` statements, so a check written as an
`assert` would pass silently in that configuration.

## Errors are not control flow

A rule that does not apply to its premises returns the protocol's
non-applicable value, and
a prover that cannot decide a question returns an abstention with its cause.
Data that is dropped is reported, either as an error or as a counted outcome,
and is never discarded silently. A loss function always returns a tensor; a
term with nothing to score in a batch reports a count of zero and is left out
of the weighted sum.

An empty set of observations also differs from an observed cost of zero.
The latter can lower an average; the former supplies no evidence to average.
[Objective denominators and empty cases](induction-objectives.md#combining-terms)
make that distinction explicit in training.
