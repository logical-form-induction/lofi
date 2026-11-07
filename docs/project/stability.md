# Stability

LoFI is pre-1.0. A minor release, such as a change from 0.1 to 0.2, may change
the arguments, return types, or import paths of public interfaces. Projects
that need repeatable runs should record the package version as well as the
[resolved configuration and seed](../explanation/documents-are-the-dsl.md#point-identity).

## Supported API

A package's `__init__` exports its supported surface, which is rendered in the
[API reference](../reference/index.md). Direct imports from underscore-prefixed
modules are private; names reexported by a package `__init__` are public.

The **[public API](../reference/index.md#reading-a-python-api-entry)** is the set of interfaces intended for callers outside the
package. For instance, import
[`SpanIndex`](../reference/deduce.md#lofi.deduce.SpanIndex) from [`lofi.deduce`](../reference/deduce.md#lofi.deduce)
through its public export. An underscore-prefixed implementation module may
move when the package is reorganized. Public exports identify supported
interfaces, though their signatures may still change under the version policy
above.

## Current implementation

The symbolic core includes [terms and the term store](../reference/terms.md#lofi.terms), [rules and rule sets](../reference/judge.md#lofi.judge),
[lexical entries and sources](../reference/lexicon.md#lofi.lexicon), [spans, and semirings](../reference/deduce.md#lofi.deduce). It requires only didactic and
panproto, together with didactic-settings, as [Python runtime dependencies](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml).
These packages manipulate explicit expressions and rules without requiring
a neural tensor framework. The distribution also includes the compiled Rust
extension; source installations need the toolchain described in
[installation](../how-to/setup/installing.md).

Above it, [`lofi.interpret`](../reference/interpret.md#lofi.interpret), [`lofi.formalism`](../reference/formalism.md#lofi.formalism), [`lofi.spec`](../reference/spec.md#lofi.spec), [`lofi.registry`](../reference/registry.md#lofi.registry),
[`lofi.stats`](../reference/stats.md#lofi.stats), [`lofi.nn`](../reference/nn.md#lofi.nn), [`lofi.data`](../reference/data.md#lofi.data), [`lofi.probe`](../reference/probe.md#lofi.probe), [`lofi.judgment`](../reference/judgment.md#lofi.judgment),
[`lofi.train`](../reference/train.md#lofi.train), [`lofi.eval`](../reference/eval.md#lofi.eval), [`lofi.run`](../reference/run.md#lofi.run), [`lofi.typecheck`](../reference/typecheck.md#lofi.typecheck), and [`lofi.lf`](../reference/lf.md#lofi.lf) have
public APIs. [`lofi.grammars`](../reference/grammars.md#lofi.grammars) exports the built-in CCG declarations; grammar
subpackages expose additional declarations.

## Structural constraints

The following constraints are structural:

**The base install remains small.** Its [Python runtime dependencies](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml) are
didactic, didactic-settings, and panproto. A dependency is another package
needed to use the installed capability. [Named extras](../how-to/setup/installing.md#add-an-extra)
request additional capabilities and their dependencies, such as `nn` for
neural models. The base dependencies suffice for the symbolic core.

**The [symbolic core remains independent](../explanation/layering.md#the-order).** Modules from [`lofi.terms`](../reference/terms.md#lofi.terms) through
[`lofi.grammars`](../reference/grammars.md#lofi.grammars) do not import tensor libraries, corpus readers, or the
logical form packages.

**A probe reaches a model through published surfaces.** `lofi.nn.sites`,
`lofi.nn.ports` and `lofi.nn.policy` define that interface.

A [probe](../explanation/probing-through-sites.md#observations-and-interventions)
observes or intervenes on a model through quantities the model deliberately
exposes. A [site](../explanation/probing-through-sites.md#a-site-is-a-published-promise)
identifies such a quantity and its declared properties. The interfaces are
documented under [sites](../reference/nn.md#sites),
[ports](../reference/nn.md#ports), and [policy](../reference/nn.md#policy);
their source is in
[`lofi/nn/sites.py`](https://github.com/logical-form-induction/lofi/blob/main/lofi/nn/sites.py),
[`lofi/nn/ports.py`](https://github.com/logical-form-induction/lofi/blob/main/lofi/nn/ports.py),
and [`lofi/nn/policy.py`](https://github.com/logical-form-induction/lofi/blob/main/lofi/nn/policy.py).
This contract lets the model change its private storage while keeping the
declared observations available to probes.

## The law suites

Implementations of a LoFI [protocol](../how-to/extending/plugins.md#register-a-component) should pass its law suite. Changes to those
laws are treated as compatibility changes.

A protocol specifies the operations required of an implementation. Its
[law suite](../reference/laws.md) tests shared relationships among those
operations on supplied or generated inputs. Passing those tests provides
evidence of conformance on the tested inputs; proving a law requires an
argument covering its full domain. The
[extension guide](../how-to/extending/laws.md) shows how to run the checks.

## Versioning

Until 1.0, a minor version may break a signature. Check the
[law suites](../reference/laws.md) when updating an implementation.
