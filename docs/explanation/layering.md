# The layer order

This page explains how LoFI's packages are ordered, which imports the order
permits, and which narrower contracts forbid imports the order alone would
allow. It also says what each contract protects, since each one exists so that
a comparison between declarations cannot be confounded by coupling between
their implementations.

## The order

LoFI is one distribution whose top-level packages sit in nineteen layers. A
layer may import any layer below it and nothing above it. Names joined by a
pipe are independent siblings: they sit at one level and may not import each
other. From the top:

```text
lofi.run                                    run orchestration, manifests, telemetry, the command line
lofi.report                                 report columns and table renderers
lofi.train | lofi.eval                      training interfaces | evaluation procedures
lofi.training_plugins                       trainer planning, construction and pricing plugins
lofi.data | lofi.probe | lofi.judgment      corpora | probes | the response model
lofi.nn                                     encoders, decoders, controllers, parsers, losses
lofi.stats                                  p-values, null models, dependence, corrections
lofi.registry                               component discovery
lofi.spec                                   run specifications, wiring, codecs
lofi.control_variate | lofi.mixing | lofi.reward   control variates | multi-step mixing | declared payoffs
lofi.grammars                               cfg, ab, ccg, tlg
lofi.formalism                              grammar declarations, profiles, budgets, regions
lofi.interpret | lofi.typecheck | lofi.lf   readings | typing | the logical-form language
lofi.deduce                                 spans, semirings, charts, the engine, outside
lofi.lexicon                                entries and lexical sources
lofi.judge                                  rules, rule sets, normal forms, reports
lofi.terms                                  sorts, operators, signatures, terms, the store
lofi._frozen                                frozen interior values
lofi._types | lofi._checks | lofi._alignment | lofi._stats   shared values and checks
```

The [import-linter](https://import-linter.readthedocs.io/) layers contract in
`.importlinter` checks this order and is exhaustive: a new top-level package
fails the check until it is placed. In import-linter's own syntax a colon joins
siblings that *may* import each other, the opposite of a pipe.

Let $V$ be the set of top-level packages, and let $(u,v)\in E$ mean that
package $u$ imports package $v$. A layer assignment
$\lambda:V\to\{0,\ldots,18\}$ numbers layers from bottom to top. The stated
order requires $\lambda(v)<\lambda(u)$ for an edge between distinct
packages; same-layer siblings in this contract have no edges between them.
Narrower contracts remove additional edges from this permitted set. This
constraint makes the package level import graph acyclic, but establishes no
claim about the statistical independence of two components' outputs.

## Narrower contracts

Further contracts forbid specific imports the layer order would otherwise
allow.

**The lexicon does not import the deduction engine.** An entry pairs a form
with a category and a meaning and names no span. The axiom edge built from an
entry belongs to the engine.

**The engine imports no formalism and no grammar pack.** `lofi.deduce` depends
on the `Rule` protocol rather than a concrete inventory, so one engine serves
every formalism under comparison. Readings and rendering in `lofi.interpret`
likewise import no grammar pack; a derivation style reads a rule's name and an
injected symbol table rather than dispatching on a combinator's identity.

**The induction half does not import the logical form half.** The layer order
would permit an import of `lofi.typecheck` by `lofi.formalism`; a narrower contract forbids
terms, rules, the lexicon, the engine, formalisms, grammar packs and
interpretation from importing `lofi.typecheck` or `lofi.lf`. The one exception
is the meaning reading, `lofi.interpret._meaning`, which a separate contract
keeps above the rest of interpretation so that nothing else in that layer
imports it. Interpretation, typing and the logical form language are mutually
independent apart from that edge, so a grammar can be compared with typing
switched off.

**The symbolic core imports no tensor library.** Terms, rules, the lexicon,
the engine, interpretation, formalisms and grammar packs import none of
`torch`, `numpy`, `quivers`, `lairs` or `bead` at any depth, and only `lofi.terms`
and `lofi.deduce` reach the native extension.

**Validation libraries stay at the boundary.** `didactic` and `panproto` are
named only by the boundary modules (`lofi.spec`, `lofi.registry`, the
`spec` modules of grammar packs, corpus readers and synthetic generators, and
the `_boundary` modules of training, evaluation, runs and probes), plus a small
listed set.
[Boundaries](boundaries.md) explains why.

**Application libraries stay in the run layer.** Telemetry, progress and
command line libraries are imported only by `lofi.run`, so importing the
library never starts a client or a user interface. The Hugging Face Hub client
is imported only by the optional Hub module.

**A probe reaches a model through published surfaces only.** Probing imports
`lofi.nn.sites`, `lofi.nn.ports` and `lofi.nn.policy` and nothing else under
`lofi.nn`. The contract is an allowlist of those public modules, since a
denylist of private paths is defeated by importing a public module beside
them.

**Implementations of one protocol are independent.** Type encoders, combinator
decoders, grammar packs and corpus readers do not import their peers, and no
type encoder imports a combinator decoder. A family that reached into its
neighbor for a live instance would run only beside that neighbor, and a
comparison between the two would then measure the coupling. No library module
imports a script.

## Why the corpus readers are under `lofi.data`

Corpus readers live at `lofi.data.corpora.*`; there is no `lofi.corpora`.
Nothing below the data layer imports a corpus, and a corpus reader imports no
neural, deduction or judge package, so the corpus interface stays independent
of any one source, CCGbank included.

Only corpus-specific readers, codecs and declarations live under
`lofi.data.corpora`. The corpus-independent vocabulary a reader emits, along
with datasets and path values, lives directly under `lofi.data`. The reader
independence contract is a glob over the package, so it needs no exceptions as
long as every child of that package is a reader.

To add a package or a contract, see the [contributor
guide](../development/index.md).
