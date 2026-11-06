# Documents are the declaration language

A run document records the choices for one training run: its data, grammar,
model, and training procedure. It is a configuration file, usually written in
[YAML](../how-to/setup/configuration.md). LoFI checks the document, constructs the requested
components, and records the resolved choices beside the results.

## The document is the program

Everything a LoFI point does is declared in one run document: the grammar that
generates or reads the data, the logical form language, the model components,
the [loss terms](induction-objectives.md#combining-terms), the trainer, the evaluators and the sinks. An experiment
document adds a sweep, a [preregistration](experiments-and-commitments.md#preregistration) and the reports to produce. There is
no other place a choice can be made. A capability that cannot be reached from a
document cannot be compared in an experiment, so the documents are the
declaration language, and their schema is the language's grammar.

That schema is the [`RunSpec`](../reference/spec.md#lofi.spec.RunSpec) model and the declaration
classes it reaches. It is open: installing a plugin adds variants to the
families whose slots the document fills (see [Swappable grammatical
assumptions](swappable-assumptions.md)). It is also closed in the sense that
matters for validation: every declaration class forbids unknown fields, so a
document can only say what an installed declaration declares.

## Composition

Suppose a document sets [`trainer.epochs`](../how-to/training/configure-a-trainer.md) to 1, a selected profile sets it to
2, and the command line includes `trainer.epochs=3`. The resolved epoch
setting is three. The last applicable setting wins; the record of where settings
came from retains all three. This is why the order of composition matters.

A document is rarely written whole. Its [`defaults` list](../how-to/setup/configuration.md) selects packaged
fragments for configuration slots, and composition merges, in ascending
precedence, the document with its selected fragments, an optional named
profile, overlay documents, and `key=value` overrides from the command line.
[`SpecLoader`](../reference/spec.md#lofi.spec.SpecLoader) performs the merge and validates the
result once, against the schema, as a whole. An unknown path or a value of the
wrong type is refused before any model or data loader is constructed, and the
refusal names the leaf, or the document when it is refused as a whole. Overrides are typed
against the schema before they are applied, so `trainer.epochs=2` sets an
integer and a group override such as `model/encoders/terms=transformer`
selects a fragment rather than setting a string.

Writing $\oplus$ for the loader's precedence-aware merge, composition has
the schematic form

$$
\operatorname{validate}\!\left(d_0\oplus d_1\oplus\cdots\oplus d_k\right).
$$

The ordered $d_i$ are the selected declaration layers. This is a partial
operation: invalid paths, incompatible variants, or failed invariants cause
refusal. Merge is order-sensitive: exchanging two layers can change the
resolved value. Provenance records the winning source
and overridden sources at each path.

The [configuration guide](../how-to/setup/configuration.md) describes each
layer and how to write it.

## Provenance

Composition records where every value came from. For each dotted path that some
layer set, the traced composition holds the source that won and every source
that was overruled, lowest precedence first. A run directory writes the
composed document as `config.resolved.json` and the provenance beside it as
`config.sources.json`, and [`lofi show-config`](../reference/cli.md#show-config) prints both. Provenance answers
the question that a merged document by itself cannot: whether a value is the
document's own, a fragment's default, a profile's setting, or an override typed
at the command line.

Two compositions that reach the same resolved document through different
files describe the same point; their provenance records retain the different
sources.

## Point identity

A point is identified by [`point_id`](../reference/run.md#lofi.run.point_id): the first sixteen
hexadecimal digits of the SHA-256 of canonical JSON over the resolved
document's scientific sections and the seed. Canonical JSON sorts keys and
fixes separators, so the order in which keys were written does not matter.
The scientific sections are every section except the observation sections the
run specification declares, which are currently `sinks`. Adding a [metric sink](../how-to/experiments/track-runs.md#declare-a-sink)
changes where a point's metrics go and nothing it computes, so the same run
with and without a sink is one point.

A hash turns the serialized settings into a compact identifier, much as a
catalog number lets a reader locate a longer record. To establish which
settings were used, inspect the resolved document itself. The hash helps
detect changes and locate results, with possible collisions between distinct
documents because its length is finite.

Identifying a point by its resolved document has three consequences.

1. **Spelling does not matter.** A value set in a fragment, a profile or an
   override yields the same point as the same value written in the document.
   Refactoring a configuration library without changing any resolved value
   leaves every point identifier unchanged.
2. **Scientific differences enter the hash.** A difference in any scientific
   leaf, a default included, changes the serialized input. The identifier is
   a truncated hash, so distinct inputs can collide.
3. **Records can be checked.** An experiment's ingestion recomputes each
   manifest's point identifier from its resolved document and refuses a
   manifest whose directory name disagrees, or whose point lies outside the
   frozen sweep (see [Experiments and
   commitments](experiments-and-commitments.md)).

The seed enters the identifier but is not part of the configuration. A point
also records the experiment seed its run seed was derived from, and running the
same point under a different root is refused rather than mixed into the same
records.

## The boundary fingerprint

A resolved document is interpreted by a schema and by the substrate packages
that validate it. The same document validated against a different set of
installed declaration classes, or a different version of the configuration
substrate, may mean something different. A run thus also records a
[`RunFingerprint`](../reference/spec.md#lofi.spec.RunFingerprint) of its configuration boundary. The
fingerprint is canonical JSON holding the schema URI of every model the run
specification reaches, with every installed plugin loaded first; every indexed
language with its theory version, specification, digest and lowering route;
and the installed versions of `didactic`, `didactic-settings`, `lairs` and
`panproto`. Its digest is the SHA-256 of that document.

To check the configuration boundary, [`require_unchanged`](../reference/spec.md#lofi.spec.require_unchanged)
compares a committed fingerprint with a fresh one and raises
[`BoundaryMismatchError`](../reference/spec.md#lofi.spec.BoundaryMismatchError) naming every member that differs. A committed run also
stores the exact bytes of its resolved configuration and its fingerprint record,
and identifies each by the SHA-256 of the stored bytes; a record whose bytes no
longer hash to the identifier it was committed with is refused as
[`TamperedRecordError`](../reference/run.md#lofi.run.TamperedRecordError). Nothing is identified by re-encoding what the bytes
decode to.

Grammars and languages carry fingerprints of their own, computed from their
compiled parts (see [Grammars as deductive systems](derivations-as-proofs.md)).
Those enter a point's identity through the resolved document that declares
them.

## What this does not establish

Matching point identifiers provide a check on declarations, subject to hash
collisions; equality can be established by comparing the resolved scientific
documents. Numerical reproducibility also depends on floating-point reductions,
nondeterministic kernels, and hardware. A run directory records the accelerator
runtime separately from the fingerprint and point identifier so that differences
between executions of the same declaration can be traced.
