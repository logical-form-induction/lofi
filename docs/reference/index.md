# Reference

Use this reference to check a command, a configuration field, or a Python
operation. The [configuration tables](configuration/index.md) connect scientific
choices to document keys. The [package entries](#packages) specify the inputs,
outputs, and errors of the Python objects that implement those choices.

For a first run, follow the [tutorial](../tutorials/01-first-run.md). For the
mathematical objects those interfaces manipulate, start with
[Categories, semantic types, and logical forms](../explanation/logical-forms-as-types.md).

The reference states what exists: the command line, the entry point groups a
plugin extends, the classes that validate each section of a configuration
document, and the public API of every package. API pages are rendered from the
NumPy-style docstrings in the source and are ordered by the
[layer order](../explanation/layering.md).

## Reading a Python API entry

A Python API is the set of objects a program can use directly. A *class*
defines a kind of object; a *function* takes inputs and returns a result;
a *method* is an operation attached to a class or object. The displayed
signature lists arguments and their type annotations. `Parameters` explains
the inputs, `Returns` explains the result, and `Raises` names conditions that
produce an error. For instance, [`SpanIndex`](deduce.md#lofi.deduce.SpanIndex)
is a class for addressing sentence spans; the [chart tutorial](../tutorials/07-charts-and-semirings.md)
constructs one with `SpanIndex(5)` for a five-token input.

| Notation | How to read it |
| --- | --- |
| `str`, `int`, `float`, `bool` | Text, an integer, a floating-point number, or a true/false value. |
| `list[T]`, `Sequence[T]` | A list, or a more general sequence, whose elements have type `T`. |
| `Mapping[K, V]` | Keys of type `K` associated with values of type `V`. |
| `tuple[A, B]` | Two ordered positions, with types `A` and `B`. |
| `T \| None` | Either a value of type `T` or the Python absence value `None`. |
| `name: T = value` | An argument with type `T` and the displayed default when omitted. |
| `*` in an argument list | Arguments after it must be supplied by name. |
| `/` in an argument list | Arguments before it must be supplied by position. |
| `-> T` | The operation returns a value of type `T`. |

An argument can be omitted when its signature supplies a default. A type
annotation containing `None` permits that value as an argument. Configuration
files use `null` for absence. A protocol, such as [`Rule`](judge.md#lofi.judge.Rule), specifies the operations
an object must provide; a concrete implementation supplies those operations. Declaration
classes describe settings, whereas built runtime objects carry out computations.

For numerical arrays, read the documented shape together with the axis names.
An `(item, span, width)` array has one dimension for observations, one for
spans, and one for representation coordinates. Check the documented value
convention separately, including any probability normalization.
[Reading the notation](../explanation/reading-the-notation.md)
explains the separate mathematical notation used in definitions.

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
  [Logical forms](lf.md) document the [logical form](../explanation/logical-forms-as-types.md#the-correspondence) layer, its finite models and
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
