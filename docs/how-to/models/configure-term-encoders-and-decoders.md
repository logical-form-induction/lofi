# Configure term encoders and decoders

A run represents categories, semantic types and logical forms as vectors
through three linked sections of `model`: `model.encoders` and
`model.decoders` declare named components, and `model.spaces` declares
representation spaces that bind one sort of term to an encoder, an identity
decoder and, optionally, one decoder per binary combinator. This guide
declares a space, chooses its encoder and decoders, relates two spaces, and
shows how each declaration is refused.

The examples compose two checked-in documents: the
[first tutorial's](../../tutorials/01-first-run.md)
[`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml),
which declares one category space, and
[`lofi/configs/run/synthetic-parser-consistency.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/run/synthetic-parser-consistency.yaml),
which adds a semantic-type space and relates the two. The setup they share:

```python
from pathlib import Path

from lofi.spec import ConfigValueError, RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("docs/examples/synthetic-parser.yaml")
```

## The mathematical interface

For a sort $S$, let $\mathcal{T}_S$ be the well-formed terms admitted by the
selected [layout](../../explanation/term-layout-capabilities.md). An encoder
is a learned map $e_{\boldsymbol{\theta}}:\mathcal{T}_S\to\mathbb{R}^{d}$.
A decoder assigns locally normalized choice probabilities and an induced
complete-term weight $w_{\boldsymbol{\phi}}(t\mid\mathbf{h})$. Calling a decoder `identity` names its
reconstruction role; it does not assert
$w_{\boldsymbol{\phi}}(t\mid e_{\boldsymbol{\theta}}(t))=1$.
A binary action decoder conditions on a declared merge
$m(\mathbf{h}_l,\mathbf{h}_r)$, whose domain and output width must match it.

Let $c_1,\ldots,c_m$ be the payload-aware choices in a term's canonical
linearization. For a sequential decoder,

$$
w_{\boldsymbol{\phi}}(t\mid\mathbf{h})
=\prod_{i\in J(t)}p_{\boldsymbol{\phi}}
(c_i\mid c_{<i},\mathbf{h}),
$$

where $J(t)$ contains occupied choices; padding is not scored. A term is
complete when no holes remain. These weights sum to one over admitted terms
only if every positive-weight path completes in an admitted term. Dead ends
and final admission failures can leave missing mass; explicit normalization
over a retained support defines a conditional PMF. The legal alternatives at each step depend on the
partial term and the selected type constraints. A factored choice head uses
$p(c_i\mid\cdot)=p(a_i\mid\cdot)p(c_i\mid a_i,\cdot)$, where $a_i$ is its
operator family. This is a factorization of a choice distribution, not an
additional independence assumption about terms.

Tree decoders pass a parent state to child slots; linear decoders pass the
prefix state; transformers attend to the slots allowed by `structure`.
Thus the conditioning information differs even when these decoders share a
codec. The masked diffusion decoder also exposes canonical-order frontier
and teacher-forced probabilities. Its separately sampled denoising bound is
not interchangeable with that complete-term weight.

A tree encoder recursively computes
$\mathbf{h}_v=f_{\boldsymbol{\theta}}(\mathbf{e}_{c_v},
\mathbf{h}_{v_1},\ldots,\mathbf{h}_{v_k})$ over ordered children.
A linear encoder instead iterates a recurrent cell over the codec order.
A bidirectional encoder combines forward and reverse states by the declared
sum, mean, or concatenation. These operations retain different structural
information; equal widths do not make their maps equivalent.

## Declare a space

The synthetic parser declares one encoder role, `terms`, two decoder roles,
`identity` and `application`, and one space over the category sort `Cat`:

```yaml
model:
  encoders:
    terms:
      width: 4
      layers: 1
      norm_cap: null
  decoders:
    identity:
      width: 4
    application:
      width: 4
  spaces:
    categories:
      sort: Cat
      encoder: terms
      identity: identity
      combinators:
        merge: concatenate
        decoders:
          "application:/": application
          "application:\\": application
```

The kinds come from the `defaults` list, which selects
`model.encoders.terms: rnn`, `model.decoders.identity: rnn` and
`model.decoders.application: rnn`. Reading the space line by line:

| Line | What it says |
| --- | --- |
| `sort: Cat` | The space holds terms of the sort `Cat`, the grammar's categories. A space over semantic types uses `Ty`, and one over logical forms uses `Tm`. |
| `encoder: terms` | The encoder role that maps a term to its vector. |
| `identity: identity` | The decoder role that reconstructs a term from its own vector. |
| `combinators.decoders` | One decoder role per binary action, keyed by the action's name. Here forward and backward application share the `application` decoder. |
| `combinators.merge` | How a combinator decoder receives its two daughter vectors: `concatenate`, or `chart-combine`, which reuses the chart's own combination. |

A space's roles must be declared. A space that names an encoder the model does
not hold is refused at composition:

```python
try:
    codec(document, overrides=("model.spaces.categories.encoder=missing",))
except ConfigValueError as refused:
    print(refused)
```

```text
model was refused: space 'categories' names encoder 'missing'; declared encoders: ['terms']
```

`merge` is declared exactly when `decoders` is nonempty; removing it from this
space is refused with "a combinator family declares a merge exactly for binary
decoders". When the model is built, every binary action of the grammar's rules
must have a decoder and no decoder may name an action that is not binary, so a
grammar that adds forward composition also adds its action, for instance
`composition:/:1:harmonic:unrestricted`, to `combinators.decoders` (the
[configuration guide](../setup/configuration.md) shows that fragment).

## Choose an encoder

Term encoders register under the `lofi.term_encoder` entry-point group. The
packaged fragments under `lofi/configs/model/encoders/terms/` are `rnn`,
`linear`, `hrr` and `transformer`, all at width 256:

| `kind` | Fragment | Reads a term | Own fields |
| --- | --- | --- | --- |
| `tree-rnn` | `rnn` | Bottom up over the codec's slots, each slot's state computed from its children. | `cell`, `layers`, `norm_cap`, `inventory` (`uncached` or `shared-dag`). |
| `linear-rnn` | `linear` | As a recurrence over the codec's linearization order. | `cell`, `layers`, `direction` (`forward`, `bidirectional`), `merge` (required exactly when bidirectional: `mean`, `sum`, or `concatenation`, which doubles the output width), `variables` (`index` or `level`, how bound variables are coded). |
| `hrr` | `hrr` | By binding each address to its constructor's code in a holographic reduced representation. | `magnitude`, `floor`. |
| `transformer` | `transformer` | With attention over the codec's slots. | `heads`, `layers`, `feedforward_width`, `structure`, `activation` (default `gelu`, from the `lofi.activation` group), `norm_cap`. |

Every encoder declares a positive `width`, which is also its output width.
`norm_cap` bounds the norm a constructor or symbol vector is read at, or reads
the vectors as stored when `null`.

A `tree-rnn` or `linear-rnn` encoder's `cell`, like a `linear-rnn` decoder's,
names a plugin in the `lofi.recurrent_cell` group, the same group the subword
pooler's context and the span decoder read: `gru` by default, `lstm`,
`elman-tanh` or `elman-relu`. Each cell steps the state one slot at a time and
names the activations it computes that state from, and a `tree-rnn` publishes
each one as a gate site keyed by layer and name:

| `cell` | Gates | Memory apart from the hidden state |
| --- | --- | --- |
| `gru` | `reset`, `update`, `candidate` | No |
| `lstm` | `input`, `forget`, `candidate`, `output` | Yes |
| `elman-tanh` | `candidate` | No |
| `elman-relu` | `candidate` | No |

A one-layer `tree-rnn` whose cell carries no memory apart from its hidden
state is the only encoder that publishes the synthetic-tree continuation that
soft controller raising needs
([Configure a controller](configure-a-controller.md)). An installed package
adds a cell by registering a `RecurrentCellPlugin` under `lofi.recurrent_cell`;
the plugin builds both the batch-first stack the token components run and the
single-step layer the term components run, so every installed name serves both.
A name that no installed plugin declares is refused when the component is built.

The transformer's `structure` names an entry of the `lofi.transformer_structure`
group, the one family the transformer encoder and decoder both read, and says
which relations of the linearization reach the attention: `positions`,
`positions-with-depth`, `ancestor-mask` (the default) or
`ancestor-and-sibling-bias` (the packaged fragment's choice). The encoder
attends over every slot at once, so it refuses `parent-recurrence` when it is
built. An installed package adds a structure by registering a
`TransformerStructurePlugin` from `lofi.nn.transformer` under that group.

Recurrent encoders use Elman, GRU, or LSTM cells
\citep{elman1990,cho-etal2014,hochreiter-schmidhuber1997}; the modern LSTM
cell includes the forget gate introduced by \citet{gers-etal2000}.
Tree-structured recurrence is a broader family \citep{tai-etal2015};
selecting a tree encoder here does not assert architectural identity with
that paper's Tree-LSTM. The transformer uses attention
\citep{vaswani-etal2017} with the selected structural restrictions.
The HRR equations and their distinction from the original circular binding
are given in [Term-layout capabilities](../../explanation/term-layout-capabilities.md).

Select the packaged transformer for the synthetic parser's `terms` role. The
document's own width of 4 still applies, and the fragment's eight heads do not
divide it:

```python
try:
    codec(document, overrides=("model/encoders/terms=transformer",))
except ConfigValueError as refused:
    print(refused)
```

```text
model.encoders.terms was refused: the heads divide the width they split between them
```

Lower the heads and the feed-forward width with it:

```python
run = codec(
    document,
    overrides=(
        "model/encoders/terms=transformer",
        "model.encoders.terms.heads=2",
        "model.encoders.terms.feedforward_width=8",
    ),
)
print(run.model.encoders["terms"])
```

```text
TransformerTermEncoderSpec(kind='transformer', width=4, heads=2, layers=1, feedforward_width=8, structure='ancestor-and-sibling-bias', activation='gelu', norm_cap=None)
```

The document's `layers: 1` and `norm_cap: null` replace the fragment's
`layers: 4` and `norm_cap: 1.0`, because the run document is composed after
its selected fragments.

## Choose decoders

Term decoders register under the `lofi.term_decoder` entry-point group, and
the packaged library holds the same five fragments under both
`lofi/configs/model/decoders/identity/` and
`lofi/configs/model/decoders/application/`:

| `kind` | Fragment | Reconstructs a term | Own fields |
| --- | --- | --- | --- |
| `tree-rnn` | `rnn` | By recursing from the term's vector down to its slots. | `width`, `type_directed`. |
| `linear-rnn` | `linear` | Autoregressively over the term's linearized choices. | `width`, `cell`, `layers`, `choice_heads`, `tie`. |
| `hrr` | `hrr` | By unbinding each slot against the encoder's codebook. | `choice_heads`; it takes its width and codes from the encoder it is wired to. |
| `transformer` | `transformer` | With attention over the slots it fills. | `width`, `heads`, `layers`, `feedforward_width`, `structure`, `activation` (default `gelu`). |
| `diffusion` | `diffusion` | By denoising every slot at once over a masking schedule. | `width`, `layers`, `feedforward_width`, `noise_steps`, `first_masking` (the fraction the first step masks, rising evenly to one), `activation` (default `gelu`). |

For the masked denoising family, let $\beta_t\in(0,1]$ be the probability
that a slot is masked at step $t$. With $T$ the sampled step, a corruption
mask $\mathbf{M}$ satisfies
$\mathbb{P}(M_i=1\mid T=t)=\beta_t$ independently across slots.
The denoiser conditions on the revealed choices, mask, step, position, and
encoded term. This uses the absorbing-mask idea from discrete diffusion
\citep{austin-etal2021}; LoFI's canonical-order complete-term decoding and
its time-sampled denoising-bound API are distinct scoring operations.

Every decoder declares `choice_heads`: `flat` (the default) reads each slot as
one categorical over the language's choices, and `factored` as a choice of
operator family followed by the member of that family.

The transformer decoder's `structure` names an entry of the same
`lofi.transformer_structure` group; the packaged fragment chooses
`parent-recurrence`. A `linear-rnn` decoder's
`tie` names a `linear-rnn` encoder role whose choice embedding the decoder
shares.

An `tree-rnn` over logical forms can restrict what it decodes by type.
`type_directed.prefix: type-grammar` lets the type grammar fix the decoded
term's outer type, admitting only choices whose partial term still has a
completion of the required type, and `type_directed.mask: known` lets known
type declarations restrict the decoded terms; both default to `none`, which
admits every choice. The L1 run document,
[`experiments/l1/l1-run.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/l1/l1-run.yaml),
declares both for its `form-identity` decoder:

```yaml
form-identity:
  kind: tree-rnn
  width: 4
  type_directed: {prefix: type-grammar, mask: known}
```

A `type-grammar` prefix requires the language to declare a truth type, and
decoding without one is refused with "type-directed term decoding requires a
declared truth type".

## Relate two spaces

A space may declare one `relation` to another space. The consistency run adds a
`semantic-space` over `Ty` with its own encoder and decoders, and relates the
category space to it:

```python
consistency = Path("lofi/configs/run/synthetic-parser-consistency.yaml")
model = codec(consistency).model
for role, encoder in model.encoders.items():
    print("encoder", role, encoder.kind, encoder.width)
for role, decoder in model.decoders.items():
    print("decoder", role, decoder.kind)
for name, space in model.spaces.items():
    print(
        name,
        space.sort,
        space.encoder,
        space.identity,
        space.combinators.merge,
        space.relation,
    )
```

```text
encoder semantic-types tree-rnn 4
encoder terms tree-rnn 4
decoder application tree-rnn
decoder identity tree-rnn
decoder type-application tree-rnn
decoder type-identity tree-rnn
categories Cat terms identity concatenate SpaceRelationSpec(space='semantic-space', via=<SpaceRelation.CONSISTENCY: 'consistency'>, mapping=None)
semantic-space Ty semantic-types type-identity concatenate None
```

`via` names the relation:

| `via` | What it says |
| --- | --- |
| `consistency` | The same vector is read by both identity decoders, so the spaces require equal widths. The decoded distributions are compared through the interpretation’s erasure. |
| `erase-then-decode` | The relation holds between decoded terms: a category term is carried into the related language by the erasure. It has no vector projection. |
| `mapped` | A learned map sends a vector of this space to a vector of the related one. It requires a `mapping` and holds no erasure. |

For a declared category-to-type erasure $\epsilon$, consistency decodes the
same vector $\mathbf{h}$ in both spaces, pushes the source distribution through
erasure, and compares the resulting term distributions:

$$
q(\tau\mid\mathbf{h})=\sum_{C:\epsilon(C)=\tau}p_{\mathrm{Cat}}(C\mid\mathbf{h}),
\qquad
\ell_{\mathrm{consistency}}=
D\!\left(q(\cdot\mid\mathbf{h}),p_{\mathrm{Ty}}(\cdot\mid\mathbf{h})\right).
$$

Here $D$ is the configured coherence kernel, with the support and residual
policies described in [Induction objectives](../../explanation/induction-objectives.md).
This does not compare two encoder vectors directly. A learned mapping
$f:\mathbb{R}^{d_1}\to\mathbb{R}^{d_2}$ instead projects the source vector
before the target decoder reads it; its existence does not establish
preservation of symbolic typing.

A `mapping` declares `form: linear`, or `form: multilayer` with `hidden`,
`depth` and `dropout`. The packaged
`lofi/configs/run/synthetic-parser-mapped.yaml` refines the consistency run
with a linear map:

```yaml
defaults:
  - synthetic-parser-consistency
model: {spaces: {categories: {relation: {via: mapped, mapping: {form: linear}}}}}
trainer:
  active_groups+: [categories]
```

The run also names the space's own role, `categories`, among the trainer's
active groups. A `mapped` relation without a mapping is refused:

```python
try:
    codec(
        document,
        overrides=(
            'model.spaces.categories.relation={"space": "categories", "via": "mapped"}',
        ),
    )
except ConfigValueError as refused:
    print(refused)
```

```text
model.spaces.categories.relation was refused: only a mapped space relation declares one mapping
```

## See also

- [Configure readouts](configure-readouts.md) to decode terms from a space.
- [Combine losses](../training/combine-losses.md) for the coherence and
  reconstruction terms that train a space.
- [Term layout capabilities](../../explanation/term-layout-capabilities.md)
  for what each encoder and decoder assumes of a term codec.
- The [model configuration reference](../../reference/configuration/model.md)
  for every encoder and decoder field.

## References

\bibliography
