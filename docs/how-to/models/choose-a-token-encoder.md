# Choose a token encoder

A run's `model.token_encoder` declares how each word of an item becomes a row
of features, and `model.subword_pooler` declares how an encoder that splits
words into subwords reads one row per word back off them. This guide selects
each of the four packaged encoders (`static`, `learned`, `pretrained` and
`llm2vec`), sets the options each one adds, and shows how a declaration is
refused.

The examples compose
[`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml)
with a [`RunSpecCodec`](../../reference/spec.md), as the
[first tutorial](../../tutorials/01-first-run.md) does, and print the composed
declaration. The setup they share:

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("docs/examples/synthetic-parser.yaml")
```

## Token and word representations

For vocabulary $\mathcal{V}$ and width $d$, a lookup encoder stores
$\mathbf{E}\in\mathbb{R}^{|\mathcal{V}|\times d}$ and returns row
$\mathbf{e}_{w_i}$ for token $w_i$. A contextual encoder instead returns
$\mathbf{H}=f_{\boldsymbol{\theta}}(w_1,\ldots,w_n)\in\mathbb{R}^{m\times d}$,
where $m$ is the number of subword positions. The tokenizer alignment assigns
a nonempty subword index set $I_i$ to each retained word. First and mean
pooling return $\mathbf{h}_{\min I_i}$ and
$|I_i|^{-1}\sum_{j\in I_i}\mathbf{h}_j$, respectively. Attention pooling uses

$$
a_{ij}=\frac{\exp s_j}{\sum_{k\in I_i}\exp s_k},\qquad
\mathbf{x}_i=\sum_{j\in I_i}a_{ij}\mathbf{h}_j,
$$

where $s_j$ is the learned scorer's output, after any declared recurrent
context. All alternatives normalize within one word, excluding masked
subword positions. A missing or incompatible alignment is not repaired by
averaging unrelated rows.

## The four encoders

Each encoder registers under the `lofi.token_encoder` entry-point group, and
the packaged library holds one or more fragments for it under
`lofi/configs/model/token_encoder/`:

| `kind` | Rows come from | Packaged fragments |
| --- | --- | --- |
| `static` | A text file of fixed vectors, one word per line. | `static` |
| `learned` | An embedding table initialized at random and trained with the model. | `learned` |
| `pretrained` | A pretrained checkpoint loaded at a pinned commit. | `pretrained` (RoBERTa), `modernbert`, `xlm-roberta`, `eurobert`, `mmbert`, `mt5` |
| `llm2vec` | A decoder language model run as a bidirectional encoder. | `llm2vec` |

The `pretrained` and `llm2vec` encoders need the `nn` extra, which installs
`transformers`. Each declares its `checkpoint` as a hub source, a repository
pinned to one commit, and a run that declares either also declares the
`artifacts` cache the release is fetched into: an absolute `root` and an
`offline` flag that, when true, refuses a release absent from the cache rather
than downloading it. Every release is fetched through the one artifact fetcher
the cache describes, and the encoder reads only the fetched directory. Neither
encoder loads its weights at composition time, so the declarations below
validate without a download.

## Read vectors from a file

The example document declares a `static` encoder over a packaged vectors file:

```yaml
model:
  token_encoder:
    vectors: lofi.resource:run/synthetic-parser.vec
    vectors_format: glove
    trainable: true
```

Its composed declaration:

```python
encoder = codec(document).model.token_encoder
print(encoder.kind, encoder.vectors_format, encoder.trainable, encoder.oov_init)
```

```text
static glove True zeros
```

| Field | What it says |
| --- | --- |
| `vectors` | The file the rows are read from: a path, or a `lofi.resource:` URI for a file shipped in an installed package. |
| `vectors_format` | The file's layout, from the `lofi.vector_format` group: `glove` (one word and its values per line) or `word2vec` (the same, after a header line). |
| `trainable` | Whether the loaded rows take gradients. It defaults to `false`. |
| `lowercase` | Whether words are lowercased before lookup. |
| `oov_init`, `oov_init_std`, `seed` | How rows for words absent from the file are initialized, from the `lofi.oov_initialization` group: `zeros`, or `normal` with standard deviation `oov_init_std` under its own `seed`. |

## Learn rows from scratch

A `learned` encoder declares only its width and, optionally, the standard
deviation of its initial rows, which defaults to $d^{-1/2}$ for width $d$. Replace the
document's encoder with a whole mapping:

```python
learned = codec(
    document, overrides=('model.token_encoder={"kind": "learned", "width": 8}',)
)
print(learned.model.token_encoder)
```

```text
LearnedTokenEncoderSpec(kind='learned', width=8, init_std=None)
```

Selecting the packaged fragment instead, with `--set model/token_encoder=learned`,
is refused for this document, because the document's own `vectors` leaf still
applies and belongs to a different variant:

```python
from lofi.spec import UnknownSettingError

try:
    codec(document, overrides=("model/token_encoder=learned",))
except UnknownSettingError as refused:
    print(refused)
```

```text
model.token_encoder.vectors is not a setting of the selected variant; it belongs to static, and the selected variant declares model.token_encoder.init_std, model.token_encoder.kind, model.token_encoder.width
```

Group selection is the right route for a document that leaves the encoder's
variant-specific leaves to the fragment; a document that sets them replaces the
mapping or removes those leaves.

## Load a pretrained checkpoint

A `pretrained` encoder names a hub repository and the full commit hash it is
fetched at. Declare RoBERTa, as the packaged `pretrained` fragment does, with
the cache it is fetched into, and pool each word by the mean of its subword
rows:

```python
roberta = (
    'model.token_encoder={"kind": "pretrained", "checkpoint": {"kind": "hub",'
    ' "repository": "FacebookAI/roberta-base",'
    ' "revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b"}}'
)
cache = 'artifacts={"root": "/data/lofi/artifacts"}'
run = codec(document, overrides=(roberta, cache, "model/subword_pooler=mean"))
encoder = run.model.token_encoder
print(encoder.pretrained_weights())
print(encoder.architecture, encoder.trainable, encoder.window, encoder.readout)
print(run.model.subword_pooler)
```

```text
FacebookAI/roberta-base@e2da8e2f811d1448a5b465c236feacd80ffbac7b
encoder-only True ContextWindowSpec(rows=None, overflow='report') LastLayerReadoutSpec(kind='last')
MeanSubwordPoolerSpec(kind='mean')
```

`trainable` is `True` here because the document's `trainable: true` carries
over to the new variant, which declares the same field. The remaining fields:

| Field | What it says |
| --- | --- |
| `architecture` | Which stack of the checkpoint runs, from the `lofi.encoder_architecture` group: `encoder-only` runs the whole checkpoint, and `t5-encoder` runs only the encoder stack of a T5-family encoder-decoder (the packaged `mt5` fragment uses it). |
| `readout` | Which hidden layers become rows: `last` (the default), `fixed` with a `layer` index, or `scalar-mix`, a trainable normalized mixture over every layer. |
| `window.rows` | How many rows one item may take, or `null` to take the checkpoint's own bound. |
| `window.overflow` | What happens to a longer item, from the `lofi.overflow_policy` group: `report` encodes the longest prefix that fits and records the dropped words, and `refuse` raises `ContextOverflowError` naming the item, the rows it needs and the rows the window holds. |
| `join` | The nonempty string placed between words before tokenization. It defaults to one space. |

Change the readout and the overflow policy:

```python
mixed = codec(
    document,
    overrides=(
        roberta,
        cache,
        'model.token_encoder.readout={"kind": "scalar-mix"}',
        "model.token_encoder.window.overflow=refuse",
    ),
)
print(mixed.model.token_encoder.readout, mixed.model.token_encoder.window)
```

```text
ScalarMixReadoutSpec(kind='scalar-mix') ContextWindowSpec(rows=None, overflow='refuse')
```

A revision must be a full lowercase commit hash, so a branch name is refused
at composition:

```python
from lofi.spec import ConfigValueError

try:
    codec(
        document,
        overrides=(roberta, cache, "model.token_encoder.checkpoint.revision=main"),
    )
except ConfigValueError as refused:
    print(refused)
```

```text
model.token_encoder.checkpoint was refused: a checkpoint's revision is a full lowercase commit hash, not 'main'
```

An encoder reads a hub repository, so a `url` source, which names one archive
file, is refused at composition too, as is a run declaring a pretrained encoder
without an `artifacts` cache.

## Convert a decoder language model

An `llm2vec` encoder fetches a decoder language model at a pinned revision and
runs it with bidirectional attention. It shares `trainable`, `window` and `join`
with `pretrained`. Its `adaptation`, when declared, is a masked next-token
training run over a text corpus that takes place before the encoder is used;
`null`, the default, uses the checkpoint as loaded.

Declare a LoRA adaptation of the packaged Sheared-LLaMA checkpoint:

```python
llm2vec = (
    'model.token_encoder={"kind": "llm2vec", "checkpoint": {"kind": "hub",'
    ' "repository": "princeton-nlp/Sheared-LLaMA-1.3B",'
    ' "revision": "a4b76938edbf571ea7d7d9904861cbdca08809b4"},'
    ' "adaptation": {"corpus": "corpus.txt", "parameterization": "lora", "rank": 8, "steps": 200}}'
)
adaptation = codec(document, overrides=(llm2vec, cache)).model.token_encoder.adaptation
print(adaptation.parameterization, adaptation.rank, adaptation.alpha, adaptation.steps)
print(
    adaptation.replacement,
    adaptation.mask_probability,
    adaptation.optimizer.learning_rate,
)
```

```text
lora 8 8.0 200
mixed 0.2 5e-05
```

| Field | What it says |
| --- | --- |
| `corpus` | The text file adapted on, one item per line, or a packaged resource. |
| `mask`, `mask_probability` | The token a chosen position is replaced with, which the tokenizer's vocabulary must hold, and the probability each position is chosen. |
| `replacement` | From the `lofi.mask_replacement` group: `mixed` replaces four in five chosen tokens with the mask, one in ten with a random token and leaves one in ten, in the proportions `mask_replacement_fraction` and `random_replacement_fraction`; `mask` replaces every chosen token with the mask. |
| `steps`, `batch_size`, `optimizer`, `seed` | The adaptation run's length, lines per step, optimizer (AdamW at `5e-5` by default) and seed. |
| `parameterization`, `rank`, `alpha` | From the `lofi.adaptation` group: `full` adapts every parameter, and `lora` adapts low-rank updates of rank `rank` scaled by `alpha`. |

With selected target positions $J$ and corrupted subword sequence
$\widetilde{\mathbf{w}}$, masked next-token adaptation minimizes

$$
\ell_{\mathrm{adapt}}=-\frac1{|J|}\sum_{j\in J}
\log p_{\boldsymbol{\theta}}(w_j\mid\widetilde{\mathbf{w}};
\text{output position }j-1).
$$

The model reads bidirectional context; the target uses the previous output
position, as in masked next-token prediction \citep{behnamghader-etal2024}.
The first position and special tokens are ineligible, and a draw with no
chosen position takes no update. This optional adaptation does not imply that
the full LLM2Vec training pipeline, including contrastive training, has run.

For an adapted linear map, LoRA replaces $\mathbf{W}$ by
$\mathbf{W}+(\alpha/r)\mathbf{B}\mathbf{A}$, with
$\mathbf{A}\in\mathbb{R}^{r\times d_{\mathrm{in}}}$ and
$\mathbf{B}\in\mathbb{R}^{d_{\mathrm{out}}\times r}$
\citep{hu-etal2022}. The original matrix is frozen under this parameterization;
`full` instead permits the original parameters to change.

## Pool subwords into words

A `pretrained` or `llm2vec` encoder produces one row per subword, and the
subword pooler from the `lofi.subword_pooler` group reduces each word's rows to
one:

| `kind` | A word's vector is |
| --- | --- |
| `first` | The row of its first subword. |
| `mean` | The mean of its subword rows. |
| `attention` | The sum of its rows under the softmax of learned scores. `scorer` lists feed-forward layers that score each row, and `context`, when declared, first reads the word's rows with a recurrence (`cell` from the `lofi.recurrent_cell` group, `gru` by default, `lstm`, `elman-tanh` or `elman-relu`, the same cells the recurrent term encoders and decoders run; `layers`, `width`, `bidirectional`). |

The packaged `attention` fragment declares `context: {}`, which takes a
one-layer bidirectional GRU.

## See also

- [Configure term encoders and decoders](configure-term-encoders-and-decoders.md)
  for the components that read the token rows.
- [Configure a trainer](../training/configure-a-trainer.md) for
  `active_groups`, which decides whether the `token_encoder` group trains in a
  stage.
- The [model configuration reference](../../reference/configuration/model.md)
  for every field of every encoder and pooler.

## References

\bibliography
