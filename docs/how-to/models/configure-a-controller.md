# Configure a controller

A run's `model.controller` declares the policy that weighs combinator choices:
for each ordered pair of adjacent child representations in the chart, it
scores every action of the grammar's rule inventory and normalizes the scores
into a distribution. This guide selects the `mlp`, `bilinear` or `transformer`
controller, sets the options they share, and shows how each is refused.

The controller registers under the `lofi.controller` entry-point group. The
packaged library holds one fragment, `lofi/configs/model/controller/mlp.yaml`,
which declares `kind: mlp` and leaves every other field at its default. The
chart parser receives the whole controller as its policy, so its parameters
train with the `controller` group named in a trainer's `active_groups`.

The examples compose
[`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml),
which selects the packaged `mlp` fragment and narrows its hidden layer:

```yaml
model:
  controller:
    hidden: [{width: 4, activation: tanh}]
```

The setup the examples share:

```python
from pathlib import Path

from lofi.spec import ConfigValueError, RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("docs/examples/synthetic-parser.yaml")
```

## Read the composed controller

```python
controller = codec(document).model.controller
print(controller)
print(controller.raising())
```

```text
MlpControllerSpec(raise='none', kind='mlp', input='inside', features=('left', 'right'), normalizer=SoftmaxControllerNormalizerSpec(kind='softmax'), hidden=(FeedForwardLayerSpec(width=4, activation='tanh', dropout=0.0),))
none
```

Every controller declares the same four shared fields:

| Field | What it says |
| --- | --- |
| `input` | Which chart representation each child contributes: `inside` (the default) or `span`. A `span` input is twice the term encoder's width. |
| `features` | The ordered features built from a left and a right child, each installed in the `lofi.pair_feature` group: `left`, `right`, `difference` (left minus right) and `product` (elementwise). The default is `[left, right]`. The list must be nonempty and must not repeat a feature. |
| `normalizer` | How scores become a distribution: `softmax` (the default) or `entmax` with an `alpha` of at least one, whose normalizing threshold is searched for in `bisections` halvings (default `50`). |
| `raise` | Whether the controller differentiably chooses type-raising targets: `none` (the default) or `soft`. Soft raising requires `input: inside`. |

Soft raising uses each rule's declared `targets`, including compound categories.
When `targets` is null, it uses the represented primitive categories. Its
inputs are learned vectors without symbolic category identities, so each raising
rule must declare `arguments: null`. A restricted argument set is refused when
the controller is built. Symbolic raising supports restricted arguments.


## Choose the scoring head

The three kinds differ only in how they turn the pair into scores:

| `kind` | Scores | Own fields |
| --- | --- | --- |
| `mlp` | A feed-forward network over the concatenated features. | `hidden`: a list of layers, each with `width`, `activation` (from the `lofi.activation` group: `tanh`, `relu`, `gelu`, `silu`) and `dropout`. The default is one layer of width 64 under `tanh`. |
| `bilinear` | A bilinear form of the left and right children, added to an affine map of the features. | None. |
| `transformer` | Self-attention over the two children. | `layers`, `heads`, `feedforward_width`, `activation` (default `gelu`), `dropout` in `[0, 1)` and `norm_placement` (`pre` or `post`, the default). Layers, heads and feed-forward width are positive. |

Replace the document's controller with a bilinear one that also reads the
elementwise product:

```python
bilinear = codec(
    document,
    overrides=(
        'model.controller={"kind": "bilinear", "features": ["left", "right", "product"]}',
    ),
).model.controller
print(bilinear)
```

```text
BilinearControllerSpec(raise='none', kind='bilinear', input='inside', features=('left', 'right', 'product'), normalizer=SoftmaxControllerNormalizerSpec(kind='softmax'))
```

The override replaces the whole mapping, so the document's `hidden` leaf, which
the bilinear head does not declare, goes with it. On the command line the same
override is `--set 'model.controller={"kind": "bilinear", ...}'`.

Declare a two-head transformer whose scores pass through 1.5-entmax:

```python
transformer = codec(
    document,
    overrides=(
        'model.controller={"kind": "transformer", "heads": 2, "feedforward_width": 8,'
        ' "normalizer": {"kind": "entmax", "alpha": 1.5}}',
    ),
).model.controller
print(
    transformer.layers,
    transformer.heads,
    transformer.activation,
    transformer.normalizer,
)
```

```text
1 2 gelu EntmaxControllerNormalizerSpec(kind='entmax', alpha=1.5, bisections=50)
```

An entmax normalizer with `alpha` above one can assign an action exactly zero
weight, where softmax keeps every action's weight positive.

## How a declaration is refused

A repeated feature and an `alpha` below one are refused at composition, with
the leaf that holds the value:

```python
for override in (
    'model.controller.features=["left", "left"]',
    'model.controller.normalizer={"kind": "entmax", "alpha": 0.5}',
):
    try:
        codec(document, overrides=(override,))
    except ConfigValueError as refused:
        print(refused)
```

```text
model.controller.features was refused: a controller repeats child-pair features: ('left', 'left')
model.controller.normalizer.alpha was refused: a controller entmax alpha is at least one, not 0.5
```

The chart decides which inputs a controller may read. An `inside-outside`
chart provides `span` only when its split weigher is `attend` and its
reduction is `log-sum-exp`, so the synthetic parser, whose chart mixes splits
by their marginals, refuses a span input:

```python
try:
    codec(document, overrides=("model.controller.input=span",))
except ConfigValueError as refused:
    print(refused)
```

```text
model was refused: model.controller: input 'span' is incompatible with the 'inside-outside' parser; it provides controller inputs ['inside']
```

Adding `model.parser.attend={"kind": "attend"}` to the same overrides makes the
document valid. Declaring `raise: soft` on top of it is refused in turn, with
"model.controller: soft raising requires input 'inside', not 'span'".

## See also

- [Configure a chart parser](configure-a-chart-parser.md) for the chart that
  consumes the controller's action weights.
- [Configure a trainer](../training/configure-a-trainer.md) for the
  `active_groups` that decide when the controller trains.
- The [model configuration reference](../../reference/configuration/model.md)
  for every controller field, and the [network reference](../../reference/nn.md)
  for the built controllers.
