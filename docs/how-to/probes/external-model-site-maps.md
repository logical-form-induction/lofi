# Instrument an external Torch model

An [`ExternalModelSpec`](../../reference/probe.md#lofi.probe.ExternalModelSpec) declares the sites of a third-party Torch model: each
names a module by its full path and gives it the role, axes, numeric class and
access a LoFI site carries. This guide finds the module path, declares and
lowers the site map, observes and patches the site for one pass, takes a
gradient there, supplies another hook mechanism, and declares a released model
as an `external` probe. The complete executable version of the first steps is
[`examples/external_site_map.py`](https://github.com/logical-form-induction/lofi/blob/main/examples/external_site_map.py).
## Find the exact module path

A Torch module is a named part of a model's computation. A hook can access
that part's output during a forward pass. The [site declaration](../../explanation/probing-through-sites.md#a-site-is-a-published-promise)
adds information the module path alone lacks: what the output means, which
axes it has, and whether a probe may read or replace it.
[`ExternalModel`](https://github.com/logical-form-induction/lofi/blob/main/lofi/probe/_external.py#L681)
uses the declared map to attach this access to the external module.

The example's `TinyModel` stores its interesting layer as `hidden`, whose
forward returns a tuple. List the paths `named_modules()` gives:

```python
import torch

from examples.external_site_map import TinyModel

model = TinyModel()
features = torch.tensor([[1.0, 2.0]])

for path, _module in model.named_modules():
    print(repr(path))
```

```text
''
'hidden'
'hidden.linear'
```

Paths must match the full declared module path. Opening a region
whose declaration names a path the model does not have, such as
`encoder.blocks.7`, raises a `ValueError` that names both the site and the
missing path.
## Declare and lower the site map

A module can return several values, such as a tuple containing a tensor and
auxiliary information. Select the tensor that corresponds to the intended
measurement before assigning its axes. Swapping two equally sized dimensions
can preserve an array's shape while changing which values belong to an item
or position. [`ExternalSiteDecl`](https://github.com/logical-form-induction/lofi/blob/main/lofi/probe/_boundary.py#L356)
records the selection and the site's contract together.

Import the boundary declarations from [`lofi.probe`](../../reference/probe.md#lofi.probe)
and the site identity types from [`lofi.nn.sites`](../../reference/nn.md#lofi.nn.sites):

```python
from lofi.nn.sites import ComponentId, Role, Site
from lofi.probe import ExternalModelSpec, ExternalSiteDecl, ExternalSiteMapIso

model_id = ComponentId("tiny-third-party")
hidden = Site(model_id, Role.RESIDUAL_STREAM, (0,))

declaration = ExternalModelSpec(
    model=str(model_id),
    sites=(
        ExternalSiteDecl(
            path="hidden",
            role="RESIDUAL_STREAM",
            layer=0,
            axes=("batch", "feature"),
            dtype_class="real",
            differentiable=True,
            access=("OBSERVE", "PATCH"),
            select=0,
        ),
    ),
)
site_map = ExternalSiteMapIso().forward(declaration)
```

`ExternalModelSpec.model` names the external component, and `checkpoint`
optionally names the source of the pinned artifact backing it (below). Each
[`ExternalSiteDecl`](../../reference/probe.md#lofi.probe.ExternalSiteDecl) declares:

| Field | What it declares |
| --- | --- |
| `path` | The full dotted module path. |
| `role` | A member name of [`lofi.nn.sites.Role`](../../reference/nn.md#lofi.nn.sites.Role). |
| `layer` | The nonnegative integer used as the site's key, so the site above is `Site(model_id, Role.RESIDUAL_STREAM, (0,))`. |
| `axes` | Every tensor axis, outermost first; names are nonempty and unique, and the observed rank must equal their number. |
| [`dtype_class`](../../reference/probe.md#lofi.probe.ExternalSiteDecl) | What the values mean, independently of storage dtype: `real`, `log-probability`, `probability`, `integer` or `mask`. |
| `differentiable` | Whether a gradient may flow through the site. A site a gradient is taken at must declare `True`. |
| `access` | A nonempty tuple of `OBSERVE`, `PATCH` or both. An operation the site does not permit is refused. |
| `select` | `None` when the module's output is the tensor; for a tuple output, the zero-based position holding it. Only that position is observed or replaced, and the others pass through unchanged. |

`ExternalSiteMapIso.forward()` is the only lowering step. It validates the
declaration and returns the runtime site map used below. An external map
carries no shared role inventory, so it is held only to single-component
ownership and to its own declarations.

## Observe and patch one pass

[`ExternalModel`](../../reference/probe.md#lofi.probe.ExternalModel) wraps the module and the lowered map. Its `run()` method takes
a nonempty mapping from forward argument names to tensors. Open an
observation region over the site:

```python
from lofi.probe import ExternalModel, instrument

external = ExternalModel(model, site_map)

with instrument(external, observe=frozenset({hidden})) as region:
    output = region(lambda: external.run({"features": features}))

reading = region.trace[hidden]
print(reading.axes)
print(reading.tensor)
```

```text
('batch', 'feature')
tensor([[1., 2.]])
```

The trace holds a detached [`Observation`](../../reference/probe.md#lofi.probe.Observation) under each requested site. A site the
map does not publish is refused before the region opens, with a `ValueError`
that names the site and how many sites the component publishes; that is
distinct from a declared path the model cannot resolve.

Pass interventions through `patch` to replace the value at a site for one
pass:

```python
from lofi.probe import Zero

with instrument(external, patch={hidden: Zero()}) as region:
    patched_output = region(lambda: external.run({"features": features}))
print(output, patched_output)
```

```text
tensor([[2., 4.]], grad_fn=<MulBackward0>) tensor([[0., 0.]])
```

Leaving the `with` block removes every installed hook, after a completed pass,
an exception, or a refusal. The checked-in example checks that the model's
ordinary output is restored after a patched pass and after a patched pass whose
model raises:

```bash
uv run python examples/external_site_map.py
```

```text
observed hidden value: [[1.0, 2.0]]
patched model output: [[0.0, 0.0]]
restored after successful patch: True
failing pass: the tiny model refused this input
restored after failing patch: True
unknown site: Site('tiny-third-party', Role.GATE, (0,)) is not published by this component, which publishes 1 points
```

## Take a gradient at a site

A detached observation stores a value for inspection. Gradient computation
requires access to the automatic-differentiation graph that produced the value.
[`differentiate`](https://github.com/logical-form-induction/lofi/blob/main/lofi/probe/_instrument.py#L734)
opens the access needed for that computation; the
[`GradientSites`](https://github.com/logical-form-induction/lofi/blob/main/lofi/probe/_instrument.py#L614)
check verifies that each requested site permits it.

Ordinary observations are detached. To take a gradient with respect to the
value at a site, check the sites once as [`GradientSites`](../../reference/probe.md#lofi.probe.GradientSites) and open a
`differentiate` region over them:

```python
from lofi.probe import GradientSites, differentiate

points = GradientSites(external.site_map(), frozenset({hidden}))
with differentiate(external, sites=points) as region:
    result = region(lambda: external.run({"features": features}))
    gradients = region.gradients(result.sum())
print(points)
print(gradients[hidden])
```

```text
GradientSites('tiny-third-party', sites=1)
tensor([[2., 2.]])
```

`GradientSites` refuses a site that is not published, does not permit both
observation and patching, or is declared nondifferentiable, so any number of
regions can then be opened over the checked set without checking it again. A
pass that never reaches a declared site is refused, and `gradients` is taken
once, of a scalar computed through the sites. EAP-IG edge attribution,
[`attribute_edges`](../../reference/probe.md#lofi.probe.attribute_edges), reads its gradients the same way.

## Supply another hook mechanism

`ExternalModel(..., hooking=...)` accepts an implementation of
[`lofi.probe.Hooking`](../../reference/probe.md#lofi.probe.Hooking), whose one method has this contract:

```python
from collections.abc import Sequence

from lofi.probe import Attachment, Recorder, Release


def install(
    attached: Sequence[Attachment],
    record: Recorder,
    /,
) -> Release: ...
```

Each [`Attachment`](../../reference/probe.md#lofi.probe.Attachment) identifies the exact `module`, `path`, `site`, `select` and
axis names, says whether the value is recorded, and supplies an optional patch.
At each reached attachment, the implementation reads the whole tensor output or
the tuple position `select` names, makes a [`lofi.probe.Observation`](../../reference/probe.md#lofi.probe.Observation) of it, and
passes it to `attachment.patch.stands_in_for()` when a patch is present. The
replacement is what the model receives and, when `attachment.observed` is
true, what is passed to `record`. The returned zero-argument [`Release`](../../reference/probe.md#lofi.probe.Release) removes
everything installed. An implementation must preserve unselected tuple
positions, roll back a partial installation, and restore the model after
normal and exceptional exits. The default, [`lofi.probe.TorchHooks`](../../reference/probe.md#lofi.probe.TorchHooks), does this
with Torch forward hooks.

## Probe a released model from a run document

The `external` probe kind runs split posterior diagnostics over a released
DIORA [checkpoint](../../reference/probe.md#lofi.probe.ExternalModelSpec). Its declaration holds:

| Field | What it declares |
| --- | --- |
| `model` | An [`ExternalModelSpec`](../../reference/probe.md#lofi.probe.ExternalModelSpec) whose `checkpoint` names where the pinned artifact is fetched from: `{kind: url, url, sha256}`, an absolute HTTP or HTTPS URL pinned to its lowercase SHA-256 digest, or `{kind: hub, repository, revision}`, a model hub repository pinned to a full commit hash. |
| `backend` | The architecture the artifact stores. The packaged kind, `released-mlp`, builds the released MLP DIORA from the state file at `member`, a path inside the fetched zip archive or hub snapshot. |
| `artifacts` | The cache the artifact is fetched into: `root`, an absolute directory, and `offline`, whether an artifact absent from the cache is refused rather than fetched (false by default). It is the same declaration a run's `artifacts` section makes for pretrained [token encoders](../models/choose-a-token-encoder.md#the-four-encoders). |
| `stimulus_lengths` | The sentence lengths of the deterministic all-zero stimuli, one pass each. |

One fetcher resolves either source kind under `artifacts.root`. An archive is
cached at `archives/<sha256>` and is used only when its bytes have the
declared digest, both when it is downloaded and when a cached copy is read
again. A hub revision is cached as a snapshot directory at
`snapshots/<repository>/<commit>`. Source kinds are listed in the
`lofi.artifact_source` group and backends in the `lofi.diora_backend` group.

The packaged fragment `probes/diora/naacl-2019` declares the release of the
NAACL 2019 DIORA implementation, so a document selects it and adds only its
cache and stimuli:

```yaml
defaults:
  - run/synthetic-parser
  - probes.diora: naacl-2019
probes:
  diora:
    artifacts:
      root: /data/lofi/artifacts
    stimulus_lengths: [5, 10, 20]
```

The stimuli exercise the released chart and its sites; rows over corpus inputs
belong to the experiment that consumes the probe rows. [Probe a trained point](run-probes.md)
covers declaring probes in a run.
