# Choose a device policy

A run document's `device_policy` section declares where a point trains and in
what precision: the device, the floating-point type it computes in, whether
torch's deterministic algorithms are enforced, the float32 matrix-multiplication
precision, and the activation-memory budget of one training step. This guide
selects a packaged policy, overrides its fields, and lists how a policy is
refused.

## Select a packaged policy

Three policies ship under `lofi/configs/device_policy/`:

| Fragment | `device` | `compute_dtype` | `deterministic` | `matmul_precision` |
| --- | --- | --- | --- | --- |
| `cpu` | `cpu` | `float32` | `true` | `highest` |
| `cuda-mixed` | `cuda` | `bfloat16` | `false` | `high` |
| `mps` | `mps` | `float32` | `false` | `medium` |

Each also declares `activation_memory_bytes`. A document selects one in its
`defaults` list, as
[`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml)
does with `- device_policy: cpu`. On the command line, a root-level group
override selects another, as in `--set /device_policy=mps`; from Python it is
one more override to the codec:

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("docs/examples/synthetic-parser.yaml")

for overrides in [("/device_policy=cuda-mixed",), ("/device_policy=mps",)]:
    policy = codec(document, overrides=overrides).device_policy
    print(policy.device, policy.compute_dtype, policy.matmul_precision)
```

```text
cuda bfloat16 high
mps float32 medium
```

The device policy is part of the resolved document, so the same document under
two policies is two points with two identifiers.

## Read the resolved policy

[`resolve_device_policy`](../../reference/run.md) turns a validated run's
declaration into the runtime `DevicePolicy` that places parameters and
activations:

```python
from lofi.run import resolve_device_policy

run = codec(document)
print(run.device_policy)
print(resolve_device_policy(run))
```

```text
DevicePolicySpec(device='cpu', compute_dtype=<ComputeDtype.FLOAT32: 'float32'>, deterministic=True, matmul_precision=<MatmulPrecision.HIGHEST: 'highest'>, activation_memory_bytes=8000000000, scaler=None, checkpoint_from_width=None)
DevicePolicy(device=device(type='cpu'), dtype=torch.float32, autocast_dtype=None, checkpoint_from_width=None)
```

Resolution maps `compute_dtype` to a parameter type and an autocast type, so
under `cuda-mixed` parameters stay in float32 and autocast computes in
bfloat16:

| `compute_dtype` | Parameters | Autocast |
| --- | --- | --- |
| `float16` | float32 | float16 |
| `bfloat16` | float32 | bfloat16 |
| `float32` | float32 | off |
| `float64` | float64 | off |

Resolution also checks that the machine holds the declared device. The
processor and the `meta` device are always present. An accelerator device must
be of the family the machine's accelerator belongs to, and an index must be
below the number of devices it holds, so a `cuda-mixed` point resolved on a
laptop is refused before any model is built, with a message such as
`device 'cuda' is not present: this machine's accelerator is mps` or
`device 'cuda:2' is not present: this machine holds 2 cuda devices`.
Every route that trains, restores or probes a point resolves its policy this
way; `lofi price` refuses a `--device` of another kind than the declared one
before it resolves anything.

## Declare the fields

The fields of [`DevicePolicySpec`](../../reference/configuration/components.md):

| Field | What it says |
| --- | --- |
| `device` | A torch device name, with an index where needed (`cuda:1`). Its family must be one of `cpu`, `cuda`, `mps`, `xpu` or `meta`. |
| `compute_dtype` | `float16`, `bfloat16`, `float32` or `float64`, as tabulated above. |
| `deterministic` | Whether `torch.use_deterministic_algorithms` is on while the point trains. |
| `matmul_precision` | `highest`, `high` or `medium`, passed to `torch.set_float32_matmul_precision` while the point trains. |
| `activation_memory_bytes` | The peak bytes one training step may hold on the device: pricing reads each stack against it, and training stops at a step above it. |
| `scaler` | Optional dynamic gradient scaling: `init_scale`, `growth_factor`, `backoff_factor` and `growth_interval`. See the note below. |
| `checkpoint_from_width` | The narrowest span width whose chart cells are recomputed in the backward pass instead of held from the forward pass, or `null` to hold every cell. |

`deterministic` and `matmul_precision` are process-wide torch settings. The run
store applies them for the duration of one point and restores the previous
settings when the point ends.

`activation_memory_bytes` is the budget that `lofi price` reports each priced
stack against, and an array launch leaves out the points of a stack whose
measured peak exceeds it; see [Run and price a sweep](../experiments/sweeps.md).
Training holds every step to it as well. On a `cuda`, `mps` or `xpu` device the
allocator keeps a count of the most bytes it has held, so each training step
resets that count when it starts and reads it when it ends. The first step whose
peak lies above the bound stops the run with an `ActivationMemoryExceededError`
such as

```text
a training step held 9126805504 bytes on cuda:0 at its peak, above the activation_memory_bytes bound of 8000000000 its device policy declares
```

The error is a `MemoryError`, so the executor treats it as exhausted memory
under the experiment's `failure_policy.out_of_memory`: `halve-batch` retries
the point at half its training batch and `retry` attempts it again unchanged
until its infrastructure attempts are spent. The point's closed record still
reports the most bytes the device held over the whole point. The processor and
the `meta` device keep no such count, so on them the bound prices the stack and
sizes an array task's memory request, but no training step is measured.
`checkpoint_from_width` trades recomputation for activation memory on long
inputs. Chart cells are composed from width two, so its least value is `2`.

Every packaged trainer builds its gradient scaler from `scaler`: its losses
are scaled before the backward pass, its gradients are unscaled before
clipping and stepping, and the scaler's state is written into each checkpoint
and restored on resume. Without a `scaler`, optimizer steps are unscaled and a
checkpoint's `scaler` reference is empty. A `float16` point usually declares
one:

```python
half = codec(
    document,
    overrides=("device_policy.compute_dtype=float16", "device_policy.scaler={}"),
)
print(half.device_policy.scaler)
```

```text
GradScalerSpec(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)
```

## How a policy is refused

Composition refuses a field outside its declared range, and refuses a gradient
scaler under bfloat16, which needs none:

```python
for overrides in [
    ("/device_policy=cuda-mixed", "device_policy.scaler={}"),
    ("device_policy.checkpoint_from_width=1",),
    ("device_policy.compute_dtype=int8",),
]:
    try:
        codec(document, overrides=overrides)
    except Exception as refused:
        print(type(refused).__name__, "|", refused)
```

```text
ConfigValueError | device_policy was refused: bfloat16 computation does not use a gradient scaler
ConfigValueError | device_policy.checkpoint_from_width was refused: chart cells are composed from width two, so checkpoint_from_width is at least two, not 1
SettingParseError | device_policy.compute_dtype expects ComputeDtype; got 'int8'
```

The device family is checked when the policy is resolved, before any model is
built:

```python
unknown = codec(document, overrides=("device_policy.device=tpu",))
try:
    resolve_device_policy(unknown)
except ValueError as refused:
    print(refused)
```

```text
undeclared device 'tpu'; declared: cpu, cuda, mps, xpu, meta
```

## Related

- [Configure a trainer](configure-a-trainer.md) declares what the policy runs.
- [Component declarations](../../reference/configuration/components.md)
  renders `DevicePolicySpec` and `GradScalerSpec`.
