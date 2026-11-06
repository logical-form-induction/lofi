# Train

An [optimizer](configuration/training.md#optimizers) updates model parameters
from a training cost's [gradients](../explanation/inference-as-evidence.md#from-a-decision-to-a-gradient). One completed update is a step; an epoch
passes through the configured training data. [Checkpoints](checkpoints.md)
save progress for recovery at completed steps.

When training has several tasks, [mixing policies](#mixing-policies) determine
how their steps are interleaved. This differs from [combining loss terms](../explanation/induction-objectives.md#combining-terms)
within one objective. [Training declarations](configuration/training.md)
selects the trainer, optimizer, schedule, and evidence sources.

This page renders [`lofi.train`](#lofi.train), [`lofi.training_plugins`](#lofi.training_plugins) and [`lofi.mixing`](#lofi.mixing): training steps, checkpoints, [optimization policies](configuration/training.md#optimizers), [training plugins](#training-plugins) and mixing policies.

::: lofi.train

## Training plugins

::: lofi.training_plugins

## Mixing policies

::: lofi.mixing
