"""Masked divergences over symbol distributions."""

from typing import TYPE_CHECKING, assert_never

import torch

from lofi.nn.loss._divergence import (
    CrossEntropy,
    FenchelYoung,
    KullbackLeibler,
    resolved,
)

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn._values import LogDistribution
    from lofi.nn.loss._divergence import Divergence

__all__ = [
    "cross_entropy",
    "fenchel_young",
    "kullback_leibler",
    "node_divergence",
]


def _reject_unpaired(reference: LogDistribution, model: LogDistribution) -> None:
    """Raise unless two distributions can be compared symbol by symbol.

    Parameters
    ----------
    reference : LogDistribution
        The distribution the comparison is anchored on.
    model : LogDistribution
        The distribution being compared against it.

    Raises
    ------
    ValueError
        If the two do not have the same axes or shape, if either names no axis
        its mass sums over, if they name different ones, or if that axis is not the
        innermost.
    """
    if reference.axes != model.axes:
        msg = (
            f"the reference is over the axes {reference.axes} and the model "
            f"over {model.axes}"
        )
        raise ValueError(msg)

    if reference.tensor.shape != model.tensor.shape:
        msg = (
            f"the reference has shape {tuple(reference.tensor.shape)} and the "
            f"model {tuple(model.tensor.shape)}"
        )
        raise ValueError(msg)

    over = reference.normalized_over

    if over is None or model.normalized_over is None:
        msg = (
            "a divergence is between distributions, and one of these "
            "normalizes over no axis"
        )
        raise ValueError(msg)

    if over != model.normalized_over:
        msg = (
            f"the reference normalizes over {over!r} and the model over "
            f"{model.normalized_over!r}"
        )
        raise ValueError(msg)

    if reference.axis(over) != reference.tensor.ndim - 1:
        msg = f"the symbol axis {over!r} is not the innermost of {reference.axes}"
        raise ValueError(msg)


def _widened(mask: Tensor, rank: int) -> Tensor:
    """Return a mask over outer axes, shaped to broadcast against a rank.

    Parameters
    ----------
    mask : Tensor
        A boolean mask over the outermost axes.
    rank : int
        The rank to broadcast against.

    Returns
    -------
    Tensor
        A boolean view of the mask with a trailing axis of extent one for every axis it
        does not cover.
    """
    return mask.reshape(tuple(mask.shape) + (1,) * (rank - mask.ndim))


def _checked_outer(
    tensor: Tensor,
    shape: tuple[int, ...],
    *,
    what: str,
) -> Tensor:
    """Return a per-node tensor over the outermost node axes.

    Parameters
    ----------
    tensor : Tensor
        One value per node, over the outermost node axes.
    shape : tuple[int, ...]
        The shape of the node axes, which is the score shape without its symbol axis.
    what : str
        What the tensor holds, for the message.

    Returns
    -------
    Tensor
        The tensor, unchanged.

    Raises
    ------
    ValueError
        If the tensor covers no axis, covers more axes than there are node axes, or
        disagrees about the extent of one it covers.
    """
    if not 1 <= tensor.ndim <= len(shape):
        msg = (
            f"a node {what} covers the outermost of {len(shape)} node axes, so "
            f"its rank is between 1 and {len(shape)}, not {tensor.ndim}"
        )
        raise ValueError(msg)

    if tuple(tensor.shape) != shape[: tensor.ndim]:
        msg = (
            f"a node {what} of shape {tuple(tensor.shape)} does not cover the "
            f"outermost of the node axes {shape}"
        )
        raise ValueError(msg)

    return tensor


def _checked_nodes(nodes: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Return an explicit node mask.

    Parameters
    ----------
    nodes : Tensor
        True where a node contributes.
    shape : tuple[int, ...]
        The shape of the node axes, which is the score shape without its symbol axis.

    Returns
    -------
    Tensor
        The mask, unchanged.

    Raises
    ------
    TypeError
        If the mask is not of boolean dtype.
    ValueError
        If the mask covers no axis, covers more axes than there are node axes, or
        disagrees about the extent of one it covers.
    """
    if nodes.dtype is not torch.bool:
        msg = f"a node mask holds booleans, not {nodes.dtype}"
        raise TypeError(msg)

    return _checked_outer(nodes, shape, what="mask")


def _weighted(per_node: Tensor, weights: Tensor | None) -> Tensor:
    """Return the per-node comparison summed over the nodes, each at its weight.

    Parameters
    ----------
    per_node : Tensor
        The comparison at every node, over the node axes.
    weights : Tensor | None
        How much each node counts, over the outermost node axes, or None where every
        node counts once. A node weighted zero contributes nothing, even where its
        comparison is infinite.

    Returns
    -------
    Tensor
        A zero-dimensional tensor of the comparison's dtype.

    Raises
    ------
    TypeError
        If the weights are not of floating dtype.
    ValueError
        If the weights do not cover the outermost node axes.
    """
    if weights is None:
        return per_node.sum()

    if not weights.is_floating_point():
        msg = f"a node weight is a floating-point number, not {weights.dtype}"
        raise TypeError(msg)

    shape = tuple(per_node.shape)
    scale = _widened(_checked_outer(weights, shape, what="weight"), len(shape))
    counted = torch.where(scale != 0.0, per_node, per_node.new_zeros(()))

    return (counted * scale).sum()


def _kept(
    reference: LogDistribution,
    model: LogDistribution,
    nodes: Tensor | None,
) -> Tensor | None:
    """Return the nodes both distributions and the caller keep, or None for all.

    Parameters
    ----------
    reference : LogDistribution
        The distribution the comparison is anchored on.
    model : LogDistribution
        The distribution being compared against it.
    nodes : Tensor | None
        True where a node contributes, over the outermost node axes, or None where the
        caller restricts nothing.

    Returns
    -------
    Tensor | None
        A boolean mask broadcasting against the node axes, or None where every node
        contributes.

    Raises
    ------
    TypeError
        If the node mask is not of boolean dtype.
    ValueError
        If the node mask does not cover the outermost node axes.
    """
    rank = reference.tensor.ndim - 1
    shape = tuple(reference.tensor.shape[:rank])
    given = None if nodes is None else _checked_nodes(nodes, shape)
    masks = [
        _widened(mask, rank)
        for mask in (reference.mask, model.mask, given)
        if mask is not None
    ]

    if not masks:
        return None

    kept = masks[0]

    for mask in masks[1:]:
        kept = kept & mask

    return kept


def _guarded(
    reference: LogDistribution,
    model: LogDistribution,
    nodes: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Return the two log score tensors with every masked node made inert.

    Parameters
    ----------
    reference : LogDistribution
        The distribution the comparison is anchored on.
    model : LogDistribution
        The distribution being compared against it.
    nodes : Tensor | None
        True where a node contributes, or None where every node does.

    Returns
    -------
    tuple[Tensor, Tensor]
        The reference and model log scores, of the same shape and dtype they arrived
        with.
    """
    kept = _kept(reference, model, nodes)

    if kept is None:
        return reference.tensor, model.tensor

    live = kept.unsqueeze(-1)
    inert = reference.tensor.new_zeros(())

    return (
        torch.where(live, reference.tensor, inert),
        torch.where(live, model.tensor, inert),
    )


def _absent(log_reference: Tensor) -> tuple[Tensor, Tensor]:
    """Return safe reference probabilities and log scores.

    Parameters
    ----------
    log_reference : Tensor
        The reference log scores.

    Returns
    -------
    tuple[Tensor, Tensor]
        The reference probabilities, and a copy of its log scores in which no position
        the reference gives no mass holds an infinity.
    """
    probabilities = log_reference.exp()
    unused = probabilities == 0.0
    safe = torch.where(unused, log_reference.new_zeros(()), log_reference)

    return probabilities, safe


def kullback_leibler(
    reference: LogDistribution,
    model: LogDistribution,
    *,
    nodes: Tensor | None = None,
    weights: Tensor | None = None,
) -> Tensor:
    """Return the forward divergence from the reference to the model.

    Parameters
    ----------
    reference : LogDistribution
        The distribution the divergence runs from, in log space, normalized over its
        innermost axis.
    model : LogDistribution
        The distribution it runs to, over the same axes.
    nodes : Tensor | None
        True where a node contributes, over the outermost node axes, or None where every
        node the two values keep does.
    weights : Tensor | None
        How much each node counts, over the outermost node axes, or None where every
        node counts once. A node weighted zero contributes nothing, even where its
        comparison is infinite.

    Returns
    -------
    Tensor
        A zero-dimensional tensor of the inputs' floating dtype: the divergence of each
        kept node, at its weight, summed over the nodes.

    Raises
    ------
    TypeError
        If the node mask is not of boolean dtype, or the weights are not of floating
        dtype.
    ValueError
        If the two distributions disagree about their axes, shape or normalized axis, if
        that axis is not the innermost, or if the node mask or the weights do not cover
        the node axes.
    """
    _reject_unpaired(reference, model)
    log_reference, log_model = _guarded(reference, model, nodes)
    probabilities, safe_reference = _absent(log_reference)
    safe_model = torch.where(
        probabilities == 0.0,
        log_model.new_zeros(()),
        log_model,
    )

    return _weighted((probabilities * (safe_reference - safe_model)).sum(-1), weights)


def cross_entropy(
    reference: LogDistribution,
    model: LogDistribution,
    *,
    nodes: Tensor | None = None,
    weights: Tensor | None = None,
) -> Tensor:
    """Return the cross-entropy of the model under the reference.

    Parameters
    ----------
    reference : LogDistribution
        The distribution the expectation is taken under, in log space, normalized over
        its innermost axis.
    model : LogDistribution
        The distribution being scored, over the same axes.
    nodes : Tensor | None
        True where a node contributes, over the outermost node axes, or None where every
        node the two values keep does.
    weights : Tensor | None
        How much each node counts, over the outermost node axes, or None where every
        node counts once. A node weighted zero contributes nothing, even where its
        comparison is infinite.

    Returns
    -------
    Tensor
        A zero-dimensional tensor of the inputs' floating dtype: the cross-entropy of
        each kept node, at its weight, summed over the nodes.

    Raises
    ------
    TypeError
        If the node mask is not of boolean dtype, or the weights are not of floating
        dtype.
    ValueError
        If the two distributions disagree about their axes, shape or normalized axis, if
        that axis is not the innermost, or if the node mask or the weights do not cover
        the node axes.
    """
    _reject_unpaired(reference, model)
    log_reference, log_model = _guarded(reference, model, nodes)
    probabilities, _ = _absent(log_reference)
    safe_model = torch.where(
        probabilities == 0.0,
        log_model.new_zeros(()),
        log_model,
    )

    return _weighted(-(probabilities * safe_model).sum(-1), weights)


def fenchel_young(
    reference: LogDistribution,
    model: LogDistribution,
    *,
    alpha: float = 1.5,
    nodes: Tensor | None = None,
    weights: Tensor | None = None,
) -> Tensor:
    """Return the Fenchel-Young loss of the Tsallis entropy indexed by alpha.

    Parameters
    ----------
    reference : LogDistribution
        The distribution the loss is measured against, in log space, normalized over its
        innermost axis.
    model : LogDistribution
        The distribution being scored, over the same axes.
    alpha : float
        Which entropy the loss is built on.
    nodes : Tensor | None
        True where a node contributes, over the outermost node axes, or None where every
        node the two values keep does.
    weights : Tensor | None
        How much each node counts, over the outermost node axes, or None where every
        node counts once. A node weighted zero contributes nothing, even where its
        comparison is infinite.

    Returns
    -------
    Tensor
        A zero-dimensional tensor of the inputs' floating dtype: the loss of each kept
        node, at its weight, summed over the nodes.

    Raises
    ------
    TypeError
        If the node mask is not of boolean dtype, the weights are not of floating dtype,
        or alpha is not a number.
    ValueError
        If distributions disagree, normalization, mask or weight axes are invalid, or
        ``alpha`` is below one.
    """
    spec = FenchelYoung(alpha)

    if isinstance(resolved(spec), KullbackLeibler):
        return kullback_leibler(reference, model, nodes=nodes, weights=weights)

    index = spec.alpha

    _reject_unpaired(reference, model)
    log_reference, log_model = _guarded(reference, model, nodes)
    reference_power = (index * log_reference).exp()
    model_power = (index * log_model).exp()
    crossed = ((index - 1.0) * log_model + log_reference).exp()
    per_node = (reference_power + (index - 1.0) * model_power - index * crossed).sum(-1)

    return _weighted(per_node, weights) / (index * (index - 1.0))


def node_divergence(
    divergence: Divergence,
    reference: LogDistribution,
    model: LogDistribution,
    *,
    nodes: Tensor | None = None,
    weights: Tensor | None = None,
) -> Tensor:
    """Return the comparison one declared divergence makes, node-summed.

    Parameters
    ----------
    divergence : Divergence
        Which comparison to make.
    reference : LogDistribution
        The distribution the comparison is anchored on, in log space, normalized over
        its innermost axis.
    model : LogDistribution
        The distribution being compared against it, over the same axes.
    nodes : Tensor | None
        True where a node contributes, over the outermost node axes, or None where every
        node the two values keep does.
    weights : Tensor | None
        How much each node counts, over the outermost node axes, or None where every
        node counts once. A node weighted zero contributes nothing, even where its
        comparison is infinite.

    Returns
    -------
    Tensor
        A zero-dimensional tensor of the inputs' floating dtype: the comparison at each
        kept node, at its weight, summed over the nodes.

    Raises
    ------
    TypeError
        If the node mask is not of boolean dtype, or the weights are not of floating
        dtype.
    ValueError
        If the two distributions disagree about their axes, shape or normalized axis, if
        that axis is not the innermost, or if the node mask or the weights do not cover
        the node axes.
    """
    match divergence:
        case KullbackLeibler():
            return kullback_leibler(reference, model, nodes=nodes, weights=weights)

        case CrossEntropy():
            return cross_entropy(reference, model, nodes=nodes, weights=weights)

        case FenchelYoung():
            return fenchel_young(
                reference,
                model,
                alpha=divergence.alpha,
                nodes=nodes,
                weights=weights,
            )
        case unreachable:
            assert_never(unreachable)
