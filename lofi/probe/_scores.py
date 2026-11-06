"""Scores over readings on each published point's declared numeric domain.

Scores convert values according to the declared numeric class, reduce only named
axes, and preserve the remaining axis order. Rows without valid data remain masked.
"""

from math import inf, log
from typing import TYPE_CHECKING

import torch

from lofi.nn.sites import DTypeClass
from lofi.probe._dtypes import (
    LOG_ZERO,
    LogProbabilities,
    Probabilities,
    as_log_probabilities,
    as_probabilities,
    entropy,
    exponentiate,
    logarithm,
)
from lofi.probe._values import Observation

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "best_two",
    "certainty",
    "consistency",
    "cosine_stability",
    "energy_concentration",
    "js_divergence",
    "js_divergence_score",
    "kl_divergence_score",
    "maximum_probability",
    "norm_preservation",
    "normalized_entropy",
    "normalized_js_divergence",
    "smoothness",
    "sparsity",
    "spectral_entropy",
    "top_decile_power",
    "top_two_margin",
]


def _ensure_class(declared: DTypeClass) -> None:
    """Refuse a declaration that is not a numeric class.

    Parameters
    ----------
    declared : DTypeClass
        What a point declared its numbers to mean.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    """
    if not isinstance(declared, DTypeClass):
        msg = f"what a point declares is a DTypeClass, not {declared!r}"
        raise TypeError(msg)


def _floating(values: Tensor) -> Tensor:
    """Return the numbers as floating-point numbers.

    Parameters
    ----------
    values : Tensor
        The numbers read.

    Returns
    -------
    Tensor
        The same numbers, converted to the default floating-point dtype where they
        were stored as whole numbers or truth values.
    """
    if values.is_floating_point():
        return values

    return values.to(torch.get_default_dtype())


def _log_distribution(
    observation: Observation, declared: DTypeClass, *, over: str
) -> LogProbabilities:
    """Return the reading as log probabilities along one axis.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    over : str
        The axis the outcomes run along, named as the reading declares it.

    Returns
    -------
    LogProbabilities
        The reading, taken as it is where the point declares log probabilities, and
        normalized by a log-softmax over the live positions of the axis where the
        point declares real numbers. An invalid position holds the log of zero, as
        does every position of a row that holds no data.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares a class that weighs no outcomes, or if the reading
        contradicts the class it declares.
    KeyError
        If the reading declares no axis of that name.
    """
    _ensure_class(declared)

    if declared is DTypeClass.LOG_PROBABILITY:
        return as_log_probabilities(observation, declared)

    if declared is DTypeClass.PROBABILITY:
        return logarithm(as_probabilities(observation, declared))

    if declared is not DTypeClass.REAL:
        msg = (
            f"{observation.site!r} declares {declared.value} numbers, which weigh "
            "no outcomes along an axis"
        )
        raise ValueError(msg)

    position = observation.position(over)
    covering = observation.covering()
    weights = _floating(observation.tensor).masked_fill(~covering, -inf)
    logs = torch.where(covering, weights.log_softmax(dim=position), LOG_ZERO)

    return LogProbabilities(
        Observation(
            observation.site, logs, axes=observation.axes, mask=observation.mask
        )
    )


def _distribution(
    observation: Observation, declared: DTypeClass, *, over: str
) -> Probabilities:
    """Return the reading as probabilities along one axis.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    over : str
        The axis the outcomes run along, named as the reading declares it.

    Returns
    -------
    Probabilities
        The reading, taken as it is where the point declares probabilities and
        raised exactly once otherwise. An invalid position holds zero.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares a class that weighs no outcomes, or if the reading
        contradicts the class it declares.
    KeyError
        If the reading declares no axis of that name.
    """
    if declared is DTypeClass.PROBABILITY:
        return as_probabilities(observation, declared)

    return exponentiate(_log_distribution(observation, declared, over=over))


def _both_domains(
    observation: Observation, declared: DTypeClass, *, over: str
) -> tuple[Tensor, Tensor]:
    """Return the probabilities along one axis beside their logarithms.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    over : str
        The axis the outcomes run along, named as the reading declares it.

    Returns
    -------
    tuple[Tensor, Tensor]
        The probabilities and the log probabilities, each raised or logged at most
        once from what was read.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares a class that weighs no outcomes, or if the reading
        contradicts the class it declares.
    KeyError
        If the reading declares no axis of that name.
    """
    if declared is DTypeClass.PROBABILITY:
        values = as_probabilities(observation, declared)

        return values.tensor, logarithm(values).tensor

    logs = _log_distribution(observation, declared, over=over)

    return exponentiate(logs).tensor, logs.tensor


def _magnitudes(observation: Observation, declared: DTypeClass) -> Tensor:
    """Return the numbers read, in the linear domain, an invalid position at zero.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.

    Returns
    -------
    Tensor
        Floating-point numbers of the reading's shape: probabilities where the point
        declares log probabilities, and the numbers as they were read otherwise.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares truth values, which have no magnitude, or if the
        reading contradicts the class it declares.
    """
    _ensure_class(declared)

    if declared is DTypeClass.LOG_PROBABILITY:
        return exponentiate(as_log_probabilities(observation, declared)).tensor

    if declared is DTypeClass.PROBABILITY:
        return as_probabilities(observation, declared).tensor

    if declared is DTypeClass.MASK:
        msg = f"{observation.site!r} declares truth values, which have no magnitude"
        raise ValueError(msg)

    return _floating(observation.tensor).masked_fill(~observation.covering(), 0.0)


def _scored(observation: Observation, scores: Tensor, *dropped: int) -> Observation:
    """Return scores as a reading over the axes a score did not reduce.

    Parameters
    ----------
    observation : Observation
        The reading the scores were computed from.
    scores : Tensor
        The scores, over the reading's other axes in their own order.
    *dropped : int
        The positions, outermost first, of the axes the scores reduced over.

    Returns
    -------
    Observation
        The scores at the reading's point, named by the axes that remain. A row is
        masked off where it held no data along the reduced axes, and the result has
        no mask where the reading had none.
    """
    kept = tuple(
        name
        for position, name in enumerate(observation.axes)
        if position not in dropped
    )
    mask = observation.mask

    if mask is not None:
        reduced = tuple(position for position in dropped if position < mask.ndim)

        if reduced:
            mask = mask.any(dim=reduced)

    return Observation(observation.site, scores, axes=kept, mask=mask)


def _paired(left: Observation, right: Observation, over: str) -> int:
    """Return which axis of two readings of one shape the named one is.

    Parameters
    ----------
    left : Observation
        One reading.
    right : Observation
        The reading it is compared with.
    over : str
        The axis the comparison runs along.

    Returns
    -------
    int
        The position of that axis, outermost first, in both readings.

    Raises
    ------
    ValueError
        If the two readings differ in their axes, their shape or their mask.
    KeyError
        If the readings declare no axis of that name.
    """
    if left.axes != right.axes:
        msg = f"{left!r} and {right!r} are not readings over one set of axes"
        raise ValueError(msg)

    if left.tensor.shape != right.tensor.shape:
        msg = f"{left!r} and {right!r} are not readings of one shape"
        raise ValueError(msg)

    if not torch.equal(left.covering(), right.covering()):
        msg = f"{left!r} and {right!r} are masked differently"
        raise ValueError(msg)

    return left.position(over)


def _sequence(observation: Observation, across: str, over: str) -> tuple[int, int]:
    """Return which axes of a reading the step axis and the vector axis are.

    Parameters
    ----------
    observation : Observation
        The reading.
    across : str
        The axis consecutive states run along.
    over : str
        The axis each state's numbers run along.

    Returns
    -------
    tuple[int, int]
        The positions of the step axis and of the vector axis.

    Raises
    ------
    ValueError
        If the two names agree.
    KeyError
        If the reading declares no axis of either name.
    """
    if across == over:
        msg = f"the steps and the numbers of a state run along one axis, {over!r}"
        raise ValueError(msg)

    return observation.position(across), observation.position(over)


def _at_least_two(observation: Observation, over: str) -> int:
    """Return which axis of a reading the named one is, if it has two positions.

    Parameters
    ----------
    observation : Observation
        The reading.
    over : str
        The axis to look up.

    Returns
    -------
    int
        Its position, outermost first.

    Raises
    ------
    ValueError
        If the axis has fewer than two positions.
    KeyError
        If the reading declares no axis of that name.
    """
    extent = observation.extent(over)

    if extent < 2:
        msg = (
            f"{observation.site!r} holds {extent} positions along {over!r}, so no "
            "gap between the two largest exists"
        )
        raise ValueError(msg)

    return observation.position(over)


def _power_spectrum(
    observation: Observation, declared: DTypeClass, over: str
) -> Tensor:
    """Return the power spectrum of each vector of a reading along one axis.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    over : str
        The axis each vector runs along.

    Returns
    -------
    Tensor
        The squared magnitude of each bin of the discrete Fourier transform, as many
        bins as the axis has positions, moved to the innermost axis. A vector of
        zeros has a spectrum of zeros.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares truth values, if the reading contradicts the class
        it declares, or if the axis has fewer than two positions.
    KeyError
        If the reading declares no axis of that name.
    """
    position = _at_least_two(observation, over)
    vectors = _magnitudes(observation, declared).movedim(position, -1)

    return torch.fft.fft(vectors, dim=-1).abs().square()


def best_two(observation: Observation, *, over: str) -> Tensor:
    """Return the two largest numbers of each row along one axis, largest first.

    Parameters
    ----------
    observation : Observation
        The reading.
    over : str
        The axis the rows run along, named as the reading declares it.

    Returns
    -------
    Tensor
        Over the reading's other axes in their own order, with an innermost axis of
        two holding the largest number and then the second largest. Every position
        is read, invalid ones included.

    Raises
    ------
    ValueError
        If the axis has fewer than two positions.
    KeyError
        If the reading declares no axis of that name.
    """
    position = _at_least_two(observation, over)

    return observation.tensor.movedim(position, -1).topk(2, dim=-1).values


def maximum_probability(
    observation: Observation, declared: DTypeClass, *, over: str
) -> Observation:
    """Return the largest outcome probability of each distribution.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    over : str
        The axis the outcomes run along, named as the reading declares it.

    Returns
    -------
    Observation
        Over the reading's other axes, the largest live probability along the axis,
        from zero to one. A row without a live outcome remains masked.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares a class that weighs no outcomes, or if the reading
        contradicts the class it declares.
    KeyError
        If the reading declares no axis of that name.
    """
    position = observation.position(over)
    values = _distribution(observation, declared, over=over)
    bounded = values.tensor.clamp(min=0.0, max=1.0)
    scores = bounded.masked_fill(~values.covering(), 0.0).amax(dim=position)

    return _scored(observation, scores, position)


def normalized_entropy(
    observation: Observation, declared: DTypeClass, *, over: str
) -> Observation:
    """Return the entropy of each distribution as a fraction of its maximum.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    over : str
        The axis the outcomes run along, named as the reading declares it.

    Returns
    -------
    Observation
        Over the reading's other axes, the entropy along the axis divided by the log
        of how many live outcomes the row has, from zero to one. A row with one live
        outcome scores zero.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares a class that weighs no outcomes, if the reading
        contradicts the class it declares, or if the axis has fewer than two
        positions.
    KeyError
        If the reading declares no axis of that name.
    """
    position = _at_least_two(observation, over)
    values = _distribution(observation, declared, over=over)
    spread = entropy(values, over=over)
    live = values.covering().sum(dim=position).to(spread.dtype)
    width = live.clamp(min=1.0).log()
    scores = torch.where(live > 1, spread / width, 0.0).clamp(min=0.0, max=1.0)

    return _scored(observation, scores, position)


def certainty(
    observation: Observation, declared: DTypeClass, *, over: str
) -> Observation:
    """Return how far each distribution is from uniform, as one minus its entropy.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    over : str
        The axis the outcomes run along, named as the reading declares it.

    Returns
    -------
    Observation
        Over the reading's other axes, one minus the normalized entropy, from zero
        for a uniform distribution to one for a distribution on one outcome.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares a class that weighs no outcomes, if the reading
        contradicts the class it declares, or if the axis has fewer than two
        positions.
    KeyError
        If the reading declares no axis of that name.
    """
    spread = normalized_entropy(observation, declared, over=over)

    return Observation(
        spread.site, 1.0 - spread.tensor, axes=spread.axes, mask=spread.mask
    )


def top_two_margin(
    observation: Observation, declared: DTypeClass, *, over: str
) -> Observation:
    """Return the gap between the two most probable outcomes of each distribution.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    over : str
        The axis the outcomes run along, named as the reading declares it.

    Returns
    -------
    Observation
        Over the reading's other axes, the largest probability minus the second
        largest, from zero to one.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares a class that weighs no outcomes, if the reading
        contradicts the class it declares, or if the axis has fewer than two
        positions.
    KeyError
        If the reading declares no axis of that name.
    """
    position = observation.position(over)
    values = _distribution(observation, declared, over=over)
    best = best_two(values.reading, over=over)
    scores = (best.select(-1, 0) - best.select(-1, 1)).clamp(min=0.0, max=1.0)

    return _scored(observation, scores, position)


def sparsity(
    observation: Observation,
    declared: DTypeClass,
    *,
    over: str,
    tolerance: float = 0.01,
) -> Observation:
    """Return the fraction of each vector that is negligible beside its largest entry.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    over : str
        The axis each vector runs along, named as the reading declares it.
    tolerance : float
        The fraction of the vector's largest magnitude at or below which an entry
        counts as negligible.

    Returns
    -------
    Observation
        Over the reading's other axes, how many live entries are negligible over how
        many are live, from zero to one. A vector of zeros scores one.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares truth values, or if the reading contradicts the class
        it declares.
    KeyError
        If the reading declares no axis of that name.
    """
    position = observation.position(over)
    size = _magnitudes(observation, declared).abs()
    covering = observation.covering()
    ceiling = size.amax(dim=position, keepdim=True)
    negligible = (size <= tolerance * ceiling) & covering
    scores = negligible.sum(dim=position) / covering.sum(dim=position).clamp(min=1)

    return _scored(observation, scores, position)


def norm_preservation(
    before: Observation, after: Observation, declared: DTypeClass, *, over: str
) -> Observation:
    """Return how nearly a transformation kept the norm of each vector.

    Parameters
    ----------
    before : Observation
        The reading the transformation was applied to.
    after : Observation
        The reading it produced, over the same axes and under the same mask.
    declared : DTypeClass
        What the points both were read at declared their numbers to mean.
    over : str
        The axis each vector runs along, named as the readings declare it.

    Returns
    -------
    Observation
        At the point the reading before was taken, over the readings' other axes,
        the sigmoid of one minus the absolute log of the ratio of the norm after to
        the norm before, from zero, where exactly one of the two norms is zero, to
        the sigmoid of one, where the norms agree.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares truth values, if a reading contradicts the class it
        declares, or if the two readings differ in their axes, shape or mask.
    KeyError
        If the readings declare no axis of that name.
    """
    position = _paired(before, after, over)
    first = _magnitudes(before, declared).norm(dim=position)
    second = _magnitudes(after, declared).norm(dim=position)
    unchanged = (first == 0) & (second == 0)
    gap = torch.where(unchanged, 0.0, (second.log() - first.log()).abs())

    return _scored(before, torch.sigmoid(1.0 - gap), position)


def consistency(
    observation: Observation, declared: DTypeClass, *, across: str, over: str
) -> Observation:
    """Return how little each distribution varies from one step to the next.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    across : str
        The axis the steps run along, named as the reading declares it.
    over : str
        The axis the outcomes run along, named as the reading declares it.

    Returns
    -------
    Observation
        Over the reading's other axes, the exponential of the negated mean, over
        outcomes, of the population variance of each outcome's probability across
        the live steps, above zero and at most one. A row with one live step scores
        one.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares a class that weighs no outcomes, if the reading
        contradicts the class it declares, or if the two axis names agree.
    KeyError
        If the reading declares no axis of either name.
    """
    step, outcome = _sequence(observation, across, over)
    values = _distribution(observation, declared, over=over)
    covering = values.covering()
    steps = covering.sum(dim=step, keepdim=True)
    mean = values.tensor.sum(dim=step, keepdim=True) / steps.clamp(min=1)
    deviation = torch.where(covering, values.tensor - mean, 0.0)
    variance = deviation.square().sum(dim=step, keepdim=True) / steps.clamp(min=1)
    present = steps > 0
    spread = variance.masked_fill(~present, 0.0).sum(
        dim=outcome, keepdim=True
    ) / present.sum(dim=outcome, keepdim=True).clamp(min=1)

    return _scored(
        observation, torch.exp(-spread.squeeze((step, outcome))), step, outcome
    )


def smoothness(
    observation: Observation, declared: DTypeClass, *, across: str, over: str
) -> Observation:
    """Return how little each distribution moves between consecutive steps.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    across : str
        The axis the steps run along, named as the reading declares it.
    over : str
        The axis the outcomes run along, named as the reading declares it.

    Returns
    -------
    Observation
        Over the reading's other axes, the exponential of the negated mean absolute
        change in probability over every outcome and every transition between two
        live steps, above zero and at most one. A row with no such transition scores
        one.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares a class that weighs no outcomes, if the reading
        contradicts the class it declares, or if the two axis names agree.
    KeyError
        If the reading declares no axis of either name.
    """
    step, outcome = _sequence(observation, across, over)
    values = _distribution(observation, declared, over=over)
    moved = values.tensor.movedim((step, outcome), (0, 1))
    covering = values.covering().movedim((step, outcome), (0, 1))
    change = (moved[1:] - moved[:-1]).abs()
    live = covering[1:] & covering[:-1]
    total = change.masked_fill(~live, 0.0).sum(dim=(0, 1))
    scores = torch.exp(-total / live.sum(dim=(0, 1)).clamp(min=1))

    return _scored(observation, scores, step, outcome)


def cosine_stability(
    observation: Observation, declared: DTypeClass, *, across: str, over: str
) -> Observation:
    """Return the mean cosine between each pair of consecutive states.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    across : str
        The axis the states run along, named as the reading declares it.
    over : str
        The axis each state's numbers run along, named as the reading declares it.

    Returns
    -------
    Observation
        Over the reading's other axes, the mean cosine similarity over every pair of
        consecutive live states, from minus one to one. A zero vector has cosine zero
        with everything, and a row with no pair of consecutive live states scores
        one.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares truth values, if the reading contradicts the class it
        declares, or if the two axis names agree.
    KeyError
        If the reading declares no axis of either name.
    """
    step, vector = _sequence(observation, across, over)
    moved = _magnitudes(observation, declared).movedim((step, vector), (0, 1))
    norms = moved.norm(dim=1, keepdim=True)
    unit = torch.where(norms > 0, moved / norms, 0.0)
    cosine = (unit[1:] * unit[:-1]).sum(dim=1).clamp(min=-1.0, max=1.0)
    live = observation.covering().movedim((step, vector), (0, 1)).any(dim=1)
    paired = live[1:] & live[:-1]
    count = paired.sum(dim=0)
    total = cosine.masked_fill(~paired, 0.0).sum(dim=0)
    scores = torch.where(count > 0, total / count.clamp(min=1), 1.0)

    return _scored(observation, scores, step, vector)


def kl_divergence_score(
    left: Observation, right: Observation, declared: DTypeClass, *, over: str
) -> Observation:
    """Return the sigmoid of the divergence of each left distribution from the right.

    Parameters
    ----------
    left : Observation
        The reading whose distributions the divergence is taken from.
    right : Observation
        The reading whose distributions it is taken against, over the same axes and
        under the same mask.
    declared : DTypeClass
        What the points both were read at declared their numbers to mean.
    over : str
        The axis the outcomes run along, named as the readings declare it.

    Returns
    -------
    Observation
        At the point the left reading was taken, over the readings' other axes, the
        sigmoid of the Kullback-Leibler divergence in nats, from one half, where the
        two agree, to one, where the right assigns zero to an outcome the left
        assigns mass.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares a class that weighs no outcomes, if a reading
        contradicts the class it declares, or if the two readings differ in their
        axes, shape or mask.
    KeyError
        If the readings declare no axis of that name.
    """
    position = _paired(left, right, over)
    p, log_p = _both_domains(left, declared, over=over)
    _, log_q = _both_domains(right, declared, over=over)
    terms = torch.where(p > 0, p * (log_p - log_q), 0.0)
    scores = torch.sigmoid(terms.sum(dim=position).clamp(min=0.0))

    return _scored(left, scores, position)


def js_divergence_score(
    left: Observation, right: Observation, declared: DTypeClass, *, over: str
) -> Observation:
    """Return the sigmoid of the divergence between each pair of distributions.

    Parameters
    ----------
    left : Observation
        One reading.
    right : Observation
        The reading it is compared with, over the same axes and under the same mask.
    declared : DTypeClass
        What the points both were read at declared their numbers to mean.
    over : str
        The axis the outcomes run along, named as the readings declare it.

    Returns
    -------
    Observation
        At the point the left reading was taken, over the readings' other axes, the
        sigmoid of the Jensen-Shannon divergence in nats, from one half, where the
        two agree, to two thirds, where they share no outcome.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares a class that weighs no outcomes, if a reading
        contradicts the class it declares, or if the two readings differ in their
        axes, shape or mask.
    KeyError
        If the readings declare no axis of that name.
    """
    divergence, position = _js_divergence(left, right, declared, over=over)
    scores = torch.sigmoid(divergence)

    return _scored(left, scores, position)


def js_divergence(
    left: Observation, right: Observation, declared: DTypeClass, *, over: str
) -> Observation:
    r"""Return the Jensen-Shannon divergence between pairs of distributions.

    Parameters
    ----------
    left : Observation
        One reading.
    right : Observation
        The reading it is compared with, over the same axes and under the same mask.
    declared : DTypeClass
        What the points both were read at declared their numbers to mean.
    over : str
        The axis the outcomes run along, named as the readings declare it.

    Returns
    -------
    Observation
        At the point the left reading was taken, over the readings' other axes, the
        Jensen-Shannon divergence in natural-log units, from zero where the two
        distributions agree to $\log 2$ where their supports are disjoint.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares a class that weighs no outcomes, if a reading
        contradicts its declaration, or if the two readings differ in axes, shape or
        mask.
    KeyError
        If the readings declare no axis of that name.
    """
    divergence, position = _js_divergence(left, right, declared, over=over)

    return _scored(left, divergence, position)


def _js_divergence(
    left: Observation, right: Observation, declared: DTypeClass, *, over: str
) -> tuple[Tensor, int]:
    """Return raw Jensen-Shannon divergence and its reduced axis.

    Parameters
    ----------
    left : Observation
        One reading.
    right : Observation
        The reading it is compared with.
    declared : DTypeClass
        What both points declare their numbers to mean.
    over : str
        The axis the outcomes run along.

    Returns
    -------
    tuple[Tensor, int]
        The divergence in nats over the remaining axes and the reduced axis position.
    """
    position = _paired(left, right, over)
    p, log_p = _both_domains(left, declared, over=over)
    q, log_q = _both_domains(right, declared, over=over)
    log_mixture = torch.logaddexp(log_p, log_q) - log(2)
    from_left = torch.where(p > 0, p * (log_p - log_mixture), 0.0)
    from_right = torch.where(q > 0, q * (log_q - log_mixture), 0.0)
    divergence = 0.5 * (from_left + from_right).sum(dim=position)

    return divergence.clamp(min=0.0, max=log(2)), position


def normalized_js_divergence(
    left: Observation, right: Observation, declared: DTypeClass, *, over: str
) -> Observation:
    r"""Return Jensen-Shannon divergence as a fraction of its maximum.

    Parameters
    ----------
    left : Observation
        One reading.
    right : Observation
        The reading it is compared with, over the same axes and under the same mask.
    declared : DTypeClass
        What the points both were read at declared their numbers to mean.
    over : str
        The axis the outcomes run along, named as the readings declare it.

    Returns
    -------
    Observation
        At the left point and over the remaining axes, Jensen-Shannon divergence in
        nats divided by $\log 2$, from zero for identical distributions to one for
        distributions with disjoint support.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares a class that weighs no outcomes, a reading contradicts
        its declaration, or the readings differ in axes, shape or mask.
    KeyError
        If the readings declare no axis of that name.
    """
    divergence, position = _js_divergence(left, right, declared, over=over)
    scores = (divergence / log(2)).clamp(min=0.0, max=1.0)

    return _scored(left, scores, position)


def energy_concentration(
    observation: Observation, declared: DTypeClass, *, over: str
) -> Observation:
    """Return how few of each vector's entries hold nine tenths of its energy.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    over : str
        The axis each vector runs along, named as the reading declares it.

    Returns
    -------
    Observation
        Over the reading's other axes, one minus the fraction of live entries that,
        taken from the largest squared magnitude down, stay within nine tenths of the
        total squared magnitude, from zero to one. A vector of zeros scores zero.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares truth values, or if the reading contradicts the class
        it declares.
    KeyError
        If the reading declares no axis of that name.
    """
    position = observation.position(over)
    energy = _magnitudes(observation, declared).movedim(position, -1).square()
    total = energy.sum(dim=-1, keepdim=True)
    ordered = energy.sort(dim=-1, descending=True).values
    within = (ordered.cumsum(dim=-1) <= 0.9 * total).sum(dim=-1)
    live = observation.covering().movedim(position, -1).sum(dim=-1)
    share = within / live.clamp(min=1)
    scores = torch.where((total.squeeze(-1) > 0) & (live > 0), 1.0 - share, 0.0)

    return _scored(observation, scores, position)


def spectral_entropy(
    observation: Observation, declared: DTypeClass, *, over: str
) -> Observation:
    """Return the entropy of each vector's power spectrum as a fraction of maximum.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    over : str
        The axis each vector runs along, named as the reading declares it.

    Returns
    -------
    Observation
        Over the reading's other axes, the entropy of the power spectrum normalized
        to unit mass, divided by the log of the number of bins, from zero for a
        spectrum on one bin to one for a flat spectrum. A vector of zeros scores
        zero.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares truth values, if the reading contradicts the class
        it declares, or if the axis has fewer than two positions.
    KeyError
        If the reading declares no axis of that name.
    """
    power = _power_spectrum(observation, declared, over)
    total = power.sum(dim=-1, keepdim=True)
    share = torch.where(total > 0, power / total, 0.0)
    spread = -torch.xlogy(share, share).sum(dim=-1)
    scores = (spread / log(power.shape[-1])).clamp(min=0.0, max=1.0)

    return _scored(observation, scores, observation.position(over))


def top_decile_power(
    observation: Observation, declared: DTypeClass, *, over: str
) -> Observation:
    """Return the fraction of each vector's power in its strongest tenth of bins.

    Parameters
    ----------
    observation : Observation
        The reading.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    over : str
        The axis each vector runs along, named as the reading declares it.

    Returns
    -------
    Observation
        Over the reading's other axes, the power in the largest tenth of the
        spectrum's bins, and at least one bin, over the total power, from zero to
        one. A vector of zeros scores zero.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declares truth values, if the reading contradicts the class
        it declares, or if the axis has fewer than two positions.
    KeyError
        If the reading declares no axis of that name.
    """
    power = _power_spectrum(observation, declared, over)
    kept = max(power.shape[-1] // 10, 1)
    strongest = power.sort(dim=-1, descending=True).values[..., :kept].sum(dim=-1)
    total = power.sum(dim=-1)
    scores = torch.where(total > 0, strongest / total, 0.0).clamp(min=0.0, max=1.0)

    return _scored(observation, scores, observation.position(over))
