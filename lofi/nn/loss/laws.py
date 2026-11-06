"""Objective and parser-output conformance laws."""

from typing import TYPE_CHECKING, Final, NamedTuple

import torch

from lofi._checks import fail_law
from lofi.nn.loss._errors import UndeclaredOutputError
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._values import OUTPUT_NAMES, RecordingOutputs

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.loss._parser import SpanMarginals, SplitPosterior
    from lofi.nn.loss._ports import LossTerm, Objective, PublishedOutputs
    from lofi.nn.loss._values import ObjectiveOutputs

__all__ = [
    "OBJECTIVE_LAWS",
    "PARSER_OUTPUT_LAWS",
    "ObjectiveFixture",
    "ParserOutputFixture",
    "a_masked_position_does_not_move_the_value",
    "an_inapplicable_term_is_a_connected_zero",
    "check_objective_laws",
    "check_parser_output_laws",
    "constituent_marginals_sum_to_one_less_than_the_length",
    "no_didactic_model_is_built",
    "only_the_declared_outputs_are_read",
    "permuting_the_items_leaves_the_value_alone",
    "split_mass_sums_to_one_over_the_valid_cutpoints",
    "the_gradient_reaches_the_declared_inputs",
    "the_objective_applies_no_weight_of_its_own",
    "the_same_seed_gives_the_same_value",
    "the_value_is_a_scalar",
    "the_value_is_monotone_where_the_fixture_says",
]


def _reseed() -> None:
    """Fix the global tensor seed for objective-law checks."""
    torch.manual_seed(0)


class ObjectiveFixture[BatchT](NamedTuple):
    """Inputs for objective conformance laws.

    Parameters
    ----------
    objective : Objective[PublishedOutputs[BatchT]]
        The bound objective under test.
    outputs : Callable[[], ObjectiveOutputs[BatchT]]
        Return a bundle the objective applies to.
    inputs : Callable[[ObjectiveOutputs[BatchT]], tuple[Tensor, ...]]
        Return the leaf tensors a gradient must reach, given a bundle.
    unweighted : Callable[[ObjectiveOutputs[BatchT]], float]
        Return the quantity the objective is supposed to report for a bundle, computed
        independently of the objective.
    monotone : Callable[ [], tuple[ObjectiveOutputs[BatchT], ObjectiveOutputs[BatchT]]]
        Return two bundles the objective must order, the first scoring strictly higher
        than the second.
    models_built : Callable[[Callable[[], LossTerm]], int]
        Run a call and return how many didactic models it constructed.
    permuted : Callable[[ObjectiveOutputs[BatchT]], ObjectiveOutputs[BatchT]]
        Return the bundle with its items in another order.
    masked : Callable[[ObjectiveOutputs[BatchT]], ObjectiveOutputs[BatchT]]
        Return the bundle with the positions its masks exclude filled with other
        numbers.
    reseed : Callable[[], None]
        Restore the random state required by the determinism law.
    inapplicable : Callable[[], ObjectiveOutputs[BatchT]] | None
        Return a bundle the objective does not apply to, or None where it applies to
        every batch, which the law then checks it does.
    tolerance : float
        How far two values that should agree may sit apart.
    """

    objective: Objective[PublishedOutputs[BatchT]]
    outputs: Callable[[], ObjectiveOutputs[BatchT]]
    inputs: Callable[[ObjectiveOutputs[BatchT]], tuple[Tensor, ...]]
    unweighted: Callable[[ObjectiveOutputs[BatchT]], float]
    monotone: Callable[
        [],
        tuple[ObjectiveOutputs[BatchT], ObjectiveOutputs[BatchT]],
    ]
    models_built: Callable[[Callable[[], LossTerm]], int]
    permuted: Callable[[ObjectiveOutputs[BatchT]], ObjectiveOutputs[BatchT]]
    masked: Callable[[ObjectiveOutputs[BatchT]], ObjectiveOutputs[BatchT]]
    reseed: Callable[[], None] = _reseed
    inapplicable: Callable[[], ObjectiveOutputs[BatchT]] | None = None
    tolerance: float = 1e-6


class ParserOutputFixture(NamedTuple):
    """Inputs for parser-output conformance laws.

    Parameters
    ----------
    lengths : tuple[int, ...]
        The sentence lengths to run at.
    parse : Callable[[int], tuple[SplitPosterior, SpanMarginals]]
        Return the split posterior and span marginals one parser published for a batch
        of items of the given length.
    tolerance : float
        How far a mass that should be one may sit from one.
    """

    lengths: tuple[int, ...]
    parse: Callable[[int], tuple[SplitPosterior, SpanMarginals]]
    tolerance: float = 1e-5


def _term[BatchT](
    fixture: ObjectiveFixture[BatchT],
    outputs: ObjectiveOutputs[BatchT],
) -> LossTerm:
    """Return the objective result for one output bundle.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.
    outputs : ObjectiveOutputs[BatchT]
        The bundle to score.

    Returns
    -------
    LossTerm
        The term the objective returned.
    """
    return fixture.objective.evaluate(outputs)


def _value[BatchT](
    fixture: ObjectiveFixture[BatchT],
    outputs: ObjectiveOutputs[BatchT],
) -> float:
    """Return the objective's scalar contribution for one bundle.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.
    outputs : ObjectiveOutputs[BatchT]
        The bundle to score.

    Returns
    -------
    float
        The term's value, read off the device.
    """
    return float(_term(fixture, outputs).value.detach())


def the_value_is_a_scalar[BatchT](fixture: ObjectiveFixture[BatchT]) -> None:
    """Check that a term reports a scalar.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.

    Raises
    ------
    LawViolationError
        If the value, the residual or a diagnostic has any dimension, or if the value is
        not of floating dtype.
    """
    law = "the_value_is_a_scalar"
    term = _term(fixture, fixture.outputs())
    reported = [("value", term.value), *term.diagnostics.items()]

    if term.residual is not None:
        reported.append(("residual", term.residual))

    for name, tensor in reported:
        if tensor.ndim != 0:
            fail_law(f"{name} has shape {tuple(tensor.shape)}", law=law)

    if not term.value.is_floating_point():
        fail_law(f"the value is of dtype {term.value.dtype}", law=law)


def an_inapplicable_term_is_a_connected_zero[BatchT](
    fixture: ObjectiveFixture[BatchT],
) -> None:
    """Check that a term that does not apply is a zero still on the graph.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.

    Raises
    ------
    LawViolationError
        If the term the fixture calls inapplicable reports items, is not zero, is
        detached from the graph, or leaves a gradient anywhere.
    """
    law = "an_inapplicable_term_is_a_connected_zero"

    if fixture.inapplicable is None:
        applied = _term(fixture, fixture.outputs()).count

        if applied == 0:
            fail_law(
                "the fixture names no inapplicable batch, and the objective "
                "did not apply to the batch it does name",
                law=law,
            )

        return

    outputs = fixture.inapplicable()
    term = _term(fixture, outputs)

    if term.count != 0:
        fail_law(f"the term reports {term.count} items", law=law)

    if float(term.value.detach()) != 0.0:
        fail_law(f"the term is {float(term.value.detach())} rather than zero", law=law)

    if term.value.grad_fn is None:
        fail_law("the zero is detached from the batch it came from", law=law)

    for position, gradient in enumerate(_gradients(fixture, outputs, term)):
        if float(gradient.abs().max()) != 0.0:
            fail_law(f"input {position} took a gradient from a zero term", law=law)


def only_the_declared_outputs_are_read[BatchT](
    fixture: ObjectiveFixture[BatchT],
) -> None:
    """Check that an objective reads only declared outputs.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.

    Raises
    ------
    LawViolationError
        If the objective declares a name outside the published vocabulary, reaches for a
        field it did not declare, or reads nothing at all.
    """
    law = "only_the_declared_outputs_are_read"
    objective = fixture.objective
    unknown = objective.requires - OUTPUT_NAMES

    if unknown:
        fail_law(f"it declares {', '.join(sorted(unknown))}", law=law)

    view = RecordingOutputs(
        fixture.outputs(),
        objective=objective.name,
        requires=objective.requires,
    )

    try:
        objective.evaluate(view)

    except UndeclaredOutputError as error:
        fail_law(str(error), law=law)

    if not view.touched:
        fail_law("it read nothing through the bundle it was handed", law=law)


def a_masked_position_does_not_move_the_value[BatchT](
    fixture: ObjectiveFixture[BatchT],
) -> None:
    """Check that masked positions do not affect the value.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.

    Raises
    ------
    LawViolationError
        If refilling the positions the masks exclude moves the value.
    """
    law = "a_masked_position_does_not_move_the_value"
    outputs = fixture.outputs()
    before = _value(fixture, outputs)
    after = _value(fixture, fixture.masked(outputs))

    if abs(before - after) > fixture.tolerance:
        fail_law(f"the value moved from {before} to {after}", law=law)


def the_value_is_monotone_where_the_fixture_says[BatchT](
    fixture: ObjectiveFixture[BatchT],
) -> None:
    """Check the ordering the objective declares it imposes.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.

    Raises
    ------
    LawViolationError
        If the bundle the fixture names as scoring higher does not.
    """
    law = "the_value_is_monotone_where_the_fixture_says"
    higher, lower = fixture.monotone()
    above = _value(fixture, higher)
    below = _value(fixture, lower)

    if not above > below + fixture.tolerance:
        fail_law(f"the higher batch scored {above} and the lower {below}", law=law)


def the_gradient_reaches_the_declared_inputs[BatchT](
    fixture: ObjectiveFixture[BatchT],
) -> None:
    """Check that a backward pass reaches every input the fixture names.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.

    Raises
    ------
    LawViolationError
        If the objective does not apply to the bundle the fixture names, if an input
        takes no gradient, or if the gradient it takes is zero or not finite everywhere.
    """
    law = "the_gradient_reaches_the_declared_inputs"
    outputs = fixture.outputs()
    term = _term(fixture, outputs)

    if term.count == 0:
        fail_law("the objective does not apply to the batch the fixture names", law=law)

    for position, gradient in enumerate(_gradients(fixture, outputs, term)):
        if not bool(torch.isfinite(gradient).all()):
            fail_law(f"input {position} took a gradient that is not finite", law=law)

        if float(gradient.abs().max()) == 0.0:
            fail_law(f"input {position} took an all-zero gradient", law=law)


def the_objective_applies_no_weight_of_its_own[BatchT](
    fixture: ObjectiveFixture[BatchT],
) -> None:
    """Check that the term is the raw quantity and that the suite scales it.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.

    Raises
    ------
    LawViolationError
        If the term is not the quantity the fixture names, if it differs between two
        compositions, or if the total is not the weight times the term.
    """
    law = "the_objective_applies_no_weight_of_its_own"
    name = fixture.objective.name
    outputs = fixture.outputs()
    wanted = fixture.unweighted(outputs)
    unweighted: list[float] = []

    # a zero weight must remove the term from the total
    for weight in (1.0, 3.0, 0.0):
        suite = ObjectiveSuite([(weight, fixture.objective)])
        result = suite.evaluate(fixture.outputs())
        term = float(result.terms[name].value.detach())
        total = float(result.total.detach())
        unweighted.append(term)

        if abs(total - weight * term) > fixture.tolerance:
            fail_law(
                f"at weight {weight} the term is {term} and the total {total}",
                law=law,
            )

    if max(unweighted) - min(unweighted) > fixture.tolerance:
        fail_law(f"the term took the values {unweighted} across weights", law=law)

    reported = unweighted[0]

    if abs(reported - wanted) > fixture.tolerance:
        fail_law(f"it reported {reported} where the quantity is {wanted}", law=law)


def permuting_the_items_leaves_the_value_alone[BatchT](
    fixture: ObjectiveFixture[BatchT],
) -> None:
    """Check that the term does not depend on how the batch was ordered.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.

    Raises
    ------
    LawViolationError
        If reordering the items moves the value.
    """
    law = "permuting_the_items_leaves_the_value_alone"
    outputs = fixture.outputs()
    before = _value(fixture, outputs)
    after = _value(fixture, fixture.permuted(outputs))

    if abs(before - after) > fixture.tolerance:
        fail_law(f"the value moved from {before} to {after}", law=law)


def the_same_seed_gives_the_same_value[BatchT](
    fixture: ObjectiveFixture[BatchT],
) -> None:
    """Check that two runs from one random state agree exactly.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.

    Raises
    ------
    LawViolationError
        If the two values differ at all.
    """
    law = "the_same_seed_gives_the_same_value"
    fixture.reseed()
    first = _value(fixture, fixture.outputs())
    fixture.reseed()
    second = _value(fixture, fixture.outputs())

    if first != second:
        fail_law(f"one run gave {first} and the next {second}", law=law)


def no_didactic_model_is_built[BatchT](fixture: ObjectiveFixture[BatchT]) -> None:
    """Check that scoring a step constructs no boundary object.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.

    Raises
    ------
    LawViolationError
        If the fixture's counter reports any construction.
    """
    law = "no_didactic_model_is_built"
    outputs = fixture.outputs()
    built = fixture.models_built(lambda: _term(fixture, outputs))

    if built != 0:
        fail_law(f"scoring one step built {built} of them", law=law)


def split_mass_sums_to_one_over_the_valid_cutpoints(
    fixture: ParserOutputFixture,
) -> None:
    """Check that each span's cutpoints hold all of the probability, or none.

    Parameters
    ----------
    fixture : ParserOutputFixture
        The parser to run and the lengths to run it at.

    Raises
    ------
    LawViolationError
        If a span's valid cutpoints hold anything other than the whole mass, or if a
        span with no cutpoint holds any.
    """
    law = "split_mass_sums_to_one_over_the_valid_cutpoints"

    for length in fixture.lengths:
        posterior, _ = fixture.parse(length)
        mass = posterior.mass()
        divides = posterior.valid.any(dim=posterior.valid.ndim - 1)
        wanted = torch.where(divides, mass.new_ones(()), mass.new_zeros(()))
        worst = float((mass - wanted).abs().max()) if mass.numel() else 0.0

        if worst > fixture.tolerance:
            fail_law(
                f"at length {length} a span's cutpoints hold {worst} more or "
                "less than the whole of its mass",
                law=law,
            )


def constituent_marginals_sum_to_one_less_than_the_length(
    fixture: ParserOutputFixture,
) -> None:
    """Check that an item's constituent mass counts the nodes a tree has.

    Parameters
    ----------
    fixture : ParserOutputFixture
        The parser to run and the lengths to run it at.

    Raises
    ------
    LawViolationError
        If an item's constituent mass is not one less than its length.
    """
    law = "constituent_marginals_sum_to_one_less_than_the_length"

    for length in fixture.lengths:
        _, marginals = fixture.parse(length)
        mass = marginals.constituent_mass()
        wanted = (marginals.lengths - 1).to(mass.dtype)
        worst = float((mass - wanted).abs().max()) if mass.numel() else 0.0

        if worst > fixture.tolerance:
            fail_law(
                f"at length {length} an item's constituent mass is {worst} "
                f"away from one less than its length",
                law=law,
            )


def _gradients[BatchT](
    fixture: ObjectiveFixture[BatchT],
    outputs: ObjectiveOutputs[BatchT],
    term: LossTerm,
) -> tuple[Tensor, ...]:
    """Return the gradient the term leaves at each input the fixture names.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.
    outputs : ObjectiveOutputs[BatchT]
        The bundle the term was computed from.
    term : LossTerm
        The term to differentiate.

    Returns
    -------
    tuple[Tensor, ...]
        One gradient per named input, each of that input's shape and dtype.

    Raises
    ------
    LawViolationError
        If the fixture names no input, if an input takes no gradient at all, or if the
        term is detached from them.
    """
    law = "the_gradient_reaches_the_declared_inputs"
    inputs = fixture.inputs(outputs)

    if not inputs:
        fail_law("the fixture names no input for a gradient to reach", law=law)

    if term.value.grad_fn is None:
        fail_law("the term is detached from the bundle it was computed from", law=law)

    gradients = torch.autograd.grad(
        term.value,
        inputs,
        allow_unused=True,
        materialize_grads=True,
    )

    return tuple(gradients)


OBJECTIVE_LAWS: Final = (
    the_value_is_a_scalar,
    an_inapplicable_term_is_a_connected_zero,
    only_the_declared_outputs_are_read,
    a_masked_position_does_not_move_the_value,
    the_value_is_monotone_where_the_fixture_says,
    the_gradient_reaches_the_declared_inputs,
    the_objective_applies_no_weight_of_its_own,
    permuting_the_items_leaves_the_value_alone,
    the_same_seed_gives_the_same_value,
    no_didactic_model_is_built,
)
"""Objective laws in execution order."""

PARSER_OUTPUT_LAWS: Final = (
    split_mass_sums_to_one_over_the_valid_cutpoints,
    constituent_marginals_sum_to_one_less_than_the_length,
)
"""Parser-output laws in execution order."""


def check_objective_laws[BatchT](fixture: ObjectiveFixture[BatchT]) -> None:
    """Run every objective law against one objective.

    Parameters
    ----------
    fixture : ObjectiveFixture[BatchT]
        What the laws run against.

    Raises
    ------
    LawViolationError
        On the first law that does not hold.
    """
    for law in OBJECTIVE_LAWS:
        law(fixture)


def check_parser_output_laws(fixture: ParserOutputFixture) -> None:
    """Run every parser-output law against one parser.

    Parameters
    ----------
    fixture : ParserOutputFixture
        The parser to run and the lengths to run it at.

    Raises
    ------
    LawViolationError
        On the first law that does not hold.
    """
    for law in PARSER_OUTPUT_LAWS:
        law(fixture)
