"""Simultaneous distributed alignment for two dominance-related variables."""

from math import ceil
from statistics import fmean
from typing import TYPE_CHECKING, override

import torch

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.nn.sites import Access, Site
from lofi.probe._causal import (
    CombinatorAt,
    NamedVariable,
    Side,
    SplitAt,
    TypeAt,
    strictly_dominates,
)
from lofi.probe._das import AlignmentObjective, Annealing, Schedule
from lofi.probe._instrument import instrument
from lofi.probe._interventions import Interchange, JointInterchange
from lofi.probe._rotation import OrthogonalOptimizer
from lofi.probe._sweep import SweepReport
from lofi.stats import Bootstrap, BootstrapResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from torch import Tensor

    from lofi.nn.ports import Reading
    from lofi.probe._causal import AlignmentPolicy, HighLevelVariable
    from lofi.probe._counterfactuals import ModelJointCausalPair
    from lofi.probe._ports import JointSubject, ModelReadyJointCausalPair

__all__ = [
    "JointDistributedAlignment",
    "JointFitStep",
    "JointFitted",
    "JointLayerSelection",
    "JointReport",
    "dominance_pairs",
    "select_joint_layer",
]


def _variable_key(variable: HighLevelVariable) -> tuple[int, int, str, int]:
    if isinstance(variable, NamedVariable):
        msg = "structural dominance does not admit NamedVariable"
        raise TypeError(msg)
    split = variable.split if isinstance(variable, CombinatorAt) else -1
    return (variable.cell[0], -variable.cell[1], variable.kind, split)


def dominance_pairs(
    variables: Iterable[HighLevelVariable], /
) -> tuple[tuple[HighLevelVariable, HighLevelVariable], ...]:
    """Return every strict dominance pair in deterministic structural order."""
    held = tuple(variables)
    for variable in held:
        if isinstance(variable, NamedVariable):
            msg = "a dominance inventory does not admit NamedVariable"
            raise TypeError(msg)
        if type(variable) not in {TypeAt, SplitAt, CombinatorAt}:
            msg = f"a dominance inventory contains no high-level variable: {variable!r}"
            raise TypeError(msg)
    if len(set(held)) != len(held):
        msg = "a dominance inventory names a variable twice"
        raise ValueError(msg)

    ordered = tuple(sorted(held, key=_variable_key))
    return tuple(
        (outer, inner)
        for outer in ordered
        for inner in ordered
        if strictly_dominates(outer, inner)
    )


class JointLayerSelection(FrozenValue):
    """A layer selected from one complete preregistered development sweep."""

    __slots__ = (
        "_development_ids",
        "_layer",
        "_means",
        "_preregistration",
        "_site",
        "_variables",
    )

    _noun = "joint layer selection"

    def __init__(
        self,
        report: SweepReport,
        development_ids: Sequence[str],
    ) -> None:
        if type(report) is not SweepReport:
            msg = f"joint layer selection requires a SweepReport, not {report!r}"
            raise TypeError(msg)
        if report.unassessed:
            msg = (
                "joint layer selection requires every preregistered development "
                f"cell; {len(report.unassessed)} are unassessed"
            )
            raise ValueError(msg)
        subgrid = report.preregistration.subgrid
        if len(subgrid.positions) != 1:
            msg = "joint layer selection requires one preregistered position per layer"
            raise ValueError(msg)
        pairs = dominance_pairs(subgrid.variables)
        if len(subgrid.variables) != 2 or len(pairs) != 1:
            msg = (
                "joint layer selection requires exactly two preregistered variables "
                "in one strict dominance relation"
            )
            raise ValueError(msg)
        variables = pairs[0]
        means = tuple(
            (
                layer,
                fmean(
                    result.result.accuracy
                    for result in report.results
                    if result.cell.layer == layer
                ),
            )
            for layer in subgrid.layers
        )
        if any(not 0.0 <= value <= 1.0 for _, value in means):
            msg = "a mean marginal IIA is a proportion"
            raise ValueError(msg)
        maximum = max(value for _, value in means)
        selected = next(at for at, value in means if value == maximum)
        site = subgrid.sites[selected, subgrid.positions[0]]
        ids = tuple(
            ensure_name("a development pair identity", identity)
            for identity in development_ids
        )
        if not ids or len(set(ids)) != len(ids):
            msg = "development identities are nonempty and unique"
            raise ValueError(msg)

        object.__setattr__(self, "_layer", selected)
        object.__setattr__(self, "_site", site)
        object.__setattr__(self, "_variables", variables)
        object.__setattr__(self, "_means", means)
        object.__setattr__(
            self,
            "_preregistration",
            ensure_name("a preregistration digest", report.preregistration.digest),
        )
        object.__setattr__(self, "_development_ids", ids)

    @property
    def layer(self) -> int:
        """Return the selected layer index."""
        return self._layer

    @property
    def site(self) -> Site:
        """Return the published site at the selected layer."""
        return self._site

    @property
    def means(self) -> tuple[tuple[int, float], ...]:
        """Return every layer's mean marginal development IIA in declared order."""
        return self._means

    @property
    def variables(self) -> tuple[HighLevelVariable, HighLevelVariable]:
        """Return the canonical dominance pair whose marginals selected the layer."""
        return self._variables

    @property
    def preregistration(self) -> str:
        """Return the digest of the complete sweep used for selection."""
        return self._preregistration

    @property
    def development_ids(self) -> tuple[str, ...]:
        """Return the stable identities used for development selection."""
        return self._development_ids


def select_joint_layer(
    report: SweepReport, development_ids: Sequence[str], /
) -> JointLayerSelection:
    """Select the maximum mean marginal-IIA layer from a complete sweep."""
    return JointLayerSelection(report, development_ids)


class _JointMasks:
    __slots__ = ("_annealing", "_logits", "_optimizer", "_side")

    def __init__(
        self,
        like: Tensor,
        learning_rate: float,
        annealing: Annealing,
    ) -> None:
        side = int(like.shape[-1])
        if side < 2:
            msg = "a joint alignment needs at least two rotated coordinates"
            raise ValueError(msg)
        logits = torch.zeros(side, 3, device=like.device, dtype=like.dtype)
        logits[::2, 1] = 1.0
        logits[1::2, 2] = 1.0
        self._logits = logits.requires_grad_()
        self._optimizer = torch.optim.Adam([self._logits], lr=learning_rate)
        self._annealing = annealing
        self._side = side

    @property
    def parameter(self) -> Tensor:
        return self._logits

    def masks(self, step: int, total: int) -> tuple[Tensor, Tensor]:
        temperature = self._annealing.temperature(step, total)
        soft = torch.softmax(self._logits / temperature, dim=-1)
        hard = torch.nn.functional.one_hot(soft.argmax(dim=-1), num_classes=3).to(
            dtype=soft.dtype
        )
        assigned = hard + soft - soft.detach()
        return assigned[:, 1], assigned[:, 2]

    def step(self, gradient: Tensor) -> None:
        self._optimizer.zero_grad(set_to_none=True)
        self._logits.grad = gradient
        self._optimizer.step()

    def finish(self) -> tuple[Tensor, Tensor]:
        assigned = self._logits.detach().argmax(dim=-1)
        first = assigned == 1
        second = assigned == 2
        if not bool(first.any()) or not bool(second.any()):
            msg = (
                "a fitted joint alignment assigns at least one coordinate to each "
                "variable"
            )
            raise ValueError(msg)
        return first, second


class JointFitStep(FrozenValue):
    """One joint optimizer step's loss and gradient norms."""

    __slots__ = ("_loss", "_mask_gradient_norm", "_rotation_gradient_norm", "_step")

    _noun = "joint fit step"

    def __init__(
        self,
        step: int,
        loss: float,
        rotation_gradient_norm: float,
        mask_gradient_norm: float,
    ) -> None:
        object.__setattr__(self, "_step", ensure_non_negative("a joint fit step", step))
        values = (
            ensure_finite("a joint fit loss", loss),
            ensure_finite("a joint rotation gradient norm", rotation_gradient_norm),
            ensure_finite("a joint mask gradient norm", mask_gradient_norm),
        )
        if values[1] < 0.0 or values[2] < 0.0:
            msg = f"a joint fit step holds negative gradient norms: {values[1:]!r}"
            raise ValueError(msg)
        object.__setattr__(self, "_loss", values[0])
        object.__setattr__(self, "_rotation_gradient_norm", values[1])
        object.__setattr__(self, "_mask_gradient_norm", values[2])

    @property
    def step(self) -> int:
        """Return the zero-based optimizer step."""
        return self._step

    @property
    def loss(self) -> float:
        """Return the joint training loss before the step."""
        return self._loss

    @property
    def rotation_gradient_norm(self) -> float:
        """Return the common rotation's gradient norm."""
        return self._rotation_gradient_norm

    @property
    def mask_gradient_norm(self) -> float:
        """Return the two-mask parameter's gradient norm."""
        return self._mask_gradient_norm


class JointFitted[PairT: ModelReadyJointCausalPair = ModelJointCausalPair](FrozenValue):
    """One common fitted rotation, two disjoint masks, and fitting identities."""

    __slots__ = ("_first", "_pairs", "_rotation", "_second", "_trace")

    _noun = "fitted joint alignment"

    def __init__(
        self,
        rotation: Tensor,
        first_mask: Tensor,
        second_mask: Tensor,
        trace: Sequence[JointFitStep],
        pairs: Sequence[PairT],
    ) -> None:
        if not isinstance(rotation, torch.Tensor) or not rotation.is_floating_point():
            msg = "a fitted joint rotation is a floating point tensor"
            raise TypeError(msg)
        side = int(rotation.shape[0]) if rotation.ndim == 2 else -1
        if tuple(rotation.shape) != (side, side):
            msg = f"a fitted joint rotation is square, not {tuple(rotation.shape)}"
            raise ValueError(msg)
        exact = rotation.detach().to(torch.float64)
        if not bool(torch.isfinite(exact).all()):
            msg = "a fitted joint rotation is not finite"
            raise ValueError(msg)
        identity = torch.eye(side, dtype=torch.float64, device=exact.device)
        if not torch.allclose(exact.T @ exact, identity, rtol=0.0, atol=1e-4):
            msg = "a fitted joint rotation is not orthogonal"
            raise ValueError(msg)
        for name, mask in (("first", first_mask), ("second", second_mask)):
            if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool:
                msg = f"a fitted joint {name} mask is boolean"
                raise TypeError(msg)
            if tuple(mask.shape) != (side,) or mask.device != rotation.device:
                msg = f"a fitted joint {name} mask does not match its rotation"
                raise ValueError(msg)
        if bool((first_mask & second_mask).any()):
            msg = "a fitted joint alignment's masks overlap"
            raise ValueError(msg)
        if not bool(first_mask.any()) or not bool(second_mask.any()):
            msg = "a fitted joint alignment has two nonempty masks"
            raise ValueError(msg)
        held_trace = tuple(trace)
        held_pairs = tuple(pairs)
        if not held_trace or tuple(step.step for step in held_trace) != tuple(
            range(len(held_trace))
        ):
            msg = "a fitted joint trace is nonempty and contiguous from zero"
            raise ValueError(msg)
        if not held_pairs:
            msg = "a joint alignment was fitted to no pair"
            raise ValueError(msg)
        object.__setattr__(self, "_rotation", rotation.detach().clone())
        object.__setattr__(self, "_first", first_mask.detach().clone())
        object.__setattr__(self, "_second", second_mask.detach().clone())
        object.__setattr__(self, "_trace", held_trace)
        object.__setattr__(self, "_pairs", held_pairs)

    @property
    def rotation(self) -> Tensor:
        """Return the common fitted rotation."""
        return self._rotation.clone()

    @property
    def masks(self) -> tuple[Tensor, Tensor]:
        """Return fresh copies of the two disjoint masks."""
        return (self._first.clone(), self._second.clone())

    @property
    def widths(self) -> tuple[int, int]:
        """Return the two mask widths in dominance order."""
        return (int(self._first.sum().item()), int(self._second.sum().item()))

    @property
    def trace(self) -> tuple[JointFitStep, ...]:
        """Return every optimizer step in order."""
        return self._trace

    @property
    def pairs(self) -> tuple[PairT, ...]:
        """Return the pairs used to fit the rotation and masks."""
        return self._pairs


class JointReport(FrozenValue):
    """The five computed quantities of one held-out joint assessment."""

    __slots__ = (
        "_dominant_accuracy",
        "_dominated_accuracy",
        "_joint_accuracy",
        "_joint_minus_product",
        "_marginal_product",
    )

    _noun = "joint assessment report"

    def __init__(
        self,
        joint_accuracy: float,
        dominant_accuracy: float,
        dominated_accuracy: float,
        marginal_product: float,
        joint_minus_product: BootstrapResult,
    ) -> None:
        values = (
            float(joint_accuracy),
            float(dominant_accuracy),
            float(dominated_accuracy),
            float(marginal_product),
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            msg = f"joint assessment accuracies are proportions, not {values!r}"
            raise ValueError(msg)
        if abs(values[3] - values[1] * values[2]) > 1e-12:
            msg = "a joint report's marginal product is computed from its marginals"
            raise ValueError(msg)
        if not isinstance(joint_minus_product, BootstrapResult):
            msg = "joint minus product is a BootstrapResult"
            raise TypeError(msg)
        if joint_minus_product.observed != values[0] - values[3]:
            msg = "joint minus product is computed from joint accuracy and marginals"
            raise ValueError(msg)
        object.__setattr__(self, "_joint_accuracy", values[0])
        object.__setattr__(self, "_dominant_accuracy", values[1])
        object.__setattr__(self, "_dominated_accuracy", values[2])
        object.__setattr__(self, "_marginal_product", values[3])
        object.__setattr__(self, "_joint_minus_product", joint_minus_product)

    @property
    def joint_accuracy(self) -> float:
        """Return the share on which both intervened decisions matched."""
        return self._joint_accuracy

    @property
    def dominant_accuracy(self) -> float:
        """Return the dominant variable's marginal accuracy."""
        return self._dominant_accuracy

    @property
    def dominated_accuracy(self) -> float:
        """Return the dominated variable's marginal accuracy."""
        return self._dominated_accuracy

    @property
    def marginal_product(self) -> float:
        """Return the product of the two marginal accuracies."""
        return self._marginal_product

    @property
    def joint_minus_product(self) -> BootstrapResult:
        """Return the paired-bootstrap interval for joint minus marginal product."""
        return self._joint_minus_product


class JointDistributedAlignment[
    ReadingT: Reading,
    PairT: ModelReadyJointCausalPair = ModelJointCausalPair,
]:
    """A common rotation and two disjoint masks at a development-selected layer."""

    __slots__ = (
        "_annealing",
        "_bootstrap",
        "_fitted",
        "_objective",
        "_schedule",
        "_selection",
        "_subject",
        "_variables",
    )

    def __init__(
        self,
        subject: JointSubject[ReadingT, PairT],
        selection: JointLayerSelection,
        variables: tuple[HighLevelVariable, HighLevelVariable],
        *,
        schedule: Schedule,
        annealing: Annealing,
        bootstrap: Bootstrap,
        objective: AlignmentObjective,
        fitted: JointFitted[PairT] | None = None,
    ) -> None:
        if type(selection) is not JointLayerSelection:
            msg = f"a joint alignment requires a JointLayerSelection, not {selection!r}"
            raise TypeError(msg)
        if type(variables) is not tuple or len(variables) != 2:
            msg = "a joint alignment names exactly two variables"
            raise TypeError(msg)
        outer, inner = variables
        if not strictly_dominates(outer, inner):
            msg = "a joint alignment orders a strict dominant and dominated variable"
            raise ValueError(msg)
        if variables != selection.variables:
            msg = (
                f"a joint alignment names variables {variables!r}, but its development "
                f"sweep selected {selection.variables!r}"
            )
            raise ValueError(msg)
        if type(schedule) is not Schedule or type(annealing) is not Annealing:
            msg = "a joint alignment requires a Schedule and an Annealing"
            raise TypeError(msg)
        if not isinstance(bootstrap, Bootstrap):
            msg = f"a joint alignment requires a Bootstrap, not {bootstrap!r}"
            raise TypeError(msg)
        if not isinstance(objective, AlignmentObjective):
            msg = f"a joint alignment objective is not one: {objective!r}"
            raise TypeError(msg)
        if fitted is not None and type(fitted) is not JointFitted:
            msg = (
                f"a joint alignment holds a JointFitted result or None, not {fitted!r}"
            )
            raise TypeError(msg)
        self._subject = subject
        self._selection = selection
        self._variables = variables
        self._schedule = schedule
        self._annealing = annealing
        self._bootstrap = bootstrap
        self._objective = objective
        self._fitted = fitted

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the subject's span-to-token policy."""
        return self._subject.policy

    @property
    def widths(self) -> tuple[int, int]:
        """Return the fitted mask widths, or zero before fitting."""
        return (0, 0) if self._fitted is None else self._fitted.widths

    @property
    def selection(self) -> JointLayerSelection:
        """Return the development-only selected layer."""
        return self._selection

    @property
    def variables(self) -> tuple[HighLevelVariable, HighLevelVariable]:
        """Return the dominant and dominated variables."""
        return self._variables

    @property
    def objective(self) -> AlignmentObjective:
        """Return the declared loss the joint fit optimizes."""
        return self._objective

    @property
    def fitted(self) -> JointFitted[PairT] | None:
        """Return the joint fit, if one has been performed."""
        return self._fitted

    def _population(self, pairs: Sequence[PairT], purpose: str) -> tuple[PairT, ...]:
        held = tuple(pairs)
        if not held:
            msg = f"a joint alignment {purpose} at least one pair"
            raise ValueError(msg)
        ids = tuple(ensure_name("a joint pair identity", pair.id) for pair in held)
        if len(set(ids)) != len(ids):
            msg = f"a joint {purpose} population repeats an identity"
            raise ValueError(msg)
        for pair in held:
            if pair.variables != self._variables:
                msg = (
                    f"a joint alignment for {self._variables!r} received "
                    f"{pair.variables!r}"
                )
                raise ValueError(msg)
            if pair.policy != self.policy:
                msg = "a joint pair's alignment policy differs from the subject's"
                raise ValueError(msg)
        return held

    def _ensure_access(self) -> None:
        site = self._selection.site
        published = self._subject.site_map()
        if site not in published:
            msg = f"{site!r} is not published by the joint subject"
            raise ValueError(msg)
        specification = published[site]
        if (
            not specification.differentiable
            or Access.OBSERVE not in specification.access
            or Access.PATCH not in specification.access
        ):
            msg = f"{site!r} is not differentiably observable and patchable"
            raise ValueError(msg)

    def _labels(self, pairs: tuple[PairT, ...]) -> tuple[Tensor, Tensor]:
        labels: list[Tensor] = []
        for variable in self._variables:
            decisions = self._subject.decisions(variable)
            rows: list[int] = []
            for pair in pairs:
                decided = pair.variant[variable]
                if decided not in decisions:
                    msg = f"joint decision {decided!r} is outside {decisions!r}"
                    raise ValueError(msg)
                rows.append(decisions.index(decided))
            labels.append(torch.tensor(rows))
        return labels[0], labels[1]

    def _sources(self, pairs: tuple[PairT, ...]) -> Tensor:
        site = self._selection.site
        with (
            torch.no_grad(),
            instrument(self._subject, observe=frozenset({site})) as run,
        ):
            run(lambda: self._subject.run(pairs, Side.VARIANT))
        return run.trace[site].tensor

    def _scores(
        self,
        pairs: tuple[PairT, ...],
        rotation: Tensor,
        masks: tuple[Tensor, Tensor],
        source: Tensor,
        mode: int,
    ) -> tuple[Tensor, Tensor]:
        if mode == 2:
            patch = JointInterchange(rotation, masks[0], masks[1], source)
        elif mode in {0, 1}:
            patch = Interchange(rotation, masks[mode], source)
        else:
            msg = f"no joint scoring mode is numbered {mode}"
            raise ValueError(msg)
        with instrument(self._subject, patch={self._selection.site: patch}) as run:
            scores = run(lambda: self._subject.run(pairs, Side.BASE))
        if type(scores) is not tuple or len(scores) != 2:
            msg = "a joint subject returns two score matrices"
            raise TypeError(msg)
        for variable, matrix in zip(self._variables, scores, strict=True):
            expected = (len(pairs), len(self._subject.decisions(variable)))
            if tuple(matrix.shape) != expected:
                msg = f"joint scores have shape {tuple(matrix.shape)}, not {expected}"
                raise ValueError(msg)
        return scores

    def _batches(
        self, pairs: tuple[PairT, ...], order: Tensor
    ) -> tuple[tuple[tuple[PairT, ...], Tensor], ...]:
        return tuple(
            (
                tuple(
                    pairs[at]
                    for at in order[start : start + self._schedule.batch_size].tolist()
                ),
                order[start : start + self._schedule.batch_size],
            )
            for start in range(0, len(pairs), self._schedule.batch_size)
        )

    def fit(self, pairs: Sequence[PairT]) -> JointDistributedAlignment[ReadingT, PairT]:
        """Fit one common rotation and two disjoint masks."""
        held = self._population(pairs, "fits")
        if {pair.id for pair in held} & set(self._selection.development_ids):
            msg = "joint fitting identities overlap development identities"
            raise ValueError(msg)
        self._ensure_access()
        labels = self._labels(held)
        draws = torch.Generator().manual_seed(self._schedule.seed)
        total = self._schedule.epochs * ceil(len(held) / self._schedule.batch_size)
        rotation: OrthogonalOptimizer | None = None
        masks: _JointMasks | None = None
        trace: list[JointFitStep] = []
        step = 0
        for _ in range(self._schedule.epochs):
            order = torch.randperm(len(held), generator=draws)
            for batch, positions in self._batches(held, order):
                source = self._sources(batch)
                if rotation is None or masks is None:
                    rotation = OrthogonalOptimizer(
                        source, self._schedule.learning_rate, draws
                    )
                    masks = _JointMasks(
                        source, self._schedule.learning_rate, self._annealing
                    )
                current_masks = masks.masks(step, total)
                joint = self._scores(batch, rotation.matrix(), current_masks, source, 2)
                first = self._scores(
                    batch, rotation.matrix(), current_masks, source, 0
                )[0]
                second = self._scores(
                    batch, rotation.matrix(), current_masks, source, 1
                )[1]
                wanted = (
                    labels[0][positions].to(first.device),
                    labels[1][positions].to(second.device),
                )
                losses = tuple(
                    self._objective.loss(scores, target)
                    for scores, target in (
                        (first, wanted[0]),
                        (second, wanted[1]),
                        (joint[0], wanted[0]),
                        (joint[1], wanted[1]),
                    )
                )
                loss = losses[0] + losses[1] + losses[2] + losses[3]
                gradients = rotation.gradients_of(loss, (masks.parameter,))
                trace.append(
                    JointFitStep(
                        step,
                        float(loss.detach()),
                        float(gradients[0].detach().norm()),
                        float(gradients[1].detach().norm()),
                    )
                )
                rotation.step(gradients[0])
                masks.step(gradients[1])
                step += 1
        if rotation is None or masks is None:
            msg = "a joint fit performed no optimizer step"
            raise RuntimeError(msg)
        first_mask, second_mask = masks.finish()
        fitted = JointFitted(rotation.finish(), first_mask, second_mask, trace, held)
        return type(self)(
            self._subject,
            self._selection,
            self._variables,
            schedule=self._schedule,
            annealing=self._annealing,
            bootstrap=self._bootstrap,
            objective=self._objective,
            fitted=fitted,
        )

    def _require_fitted(self) -> JointFitted[PairT]:
        if self._fitted is None:
            msg = "a joint alignment must be fitted before held-out assessment"
            raise ValueError(msg)
        return self._fitted

    def _held_out_outcomes(
        self, pairs: Sequence[PairT]
    ) -> tuple[tuple[bool, bool, bool], ...]:
        """Return joint, dominant-only, and dominated-only outcomes per pair."""
        fitted = self._require_fitted()
        held = self._population(pairs, "evaluates")
        training = {pair.id for pair in fitted.pairs}
        evaluation = {pair.id for pair in held}
        if evaluation & training:
            msg = "joint evaluation identities overlap fitting identities"
            raise ValueError(msg)
        if evaluation & set(self._selection.development_ids):
            msg = "joint evaluation identities overlap development identities"
            raise ValueError(msg)
        labels = self._labels(held)
        fitted_first, fitted_second = fitted.masks
        dtype = fitted.rotation.dtype
        masks = (
            fitted_first.to(dtype=dtype),
            fitted_second.to(dtype=dtype),
        )
        order = torch.arange(len(held))
        outcomes: list[tuple[bool, bool, bool]] = [(False, False, False)] * len(held)
        with torch.no_grad():
            for batch, positions in self._batches(held, order):
                source = self._sources(batch)
                rotation = fitted.rotation
                joint_scores = self._scores(batch, rotation, masks, source, 2)
                dominant_scores = self._scores(batch, rotation, masks, source, 0)[0]
                dominated_scores = self._scores(batch, rotation, masks, source, 1)[1]
                wanted_dominant = labels[0][positions].to(joint_scores[0].device)
                wanted_dominated = labels[1][positions].to(joint_scores[1].device)
                joint_dominant = joint_scores[0].argmax(dim=-1) == wanted_dominant
                joint_dominated = joint_scores[1].argmax(dim=-1) == wanted_dominated
                dominant = dominant_scores.argmax(dim=-1) == labels[0][positions].to(
                    dominant_scores.device
                )
                dominated = dominated_scores.argmax(dim=-1) == labels[1][positions].to(
                    dominated_scores.device
                )
                for position, both, one, two in zip(
                    positions.tolist(),
                    (joint_dominant & joint_dominated).tolist(),
                    dominant.tolist(),
                    dominated.tolist(),
                    strict=True,
                ):
                    outcomes[position] = (both, one, two)
        return tuple(outcomes)

    def matches(self, pairs: Sequence[PairT]) -> tuple[tuple[bool, bool], ...]:
        """Return dominant-only and dominated-only held-out outcomes per pair."""
        return tuple(
            (dominant, dominated)
            for _, dominant, dominated in self._held_out_outcomes(pairs)
        )

    def report(self, pairs: Sequence[PairT]) -> JointReport:
        """Compute the two marginals, product, joint accuracy, and paired interval."""
        outcomes = self._held_out_outcomes(pairs)
        joint = sum(both for both, _, _ in outcomes) / len(outcomes)
        dominant = sum(first for _, first, _ in outcomes) / len(outcomes)
        dominated = sum(second for _, _, second in outcomes) / len(outcomes)
        product = dominant * dominated

        def difference(sample: Sequence[tuple[bool, bool, bool]]) -> float:
            together = sum(both for both, _, _ in sample) / len(sample)
            first = sum(one for _, one, _ in sample) / len(sample)
            second = sum(two for _, _, two in sample) / len(sample)
            return together - first * second

        interval = self._bootstrap.interval(outcomes, difference)
        return JointReport(joint, dominant, dominated, product, interval)

    @override
    def __repr__(self) -> str:
        """Return the selected layer, variables, widths, and fit state."""
        return (
            f"JointDistributedAlignment(layer={self._selection.layer!r}, "
            f"variables={self._variables!r}, widths={self.widths!r}, "
            f"objective={self._objective.value!r}, "
            f"fitted={self._fitted is not None!r})"
        )
