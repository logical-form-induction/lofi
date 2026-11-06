"""Pure simulation-calibration and parameter-recovery summaries."""

# The reducer and its input value intentionally share immutable tensor snapshots.
# ruff: noqa: SLF001

from collections.abc import Mapping, Sequence
from math import sqrt
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

import torch
from scipy.stats import kstest

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.judgment._spec import InferenceMethod

if TYPE_CHECKING:
    from torch import Tensor

    from lofi._types import JsonValue

__all__ = [
    "CalibrationReplicate",
    "CalibrationReport",
    "CoverageSummary",
    "CutpointRecovery",
    "SbcBlockSummary",
    "calibration_report",
]

_METHODS: Final = (
    InferenceMethod.NUTS,
    InferenceMethod.STRUCTURED,
    InferenceMethod.MEAN_FIELD,
)
_METHOD_SET: Final = frozenset(_METHODS)
_COVERAGE_LEVEL: Final = 0.9
_RECOVERY_FLOOR: Final = 0.8
_KS_ALPHA: Final = 0.05


def _tensor(field: str, value: Tensor, *, dimensions: int | None = None) -> Tensor:
    """Return a finite floating-point CPU snapshot of one tensor."""
    if not isinstance(value, torch.Tensor):
        msg = f"{field} is not a tensor: {value!r}"
        raise TypeError(msg)

    if dimensions is not None and value.ndim != dimensions:
        msg = f"{field} has shape {tuple(value.shape)!r}, not {dimensions} dimensions"
        raise ValueError(msg)

    if value.numel() == 0:
        msg = f"{field} is empty"
        raise ValueError(msg)

    if not value.is_floating_point() or value.is_complex():
        msg = f"{field} is not a real floating-point tensor"
        raise TypeError(msg)

    if not bool(torch.isfinite(value).all()):
        msg = f"{field} contains a nonfinite value"
        raise ValueError(msg)

    return value.detach().to(device="cpu").clone()


def _method_mapping[T](
    field: str,
    values: Mapping[InferenceMethod, T],
) -> dict[InferenceMethod, T]:
    """Return a complete method mapping in preregistered order."""
    if not isinstance(values, Mapping):
        msg = f"{field} is not a mapping"
        raise TypeError(msg)

    invalid = tuple(key for key in values if not isinstance(key, InferenceMethod))
    if invalid:
        msg = f"{field} has non-InferenceMethod keys: {invalid!r}"
        raise TypeError(msg)

    methods = frozenset(values)
    if methods != _METHOD_SET:
        missing = tuple(method.value for method in _METHODS if method not in methods)
        extra = tuple(sorted(method.value for method in methods - _METHOD_SET))
        msg = (
            f"{field} must contain exactly {_METHODS!r}; "
            f"missing={missing!r}, extra={extra!r}"
        )
        raise ValueError(msg)

    return {method: values[method] for method in _METHODS}


@final
class CalibrationReplicate(FrozenValue):
    """One prior predictive truth and three posterior approximations.

    Posterior tensors have shape ``(chain, draw, *truth_shape)``. Cutpoint
    truths have shape ``(participant, cutpoint)`` and their draws prepend the
    same explicit chain and draw axes.
    """

    _noun = "calibration replicate"

    __slots__ = (
        "_cutpoint_draws",
        "_posterior_draws",
        "_seed",
        "_true_cutpoints",
        "_truths",
    )

    def __init__(
        self,
        *,
        seed: int,
        truths: Mapping[str, Tensor],
        posterior_draws: Mapping[InferenceMethod, Mapping[str, Tensor]],
        true_cutpoints: Tensor,
        cutpoint_draws: Mapping[InferenceMethod, Tensor],
    ) -> None:
        checked_seed = ensure_non_negative("a calibration replicate's seed", seed)
        if not isinstance(truths, Mapping):
            msg = "a calibration replicate's truths are not a mapping"
            raise TypeError(msg)
        if not truths:
            msg = "a calibration replicate has no parameter blocks"
            raise ValueError(msg)

        held_truths: dict[str, Tensor] = {}
        for raw_block, truth in truths.items():
            block = ensure_name("a calibration parameter block", raw_block)
            if block in held_truths:
                msg = f"a calibration replicate repeats block {block!r}"
                raise ValueError(msg)
            held_truths[block] = _tensor(f"truth block {block!r}", truth)
        held_truths = {block: held_truths[block] for block in sorted(held_truths)}

        supplied_draws = _method_mapping("posterior draws", posterior_draws)
        held_draws: dict[InferenceMethod, Mapping[str, Tensor]] = {}
        for method, method_draws in supplied_draws.items():
            if not isinstance(method_draws, Mapping):
                msg = f"posterior draws for {method.value!r} are not a mapping"
                raise TypeError(msg)
            if frozenset(method_draws) != frozenset(held_truths):
                msg = (
                    f"posterior draws for {method.value!r} do not have the truth "
                    f"blocks {tuple(held_truths)!r}"
                )
                raise ValueError(msg)

            checked: dict[str, Tensor] = {}
            leading: tuple[int, int] | None = None
            for block, truth in held_truths.items():
                draws = _tensor(
                    f"posterior draws for {method.value!r} block {block!r}",
                    method_draws[block],
                )
                if draws.ndim != truth.ndim + 2 or tuple(draws.shape[2:]) != tuple(
                    truth.shape
                ):
                    msg = (
                        f"posterior draws for {method.value!r} block {block!r} have "
                        f"shape {tuple(draws.shape)!r}, not (chain, draw, "
                        f"*{tuple(truth.shape)!r})"
                    )
                    raise ValueError(msg)
                if draws.shape[0] < 1 or draws.shape[1] < 1:
                    msg = (
                        f"posterior draws for {method.value!r} block {block!r} "
                        "have an empty chain or draw axis"
                    )
                    raise ValueError(msg)
                current = (draws.shape[0], draws.shape[1])
                if leading is not None and current != leading:
                    msg = (
                        f"posterior blocks for {method.value!r} have unequal "
                        "chain/draw axes"
                    )
                    raise ValueError(msg)
                leading = current
                checked[block] = draws
            held_draws[method] = MappingProxyType(checked)

        truth_cutpoints = _tensor(
            "a calibration replicate's true cutpoints", true_cutpoints, dimensions=2
        )
        supplied_cutpoints = _method_mapping("cutpoint draws", cutpoint_draws)
        held_cutpoints: dict[InferenceMethod, Tensor] = {}
        for method, raw_draws in supplied_cutpoints.items():
            draws = _tensor(
                f"cutpoint draws for {method.value!r}", raw_draws, dimensions=4
            )
            if tuple(draws.shape[2:]) != tuple(truth_cutpoints.shape):
                msg = (
                    f"cutpoint draws for {method.value!r} have shape "
                    f"{tuple(draws.shape)!r}, not (chain, draw, "
                    f"*{tuple(truth_cutpoints.shape)!r})"
                )
                raise ValueError(msg)
            if draws.shape[0] < 1 or draws.shape[1] < 1:
                msg = (
                    f"cutpoint draws for {method.value!r} have an empty chain "
                    "or draw axis"
                )
                raise ValueError(msg)
            block_axes = next(iter(held_draws[method].values())).shape[:2]
            if draws.shape[:2] != block_axes:
                msg = (
                    f"cutpoint draws for {method.value!r} have chain/draw axes "
                    f"{tuple(draws.shape[:2])!r}, not {tuple(block_axes)!r}"
                )
                raise ValueError(msg)
            held_cutpoints[method] = draws

        object.__setattr__(self, "_seed", checked_seed)
        object.__setattr__(self, "_truths", MappingProxyType(held_truths))
        object.__setattr__(self, "_posterior_draws", MappingProxyType(held_draws))
        object.__setattr__(self, "_true_cutpoints", truth_cutpoints)
        object.__setattr__(self, "_cutpoint_draws", MappingProxyType(held_cutpoints))

    @property
    def seed(self) -> int:
        """Return the simulation seed."""
        return self._seed

    @property
    def truths(self) -> Mapping[str, Tensor]:
        """Return snapshots of the true parameter blocks."""
        return MappingProxyType(
            {name: value.clone() for name, value in self._truths.items()}
        )

    @property
    def posterior_draws(self) -> Mapping[InferenceMethod, Mapping[str, Tensor]]:
        """Return snapshots of posterior draws by method and block."""
        return MappingProxyType(
            {
                method: MappingProxyType(
                    {name: value.clone() for name, value in blocks.items()}
                )
                for method, blocks in self._posterior_draws.items()
            }
        )

    @property
    def true_cutpoints(self) -> Tensor:
        """Return a snapshot of the participant-by-cutpoint truth."""
        return self._true_cutpoints.clone()

    @property
    def cutpoint_draws(self) -> Mapping[InferenceMethod, Tensor]:
        """Return snapshots of cutpoint draws by method."""
        return MappingProxyType(
            {method: draws.clone() for method, draws in self._cutpoint_draws.items()}
        )


@final
class SbcBlockSummary(FrozenValue):
    """Randomized NUTS SBC ranks and their uniformity test for one block."""

    _noun = "SBC block summary"
    __slots__ = (
        "_block",
        "_histogram",
        "_ks_p_value",
        "_ks_statistic",
        "_posterior_samples",
        "_ranks",
    )

    def __init__(
        self,
        block: str,
        ranks: Sequence[int],
        posterior_samples: int,
        ks_statistic: float,
        ks_p_value: float,
    ) -> None:
        checked_block = ensure_name("an SBC parameter block", block)
        if posterior_samples < 1:
            msg = f"an SBC summary has no posterior samples: {posterior_samples!r}"
            raise ValueError(msg)
        checked_ranks = tuple(ranks)
        if not checked_ranks:
            msg = "an SBC summary has no ranks"
            raise ValueError(msg)
        if any(
            isinstance(rank, bool) or not isinstance(rank, int)
            for rank in checked_ranks
        ):
            msg = f"an SBC summary has a noninteger rank: {checked_ranks!r}"
            raise TypeError(msg)
        if any(rank < 0 or rank > posterior_samples for rank in checked_ranks):
            msg = f"an SBC summary has a rank outside [0, {posterior_samples}]"
            raise ValueError(msg)
        statistic = float(ks_statistic)
        p_value = float(ks_p_value)
        if not 0.0 <= statistic <= 1.0 or not 0.0 <= p_value <= 1.0:
            msg = "an SBC summary has an invalid KS statistic or p-value"
            raise ValueError(msg)
        histogram = tuple(
            checked_ranks.count(rank) for rank in range(posterior_samples + 1)
        )
        object.__setattr__(self, "_block", checked_block)
        object.__setattr__(self, "_ranks", checked_ranks)
        object.__setattr__(self, "_posterior_samples", posterior_samples)
        object.__setattr__(self, "_histogram", histogram)
        object.__setattr__(self, "_ks_statistic", statistic)
        object.__setattr__(self, "_ks_p_value", p_value)

    @property
    def block(self) -> str:
        """Return the parameter block."""
        return self._block

    @property
    def ranks(self) -> tuple[int, ...]:
        """Return randomized ranks in seed and tensor-index order."""
        return self._ranks

    @property
    def posterior_samples(self) -> int:
        """Return the number of posterior samples behind every rank."""
        return self._posterior_samples

    @property
    def histogram(self) -> tuple[int, ...]:
        """Return counts for ranks zero through ``posterior_samples``."""
        return self._histogram

    @property
    def ks_statistic(self) -> float:
        """Return the normalized-rank Kolmogorov-Smirnov statistic."""
        return self._ks_statistic

    @property
    def ks_p_value(self) -> float:
        """Return the normalized-rank Kolmogorov-Smirnov p-value."""
        return self._ks_p_value

    @property
    def passes(self) -> bool:
        """Return whether uniformity is retained at the five-percent level."""
        return self._ks_p_value >= _KS_ALPHA


@final
class CoverageSummary(FrozenValue):
    """Equal-tail coverage for one inference method and parameter block."""

    _noun = "coverage summary"
    __slots__ = ("_block", "_covered", "_method", "_total")

    def __init__(
        self, method: InferenceMethod, block: str, covered: int, total: int
    ) -> None:
        if method not in _METHOD_SET:
            msg = f"a coverage summary has an unregistered method: {method!r}"
            raise ValueError(msg)
        checked_block = ensure_name("a coverage parameter block", block)
        if total < 1 or covered < 0 or covered > total:
            msg = f"a coverage summary has impossible counts: {covered!r}/{total!r}"
            raise ValueError(msg)
        object.__setattr__(self, "_method", method)
        object.__setattr__(self, "_block", checked_block)
        object.__setattr__(self, "_covered", covered)
        object.__setattr__(self, "_total", total)

    @property
    def method(self) -> InferenceMethod:
        """Return the inference method."""
        return self._method

    @property
    def block(self) -> str:
        """Return the parameter block."""
        return self._block

    @property
    def level(self) -> float:
        """Return the preregistered equal-tail level."""
        return _COVERAGE_LEVEL

    @property
    def covered(self) -> int:
        """Return how many true scalars fell inside their intervals."""
        return self._covered

    @property
    def total(self) -> int:
        """Return how many true scalars were checked."""
        return self._total

    @property
    def fraction(self) -> float:
        """Return the covered fraction."""
        return self._covered / self._total


@final
class CutpointRecovery(FrozenValue):
    """Participant-level recovery for one method and one cutpoint."""

    _noun = "cutpoint recovery"
    __slots__ = ("_cutpoint", "_method", "_participants", "_pearson", "_spearman")

    def __init__(
        self,
        method: InferenceMethod,
        cutpoint: int,
        participants: int,
        pearson: float,
        spearman: float,
    ) -> None:
        if method not in _METHOD_SET:
            msg = f"a cutpoint recovery has an unregistered method: {method!r}"
            raise ValueError(msg)
        if cutpoint < 0 or participants < 2:
            msg = (
                "a cutpoint recovery has invalid dimensions: "
                f"{cutpoint!r}, {participants!r}"
            )
            raise ValueError(msg)
        checked_pearson = float(pearson)
        checked_spearman = float(spearman)
        if not -1.0 <= checked_pearson <= 1.0 or not -1.0 <= checked_spearman <= 1.0:
            msg = "a cutpoint recovery has an invalid correlation"
            raise ValueError(msg)
        object.__setattr__(self, "_method", method)
        object.__setattr__(self, "_cutpoint", cutpoint)
        object.__setattr__(self, "_participants", participants)
        object.__setattr__(self, "_pearson", checked_pearson)
        object.__setattr__(self, "_spearman", checked_spearman)

    @property
    def method(self) -> InferenceMethod:
        """Return the inference method."""
        return self._method

    @property
    def cutpoint(self) -> int:
        """Return the zero-based cutpoint index."""
        return self._cutpoint

    @property
    def participants(self) -> int:
        """Return the number of participant observations compared."""
        return self._participants

    @property
    def pearson(self) -> float:
        """Return Pearson recovery across participants."""
        return self._pearson

    @property
    def spearman(self) -> float:
        """Return Spearman recovery across participants."""
        return self._spearman

    @property
    def floor(self) -> float:
        """Return the preregistered NUTS Pearson floor."""
        return _RECOVERY_FLOOR

    @property
    def clears_floor(self) -> bool:
        """Return whether Pearson recovery clears the preregistered floor."""
        return self._pearson >= _RECOVERY_FLOOR


@final
class CalibrationReport(FrozenValue):
    """Canonical SBC, coverage, and cutpoint-recovery summaries."""

    _noun = "calibration report"
    __slots__ = ("_coverage", "_cutpoints", "_rank_seed", "_replicates", "_sbc")

    def __init__(
        self,
        *,
        replicates: int,
        rank_seed: int,
        sbc: Sequence[SbcBlockSummary],
        coverage: Sequence[CoverageSummary],
        cutpoints: Sequence[CutpointRecovery],
    ) -> None:
        if replicates < 1:
            msg = "a calibration report has no replicates"
            raise ValueError(msg)
        held_sbc = tuple(sbc)
        held_coverage = tuple(coverage)
        held_cutpoints = tuple(cutpoints)
        if not held_sbc or not held_coverage or not held_cutpoints:
            msg = "a calibration report has an empty summary family"
            raise ValueError(msg)
        if not all(isinstance(value, SbcBlockSummary) for value in held_sbc):
            msg = "a calibration report contains a non-SBC summary"
            raise TypeError(msg)
        if not all(isinstance(value, CoverageSummary) for value in held_coverage):
            msg = "a calibration report contains a noncoverage summary"
            raise TypeError(msg)
        if not all(isinstance(value, CutpointRecovery) for value in held_cutpoints):
            msg = "a calibration report contains a nonrecovery summary"
            raise TypeError(msg)

        blocks = tuple(summary.block for summary in held_sbc)
        if blocks != tuple(sorted(set(blocks))):
            msg = (
                "a calibration report's SBC summaries must contain distinct blocks "
                "in canonical order"
            )
            raise ValueError(msg)
        posterior_samples = {summary.posterior_samples for summary in held_sbc}
        if len(posterior_samples) != 1:
            msg = (
                "a calibration report's SBC blocks have unequal posterior sample counts"
            )
            raise ValueError(msg)

        coverage_cells = tuple(
            (summary.method, summary.block) for summary in held_coverage
        )
        expected_coverage = tuple(
            (method, block) for method in _METHODS for block in blocks
        )
        if coverage_cells != expected_coverage:
            msg = (
                "a calibration report's coverage summaries must contain exactly "
                "every method-by-block cell in canonical order"
            )
            raise ValueError(msg)

        nuts_cutpoints = tuple(
            summary.cutpoint
            for summary in held_cutpoints
            if summary.method is InferenceMethod.NUTS
        )
        if nuts_cutpoints != tuple(range(len(nuts_cutpoints))):
            msg = (
                "a calibration report's NUTS recovery summaries must contain one "
                "canonical contiguous cutpoint set"
            )
            raise ValueError(msg)
        expected_cutpoints = tuple(
            (method, cutpoint) for method in _METHODS for cutpoint in nuts_cutpoints
        )
        recovery_cells = tuple(
            (summary.method, summary.cutpoint) for summary in held_cutpoints
        )
        if recovery_cells != expected_cutpoints:
            msg = (
                "a calibration report's recovery summaries must contain exactly "
                "every method-by-cutpoint cell in canonical order"
            )
            raise ValueError(msg)
        participants = {summary.participants for summary in held_cutpoints}
        if len(participants) != 1:
            msg = (
                "a calibration report's cutpoint summaries have unequal "
                "participant counts"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_replicates", replicates)
        object.__setattr__(
            self, "_rank_seed", ensure_non_negative("rank seed", rank_seed)
        )
        object.__setattr__(self, "_sbc", held_sbc)
        object.__setattr__(self, "_coverage", held_coverage)
        object.__setattr__(self, "_cutpoints", held_cutpoints)

    @property
    def replicates(self) -> int:
        """Return the number of reduced replicates."""
        return self._replicates

    @property
    def rank_seed(self) -> int:
        """Return the caller-owned randomized-rank seed."""
        return self._rank_seed

    @property
    def sbc(self) -> tuple[SbcBlockSummary, ...]:
        """Return per-block SBC summaries in block order."""
        return self._sbc

    @property
    def coverage(self) -> tuple[CoverageSummary, ...]:
        """Return coverage summaries in method and block order."""
        return self._coverage

    @property
    def cutpoints(self) -> tuple[CutpointRecovery, ...]:
        """Return recovery summaries in method and cutpoint order."""
        return self._cutpoints

    @property
    def clears_floor(self) -> bool:
        """Return whether every NUTS cutpoint's recovery clears the floor."""
        return all(
            summary.clears_floor
            for summary in self._cutpoints
            if summary.method is InferenceMethod.NUTS
        )

    @property
    def long_rows(self) -> tuple[dict[str, JsonValue], ...]:
        """Return deterministic records containing JSON scalar values only."""
        rows: list[dict[str, JsonValue]] = [
            {
                "summary": "sbc",
                "method": InferenceMethod.NUTS.value,
                "block": summary.block,
                "rank_count": len(summary.ranks),
                "posterior_samples": summary.posterior_samples,
                "ks_statistic": summary.ks_statistic,
                "ks_p_value": summary.ks_p_value,
                "passes": summary.passes,
                "replicates": self._replicates,
                "rank_seed": self._rank_seed,
            }
            for summary in self._sbc
        ]
        rows.extend(
            {
                "summary": "sbc_rank_bin",
                "method": InferenceMethod.NUTS.value,
                "block": summary.block,
                "rank": rank,
                "count": count,
                "posterior_samples": summary.posterior_samples,
                "replicates": self._replicates,
                "rank_seed": self._rank_seed,
            }
            for summary in self._sbc
            for rank, count in enumerate(summary.histogram)
        )
        rows.extend(
            {
                "summary": "coverage",
                "method": summary.method.value,
                "block": summary.block,
                "level": summary.level,
                "covered": summary.covered,
                "total": summary.total,
                "fraction": summary.fraction,
                "replicates": self._replicates,
                "rank_seed": self._rank_seed,
            }
            for summary in self._coverage
        )
        rows.extend(
            {
                "summary": "cutpoint_recovery",
                "method": summary.method.value,
                "cutpoint": summary.cutpoint,
                "participants": summary.participants,
                "pearson": summary.pearson,
                "spearman": summary.spearman,
                "floor": summary.floor,
                "clears_floor": summary.clears_floor,
                "replicates": self._replicates,
                "rank_seed": self._rank_seed,
            }
            for summary in self._cutpoints
        )
        return tuple(rows)

    @override
    def __repr__(self) -> str:
        """Return summary-family and replicate counts."""
        return (
            f"CalibrationReport(replicates={self._replicates}, "
            f"blocks={len(self._sbc)}, coverage={len(self._coverage)}, "
            f"cutpoints={len(self._cutpoints)})"
        )


def _same_structure(
    reference: CalibrationReplicate, candidate: CalibrationReplicate
) -> bool:
    """Return whether two replicates have exactly the same tensor structure."""
    if tuple(reference._truths) != tuple(candidate._truths):
        return False
    if reference._true_cutpoints.shape != candidate._true_cutpoints.shape:
        return False
    for block in reference._truths:
        if reference._truths[block].shape != candidate._truths[block].shape:
            return False
    for method in _METHODS:
        if (
            reference._cutpoint_draws[method].shape
            != candidate._cutpoint_draws[method].shape
        ):
            return False
        for block in reference._truths:
            if (
                reference._posterior_draws[method][block].shape
                != candidate._posterior_draws[method][block].shape
            ):
                return False
    return True


def _randomized_rank(truth: Tensor, draws: Tensor, generator: torch.Generator) -> int:
    """Return one rank, breaking posterior ties from the local stream."""
    below = int((draws < truth).sum().item())
    tied = int((draws == truth).sum().item())
    if tied == 0:
        return below
    offset = int(torch.randint(tied + 1, (), generator=generator).item())
    return below + offset


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Pearson correlation or refuse an undefined comparison."""
    if len(left) != len(right) or len(left) < 2:
        msg = "cutpoint recovery needs at least two paired participants"
        raise ValueError(msg)
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = tuple(value - left_mean for value in left)
    right_centered = tuple(value - right_mean for value in right)
    denominator = sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        msg = "cutpoint recovery is undefined for a constant participant vector"
        raise ValueError(msg)
    correlation = (
        sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
        / denominator
    )
    return max(-1.0, min(1.0, correlation))


def _ranks(values: Sequence[float]) -> tuple[float, ...]:
    """Return one-based average ranks with explicit tie handling."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranked = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranked[order[position]] = average
        start = end
    return tuple(ranked)


def calibration_report(
    replicates: Sequence[CalibrationReplicate],
    /,
    *,
    rank_seed: int,
) -> CalibrationReport:
    """Reduce completed prior predictive replicates without fitting or I/O.

    Randomized ranks use only NUTS, the reference posterior, and draw from a
    private stream initialized by ``rank_seed``. Coverage and participant-level
    cutpoint recovery are reported for NUTS, structured, and mean-field
    inference. Each cutpoint's recovery records whether its Pearson
    correlation reaches the preregistered floor of 0.80, and the report
    records whether every NUTS cutpoint does.
    """
    checked_rank_seed = ensure_non_negative("rank seed", rank_seed)
    held = tuple(replicates)
    if not held:
        msg = "calibration needs at least one replicate"
        raise ValueError(msg)
    if not all(isinstance(replicate, CalibrationReplicate) for replicate in held):
        msg = "calibration received a nonreplicate value"
        raise TypeError(msg)
    seeds = tuple(replicate.seed for replicate in held)
    if len(set(seeds)) != len(seeds):
        msg = f"calibration replicate seeds are not unique: {seeds!r}"
        raise ValueError(msg)
    ordered = tuple(sorted(held, key=lambda replicate: replicate.seed))
    reference = ordered[0]
    if any(not _same_structure(reference, replicate) for replicate in ordered[1:]):
        msg = "calibration replicates do not have identical block and tensor shapes"
        raise ValueError(msg)

    blocks = tuple(reference._truths)
    generator = torch.Generator(device="cpu").manual_seed(checked_rank_seed)
    sbc: list[SbcBlockSummary] = []
    for block in blocks:
        shape = reference._posterior_draws[InferenceMethod.NUTS][block].shape
        posterior_samples = shape[0] * shape[1]
        ranks: list[int] = []
        for replicate in ordered:
            truth = replicate._truths[block].reshape(-1)
            draws = replicate._posterior_draws[InferenceMethod.NUTS][block].reshape(
                posterior_samples, -1
            )
            ranks.extend(
                _randomized_rank(truth[index], draws[:, index], generator)
                for index in range(truth.numel())
            )
        normalized = tuple((rank + 0.5) / (posterior_samples + 1) for rank in ranks)
        test = kstest(normalized, "uniform")
        sbc.append(
            SbcBlockSummary(
                block,
                ranks,
                posterior_samples,
                float(test.statistic),
                float(test.pvalue),
            )
        )

    tail = (1.0 - _COVERAGE_LEVEL) / 2.0
    coverage: list[CoverageSummary] = []
    for method in _METHODS:
        for block in blocks:
            covered = 0
            total = 0
            for replicate in ordered:
                truth = replicate._truths[block]
                draws = replicate._posterior_draws[method][block]
                flattened = draws.reshape(draws.shape[0] * draws.shape[1], *truth.shape)
                lower = torch.quantile(flattened, tail, dim=0)
                upper = torch.quantile(flattened, 1.0 - tail, dim=0)
                covered += int(((truth >= lower) & (truth <= upper)).sum().item())
                total += truth.numel()
            coverage.append(CoverageSummary(method, block, covered, total))

    cutpoints: list[CutpointRecovery] = []
    participant_observations = len(ordered) * reference._true_cutpoints.shape[0]
    for method in _METHODS:
        truths = torch.cat(
            tuple(replicate._true_cutpoints for replicate in ordered), dim=0
        )
        means = torch.cat(
            tuple(
                replicate._cutpoint_draws[method].mean(dim=(0, 1))
                for replicate in ordered
            ),
            dim=0,
        )
        for cutpoint in range(truths.shape[1]):
            actual = tuple(float(value) for value in truths[:, cutpoint])
            recovered = tuple(float(value) for value in means[:, cutpoint])
            cutpoints.append(
                CutpointRecovery(
                    method,
                    cutpoint,
                    participant_observations,
                    _correlation(actual, recovered),
                    _correlation(_ranks(actual), _ranks(recovered)),
                )
            )

    return CalibrationReport(
        replicates=len(ordered),
        rank_seed=checked_rank_seed,
        sbc=sbc,
        coverage=coverage,
        cutpoints=cutpoints,
    )
