"""Declared constraint sources compose with policies and synthetic datasets."""

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl, SyntheticCorpusLowerer
from lofi.spec import RunSpecCodec, config_search_path
from lofi.train import constraint_source, select_constraints

if TYPE_CHECKING:
    from lofi.data import PinnedConstraintSource
    from lofi.data.synthetic import SyntheticDataset
    from lofi.spec import RunSpec
    from lofi.spec.constraint_sources import DeclaredConstraintSourceSpec

ROOT: Final = Path(__file__).resolve().parents[2]


def configured(name: str, /) -> RunSpec:
    """Return one packaged declared-constraint run."""
    return RunSpecCodec(search_path=config_search_path(ROOT))(
        ROOT / "lofi/configs/run" / name
    )


def drawn(run: RunSpec, /) -> SyntheticDataset:
    """Lower the run's synthetic dataset under its declared parse policy."""
    lowerer = SyntheticCorpusLowerer()
    declaration = cast("SyntheticCorpusDecl", run.data.datasets["train"].dataset)

    return lowerer.lower(
        lowerer.check(declaration),
        run.parse.build().policy,
    )


def test_all_declared_pin_locators_compose_through_one_source() -> None:
    run = configured("synthetic-pinning-declared.yaml")
    dataset = drawn(run)
    source = cast(
        "PinnedConstraintSource[str, str]",
        constraint_source(run, "train", dataset),
    )
    selection = select_constraints(run, "train", dataset)

    assert source.patterns[0].pattern == ("word-0",)
    assert source.patterns[1].pattern == ("word-2", None)
    assert source.patterns[2].pattern is None
    assert selection is not None
    assert selection.budget.types_pinned > 0
    assert selection.budget.tokens_covered > 0


def test_inversion_and_score_quantiles_compose_without_another_source() -> None:
    inverted = configured("synthetic-pinning-inverted.yaml")
    quantile = configured("synthetic-pinning-quantile.yaml")
    selected = select_constraints(quantile, "train", drawn(quantile))
    declaration = cast(
        "DeclaredConstraintSourceSpec",
        inverted.data.datasets["train"].constraint_source,
    )

    assert declaration.clamp.negate
    assert selected is not None
    assert selected.budget.types_pinned > 0
    assert selected.budget.tokens_covered > 0
    assert selected.budget.mass_share == 0.5
