import importlib
import pkgutil
from typing import TYPE_CHECKING, ClassVar, Final, cast

import pytest

import lofi.probe
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._catalog import DiagnosticCatalog, installed_diagnostics
from lofi.probe._chart import CutpointEntropy, SpanLengthBias
from lofi.probe._diagnose import DiagnosticContext, DiagnosticResult
from lofi.probe._transformer import (
    AttentionPatterns,
    FeedForwardSpecialization,
    ResidualContributions,
)
from lofi.probe._values import Trace
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from pathlib import Path

PARSER: Final = ComponentId("cky-parser")
CELL: Final = Site(PARSER, Role.CHART_CELL, (0, 2))
GATING: Final = Site(PARSER, Role.GATE, (0, 2))

REAL: Final = SiteSpec(
    ("item", "width"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)


class Counting:
    """A diagnostic that applies at every published site."""

    tag: ClassVar[str] = "counting"
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return a result with no measurement."""
        context.readings(trace)

        return DiagnosticResult(self.tag, [])


class Gated:
    """A diagnostic that applies only where a gate is published."""

    tag: ClassVar[str] = "gated"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.GATE})
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return a result with no measurement."""
        context.readings(trace)

        return DiagnosticResult(self.tag, [])


class Colliding:
    """A second diagnostic claiming an occupied tag."""

    tag: ClassVar[str] = "counting"
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return a result with no measurement."""
        context.readings(trace)

        return DiagnosticResult(self.tag, [])


class Unnamed:
    """A diagnostic without a registration tag."""

    tag: ClassVar[str] = ""
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return a result with no measurement."""
        context.readings(trace)

        return DiagnosticResult("unnamed", [])


class Loose:
    """A diagnostic with mutable role metadata."""

    tag: ClassVar[str] = "loose"
    requires: ClassVar[frozenset[Role]] = cast("frozenset[Role]", {Role.GATE})
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return a result with no measurement."""
        context.readings(trace)

        return DiagnosticResult(self.tag, [])


class Unscored:
    """A diagnostic with mutable score metadata."""

    tag: ClassVar[str] = "unscored"
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = cast("frozenset[str]", ["share"])

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return a result with no measurement."""
        context.readings(trace)

        return DiagnosticResult(self.tag, [])


class Anonymous:
    """A diagnostic with an unnamed measurement."""

    tag: ClassVar[str] = "anonymous"
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = frozenset({""})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return a result with no measurement."""
        context.readings(trace)

        return DiagnosticResult(self.tag, [])


def a_catalog() -> DiagnosticCatalog:
    return DiagnosticCatalog({"gated": Gated, "counting": Counting})


def test_a_diagnostic_is_found_under_the_tag_it_declares() -> None:
    catalog = a_catalog()

    assert catalog.tags == ("counting", "gated")
    assert catalog.kind("gated") is Gated
    assert "counting" in catalog
    assert "colliding" not in catalog
    assert len(catalog) == 2


def test_a_second_diagnostic_claiming_one_tag_is_refused_naming_both() -> None:
    with pytest.raises(RegistryError, match="Colliding") as collision:
        DiagnosticCatalog({"counting": Counting, "colliding": Colliding})

    assert "'counting'" in str(collision.value)
    assert "'colliding'" in str(collision.value)


def test_a_diagnostic_registering_under_no_tag_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        DiagnosticCatalog({"": Unnamed})


def test_a_diagnostic_whose_required_roles_are_not_a_frozen_set_is_refused() -> None:
    with pytest.raises(TypeError, match="frozenset of roles"):
        DiagnosticCatalog({"loose": Loose})


def test_a_diagnostic_whose_scores_are_not_a_frozen_set_is_refused() -> None:
    with pytest.raises(TypeError, match="frozenset of names"):
        DiagnosticCatalog({"unscored": Unscored})


def test_a_diagnostic_scoring_a_measurement_it_does_not_name_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        DiagnosticCatalog({"anonymous": Anonymous})


def test_an_unregistered_tag_names_the_tags_that_are() -> None:
    catalog = a_catalog()

    with pytest.raises(KeyError, match="gated") as missing:
        catalog.kind("attention")

    assert "attention" in str(missing.value)


def test_resolution_is_the_diagnostics_whose_roles_the_model_publishes() -> None:
    catalog = a_catalog()
    without = catalog.matching(DiagnosticContext({CELL: REAL}))
    with_gate = catalog.matching(DiagnosticContext({CELL: REAL, GATING: REAL}))

    assert [type(one) for one in without] == [Counting]
    assert [type(one) for one in with_gate] == [Counting, Gated]


def test_a_model_publishing_nothing_supports_what_requires_nothing() -> None:
    resolved = a_catalog().matching(DiagnosticContext({}))

    assert [type(one) for one in resolved] == [Counting]


def _install_toy_diagnostic(root: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Install a distribution listing one gate diagnostic, returning its tag."""
    site = root / "site"
    site.mkdir()
    module = f"lofi_diagnostic_plugin_{root.name.replace('-', '_')}"
    tag = f"toy-gate-{root.name.replace('_', '-')}"
    (site / f"{module}.py").write_text(
        "from typing import ClassVar\n"
        "\n"
        "from lofi.nn.sites import Role\n"
        "from lofi.probe import DiagnosticResult\n"
        "\n"
        "\n"
        "class ToyGate:\n"
        f"    tag: ClassVar[str] = {tag!r}\n"
        "    requires: ClassVar[frozenset[Role]] = frozenset({Role.GATE})\n"
        "    scores: ClassVar[frozenset[str]] = frozenset()\n"
        "\n"
        "    def run(self, trace, /, *, context):\n"
        "        return DiagnosticResult(self.tag, [])\n",
        encoding="utf-8",
        newline="\n",
    )
    metadata = site / f"{module}-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Name: {module.replace('_', '-')}\nVersion: 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.diagnostic]\n{tag} = {module}:ToyGate\n", encoding="utf-8", newline="\n"
    )
    monkeypatch.syspath_prepend(str(site))

    return tag


def test_a_diagnostic_a_test_package_lists_is_found_by_the_role_it_requires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tag = _install_toy_diagnostic(tmp_path, monkeypatch)
    catalog = installed_diagnostics()
    without = {one.tag for one in catalog.matching(DiagnosticContext({CELL: REAL}))}
    with_gate = {
        one.tag
        for one in catalog.matching(DiagnosticContext({CELL: REAL, GATING: REAL}))
    }

    assert tag in catalog
    assert tag not in without
    assert tag in with_gate
    assert catalog.kind(tag).requires == frozenset({Role.GATE})


def test_importing_the_probe_package_registers_nothing() -> None:
    modules = [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(lofi.probe.__path__, prefix="lofi.probe.")
    ]
    held = [
        f"{module.__name__}.{name}"
        for module in (lofi.probe, *modules)
        for name, value in vars(module).items()
        if isinstance(value, (Registry, DiagnosticCatalog))
    ]

    assert held == []


def test_every_diagnostic_the_package_registers_constructs_and_runs() -> None:
    context = DiagnosticContext({CELL: REAL})
    trace = Trace()
    trace.close()

    installed = installed_diagnostics()

    assert installed.tags == (
        "action-margin",
        "atomic-complex-differentiation",
        "attention-patterns",
        "constructor-embedding",
        "controller-choice",
        "controller-relations",
        "cutpoint-entropy",
        "depth-graded-encoding",
        "feed-forward-specialization",
        "finite-readings",
        "gate-patterns",
        "hidden-state-evolution",
        "memory-cell-dynamics",
        "recurrent-connection-analysis",
        "recursion-spread",
        "residual-contributions",
        "span-length-bias",
        "split-posterior-kl",
        "superposition-interference",
    )

    for tag in installed.tags:
        built = installed.kind(tag)()

        assert built.run(trace, context=context).tag == tag


def test_transformer_diagnostics_declare_their_exact_capabilities() -> None:
    assert AttentionPatterns.requires == frozenset(
        {Role.ATTENTION_LOGITS, Role.ATTENTION_WEIGHTS}
    )


def test_chart_diagnostics_declare_the_exact_readout_roles_they_consume() -> None:
    assert CutpointEntropy.requires == frozenset({Role.SPLIT_DISTRIBUTION})
    assert SpanLengthBias.requires == frozenset({Role.CONSTITUENT_MARGINAL})
    assert FeedForwardSpecialization.requires == frozenset({Role.FFN_PREACTIVATION})
    assert ResidualContributions.requires == frozenset(
        {
            Role.ATTENTION_LOGITS,
            Role.ATTENTION_WEIGHTS,
            Role.FFN_PREACTIVATION,
            Role.RESIDUAL_STREAM,
        }
    )


def test_the_installed_catalog_says_which_tags_it_holds() -> None:
    assert "finite-readings" in repr(installed_diagnostics())
