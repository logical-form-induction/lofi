"""Plugin bridges for declared entailment oracles."""

from typing import TYPE_CHECKING, Final

from lofi.lf import LogicFragment, ProverBound, TruthTableOracle
from lofi.lf.provers import (
    QUANTIFIER_FREE_CAPABILITY,
    SmtEngine,
    SmtOracle,
    TptpEngine,
    TptpOracle,
    TranslatorCapability,
)
from lofi.registry import ComponentPlugin, Registry
from lofi.spec.provers.smt import SmtProverSettings, SmtProverSpec
from lofi.spec.provers.tptp import TptpProverSettings, TptpProverSpec
from lofi.spec.provers.truth_table import (
    TruthTableProverSettings,
    TruthTableProverSpec,
)

if TYPE_CHECKING:
    from lofi.spec.language import LanguageAssembly

__all__ = ["SMT_PLUGIN", "TPTP_PLUGIN", "TRUTH_TABLE_PLUGIN"]


def _bound(
    prover: str,
    serves: TranslatorCapability,
    language: LanguageAssembly,
    /,
) -> ProverBound:
    """Build and check a configured prover capability against one language."""
    profile = language.profile
    world_types = (
        profile.world_types
        if "*" in serves.world_types
        else frozenset(serves.world_types)
    )
    return ProverBound(
        prover,
        LogicFragment(
            order=serves.order,
            quantifiers=profile.fragment.quantifiers if serves.quantifiers else (),
            connectives=profile.fragment.connectives,
            functions=serves.functions,
            equality=serves.equality,
        ),
        world_types=world_types,
    )


def _build_truth_table(
    settings: TruthTableProverSettings,
    language: LanguageAssembly,
    /,
) -> TruthTableOracle:
    """Build exhaustive entailment over the declared truth-functional roles."""
    bound = _bound("truth-table", QUANTIFIER_FREE_CAPABILITY, language)
    return TruthTableOracle(
        language.store,
        language.profile,
        bound=bound,
        atoms=settings.atoms,
    )


def _build_smt(
    settings: SmtProverSettings,
    language: LanguageAssembly,
    /,
) -> SmtOracle:
    """Build the declared SMT oracle over one language."""
    engine = Registry[SmtEngine]("smt_engine").get(settings.engine)
    return SmtOracle(
        language,
        engine=engine,
        executable=settings.executable,
        bound=_bound(f"smt:{settings.engine}", engine.serves, language),
        logic=settings.logic,
        options=settings.options,
        prefix=settings.prefix,
        arguments=settings.arguments,
        terminate_grace=settings.terminate_grace,
        polymorphism=settings.service.polymorphism,
        certificates=settings.service.certificates,
        memory_mb=settings.service.memory_mb,
        cpu_seconds=settings.service.cpu_seconds,
        seed=settings.service.seed,
    )


def _build_tptp(
    settings: TptpProverSettings,
    language: LanguageAssembly,
    /,
) -> TptpOracle:
    """Build the declared TPTP oracle over one language."""
    engine = Registry[TptpEngine]("tptp_engine").get(settings.engine)
    return TptpOracle(
        language,
        engine=engine,
        executable=settings.executable,
        dialect=settings.dialect,
        bound=_bound(
            f"tptp:{settings.engine}:{settings.dialect}",
            engine.serves(settings.dialect),
            language,
        ),
        prefix=settings.prefix,
        options=settings.options,
        problem_file=settings.problem_file,
        memory_mb=settings.service.memory_mb,
        cpu_seconds=settings.service.cpu_seconds,
        seed=settings.service.seed,
        polymorphism=settings.service.polymorphism,
        certificates=settings.service.certificates,
    )


SMT_PLUGIN: Final = ComponentPlugin(spec=SmtProverSpec, build=_build_smt)
"""The incremental SMT prover plugin."""

TPTP_PLUGIN: Final = ComponentPlugin(spec=TptpProverSpec, build=_build_tptp)
"""The TPTP prover plugin."""

TRUTH_TABLE_PLUGIN: Final = ComponentPlugin(
    spec=TruthTableProverSpec, build=_build_truth_table
)
"""The exhaustive propositional prover plugin."""
