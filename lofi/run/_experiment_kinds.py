"""The installed experiment kinds whose points need an optional capability.

Each plugin pairs an experiment declaration with a builder that loads the
configuration implementation, and with it the judgments capability, only when
an experiment declares that kind. Discovering the plugins therefore imports no
configuration implementation, and the builders name the built configurations
through the module that defines them.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Final, Protocol, cast

from lofi.run._boundary.experiment import (
    ActiveLearningExperimentConfig,
    JudgmentFitExperimentConfig,
)
from lofi.run._experiment_config import ExperimentConfigPlugin

if TYPE_CHECKING:
    from pathlib import Path

    import lofi.run._active_learning_configuration
    import lofi.run._judgment_fit_configuration
    from lofi.run._experiment import ExperimentConfiguration

__all__ = ["ACTIVE_LEARNING_PLUGIN", "JUDGMENT_FIT_PLUGIN"]


class _JudgmentFit(Protocol):
    """The judgment-fit implementation loaded when an experiment declares it."""

    JudgmentFitConfiguration: type[
        lofi.run._judgment_fit_configuration.JudgmentFitConfiguration
    ]


class _ActiveLearning(Protocol):
    """The active-learning implementation loaded when an experiment declares it."""

    ActiveLearningConfiguration: type[
        lofi.run._active_learning_configuration.ActiveLearningConfiguration
    ]


def _judgment_fit(
    declaration: JudgmentFitExperimentConfig, path: Path, /
) -> ExperimentConfiguration:
    """Build the judgment-fit configuration of one experiment."""
    module = cast("_JudgmentFit", import_module("lofi.run._judgment_fit_configuration"))
    return module.JudgmentFitConfiguration(declaration.kind, path)


def _active_learning(
    declaration: ActiveLearningExperimentConfig, path: Path, /
) -> ExperimentConfiguration:
    """Build the active-learning configuration of one experiment."""
    module = cast(
        "_ActiveLearning", import_module("lofi.run._active_learning_configuration")
    )
    return module.ActiveLearningConfiguration(declaration.kind, path)


JUDGMENT_FIT_PLUGIN: Final = ExperimentConfigPlugin(
    spec=JudgmentFitExperimentConfig, build=_judgment_fit
)
"""The installed judgment-fit experiment kind."""

ACTIVE_LEARNING_PLUGIN: Final = ExperimentConfigPlugin(
    spec=ActiveLearningExperimentConfig, build=_active_learning
)
"""The installed active-learning experiment kind."""
