"""Declaration-family and plugin discovery."""

from typing import TYPE_CHECKING, Final

from lofi.registry import Registry, entry_point_group

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["FAMILY_GROUP", "component_groups", "load_plugins"]

_FAMILIES: Final = "families"
"""The family whose entries name every configurable component family's root."""

FAMILY_GROUP: Final = entry_point_group(_FAMILIES)
"""Entry-point group whose names declare configurable component families."""

_DECLARATION_FAMILIES: Final = (
    "rule",
    "erasure",
    "matching",
    "normal_form",
    "goal",
    "interpretation",
    "parse_strategy",
    "span_index",
    "chart_store",
    "cell_pruner",
    "semiring",
    "astar_strategy",
    "astar_root_mode",
    "astar_heuristic",
    "astar_proposer",
    "mbr_utility",
    "decode_method",
    "intervention",
    "acdc_threshold",
    "das_search",
    "artifact_source",
    "diora_backend",
    "formalism",
    "unsatisfiable_policy",
    "logical_role",
    "logical_constant",
    "inference_response",
    "inference_labeling",
    "control_variate",
    "rule_family",
    "pair_feature",
    "fuzzy_logic",
    "fuzzy_implication",
    "fuzzy_quantifier",
    "fuzzy_equality",
    "structure_source",
    "knowledge_base",
    "mixing",
    "step",
    "chart_composition",
    "split_weigher",
    "span_function",
    "lf_class",
    "pair_strategy",
    "corruption",
    "ill_typing",
    "reducer",
    "description_cost",
    "correction",
    "experiment_config",
    "launcher",
    "sweep_design",
)
"""Non-component declaration families loaded before document validation."""


def _component_families() -> tuple[str, ...]:
    """Return every family whose entries are plugins, in family order.

    Raises
    ------
    TagCollisionError
        If a family is registered more than once.
    """
    return Registry[object](_FAMILIES).available()


def component_groups() -> tuple[str, ...]:
    """Return the entry-point group of every family whose entries are plugins.

    Returns
    -------
    tuple[str, ...]
        One group per component family, in family order.

    Raises
    ------
    TagCollisionError
        If a family is registered more than once.
    """
    return tuple(entry_point_group(name) for name in _component_families())


def _load(families: Iterable[str]) -> tuple[str, ...]:
    """Load the entry points of several families, each in stable order.

    Parameters
    ----------
    families : Iterable[str]
        The families to load, in load order.

    Returns
    -------
    tuple[str, ...]
        Entry-point names in load order.
    """
    names: list[str] = []

    for family in families:
        for point in Registry[object](family).entries():
            point.load()
            names.append(point.name)

    return tuple(names)


def load_plugins() -> tuple[str, ...]:
    """Load every installed declaration in stable order.

    Family roots load first, then non-component declaration plugins, then the
    plugin objects of every component family. Loading a family root before its
    variants lets another distribution contribute both without a LoFI edit.

    Returns
    -------
    tuple[str, ...]
        Entry-point names in load order.

    Raises
    ------
    Exception
        If a plugin fails to load.
    """
    return _load((_FAMILIES, *_DECLARATION_FAMILIES, *_component_families()))
