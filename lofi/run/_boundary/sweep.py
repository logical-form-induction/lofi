"""Sweep documents, the points they expand to, and configuration overlays."""

import itertools
import json
import tomllib
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Final, Literal, Self, cast, override

import didactic.api as dx
from didactic.settings import ConfigError as EngineConfigError
from didactic.settings import load_document

from lofi._checks import ensure_name, ensure_non_negative, ensure_positive
from lofi._types import JsonValue
from lofi.spec import (
    ConfigDocumentError,
    Provenance,
    RunContext,
    RunSpec,
    RunSpecCodec,
    run_context,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


__all__ = [
    "RESOLUTION_IV",
    "Axis",
    "AxisValue",
    "FractionalFactorialDesignSpec",
    "GeneratorSpec",
    "GridDesignSpec",
    "LatinHypercubeDesignSpec",
    "RandomDesignSpec",
    "Sweep",
    "SweepDesignSpec",
    "SweepPoint",
    "overlay_settings",
    "point_id",
    "point_record",
    "read_overlay",
    "read_point",
    "resolve_point",
    "validate_resolution",
]

type AxisValue = (
    str | int | float | bool | list[AxisValue] | dict[str, AxisValue] | None
)
"""Value assigned by a sweep axis: a scalar, or a list or mapping of such values."""

type _StoredValue = (
    str | int | float | bool | tuple[_StoredValue, ...] | dict[str, _StoredValue] | None
)
"""Axis point as a model holds it, which is with each array as a tuple."""

RESOLUTION_IV: Final = 4
"""The least resolution a fractional design or resolution-IV rung admits."""


class GeneratorSpec(dx.Model, extra="forbid"):
    """Declare one generated factor of a two-level fractional factorial."""

    factor: str
    word: tuple[str, ...]

    @dx.validates("factor")
    @classmethod
    def _factor_is_named(cls, factor: str) -> str:
        return ensure_name("a generated factor", factor)

    @dx.validates("word")
    @classmethod
    def _word_is_named(cls, word: tuple[str, ...]) -> tuple[str, ...]:
        checked = tuple(ensure_name("a generator word factor", name) for name in word)

        if not checked or len(checked) != len(set(checked)):
            msg = f"a generator word is empty or repeats a factor: {checked!r}"
            raise ValueError(msg)

        return checked

    @dx.model_validator()
    def _factor_is_outside_its_word(self) -> Self:
        if self.factor in self.word:
            msg = f"the generated factor {self.factor!r} appears in its own word"
            raise ValueError(msg)

        return self


def validate_resolution(
    generators: tuple[GeneratorSpec, ...], /, *, minimum: int
) -> tuple[GeneratorSpec, ...]:
    """Return generators whose defining relation reaches a minimum resolution."""
    if not generators:
        msg = "a fractional-factorial design declares no generator"
        raise ValueError(msg)

    generated = tuple(generator.factor for generator in generators)

    if len(generated) != len(set(generated)):
        msg = f"two generators set one factor: {generated}"
        raise ValueError(msg)

    basics = {key for generator in generators for key in generator.word}
    clash = sorted(basics.intersection(generated))

    if clash:
        msg = f"generated factors also appear in a word: {clash}"
        raise ValueError(msg)

    words = tuple(
        frozenset((generator.factor, *generator.word)) for generator in generators
    )

    for size in range(1, len(words) + 1):
        for chosen in itertools.combinations(words, size):
            defining: frozenset[str] = frozenset()

            for word in chosen:
                defining = defining.symmetric_difference(word)

            if len(defining) < minimum:
                msg = (
                    f"the defining relation has the word {sorted(defining)}, shorter "
                    f"than the {minimum} letters resolution {minimum} needs"
                )
                raise ValueError(msg)

    return generators


class SweepDesignSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare how a sweep selects assignments from its axes."""


class GridDesignSpec(SweepDesignSpec):
    """Declare the complete Cartesian product of the sweep axes."""

    kind: Literal["grid"] = "grid"


class RandomDesignSpec(SweepDesignSpec):
    """Declare a seeded sample without replacement from the complete grid."""

    kind: Literal["random"] = "random"
    points: int
    seed: int

    @dx.validates("points")
    @classmethod
    def _points_are_positive(cls, points: int) -> int:
        return ensure_positive("a random-design point count", points)

    @dx.validates("seed")
    @classmethod
    def _seed_is_nonnegative(cls, seed: int) -> int:
        return ensure_non_negative("a random-design seed", seed)


class LatinHypercubeDesignSpec(SweepDesignSpec):
    """Declare a seeded discrete Latin hypercube sample."""

    kind: Literal["latin-hypercube"] = "latin-hypercube"
    points: int
    seed: int

    @dx.validates("points")
    @classmethod
    def _points_are_positive(cls, points: int) -> int:
        return ensure_positive("a Latin-hypercube point count", points)

    @dx.validates("seed")
    @classmethod
    def _seed_is_nonnegative(cls, seed: int) -> int:
        return ensure_non_negative("a Latin-hypercube seed", seed)


class FractionalFactorialDesignSpec(SweepDesignSpec):
    """Declare a resolution-IV two-level fractional factorial."""

    kind: Literal["fractional-factorial"] = "fractional-factorial"
    generators: tuple[GeneratorSpec, ...]

    @dx.validates("generators")
    @classmethod
    def _generators_reach_resolution_iv(
        cls, generators: tuple[GeneratorSpec, ...]
    ) -> tuple[GeneratorSpec, ...]:
        return validate_resolution(generators, minimum=RESOLUTION_IV)


def _as_point(stored: JsonValue | _StoredValue, /) -> AxisValue:
    """Return a stored axis point with each of its arrays as a list.

    A model stores a nested array as a tuple, while override parsing and
    document loading produce lists, so a point read back out of a model is
    converted before it is compared, rendered or resolved.

    Parameters
    ----------
    stored : JsonValue | _StoredValue
        The point as a model holds it, or as a document supplied it.
    """
    if isinstance(stored, list | tuple):
        return [_as_point(item) for item in stored]

    if isinstance(stored, dict):
        return {key: _as_point(value) for key, value in stored.items()}

    return cast("AxisValue", stored)


def _point_key(point: AxisValue, /) -> str:
    """Return text that is equal for two points exactly when the points agree.

    Two points agree when they have the same type and value at every depth, so
    ``1``, ``1.0`` and ``True`` are three points, and two mappings agree
    whatever order their keys were written in.

    Parameters
    ----------
    point : AxisValue
        The point, with each array as a list.
    """
    return json.dumps(point, allow_nan=False, separators=(",", ":"), sort_keys=True)


_MIN_YOKED_AXES = 2
"""The minimum axis count in a yoked group."""


def _canonical(config: Mapping[str, JsonValue]) -> str:
    """Return canonical JSON for a resolved configuration."""
    return json.dumps(
        dict(config),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def point_id(config: Mapping[str, JsonValue], *, seed: int = 0) -> str:
    """Return the stable identifier for a resolved configuration and seed.

    The identifier is computed over the configuration's scientific sections,
    which are all but the observation sections ``RunSpec`` declares, so the
    same run with and without a metric sink is one point.

    Parameters
    ----------
    config : Mapping[str, JsonValue]
        The resolved configuration, in any key order.
    seed : int
        The run seed, which is not part of the configuration.

    Returns
    -------
    str
        Sixteen hexadecimal digits that differ whenever a scientific section
        or the seed differs.
    """
    observation = RunSpec.observation_sections()
    scientific = {
        section: value
        for section, value in config.items()
        if section not in observation
    }

    return sha256(
        _canonical({"configuration": scientific, "seed": seed}).encode()
    ).hexdigest()[:16]


class Axis(dx.Model, extra="forbid"):
    """Declares values for one configuration leaf or one configuration group.

    An axis declares exactly one of ``path`` and ``group``, each a dotted
    path. An axis with a ``path`` assigns one leaf of the resolved document,
    so its values are the scalars, lists or mappings that leaf takes; a
    mapping point sets a whole variant of a tagged union leaf and is
    validated against the schema as a group fragment is. An axis with a
    ``group`` selects the fragment that fills a configuration slot, so its
    values are fragment names, or None to leave the slot unfilled. A group may
    be any slot, including one at the root of the document such as
    ``device_policy`` or ``trainer``. Leaf assignments apply over every
    selected fragment whatever order the axes are declared in.

    Attributes
    ----------
    path : str | None
        The dotted leaf the axis assigns, or None for a group axis.
    group : str | None
        The dotted slot whose fragment the axis selects, or None for a leaf
        axis.
    values : tuple[JsonValue, ...]
        The declared points as the model holds them, which is with each array
        as a tuple. Read :attr:`points` to compare, render or resolve them.
    """

    path: str | None = None
    group: str | None = None
    values: tuple[JsonValue, ...]

    @property
    def key(self) -> str:
        """Return the dotted path or slot the axis assigns.

        An exclusion clause names an axis by this key.
        """
        return cast("str", self.group if self.path is None else self.path)

    @property
    def selects_group(self) -> bool:
        """Return whether the axis selects a config group rather than a leaf."""
        return self.group is not None

    @property
    def points(self) -> tuple[AxisValue, ...]:
        """Return the declared points with each array as a list.

        A list-valued point comes back as a list of the same length, so a
        one-element list stays a one-element list rather than collapsing to the
        scalar it holds, and a scalar point comes back as that scalar.
        """
        return tuple(_as_point(value) for value in self.values)

    @property
    def slot(self) -> tuple[str, ...]:
        """Return the segments of the dotted path or slot."""
        return tuple(self.key.split("."))

    @dx.validates("path")
    @classmethod
    def _path_is_dotted(cls, path: str | None) -> str | None:
        """Return a dotted leaf path with no empty segment, or None."""
        return None if path is None else _dotted("axis path", path)

    @dx.validates("group")
    @classmethod
    def _group_is_dotted(cls, group: str | None) -> str | None:
        """Return a dotted slot with no empty segment, or None."""
        return None if group is None else _dotted("axis group", group)

    @dx.validates("values")
    @classmethod
    def _values_are_declared(
        cls, values: tuple[_StoredValue, ...]
    ) -> tuple[_StoredValue, ...]:
        """Return a nonempty sequence of points."""
        if not values:
            msg = "an axis declares no value"
            raise ValueError(msg)

        return values

    @dx.model_validator()
    def _assigns_one_target(self) -> Self:
        """Validate what the axis assigns and the points it assigns.

        Returns
        -------
        Self
            This axis after validation.

        Raises
        ------
        ValueError
            If the axis declares both a path and a group or neither, or if a
            group axis takes a value that is neither a fragment name nor None.
        """
        if (self.path is None) == (self.group is None):
            msg = (
                "an axis declares exactly one of a path and a group, "
                f"not path={self.path!r} and group={self.group!r}"
            )
            raise ValueError(msg)

        if not self.selects_group:
            return self

        unnamed = tuple(
            value
            for value in self.points
            if value is not None and (not isinstance(value, str) or not value)
        )

        if unnamed:
            msg = f"a group axis takes fragment names, not {unnamed}"
            raise ValueError(msg)

        return self


def _dotted(subject: str, path: str, /) -> str:
    """Return a dotted path with no empty segment.

    Parameters
    ----------
    subject : str
        What the path names, for the refusal.
    path : str
        The declared path.

    Raises
    ------
    ValueError
        If the path is empty, spelled with a slash, or has an empty segment.
    """
    named = ensure_name(subject, path)

    if "/" in named:
        msg = f"an {subject} is dotted, not {path!r}"
        raise ValueError(msg)

    if any(not part for part in named.split(".")):
        msg = f"an {subject} has an empty segment: {path!r}"
        raise ValueError(msg)

    return named


def _lies_under(inner: Axis, outer: Axis) -> bool:
    """Return whether one axis assigns a slot strictly under another's."""
    depth = len(outer.slot)

    return len(inner.slot) > depth and inner.slot[:depth] == outer.slot


def _composes_with(
    inner: Axis, outer: Axis, together: tuple[frozenset[str], ...]
) -> bool:
    """Return whether a nested assignment composes with the one it lies under.

    A leaf assignment composes with a group selection above it, since the
    configuration engine applies every selection before any leaf. A nested
    selection composes with the selection above it only while the two advance
    together in one yoked group, where the pairing is declared rather than
    crossed.
    """
    if not outer.selects_group:
        return False

    if not inner.selects_group:
        return True

    return any({inner.key, outer.key} <= group for group in together)


class Sweep(dx.Model, extra="forbid"):
    """Declares grid, yoked, excluded, and seeded sweep dimensions.

    Attributes
    ----------
    grid : tuple[Axis, ...]
        Axes crossed with one another.
    yoked : tuple[tuple[Axis, ...], ...]
        Groups of axes whose values advance together.
    exclude : tuple[dict[str, JsonValue], ...]
        Clauses removing every assignment that agrees with all of a clause's
        key-value pairs, each key naming an axis by its :attr:`Axis.key`; any
        one clause suffices. The values are held as the
        model holds an axis point, so read :attr:`clauses` to compare them.
    seeds : tuple[int, ...]
        Seeds crossed with every surviving assignment.
    expected_points : int | None
        The point count the expansion must produce, or None to accept any.
        An experiment document requires it.
    """

    design: SweepDesignSpec = GridDesignSpec()
    grid: tuple[Axis, ...] = ()
    yoked: tuple[tuple[Axis, ...], ...] = ()
    exclude: tuple[dict[str, JsonValue], ...] = ()
    seeds: tuple[int, ...] = (0,)
    expected_points: int | None = None

    @property
    def clauses(self) -> tuple[dict[str, AxisValue], ...]:
        """Return the exclusion clauses with each array value as a list.

        A clause value is compared against an assigned point in type as well
        as in value, so the two sides have to agree on how an array is spelled.
        """
        return tuple(
            {path: _as_point(value) for path, value in clause.items()}
            for clause in self.exclude
        )

    @dx.validates("grid")
    @classmethod
    def _grid_axes_are_distinct(cls, axes: tuple[Axis, ...]) -> tuple[Axis, ...]:
        """Return grid axes with distinct keys, each declaring distinct values.

        A repeated value would expand to a repeated point, which collapses
        into the first rather than adding one.
        """
        keys = tuple(axis.key for axis in axes)

        if len(keys) != len(set(keys)):
            msg = f"grid axes repeat a key: {keys}"
            raise ValueError(msg)

        for axis in axes:
            texts = tuple(_point_key(point) for point in axis.points)

            if len(texts) != len(set(texts)):
                msg = f"the grid axis {axis.key!r} repeats a value: {texts}"
                raise ValueError(msg)

        return axes

    @dx.validates("yoked")
    @classmethod
    def _yoked_groups_align(
        cls, groups: tuple[tuple[Axis, ...], ...]
    ) -> tuple[tuple[Axis, ...], ...]:
        """Return yoked groups with aligned value counts and distinct tuples.

        One position of a group is one tuple of values, one per axis. A
        repeated tuple would expand to a repeated point, so the tuples of a
        group are distinct even where one axis repeats a value.
        """
        for group in groups:
            if len(group) < _MIN_YOKED_AXES:
                msg = "a yoked group needs at least two axes"
                raise ValueError(msg)

            lengths = {len(axis.points) for axis in group}

            if len(lengths) != 1:
                msg = f"yoked axes have different lengths: {sorted(lengths)}"
                raise ValueError(msg)

            tuples = tuple(
                tuple(_point_key(point) for point in position)
                for position in zip(*(axis.points for axis in group), strict=True)
            )

            if len(tuples) != len(set(tuples)):
                keys = tuple(axis.key for axis in group)
                msg = f"the yoked group {keys} repeats a tuple: {tuples}"
                raise ValueError(msg)

        keys = tuple(axis.key for group in groups for axis in group)

        if len(keys) != len(set(keys)):
            msg = f"yoked axes repeat a key: {keys}"
            raise ValueError(msg)

        return groups

    @dx.validates("seeds")
    @classmethod
    def _seeds_are_distinct(cls, seeds: tuple[int, ...]) -> tuple[int, ...]:
        """Return nonempty, distinct, nonnegative seeds."""
        if not seeds:
            msg = "a sweep declares no seed"
            raise ValueError(msg)

        checked = tuple(ensure_non_negative("a sweep seed", seed) for seed in seeds)

        if len(checked) != len(set(checked)):
            msg = f"a sweep repeats a seed: {checked}"
            raise ValueError(msg)

        return checked

    @dx.validates("exclude")
    @classmethod
    def _clauses_compare_points(
        cls, clauses: tuple[dict[str, _StoredValue], ...]
    ) -> tuple[dict[str, _StoredValue], ...]:
        """Return nonempty clauses."""
        if any(not clause for clause in clauses):
            msg = "an exclusion clause names no axis"
            raise ValueError(msg)

        return clauses

    @dx.validates("expected_points")
    @classmethod
    def _expected_points_are_countable(cls, expected: int | None) -> int | None:
        """Return a nonnegative expected point count, or None."""
        if expected is None:
            return None

        return ensure_non_negative("an expected point count", expected)

    @dx.model_validator()
    def _dimensions_assign_distinct_paths(self) -> Sweep:
        """Validate the keys across grid, yoked, and exclusion dimensions.

        Returns
        -------
        Sweep
            This sweep after validation.

        Raises
        ------
        ValueError
            If two axes assign one key, whether as a path or as a group, if
            one declared assignment lies under another, or if an exclusion
            clause names an undeclared key.
        """
        axes = (*self.grid, *(axis for group in self.yoked for axis in group))
        keys = tuple(axis.key for axis in axes)

        if len(keys) != len(set(keys)):
            msg = f"grid and yoked axes repeat a key: {keys}"
            raise ValueError(msg)

        together = tuple(frozenset(axis.key for axis in group) for group in self.yoked)

        for outer in axes:
            nested = tuple(
                inner.key
                for inner in axes
                if _lies_under(inner, outer)
                and not _composes_with(inner, outer, together)
            )

            if nested:
                msg = f"assignments to {outer.key!r} conflict with {nested}"
                raise ValueError(msg)

        declared = frozenset(keys)

        for clause in self.exclude:
            unknown = tuple(sorted(set(clause) - declared))

            if unknown:
                msg = f"an exclusion clause names undeclared axes: {unknown}"
                raise ValueError(msg)

        return self


class SweepPoint:
    """Identifies one resolved configuration, its seed, and its overrides.

    Two points are equal when their resolved configurations and seeds agree;
    the overrides that produced a point are provenance and stay outside its
    identity.
    """

    __slots__ = (
        "_config_json",
        "_overrides",
        "_point_id",
        "_provenance_json",
        "_seed",
    )

    def __init__(
        self,
        config: Mapping[str, JsonValue],
        *,
        overrides: tuple[str, ...] = (),
        provenance: Provenance | None = None,
        seed: int = 0,
    ) -> None:
        self._config_json = _canonical(config)
        self._overrides = overrides
        self._provenance_json = (
            None if provenance is None else provenance.model_dump_json()
        )
        self._seed = ensure_non_negative("a sweep point seed", seed)
        self._point_id = point_id(config, seed=self._seed)

    @property
    def config(self) -> dict[str, JsonValue]:
        """Return a copy of the resolved configuration."""
        return cast("dict[str, JsonValue]", json.loads(self._config_json))

    @property
    def overrides(self) -> tuple[str, ...]:
        """Return the overrides that produced the point."""
        return self._overrides

    @property
    def provenance(self) -> Provenance | None:
        """Return the source of each resolved leaf, when the point carries it."""
        if self._provenance_json is None:
            return None

        return Provenance.model_validate_json(self._provenance_json)

    @property
    def seed(self) -> int:
        """Return the run seed."""
        return self._seed

    @property
    def point_id(self) -> str:
        """Return the identifier over the resolved configuration and seed."""
        return self._point_id

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another point has the same configuration and seed."""
        if type(other) is not SweepPoint:
            return NotImplemented

        return (self._config_json, self._seed) == (other._config_json, other._seed)

    @override
    def __hash__(self) -> int:
        """Return a hash over the resolved configuration and seed."""
        return hash((SweepPoint, self._config_json, self._seed))

    @override
    def __repr__(self) -> str:
        """Return the point identifier and seed."""
        return f"SweepPoint({self._point_id!r}, seed={self._seed!r})"


def point_record(point: SweepPoint, /) -> dict[str, JsonValue]:
    """Return the complete wire record for one sweep point."""
    provenance = point.provenance

    return {
        "point_id": point.point_id,
        "seed": point.seed,
        "overrides": list(point.overrides),
        "config": point.config,
        "provenance": (
            None
            if provenance is None
            else cast("dict[str, JsonValue]", json.loads(provenance.model_dump_json()))
        ),
    }


def _point_field(fields: dict[str, JsonValue], name: str, /) -> JsonValue:
    """Return one required point-record field."""
    if name not in fields:
        msg = f"a listed point lacks its {name}"
        raise TypeError(msg)

    return fields[name]


def read_point(record: JsonValue, /) -> SweepPoint:
    """Reconstruct one complete sweep point from its wire record."""
    if not isinstance(record, dict):
        msg = f"a listed point is not a mapping: {record!r}"
        raise TypeError(msg)

    fields = record
    expected = {"point_id", "seed", "overrides", "config", "provenance"}
    missing = sorted(expected - set(fields))
    extra = sorted(set(fields) - expected)

    if missing or extra:
        msg = f"a listed point has missing fields {missing} and extra fields {extra}"
        raise TypeError(msg)

    listed_id = _point_field(fields, "point_id")
    seed = _point_field(fields, "seed")
    overrides = _point_field(fields, "overrides")
    config = _point_field(fields, "config")
    provenance_record = _point_field(fields, "provenance")

    if not isinstance(listed_id, str):
        msg = f"a listed point id is not a string: {listed_id!r}"
        raise TypeError(msg)

    if not isinstance(seed, int) or isinstance(seed, bool):
        msg = f"a listed seed is not an integer: {seed!r}"
        raise TypeError(msg)

    if not isinstance(overrides, list) or not all(
        isinstance(override, str) for override in overrides
    ):
        msg = f"listed overrides are not strings: {overrides!r}"
        raise TypeError(msg)

    if not isinstance(config, dict):
        msg = f"a listed configuration is not a mapping: {config!r}"
        raise TypeError(msg)

    if provenance_record is not None and not isinstance(provenance_record, dict):
        msg = f"listed provenance is not a mapping: {provenance_record!r}"
        raise TypeError(msg)

    provenance = (
        None
        if provenance_record is None
        else Provenance.model_validate(provenance_record)
    )
    point = SweepPoint(
        config,
        overrides=tuple(cast("list[str]", overrides)),
        provenance=provenance,
        seed=seed,
    )

    if point.point_id != listed_id:
        msg = (
            f"the listed id {listed_id!r} is not the id of its "
            f"configuration and seed, {point.point_id!r}"
        )
        raise ValueError(msg)

    return point


def resolve_point(point: SweepPoint, codec: RunSpecCodec, /) -> RunSpec:
    """Reconstruct the validated run declaration carried by one point."""
    with TemporaryDirectory() as temporary:
        document = Path(temporary) / "point.json"
        document.write_text(json.dumps(point.config), encoding="utf-8", newline="\n")
        root = codec.search_path[0].parent if codec.search_path else Path.cwd()

        with run_context(RunContext(root, point.seed)):
            return codec(document)


def read_overlay(path: Path, /) -> dict[str, JsonValue]:
    """Return the mapping one overlay document holds.

    Parameters
    ----------
    path : Path
        A JSON, TOML or YAML file whose top-level value is a mapping.

    Raises
    ------
    ConfigDocumentError
        If the file is absent, does not parse, or holds something other
        than a mapping.
    """
    try:
        document = load_document(path)

    except OSError as refused:
        raise ConfigDocumentError(document=path, reason=str(refused)) from refused

    except (
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
        EngineConfigError,
    ) as refused:
        raise ConfigDocumentError(
            document=path, reason=f"the overlay does not parse: {refused}"
        ) from refused

    return cast("dict[str, JsonValue]", document)


def _leaves(path: str, value: JsonValue, into: dict[str, JsonValue]) -> None:
    """Record the leaves under a dotted path."""
    if isinstance(value, dict) and value:
        for key, nested in value.items():
            _leaves(f"{path}.{key}", nested, into)

        return

    into[path] = value


def overlay_settings(paths: Sequence[Path], /) -> dict[str, JsonValue]:
    """Return the settings a sequence of overlay documents sets, keyed by dotted path.

    A later overlay sets a leaf over an earlier one at the same path; a
    list and an empty mapping are each one leaf.

    Parameters
    ----------
    paths : Sequence[Path]
        The overlay documents in ascending precedence.

    Raises
    ------
    ConfigDocumentError
        If an overlay is absent, does not parse, or is not a mapping.
    """
    settings: dict[str, JsonValue] = {}

    for path in paths:
        for key, value in read_overlay(path).items():
            _leaves(key, value, settings)

    return settings
