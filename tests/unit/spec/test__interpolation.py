import os
import subprocess
import sys
from contextlib import nullcontext
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
from didactic.settings import (
    ConfigValue,
    InterpolationError,
    ResolverFn,
    list_resolvers,
    register_resolver,
    resolve,
    unregister_resolver,
)

from lofi.data.paths import DataRoot, PathDeclarationError
from lofi.registry import TagCollisionError
from lofi.registry import _registry as registry_module
from lofi.spec import (
    PACKAGED_CONFIG_ROOT,
    RESOLVER_GROUP,
    RunContext,
    active_run_context,
    derive_seed,
    register_resolvers,
    run_context,
)
from lofi.spec import _interpolation as interpolation_module

if TYPE_CHECKING:
    from collections.abc import Iterator

DATA_ROOT: Final = Path("/srv/lofi-data").absolute()

SEED: Final = 4321

TIMEOUT_SECONDS: Final = 120.0

SEED_CEILING: Final = 2**32

BUILTIN_RESOLVERS: Final = ("lofi.path", "lofi.pkg", "lofi.seed")


class ResolverPoint:
    """Test resolver entry point."""

    group = RESOLVER_GROUP
    dist = None

    def __init__(self, name: str, value: str, member: object) -> None:
        self.name = name
        self.value = value
        self._member = member

    def load(self) -> object:
        """Return the entry point's member."""
        return self._member


@pytest.fixture(autouse=True)
def registered() -> None:
    """Register LoFI's resolvers, as the command line does when it starts."""
    register_resolvers(replace=True)


@pytest.fixture
def context() -> RunContext:
    return RunContext(data_root=DATA_ROOT, seed=SEED)


@pytest.fixture
def restored_registry() -> Iterator[None]:
    """Put LoFI's resolvers back however the test leaves the registry."""
    try:
        yield

    finally:
        register_resolvers(replace=True)


def resolved(text: str, context: RunContext) -> ConfigValue:
    with run_context(context):
        return resolve(text, root={})


def refusal(text: str, context: RunContext | None = None) -> str:
    installed = nullcontext() if context is None else run_context(context)

    with installed, pytest.raises(InterpolationError) as refused:
        resolve(text, root={})

    return str(refused.value)


def test_importing_the_spec_package_registers_no_resolver() -> None:
    program = (
        "import lofi.spec\n"
        "from didactic.settings import list_resolvers\n"
        "print(sorted(name for name in list_resolvers() if name.startswith('lofi.')))"
    )
    answered = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        encoding="utf-8",
        timeout=TIMEOUT_SECONDS,
    )

    assert answered.stdout.strip() == "[]"


def test_the_installed_resolvers_are_registered_on_top_of_the_builtins() -> None:
    registered = list_resolvers()

    assert set(BUILTIN_RESOLVERS) <= set(registered)
    assert {"oc.env", "oc.select", "oc.create"} <= set(registered)


def test_registering_again_over_a_taken_name_is_refused() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_resolvers()


def test_two_entry_points_cannot_claim_one_resolver_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = (
        ResolverPoint("third.resolve", "third_a:resolve", lambda *_: "a"),
        ResolverPoint("third.resolve", "third_b:resolve", lambda *_: "b"),
    )
    monkeypatch.setattr(registry_module, "entry_points", lambda: points)

    with pytest.raises(TagCollisionError, match=r"each declare 'third\.resolve'"):
        register_resolvers(replace=True)


def test_a_resolver_entry_point_must_name_a_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = (ResolverPoint("third.resolve", "third:resolve", "not callable"),)
    monkeypatch.setattr(registry_module, "entry_points", lambda: points)

    with pytest.raises(TypeError, match="not callable"):
        register_resolvers(replace=True)


@pytest.mark.usefixtures("restored_registry")
def test_registering_puts_back_resolvers_that_were_removed(
    context: RunContext,
) -> None:
    for name in BUILTIN_RESOLVERS:
        unregister_resolver(name)

    assert not set(BUILTIN_RESOLVERS) & set(list_resolvers())
    assert "Unknown resolver" in refusal("${lofi.path:a/b}", context)
    assert register_resolvers() == ("lofi.path", "lofi.pkg", "lofi.seed")
    assert set(BUILTIN_RESOLVERS) <= set(list_resolvers())
    assert resolved("${lofi.path:a/b}", context) == str(DATA_ROOT / "a" / "b")


@pytest.mark.usefixtures("restored_registry")
def test_registering_with_replace_takes_back_a_name_another_resolver_holds(
    context: RunContext,
) -> None:
    for name in BUILTIN_RESOLVERS:
        register_resolver(name, lambda *_: "shadowed", replace=True)

    assert resolved("${lofi.path:a/b}", context) == "shadowed"
    assert resolved("${lofi.seed:shuffle}", context) == "shadowed"
    assert register_resolvers(replace=True) == ("lofi.path", "lofi.pkg", "lofi.seed")
    assert resolved("${lofi.path:a/b}", context) == str(DATA_ROOT / "a" / "b")
    assert resolved("${lofi.seed:shuffle}", context) == derive_seed(SEED, "shuffle")
    assert Path(str(resolved("${lofi.pkg:device_policy/cpu}", context))).is_file()


def test_a_path_is_joined_against_the_run_data_root(context: RunContext) -> None:
    assert resolved("${lofi.path:ccgbank/train.auto}", context) == str(
        DATA_ROOT / "ccgbank" / "train.auto"
    )


def test_whitespace_around_an_argument_does_not_change_what_it_names(
    context: RunContext,
) -> None:
    assert resolved("${lofi.path: ccgbank/train.auto }", context) == resolved(
        "${lofi.path:ccgbank/train.auto}", context
    )


def test_a_path_inside_surrounding_text_is_substituted(context: RunContext) -> None:
    assert resolved("read ${lofi.path:a/b} now", context) == (
        f"read {DATA_ROOT / 'a' / 'b'} now"
    )


def test_a_redundant_separator_does_not_change_the_joined_path(
    context: RunContext,
) -> None:
    assert resolved("${lofi.path:./a//b/}", context) == resolved(
        "${lofi.path:a/b}", context
    )


@pytest.mark.parametrize(
    ("declared", "complaint"),
    [
        ("/etc/passwd", "absolute"),
        ("//server/share/corpora", "absolute"),
        ("C:/corpora/train.txt", "drive-qualified"),
        ("C:corpora", "drive-qualified"),
        ("../outside", "'..'"),
        ("~/corpora", "home-relative"),
        ("a\\\\b", "backslash"),
        (".", "no path"),
    ],
)
def test_a_path_that_does_not_lie_under_the_root_is_refused(
    declared: str, complaint: str, context: RunContext
) -> None:
    assert complaint in refusal(f"${{lofi.path:{declared}}}", context)


@pytest.mark.parametrize(
    "declared", ["C:/corpora/train.txt", "C:corpora", "/etc/passwd", "../outside"]
)
def test_a_path_the_data_root_refuses_is_refused_by_the_resolver_too(
    declared: str, context: RunContext
) -> None:
    root = DataRoot(DATA_ROOT)

    with pytest.raises(PathDeclarationError):
        root.resolve(declared, key="data.corpus")

    assert refusal(f"${{lofi.path:{declared}}}", context)


@pytest.mark.usefixtures("restored_registry")
def test_a_windows_spelled_path_is_refused_rather_than_silently_joined(
    context: RunContext,
) -> None:
    received: list[tuple[str, ...]] = []
    point = next(
        point
        for point in entry_points(group=RESOLVER_GROUP)
        if point.name == "lofi.path"
    )
    lofi_path = cast("ResolverFn", point.load())

    def recording_path(*args: str) -> ConfigValue:
        received.append(args)

        return lofi_path(*args)

    register_resolver("lofi.path", recording_path, replace=True)
    reason = refusal("${lofi.path:corpora\\train.txt}", context)

    assert received == [("corpora\\train.txt",)]
    assert "${lofi.path:corpora\\train.txt}" in reason
    assert "backslash" in reason


def test_a_path_outside_a_run_is_refused_for_want_of_a_data_root() -> None:
    assert "run context" in refusal("${lofi.path:a/b}")


def test_a_packaged_fragment_resolves_to_a_file_the_distribution_holds(
    context: RunContext,
) -> None:
    found = resolved("${lofi.pkg:model/encoders/terms/hrr}", context)

    assert isinstance(found, str)
    assert Path(found).is_file()
    assert Path(found).parent == (PACKAGED_CONFIG_ROOT / "model" / "encoders" / "terms")
    assert Path(found).stem == "hrr"


def test_a_packaged_fragment_needs_no_run_context() -> None:
    assert Path(str(resolve("${lofi.pkg:device_policy/cpu}", root={}))).is_file()


def test_the_packaged_fragment_resolver_does_not_resolve_resource_files() -> None:
    reason = refusal("${lofi.pkg:run/synthetic-parser.vec}")

    assert "packaged fragment" in reason
    assert "synthetic-parser.vec.yaml" in reason


@pytest.mark.parametrize(
    "resource",
    [
        "lofi.resource:",
        "lofi.resource:../synthetic-parser.vec",
        "lofi.resource:run/absent.vec",
    ],
)
def test_an_invalid_package_resource_is_refused(resource: str) -> None:
    with pytest.raises(ValueError, match=r"resource|path|file"):
        interpolation_module.lower_package_resource(resource)


def test_an_option_the_library_does_not_hold_names_every_suffix_tried(
    context: RunContext,
) -> None:
    reason = refusal("${lofi.pkg:device_policy/nothing}", context)
    directory = PACKAGED_CONFIG_ROOT / "device_policy"
    positions = [
        reason.find(str(directory / f"nothing{suffix}"))
        for suffix in (".yaml", ".yml", ".toml", ".json")
    ]

    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)


def test_an_option_held_under_the_last_suffix_tried_is_still_found(
    tmp_path: Path, context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(interpolation_module, "PACKAGED_CONFIG_ROOT", tmp_path)
    group = tmp_path / "trainer"
    group.mkdir()
    (group / "quick.json").write_text("{}", encoding="utf-8", newline="\n")

    assert resolved("${lofi.pkg:trainer/quick}", context) == str(group / "quick.json")


def test_two_fragments_sharing_a_stem_are_refused_as_ambiguous(
    tmp_path: Path, context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(interpolation_module, "PACKAGED_CONFIG_ROOT", tmp_path)
    group = tmp_path / "trainer"
    group.mkdir()
    (group / "quick.yaml").write_text("{}", encoding="utf-8", newline="\n")
    (group / "quick.toml").write_text("", encoding="utf-8", newline="\n")

    reason = refusal("${lofi.pkg:trainer/quick}", context)

    assert "ambiguous" in reason
    assert "quick.yaml" in reason
    assert "quick.toml" in reason


def test_an_argument_naming_no_group_is_refused(context: RunContext) -> None:
    assert "names no group" in refusal("${lofi.pkg:cpu}", context)


def test_a_sub_seed_is_a_function_of_its_name_and_the_run_seed(
    context: RunContext,
) -> None:
    assert resolved("${lofi.seed:shuffle}", context) == derive_seed(SEED, "shuffle")
    assert resolved("${lofi.seed:shuffle}", context) != resolved(
        "${lofi.seed:init}", context
    )


def test_the_sub_seed_a_name_takes_does_not_drift() -> None:
    assert derive_seed(0, "a") == 221181804
    assert derive_seed(0, "shuffle") == 943667519
    assert derive_seed(1, "init") == 2017059705
    assert derive_seed(4321, "shuffle") == 2306295471


def test_a_sub_seed_does_not_depend_on_what_was_asked_before_it() -> None:
    forward = [derive_seed(SEED, name) for name in ("a", "b", "c")]
    backward = [derive_seed(SEED, name) for name in ("c", "b", "a")]

    assert forward == backward[::-1]


def test_a_sub_seed_moves_with_the_run_seed() -> None:
    assert derive_seed(SEED, "shuffle") != derive_seed(SEED + 1, "shuffle")


def test_a_sub_seed_is_a_seed_every_stream_accepts() -> None:
    drawn = [derive_seed(SEED, f"stream-{index}") for index in range(64)]

    assert all(0 <= seed < SEED_CEILING for seed in drawn)
    assert len(set(drawn)) == len(drawn)


def test_a_sub_seed_is_the_same_in_another_process() -> None:
    program = (
        "from lofi.spec import derive_seed;"
        f"print(derive_seed({SEED}, 'shuffle'), derive_seed({SEED}, 'init'))"
    )
    drawn = []

    for hash_seed in ("0", "1"):
        answered = subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            capture_output=True,
            encoding="utf-8",
            timeout=TIMEOUT_SECONDS,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        drawn.append(answered.stdout.split())

    assert drawn[0] == drawn[1]
    assert drawn[0] == [
        str(derive_seed(SEED, "shuffle")),
        str(derive_seed(SEED, "init")),
    ]


def test_a_sub_seed_outside_a_run_is_refused_for_want_of_a_seed() -> None:
    assert "run context" in refusal("${lofi.seed:shuffle}")


@pytest.mark.parametrize("seed", [-1, -4321])
def test_a_negative_run_seed_derives_nothing(seed: int) -> None:
    with pytest.raises(ValueError, match="negative"):
        derive_seed(seed, "shuffle")


def test_an_unnamed_stream_derives_nothing() -> None:
    with pytest.raises(ValueError, match="empty"):
        derive_seed(SEED, "")


@pytest.mark.parametrize("resolver", ["lofi.path", "lofi.pkg", "lofi.seed"])
def test_a_resolver_takes_exactly_one_argument(
    resolver: str, context: RunContext
) -> None:
    assert "one argument" in refusal(f"${{{resolver}:}}", context)
    assert "one argument" in refusal(f"${{{resolver}:a,b}}", context)


def test_no_run_context_is_installed_outside_a_run() -> None:
    assert active_run_context() is None


def test_the_innermost_run_context_is_the_one_read(context: RunContext) -> None:
    inner = RunContext(data_root=Path("/tmp/other"), seed=SEED + 1)  # noqa: S108

    with run_context(context):
        assert active_run_context() == context

        with run_context(inner) as installed:
            assert installed is inner
            assert active_run_context() == inner

        assert active_run_context() == context

    assert active_run_context() is None


def test_a_run_context_is_lifted_when_the_block_raises(context: RunContext) -> None:
    with pytest.raises(RuntimeError), run_context(context):
        raise RuntimeError

    assert active_run_context() is None
