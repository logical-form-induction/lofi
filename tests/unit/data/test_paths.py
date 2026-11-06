from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lofi.data.paths import DataRoot, DataRootError, PathDeclarationError


@pytest.fixture
def root() -> DataRoot:
    return DataRoot(Path("/srv/lofi-data").absolute())


def test_declared_path_resolves_under_the_root(root: DataRoot) -> None:
    resolved = root.resolve("corpora/layers/train.jsonl", key="corpus.train")

    assert resolved == root.directory / "corpora/layers/train.jsonl"
    assert resolved.is_relative_to(root.directory)


def test_resolution_ignores_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = DataRoot(tmp_path / "data")
    here = tmp_path / "here"
    there = tmp_path / "there" / "deeper"
    here.mkdir()
    there.mkdir(parents=True)

    monkeypatch.chdir(here)
    first = root.resolve("corpora/train", key="corpus.train")

    monkeypatch.chdir(there)
    second = root.resolve("corpora/train", key="corpus.train")

    assert first == second == tmp_path / "data" / "corpora" / "train"


def test_resolution_reads_no_filesystem(root: DataRoot) -> None:
    resolved = root.resolve("nowhere/at/all.jsonl", key="corpus.dev")

    assert not resolved.exists()


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("corpora/train", "corpora/train"),
        ("corpora/./train", "corpora/train"),
        ("corpora//train", "corpora/train"),
        ("corpora/train/", "corpora/train"),
        ("  corpora/train  ", "corpora/train"),
        ("train.jsonl", "train.jsonl"),
    ],
)
def test_equivalent_spellings_resolve_alike(
    root: DataRoot, declared: str, expected: str
) -> None:
    assert root.resolve(declared, key="corpus.train") == root.directory / expected


@pytest.mark.parametrize(
    ("declared", "reason"),
    [
        ("/srv/corpora/train", "an absolute path"),
        ("/", "an absolute path"),
        ("//host/share/train", "an absolute path"),
        ("C:/corpora/train", "an absolute path"),
        ("C:\\corpora\\train", "spelled with a backslash"),
        ("corpora\\train", "spelled with a backslash"),
        ("~/corpora/train", "a home-relative path"),
        ("~someone/corpora", "a home-relative path"),
        ("../corpora/train", "spelled with a '..' component"),
        ("corpora/../../elsewhere", "spelled with a '..' component"),
        ("corpora/../train", "spelled with a '..' component"),
        ("", "empty"),
        ("   ", "empty"),
    ],
)
def test_unresolvable_declarations_are_refused(
    root: DataRoot, declared: str, reason: str
) -> None:
    with pytest.raises(PathDeclarationError) as caught:
        root.resolve(declared, key="corpus.train")

    assert caught.value.reason == reason
    assert caught.value.declared == declared


def test_a_refusal_names_the_offending_key(root: DataRoot) -> None:
    with pytest.raises(PathDeclarationError) as caught:
        root.resolve("/home/someone/ccgbank", key="corpus.ccgbank.source")

    assert caught.value.key == "corpus.ccgbank.source"
    assert "corpus.ccgbank.source" in str(caught.value)
    assert "/home/someone/ccgbank" in str(caught.value)


@pytest.mark.parametrize(
    ("directory", "reason"),
    [
        (Path("data"), "relative to a working directory"),
        (Path("./data"), "relative to a working directory"),
        (Path("/srv/../srv/data").absolute(), "spelled with a '..' component"),
    ],
)
def test_unusable_roots_are_refused(directory: Path, reason: str) -> None:
    with pytest.raises(DataRootError) as caught:
        DataRoot(directory)

    assert caught.value.reason == reason
    assert caught.value.directory == directory


def test_roots_naming_one_directory_are_one_value() -> None:
    one = DataRoot(Path("/srv/lofi-data").absolute())
    same = DataRoot(Path("/srv/lofi-data/").absolute())
    other = DataRoot(Path("/srv/other-data").absolute())

    assert one == same
    assert hash(one) == hash(same)
    assert one != other
    assert one != Path("/srv/lofi-data").absolute()
    assert repr(one) == f"DataRoot({one.directory!r})"


@given(declared=st.text(max_size=40))
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_any_declaration_either_stays_under_the_root_or_is_refused(
    root: DataRoot, declared: str
) -> None:
    try:
        resolved = root.resolve(declared, key="corpus.train")
    except PathDeclarationError:
        resolved = None

    if resolved is not None:
        assert resolved.is_relative_to(root.directory)


@given(
    components=st.lists(
        st.text(alphabet=st.characters(categories=["Ll", "Nd"]), min_size=1),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_ordinary_components_resolve_to_the_root_joined_with_them(
    root: DataRoot, components: list[str]
) -> None:
    resolved = root.resolve("/".join(components), key="corpus.train")

    assert resolved == root.directory.joinpath(*components)
