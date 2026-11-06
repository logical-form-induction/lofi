"""No compute site is named in the released tree.

A site supplies its own account, placement, storage roots and host in a profile
file that lives outside the repository, so a value of any of those shapes inside
the tree is a leak of one installation into a distribution meant for all of
them. The shapes are declared once, in ``SHAPES``, and the words a value of such
a shape may be built from are declared once, in ``NEUTRAL_WORDS``: forbidding a
further shape, or admitting a further placeholder, is an edit to that data. A
shape may decline a file suffix, which is how the unquoted shapes stay off
source code, where annotations and expressions do not declare site values.

The scan reads the contents of the files the repository tracks. It never reads
the path of the checkout itself, so a clone under a user's home directory is not
an offense.
"""

import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable

TREE: Final = Path(__file__).resolve().parents[2]

TOP_LEVEL_DOMAINS: Final = (
    "ca",
    "com",
    "de",
    "edu",
    "gov",
    "io",
    "local",
    "mil",
    "net",
    "org",
    "uk",
)
"""The domain endings a host is recognized by."""

_TLD: Final = rf"(?:{'|'.join(TOP_LEVEL_DOMAINS)})"

_VALUE: Final = r"[^\s\"'`{}()\[\],;]+"
"""The characters a value assigned on a command line runs over.

Braces are excluded, so a spelling whose value is a format field is no
assignment of one.
"""

_HOST_KEY: Final = (
    r"(?:host|hostname|node|server|login_node|head_node|submit_host|remote_host)"
)
"""The keys whose value is the name of a machine."""

_DOTTED_HOST: Final = r"[A-Za-z][\w-]*(?:\.[\w-]+)+"
"""A host written with at least one dot, such as a fully qualified name."""

EXPRESSION_SOURCE: Final = frozenset({".py", ".pyi", ".js"})
"""Source suffixes whose unquoted values are expressions or annotations.

``node: DerivationNode`` and ``queue = queue.then(callback)`` do not declare a
host or a scheduler queue. Quoted site declarations remain subject to the
quoted shapes in these files.
"""


class Shape(NamedTuple):
    """A textual shape that a site-specific identifier takes.

    Attributes
    ----------
    name : str
        The name a failure reports.
    pattern : re.Pattern[str]
        A pattern whose first group captures the value carrying the identifier.
    plants : tuple[str, ...]
        Format strings that each yield an instance of the shape when formatted
        with a name, used to show that the shape is caught.
    declined_suffixes : frozenset[str]
        The suffixes of files the shape is not read against. Empty by default,
        so a shape reads every file.
    """

    name: str
    pattern: re.Pattern[str]
    plants: tuple[str, ...]
    declined_suffixes: frozenset[str] = frozenset()


SHAPES: Final = (
    Shape(
        "cluster account",
        re.compile(rf"--(?:account|allocation)=({_VALUE})"),
        ("--account={}", "--allocation={}"),
    ),
    Shape(
        "partition or queue",
        re.compile(rf"--(?:partition|queue|qos|reservation)=({_VALUE})"),
        ("--partition={}", "--queue={}", "--qos={}", "--reservation={}"),
    ),
    Shape(
        "submission flag",
        re.compile(
            rf"(?mi)\b(?:sbatch|srun|salloc|qsub|bsub)\b[^\n]*?[ \t]-[apq][ \t]+"
            rf"(?!-)({_VALUE})"
        ),
        (
            "sbatch -A {} job.sh",
            "srun -p {} run.sh",
            "qsub -q {} job.sh",
            "#SBATCH -A {}",
            "#SBATCH -p {}",
        ),
    ),
    Shape(
        "declared placement",
        re.compile(
            r"""(?mi)^[ \t]*(?:account|partition|queue|qos)[ \t]*[:=][ \t]*"""
            r"""["']([^"'{}\n]+)"""
        ),
        ('account = "{}"', 'partition: "{}"', 'queue = "{}"', 'qos: "{}"'),
    ),
    Shape(
        "unquoted placement",
        re.compile(
            r"""(?mi)^[ \t]*(?:account|partition|queue|qos)[ \t]*[:=][ \t]*"""
            rf"""(?!["'])({_VALUE})"""
        ),
        ("account = {}", "partition: {}", "queue = {}", "qos: {}"),
        EXPRESSION_SOURCE,
    ),
    Shape(
        "declared host",
        re.compile(
            rf"""(?mi)^[ \t]*["']?{_HOST_KEY}["']?[ \t]*[:=][ \t]*["']([^"'{{}}\n]+)"""
        ),
        ('host = "{}"', 'hostname: "{}"', 'node = "{}"', '"node": "{}"'),
    ),
    Shape(
        "unquoted host",
        re.compile(
            rf"""(?mi)^[ \t]*["']?{_HOST_KEY}["']?[ \t]*[:=][ \t]*"""
            rf"""(?!["'])({_VALUE})"""
        ),
        ("host = {}", "hostname: {}", "node = {}", "server: {}"),
        EXPRESSION_SOURCE,
    ),
    Shape(
        "session host",
        re.compile(
            rf"(?m)\b(?:ssh|scp|sftp|rsync)\b[^\n]*?(?<![\w.:/@-])"
            rf"(?:[A-Za-z][\w.-]*@)?({_DOTTED_HOST})(?![\w.-])"
        ),
        ("ssh {}.example.edu", "scp out someone@{}.example.org:/work/out"),
    ),
    Shape(
        "remote path",
        re.compile(r"(?m)(?<![\w.:/@-])([A-Za-z][\w.-]+):(?=/[\w.-])"),
        ("rsync -a data/ {}:/work/out",),
    ),
    Shape(
        "compute host",
        re.compile(
            rf"\b(?:login|head|node|compute|gpu|hpc|cluster|slurm|submit|batch)[\w-]*"
            rf"\.([\w.-]*[\w-])\.{_TLD}\b"
        ),
        ("login.{}.edu", "gpu-node.{}.gov"),
    ),
    Shape(
        "site filesystem root",
        re.compile(
            r"(?<![\w.-])/(?:scratch|gpfs|lustre|panfs|nobackup|work|projects"
            r"|software)/([\w.-]+)"
        ),
        ("/scratch/{}", "/gpfs/{}/runs", "/lustre/{}", "/projects/{}"),
    ),
    Shape(
        "user-bearing path",
        re.compile(
            r"(?:(?<![\w.-])/(?:home|Users)/|[A-Za-z]:\\Users\\)([A-Za-z][\w.-]*)"
        ),
        ("/home/{}", "/Users/{}/lofi", "C:\\Users\\{}"),
    ),
    Shape(
        "home shorthand",
        re.compile(r"(?<![\w./])~([A-Za-z][\w.-]*)/"),
        ("~{}/lofi",),
    ),
)
"""Every shape a site-specific identifier is recognized by."""

NEUTRAL_WORDS: Final = frozenset(TOP_LEVEL_DOMAINS) | frozenset(
    {
        "account",
        "allocation",
        "any",
        "cache",
        "cluster",
        "compute",
        "default",
        "example",
        "first",
        "gpu",
        "home",
        "host",
        "lofi",
        "lsf",
        "macbookpro",
        "name",
        "node",
        "out",
        "partition",
        "pbs",
        "placeholder",
        "project",
        "qos",
        "queue",
        "run",
        "sample",
        "scratch",
        "second",
        "sge",
        "shared",
        "site",
        "slurm",
        "someone",
        "storage",
        "test",
        "third",
        "user",
        "work",
    }
)
"""The words a value of a forbidden shape may be built from.

A value assembled from these names the kind of thing it is rather than one
installation of it, which is what a placeholder in an example and a fixture in a
test are both doing. A hardware model and a top-level domain name a kind in the
same way. A word outside this vocabulary is the name of a site.
"""

PLANTED_NAME: Final = "arcturus"
"""A name no vocabulary holds, formatted into a shape's plants."""

SITE_SCHEDULERS: Final = re.compile(
    r"(?i)\b(?:slurm|sbatch|srun|salloc|pbs|qsub|bsub|lsf|sge)\b"
)
"""Scheduler brands and commands the site-agnostic library must never select."""

LEAKS: Final = (
    ("--account={}-2109", "cluster account"),
    ("--partition={}-second", "partition or queue"),
    ("sbatch -A {}-2109 --nodes 2 job.sh", "submission flag"),
    ("#SBATCH -A {}-2109", "submission flag"),
    ('account = "{}-2109"', "declared placement"),
    ("partition: {}-gpu", "unquoted placement"),
    ("host: {}-login.example.edu", "unquoted host"),
    ('"node": "{}-dhcp.wireless-site.edu"', "declared host"),
    ("ssh someone@{}-node042.example.edu", "session host"),
    ("rsync -a runs/ {}-site:/work/out", "remote path"),
    ("login.{}-site.edu", "compute host"),
    ("/scratch/{}-user/runs", "site filesystem root"),
    ("/home/{}/lofi", "user-bearing path"),
    ("~{}/lofi", "home shorthand"),
)
"""Whole lines a launch script carries when a site has been written into it.

Each pairs a line, formatted with a name outside the vocabulary, with the shape
that has to catch it, so that every shape is shown to catch an identifier as it
is really spelled rather than only in isolation.
"""


class Offense(NamedTuple):
    """One site-specific identifier found in one file.

    Attributes
    ----------
    path : str
        The file's path relative to the scanned root.
    line : int
        The one-based line the identifier sits on.
    shape : str
        The name of the shape that matched.
    value : str
        The matched value.
    words : tuple[str, ...]
        The words of the value that name no neutral concept.
    """

    path: str
    line: int
    shape: str
    value: str
    words: tuple[str, ...]


def foreign_words(value: str) -> tuple[str, ...]:
    """Return the words of a value that name no neutral concept.

    Parameters
    ----------
    value : str
        A value a shape captured.

    Returns
    -------
    tuple[str, ...]
        Its words, lowercased and stripped of a trailing number, that
        ``NEUTRAL_WORDS`` does not hold. A purely numeric word is neutral.
    """
    parts = re.split(r"[^A-Za-z0-9]+", value)
    stripped = (re.sub(r"\d+$", "", part).lower() for part in parts)

    return tuple(word for word in stripped if word and word not in NEUTRAL_WORDS)


def offenses_in_text(text: str, path: str) -> list[Offense]:
    """Return the site-specific identifiers one file's contents carry.

    Parameters
    ----------
    text : str
        The file's contents.
    path : str
        The path reported with each offense, and whose suffix decides which
        shapes read the text.

    Returns
    -------
    list[Offense]
        One offense per match whose value carries a word outside the
        vocabulary, in the order the shapes are declared. A shape that declines
        the path's suffix contributes none.
    """
    found = []
    suffix = PurePosixPath(path).suffix

    for shape in SHAPES:
        if suffix in shape.declined_suffixes:
            continue

        for match in shape.pattern.finditer(text):
            value = match.group(1)
            words = foreign_words(value)

            if words:
                line = text.count("\n", 0, match.start()) + 1
                found.append(Offense(path, line, shape.name, value, words))

    return found


def offenses_in_files(paths: Iterable[Path], *, root: Path) -> list[Offense]:
    """Return the site-specific identifiers a set of files carries.

    Parameters
    ----------
    paths : Iterable[pathlib.Path]
        The files to read. One that is unreadable or is not UTF-8 text is
        skipped.
    root : pathlib.Path
        The directory offense paths are reported relative to.

    Returns
    -------
    list[Offense]
        Every offense across the files, in the order the files are given.
    """
    found = []

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError, UnicodeDecodeError:
            continue

        found.extend(offenses_in_text(text, path.relative_to(root).as_posix()))

    return found


def released_files() -> list[Path]:
    """Return the tracked files the distribution releases.

    Returns
    -------
    list[pathlib.Path]
        Every path ``git`` tracks in the checkout.
    """
    git = shutil.which("git")

    assert git is not None, "the scan lists the tracked files and git is absent"

    listing = subprocess.run(
        [git, "-C", str(TREE), "ls-files", "-z"],
        capture_output=True,
        encoding="utf-8",
        check=True,
    ).stdout

    return [TREE / name for name in listing.split("\0") if name]


def describe(offenses: Iterable[Offense]) -> str:
    """Return a failure message naming each offense and where it sits.

    Parameters
    ----------
    offenses : Iterable[Offense]
        The offenses to report.

    Returns
    -------
    str
        One line per offense, giving its path, line, shape and value.
    """
    lines = [
        f"{found.path}:{found.line}: {found.shape}: {found.value}" for found in offenses
    ]

    return "\n".join(["a compute site is named in the tree:", *lines])


def test_the_released_tree_names_no_compute_site() -> None:
    found = offenses_in_files(released_files(), root=TREE)

    assert not found, describe(found)


def test_the_listing_covers_the_released_tree() -> None:
    names = {path.relative_to(TREE).as_posix() for path in released_files()}

    assert "lofi/run/_boundary/__init__.py" in names
    assert "docs/examples/site-profile.toml" in names
    assert not [name for name in names if name.startswith(".local/")]


def test_the_declaring_module_names_no_site() -> None:
    module = Path(__file__).resolve()

    assert not offenses_in_files([module], root=module.parent)


def test_the_example_site_profile_names_no_site() -> None:
    profile = TREE / "docs" / "examples" / "site-profile.toml"

    assert not offenses_in_files([profile], root=TREE)


def test_the_worked_site_profiles_name_no_site() -> None:
    profiles = sorted((TREE / "config" / "sites").glob("*.toml"))

    assert {profile.name for profile in profiles} == {
        "array.toml",
        "local-accelerator.toml",
    }
    assert not offenses_in_files(profiles, root=TREE)


def test_the_library_selects_no_site_scheduler() -> None:
    sources = sorted((TREE / "lofi").rglob("*.py"))
    named = [
        path.relative_to(TREE).as_posix()
        for path in sources
        if SITE_SCHEDULERS.search(path.read_text(encoding="utf-8"))
    ]

    assert not named, f"site schedulers named in the library: {', '.join(named)}"


def test_a_site_value_in_the_example_profile_is_caught(tmp_path: Path) -> None:
    profile = TREE / "docs" / "examples" / "site-profile.toml"
    planted = tmp_path / "site-profile.toml"
    planted.write_text(
        profile.read_text(encoding="utf-8").replace(
            "placeholder-account", f"{PLANTED_NAME}-2109"
        ),
        encoding="utf-8",
        newline="\n",
    )

    found = offenses_in_files([planted], root=tmp_path)

    assert [offense.shape for offense in found] == ["declared placement"]
    assert found[0].words == (PLANTED_NAME,)


@pytest.mark.parametrize(
    ("shape", "plant"),
    [(shape, plant) for shape in SHAPES for plant in shape.plants],
    ids=[f"{shape.name}: {plant}" for shape in SHAPES for plant in shape.plants],
)
def test_a_planted_identifier_is_named_with_its_location(
    tmp_path: Path, shape: Shape, plant: str
) -> None:
    planted = tmp_path / "launch.conf"
    planted.write_text(
        f"a line with nothing in it\n{plant.format(PLANTED_NAME)}\n",
        encoding="utf-8",
        newline="\n",
    )

    found = offenses_in_files([planted], root=tmp_path)

    assert len(found) == 1, describe(found)

    offense = found[0]

    assert (offense.path, offense.line, offense.shape) == (
        "launch.conf",
        2,
        shape.name,
    )
    assert offense.words == (PLANTED_NAME,)
    assert PLANTED_NAME in offense.value
    assert describe(found).endswith(f"launch.conf:2: {shape.name}: {offense.value}")


@pytest.mark.parametrize(
    ("shape", "plant"),
    [(shape, plant) for shape in SHAPES for plant in shape.plants],
    ids=[f"{shape.name}: {plant}" for shape in SHAPES for plant in shape.plants],
)
def test_a_planted_identifier_is_captured_past_a_neutral_prefix(
    tmp_path: Path, shape: Shape, plant: str
) -> None:
    compound = f"default-{PLANTED_NAME}-second"
    planted = tmp_path / "launch.conf"
    planted.write_text(f"{plant.format(compound)}\n", encoding="utf-8", newline="\n")

    found = offenses_in_files([planted], root=tmp_path)

    assert [offense.shape for offense in found] == [shape.name], describe(found)
    assert compound in found[0].value
    assert found[0].words == (PLANTED_NAME,)


@pytest.mark.parametrize(
    ("shape", "plant"),
    [(shape, plant) for shape in SHAPES for plant in shape.plants],
    ids=[f"{shape.name}: {plant}" for shape in SHAPES for plant in shape.plants],
)
def test_a_neutral_value_of_a_forbidden_shape_is_no_offense(
    tmp_path: Path, shape: Shape, plant: str
) -> None:
    planted = tmp_path / "launch.conf"
    planted.write_text(
        plant.format("placeholder-second-partition"), encoding="utf-8", newline="\n"
    )

    assert not offenses_in_files([planted], root=tmp_path)
    assert shape.pattern.search(planted.read_text(encoding="utf-8")) is not None


@pytest.mark.parametrize(("spelling", "shape"), LEAKS, ids=[name for _, name in LEAKS])
def test_a_realistic_leak_is_caught(tmp_path: Path, spelling: str, shape: str) -> None:
    planted = tmp_path / "launch.sh"
    planted.write_text(
        f"{spelling.format(PLANTED_NAME)}\n", encoding="utf-8", newline="\n"
    )

    found = offenses_in_files([planted], root=tmp_path)

    assert [offense.shape for offense in found] == [shape], describe(found)
    assert (found[0].path, found[0].line) == ("launch.sh", 1)
    assert PLANTED_NAME in found[0].words


def test_every_shape_is_shown_to_catch_a_whole_line() -> None:
    assert {name for _, name in LEAKS} == {shape.name for shape in SHAPES}


@pytest.mark.parametrize(
    "spelling",
    [
        "see https://github.com/logical-form-induction/lofi",
        "ssh is described at https://github.com/logical-form-induction/lofi",
        "url = https://files.pythonhosted.org/packages/x.whl",
        'declared = "C:/corpora/train"',
        "pytest -p no:randomly",
        f"https://www.example.edu/~{PLANTED_NAME}/paper.pdf",
        f"http://www.example.org/archive/~{PLANTED_NAME}/paper.pdf",
    ],
)
def test_a_spelling_that_names_no_site_is_no_offense(
    tmp_path: Path, spelling: str
) -> None:
    planted = tmp_path / "notes.md"
    planted.write_text(f"{spelling}\n", encoding="utf-8", newline="\n")

    assert not offenses_in_files([planted], root=tmp_path), spelling


def test_a_checkout_under_a_user_path_is_no_offense(tmp_path: Path) -> None:
    clone = tmp_path / "checkout+" / "home" / PLANTED_NAME / "lofi"
    clone.mkdir(parents=True)
    tracked = clone / "settings.toml"
    tracked.write_text(
        'scratch_root = "/placeholder/scratch"\n', encoding="utf-8", newline="\n"
    )

    absolute = offenses_in_text(tracked.as_posix(), tracked.name)
    relative = offenses_in_text(tracked.relative_to(tmp_path).as_posix(), tracked.name)

    assert PLANTED_NAME in {word for found in absolute for word in found.words}
    assert PLANTED_NAME in {word for found in relative for word in found.words}
    assert not offenses_in_files([tracked], root=tmp_path)


def test_a_file_that_is_not_utf8_text_is_skipped(tmp_path: Path) -> None:
    binary = tmp_path / "objects.bin"
    binary.write_bytes(b"\xff\xfe\x00" + f"/scratch/{PLANTED_NAME}".encode())

    assert not offenses_in_files([binary, tmp_path / "absent.txt"], root=tmp_path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("placeholder-account", ()),
        ("account-slurm", ()),
        ("partition2", ()),
        ("scratch_1024", ()),
        (f"{PLANTED_NAME}7", (PLANTED_NAME,)),
        (f"lofi-{PLANTED_NAME}.queue", (PLANTED_NAME,)),
    ],
)
def test_a_value_is_foreign_where_a_word_is(
    value: str, expected: tuple[str, ...]
) -> None:
    assert foreign_words(value) == expected


@pytest.mark.parametrize("suffix", [".py", ".pyi", ".js"])
@pytest.mark.parametrize(
    "expression", ["queue = queue.then(callback)", "node = document.body"]
)
def test_source_expressions_are_not_site_declarations(
    suffix: str, expression: str
) -> None:
    assert not offenses_in_text(expression, f"module{suffix}")


@pytest.mark.parametrize("suffix", [".py", ".pyi", ".js"])
@pytest.mark.parametrize(
    ("spelling", "shape"),
    [
        ('queue = "{}"', "declared placement"),
        ('host = "{}"', "declared host"),
    ],
)
def test_quoted_site_declarations_are_caught_in_source(
    suffix: str, spelling: str, shape: str
) -> None:
    found = offenses_in_text(spelling.format(PLANTED_NAME), f"module{suffix}")

    assert [offense.shape for offense in found] == [shape]
    assert found[0].words == (PLANTED_NAME,)
