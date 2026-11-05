import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest

from lofi._checks import LawViolationError
from lofi.data.laws import READER_LAWS

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from lofi.data.laws import ReaderFixture

__all__ = [
    "CorpusReaderConformance",
    "CorpusReaderSubject",
    "imports_no_tensor_library",
]

_LAW_NAMES: Final = tuple(law.__name__ for law in READER_LAWS)

_TREE: Final = Path(__file__).resolve().parents[2]

_TENSOR_LIBRARY: Final = "torch"

_TIMEOUT_SECONDS: Final = 120.0


def imports_no_tensor_library(package: str) -> None:
    """Check that a package and everything under it import with torch rejected.

    Parameters
    ----------
    package : str
        The package the reader is defined in.

    Raises
    ------
    LawViolationError
        If the package, or any module under it, asks for the tensor library.
    """
    finished = subprocess.run(
        [sys.executable, "-m", "tests.independence._import_under_poison"],
        input=f"{package}\n{_TENSOR_LIBRARY}",
        capture_output=True,
        encoding="utf-8",
        check=False,
        cwd=_TREE,
        timeout=_TIMEOUT_SECONDS,
    )

    if finished.returncode != 0:
        message = (
            f"the reader imports no tensor library: {package} did not import "
            f"with {_TENSOR_LIBRARY} refused:\n{finished.stderr}"
        )

        raise LawViolationError(message)


class CorpusReaderSubject[L: Hashable, G](NamedTuple):
    """Inputs for the corpus-reader conformance laws.

    Parameters
    ----------
    reader : ReaderFixture[L, G]
        How to load the corpus, which two of the laws do twice.
    package : str
        The package the reader is defined in, imported in a fresh interpreter by the law
        about the tensor library.
    """

    reader: ReaderFixture[L, G]
    package: str


class CorpusReaderConformance:
    pytestmark = pytest.mark.conformance

    @pytest.mark.parametrize("law", READER_LAWS, ids=_LAW_NAMES)
    def test_the_reader_holds_to_each_law[L: Hashable, G](
        self,
        law: Callable[[ReaderFixture[L, G]], None],
        subject: CorpusReaderSubject[L, G],
    ) -> None:
        """Check one in-process law against the subject.

        Parameters
        ----------
        law : Callable[[ReaderFixture[L, G]], None]
            The law to run.
        subject : CorpusReaderSubject[L, G]
            What the laws run against.
        """
        law(subject.reader)

    def test_the_reader_imports_no_tensor_library[L: Hashable, G](
        self,
        subject: CorpusReaderSubject[L, G],
    ) -> None:
        """Check that the reader's package imports with the tensor library rejected.

        Parameters
        ----------
        subject : CorpusReaderSubject[L, G]
            What the laws run against.
        """
        imports_no_tensor_library(subject.package)
