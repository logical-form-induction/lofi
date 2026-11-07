from typing import TYPE_CHECKING, Final, NamedTuple

import pytest

from lofi.data.laws import CODEC_LAWS, ISO_LAWS, CodecFixture, IsoFixture

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["CodecConformance", "CodecSubject"]

_ISO_LAW_NAMES: Final = tuple(law.__name__ for law in ISO_LAWS)

_CODEC_LAW_NAMES: Final = tuple(law.__name__ for law in CODEC_LAWS)


class CodecSubject[Source, Fragment, Record](NamedTuple):
    """The two fixtures a codec's laws run against.

    Parameters
    ----------
    iso
        The translation under test, with strategies over both of its sides.
    codec
        The codec under test, with a strategy over the external sources it decodes and
        the way into a fragment's records.
    """

    iso: IsoFixture[Source, Fragment]
    codec: CodecFixture[Fragment, Record]


class CodecConformance:
    pytestmark = pytest.mark.conformance

    @pytest.mark.parametrize("law", ISO_LAWS, ids=_ISO_LAW_NAMES)
    def test_the_translation_holds_to_each_law[Source, Fragment, Record](
        self,
        law: Callable[[IsoFixture[Source, Fragment]], None],
        subject: CodecSubject[Source, Fragment, Record],
    ) -> None:
        """Check one law of the translation against the subject.

        Parameters
        ----------
        law
            The law to run.
        subject
            What the laws run against.
        """
        law(subject.iso)

    @pytest.mark.parametrize("law", CODEC_LAWS, ids=_CODEC_LAW_NAMES)
    def test_the_codec_holds_to_each_law[Source, Fragment, Record](
        self,
        law: Callable[[CodecFixture[Fragment, Record]], None],
        subject: CodecSubject[Source, Fragment, Record],
    ) -> None:
        """Check one law of the codec against the subject.

        Parameters
        ----------
        law
            The law to run.
        subject
            What the laws run against.
        """
        law(subject.codec)
