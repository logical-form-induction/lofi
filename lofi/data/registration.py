"""Codec tag declaration against the shared codec registry.

A codec declares the tag it answers to in its own file, and the declaration puts
it in the registry ``lairs`` ships, under the ``lairs.codecs`` entry point group.
A distribution that is neither ``lairs`` nor ``lofi`` adds a corpus by declaring
a tag the same way and listing its class in that group, so nothing here or in
``lairs`` is edited when a corpus arrives.
"""

from typing import TYPE_CHECKING

from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.integrations.ports import Codec
from lairs.integrations.registry import get_codec, register_codec

from lofi._checks import ensure_name

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["declared_codec_tags", "registers_as"]

_CLAIMED: set[str] = set()
"""The tags declared so far, which is what a second declaration is refused against."""


def declared_codec_tags() -> tuple[str, ...]:
    """Return the tags declared through :func:`registers_as`, in lexical order.

    A tag appears here once the module holding its decorated class has been
    imported, which is what loading the codec's entry point does. The registry
    holds more than this: a codec ``lairs`` ships, or one a third party
    registers directly, answers to a tag that was never declared here.

    Returns
    -------
    tuple[str, ...]
        The declared tags, in lexical order.
    """
    return tuple(sorted(_CLAIMED))


def registers_as[C: Codec[CorpusFragment, FragmentRecord]](
    tag: str,
) -> Callable[[type[C]], type[C]]:
    """Return a decorator declaring the tag one codec is resolved under.

    Parameters
    ----------
    tag : str
        The tag the decorated codec answers to.

    Returns
    -------
    Callable[[type[C]], type[C]]
        A class decorator that registers the codec and returns it unchanged.

    Raises
    ------
    TypeError
        If the tag is not a string.
    ValueError
        If the tag is empty.
    """
    named = ensure_name("tag", tag)

    def declare(codec: type[C]) -> type[C]:
        """Register one codec class under the declared tag.

        Parameters
        ----------
        codec : type[C]
            The class being created.

        Returns
        -------
        type[C]
            That same class.

        Raises
        ------
        TypeError
            If another codec already declared the tag.
        """
        if named in _CLAIMED:
            held = get_codec(named)
            msg = (
                f"{codec.__qualname__} declares the codec tag {named!r}, which "
                f"{held.__qualname__} already declared"
            )

            raise TypeError(msg)

        register_codec(named, codec)
        _CLAIMED.add(named)

        return codec

    return declare
