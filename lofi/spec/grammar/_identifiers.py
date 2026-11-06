"""A bijection between declared names and Python identifiers.

Nonalphanumeric characters are encoded as lowercase hexadecimal code points
between underscores. Leading digits and keywords also escape their first
character. Decoding accepts only spellings produced by the encoder.
"""

import keyword

from lofi.spec._errors import IdentifierError

__all__ = ["decode_identifier", "encode_identifier"]

_HEX_DIGITS = frozenset("0123456789abcdef")


def _plain(character: str) -> bool:
    """Return whether a character stands for itself in an identifier.

    Parameters
    ----------
    character : str
        The character to inspect.
    """
    return character.isascii() and character.isalnum()


def _escaped(character: str) -> str:
    """Return the escape sequence for one character.

    Parameters
    ----------
    character : str
        The character to escape.
    """
    return f"_{ord(character):x}_"


def encode_identifier(name: str) -> str:
    """Return the identifier that encodes a declared name.

    Parameters
    ----------
    name : str
        The declared name.

    Returns
    -------
    str
        A non-keyword Python identifier from which ``name`` is recoverable.

    Raises
    ------
    IdentifierError
        If ``name`` is empty.
    """
    if not name:
        msg = "an empty name has no identifier"
        raise IdentifierError(msg)

    encoded = "".join(
        character if _plain(character) else _escaped(character) for character in name
    )

    if encoded[0].isdigit() or keyword.iskeyword(encoded):
        encoded = _escaped(name[0]) + encoded[1:]

    return encoded


def _code_point(digits: str, identifier: str) -> str:
    """Return the character an escape sequence stands for.

    Parameters
    ----------
    digits : str
        The hexadecimal code-point digits.
    identifier : str
        The full identifier used in errors.
    """
    if not digits or any(digit not in _HEX_DIGITS for digit in digits):
        msg = f"{identifier!r} contains the escape {digits!r}, which is not hexadecimal"
        raise IdentifierError(msg)

    try:
        return chr(int(digits, 16))
    except (OverflowError, ValueError) as error:
        msg = f"{identifier!r} escapes {digits!r}, which is not a code point"
        raise IdentifierError(msg) from error


def decode_identifier(identifier: str) -> str:
    """Return the declared name an identifier encodes.

    Parameters
    ----------
    identifier : str
        An identifier produced by :func:`encode_identifier`.

    Returns
    -------
    str
        The declared name.

    Raises
    ------
    IdentifierError
        If the identifier is not the canonical encoding of any name: an
        unterminated or non-hexadecimal escape, an escape of a character
        that is written plainly, or a spelling with uppercase or leading
        zeros in an escape.
    """
    characters: list[str] = []
    at = 0

    while at < len(identifier):
        character = identifier[at]

        if character != "_":
            characters.append(character)
            at += 1

            continue

        close = identifier.find("_", at + 1)

        if close < 0:
            msg = f"{identifier!r} has an unterminated escape"
            raise IdentifierError(msg)

        characters.append(_code_point(identifier[at + 1 : close], identifier))
        at = close + 1

    name = "".join(characters)

    if not name or encode_identifier(name) != identifier:
        msg = f"{identifier!r} is not the canonical spelling of any declared name"
        raise IdentifierError(msg)

    return name
