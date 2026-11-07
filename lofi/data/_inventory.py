"""Symbol inventories with reserved prefixes."""

from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_name

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = ["DuplicateSymbolError", "Inventory", "UnknownSymbolError"]


class DuplicateSymbolError(ValueError):
    """An error for a duplicate inventory symbol.

    Parameters
    ----------
    inventory : str
        The inventory's name.
    symbol : str
        The symbol declared twice.
    """

    def __init__(self, inventory: str, symbol: str) -> None:
        super().__init__(
            f"the inventory {inventory} declares {symbol!r} twice, so the "
            "symbol would have two codes and a lookup would find one of them"
        )

        #: The inventory's name.
        self.inventory = inventory

        #: The symbol declared twice.
        self.symbol = symbol


class UnknownSymbolError(ValueError):
    """An error for an unknown inventory symbol.

    Parameters
    ----------
    inventory : str
        The inventory's name.
    symbol : str
        The symbol asked for.
    """

    def __init__(self, inventory: str, symbol: str) -> None:
        super().__init__(
            f"the inventory {inventory} does not declare {symbol!r}, and an "
            "inventory that grows while a corpus is read gives codes that "
            "depend on the order the corpus arrived in"
        )

        #: The inventory's name.
        self.inventory = inventory

        #: The symbol asked for.
        self.symbol = symbol


@final
class Inventory:
    """A stable integer coding for declared symbols.

    Parameters
    ----------
    name : str
        Inventory name used in messages.
    ordinary : Sequence[str]
        Ordinary symbols in declaration order.
    reserved : Sequence[str]
        Reserved symbols forming the index prefix.

    Raises
    ------
    ValueError
        If the name or a symbol is empty.
    DuplicateSymbolError
        If a symbol is declared twice, whether within one half or across both.
    """

    __slots__ = ("_by_symbol", "_name", "_ordinary", "_reserved")

    def __init__(
        self,
        name: str,
        ordinary: Sequence[str],
        reserved: Sequence[str] = (),
    ) -> None:
        self._name = ensure_name("an inventory's name", name)
        self._reserved = tuple(
            ensure_name("a reserved symbol", symbol) for symbol in reserved
        )
        self._ordinary = tuple(
            ensure_name("an ordinary symbol", symbol) for symbol in ordinary
        )
        codes: dict[str, int] = {}

        for code, symbol in enumerate((*self._reserved, *self._ordinary)):
            if symbol in codes:
                raise DuplicateSymbolError(self._name, symbol)

            codes[symbol] = code

        self._by_symbol: Mapping[str, int] = codes

    @property
    def name(self) -> str:
        """Return the inventory's name."""
        return self._name

    @property
    def reserved(self) -> tuple[str, ...]:
        """Return the symbols whose analysis is stipulated, in code order."""
        return self._reserved

    @property
    def ordinary(self) -> tuple[str, ...]:
        """Return the symbols whose analysis is learned, in code order."""
        return self._ordinary

    @property
    def size(self) -> int:
        """Return how many codes the inventory declares, reserved and ordinary."""
        return len(self._by_symbol)

    def __len__(self) -> int:
        """Return how many codes the inventory declares."""
        return len(self._by_symbol)

    def __contains__(self, symbol: str) -> bool:
        """Return whether a symbol is one the inventory declares."""
        return symbol in self._by_symbol

    def index(self, symbol: str) -> int:
        """Return the code a symbol is declared under.

        Parameters
        ----------
        symbol : str
            The symbol to look up.

        Returns
        -------
        int
            The symbol's integer code.

        Raises
        ------
        UnknownSymbolError
            If the inventory does not declare the symbol.
        """
        code = self._by_symbol.get(symbol)

        if code is None:
            raise UnknownSymbolError(self._name, symbol)

        return code

    def symbol(self, index: int) -> str:
        """Return the symbol a code stands for.

        Parameters
        ----------
        index : int
            The code to read.

        Returns
        -------
        str
            The symbol.

        Raises
        ------
        IndexError
            If the code is outside the inventory.
        """
        if not 0 <= index < len(self._by_symbol):
            msg = (
                f"the inventory {self._name} declares {len(self._by_symbol)} "
                f"codes and was asked for {index}"
            )
            raise IndexError(msg)

        if index < len(self._reserved):
            return self._reserved[index]

        return self._ordinary[index - len(self._reserved)]

    def is_reserved(self, symbol: str) -> bool:
        """Return whether a symbol is reserved.

        Parameters
        ----------
        symbol : str
            The symbol to place.

        Returns
        -------
        bool
            Whether the symbol is in the reserved prefix.

        Raises
        ------
        UnknownSymbolError
            If the inventory does not declare the symbol.
        """
        return self.index(symbol) < len(self._reserved)

    def extended(self, symbols: Iterable[str]) -> Inventory:
        """Return the inventory grown by more ordinary symbols.

        Parameters
        ----------
        symbols : Iterable[str]
            Ordinary symbols to append in code order.

        Returns
        -------
        Inventory
            The extended inventory.

        Raises
        ------
        ValueError
            If one of the symbols is empty.
        DuplicateSymbolError
            If the symbols name one of their own twice.
        """
        added: list[str] = []

        for symbol in symbols:
            if symbol in self._by_symbol:
                continue

            if symbol in added:
                raise DuplicateSymbolError(self._name, symbol)

            added.append(symbol)

        return Inventory(self._name, (*self._ordinary, *added), self._reserved)

    def _identity(self) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        """Return the three fields an inventory is equal on."""
        return (self._name, self._reserved, self._ordinary)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same inventory."""
        if type(other) is not Inventory:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the name and the two halves."""
        return hash((Inventory, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this inventory."""
        return f"Inventory({self._name!r}, {self._ordinary!r}, {self._reserved!r})"
