"""Selection state for ordered multi-card actions in the GUI."""

from dataclasses import dataclass, field

Card = tuple[int, int]


@dataclass
class CardSelection:
    """Tracks ordered card selections for multi-card actions.

    Attributes:
        _cards: Ordered list of selected cards as (rank, suit).
    """

    _cards: list[Card] = field(default_factory=list)

    def toggle(self, card: Card) -> None:
        """Add or remove a card from the ordered selection.

        Args:
            card: Card to toggle as (rank, suit).
        """
        if card in self._cards:
            self._cards.remove(card)
        else:
            self._cards.append(card)

    def clear(self) -> None:
        """Clear the current selection."""
        self._cards.clear()

    def index_of(self, card: Card) -> int | None:
        """Get the 1-based selection index for a card.

        Args:
            card: Card to look up.

        Returns:
            1-based index if selected, otherwise None.
        """
        if card not in self._cards:
            return None
        return self._cards.index(card) + 1

    def ordered(self) -> list[Card]:
        """Return the ordered selection.

        Returns:
            Ordered list of selected cards.
        """
        return list(self._cards)
