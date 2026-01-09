"""Card rendering and action-card conversion utilities for the GUI."""

from pathlib import Path

import pygame

from src.game_logic import (
    ACTION_TAKE_CARDS,
    COUNT_FOUR_CARDS,
    NUM_RANKS,
    NUM_SUITS,
    OFFSET_FOUR_CARDS,
    OFFSET_FOUR_NINES,
    OFFSET_SINGLE_CARD,
    OFFSET_THREE_NINES,
    SUITS,
)


class CardRenderer:
    """Handles loading and rendering card images for the game GUI.

    Attributes:
        card_images_dir: Directory containing card image files.
        card_width: Width of rendered cards in pixels.
        card_height: Height of rendered cards in pixels.
        card_images: Mapping from (rank, suit) to loaded pygame surfaces.
        card_back: Surface for the card back image.
    """

    def __init__(self, card_images_dir: str | Path, card_width: int = 80, card_height: int = 120):
        """Initialize the card renderer.

        Args:
            card_images_dir: Directory containing card image files.
            card_width: Width of rendered cards in pixels.
            card_height: Height of rendered cards in pixels.
        """
        self.card_images_dir = Path(card_images_dir)
        self.card_width = card_width
        self.card_height = card_height
        self.card_images: dict[tuple[int, int], pygame.Surface] = {}
        self.card_back: pygame.Surface | None = None

    def load_images(self) -> None:
        """Load all card images from the card images directory."""
        for rank in range(NUM_RANKS):
            for suit_idx, suit in enumerate(SUITS):
                image_path = self.card_images_dir / f'{rank}{suit}.png'
                if image_path.exists():
                    img = pygame.image.load(str(image_path))
                    img = pygame.transform.scale(img, (self.card_width, self.card_height))
                    self.card_images[(rank, suit_idx)] = img

        back_path = self.card_images_dir / 'back.png'
        if back_path.exists():
            img = pygame.image.load(str(back_path))
            self.card_back = pygame.transform.scale(img, (self.card_width, self.card_height))

    def get_card_image(self, rank: int, suit: int) -> pygame.Surface | None:
        """Get the image surface for a specific card.

        Args:
            rank: Card rank index (0=9, 1=10, ..., 5=A).
            suit: Card suit index (0=H, 1=D, 2=C, 3=S).

        Returns:
            Pygame surface for the card, or None if not loaded.
        """
        return self.card_images.get((rank, suit))

    def get_card_back(self) -> pygame.Surface | None:
        """Get the card back image surface.

        Returns:
            Pygame surface for card back, or None if not loaded.
        """
        return self.card_back

    @staticmethod
    def action_to_cards(action: int) -> list[tuple[int, int]]:
        """Decode an action ID to a list of (rank, suit) tuples.

        Args:
            action: Action ID from the game logic.

        Returns:
            List of (rank, suit) tuples representing the cards played.
        """
        if OFFSET_SINGLE_CARD <= action < OFFSET_THREE_NINES:
            rank = (action - OFFSET_SINGLE_CARD) % NUM_RANKS
            suit = (action - OFFSET_SINGLE_CARD) // NUM_RANKS
            return [(rank, suit)]

        elif OFFSET_THREE_NINES <= action < OFFSET_FOUR_NINES:
            spade_index = action - OFFSET_THREE_NINES
            card_order = [1, 2]  # D, C indices
            card_order.insert(spade_index, 3)  # S index
            return [(0, suit) for suit in card_order]

        elif OFFSET_FOUR_NINES <= action < OFFSET_FOUR_CARDS:
            spade_index = action - OFFSET_FOUR_NINES + 1
            card_order = [0, 1, 2]  # H, D, C indices
            card_order.insert(spade_index, 3)  # S index
            return [(0, suit) for suit in card_order]

        elif OFFSET_FOUR_CARDS <= action < OFFSET_FOUR_CARDS + COUNT_FOUR_CARDS:
            spade_index = (action - OFFSET_FOUR_CARDS) % NUM_SUITS
            rank = (action - OFFSET_FOUR_CARDS) // NUM_SUITS + 1
            card_order = [0, 1, 2]  # H, D, C indices
            card_order.insert(spade_index, 3)  # S index
            return [(rank, suit) for suit in card_order]

        elif action == ACTION_TAKE_CARDS:
            return []

        return []

    @staticmethod
    def card_to_single_action(rank: int, suit: int) -> int:
        """Convert a single card to its action ID.

        Args:
            rank: Card rank index.
            suit: Card suit index.

        Returns:
            Action ID for playing this single card.
        """
        return OFFSET_SINGLE_CARD + suit * NUM_RANKS + rank
