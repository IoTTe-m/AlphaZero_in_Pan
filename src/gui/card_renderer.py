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
    def __init__(self, card_images_dir: str | Path, card_width: int = 80, card_height: int = 120):
        self.card_images_dir = Path(card_images_dir)
        self.card_width = card_width
        self.card_height = card_height
        self.card_images: dict[tuple[int, int], pygame.Surface] = {}
        self.card_back: pygame.Surface | None = None

    def load_images(self):
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
        return self.card_images.get((rank, suit))

    def get_card_back(self) -> pygame.Surface | None:
        return self.card_back

    @staticmethod
    def action_to_cards(action: int) -> list[tuple[int, int]]:
        """Decode action ID to list of (rank, suit) tuples."""
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
        """Convert a single card to its action ID."""
        return OFFSET_SINGLE_CARD + suit * NUM_RANKS + rank
