import argparse
import os
import sys
import threading
import time
from pathlib import Path

import pygame

from src.game_logic import (
    ACTION_TAKE_CARDS,
    OFFSET_FOUR_CARDS,
    OFFSET_FOUR_NINES,
    OFFSET_THREE_NINES,
    RANKS,
    SUIT_SYMBOLS,
)
from src.gui.card_renderer import CardRenderer
from src.gui.config import PlayConfig
from src.gui.game_controller import GameController

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / 'configs' / 'play.yaml'

# Colors
WHITE = (255, 255, 255)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)
LIGHT_BLUE = (173, 216, 230)


class PanGameApp:
    def __init__(self, config: PlayConfig):
        self.config = config
        self.controller = GameController(config)
        self.card_renderer = CardRenderer(config.card_images_dir)

        pygame.init()
        self.screen = pygame.display.set_mode((config.window_width, config.window_height))
        pygame.display.set_caption('Pan - AlphaZero')
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        self.card_renderer.load_images()

        self.selected_cards: list[tuple[int, int]] = []
        self.card_rects: list[tuple[pygame.Rect, int, int]] = []  # (rect, rank, suit)
        self.take_button_rect: pygame.Rect | None = None
        self.restart_button_rect: pygame.Rect | None = None

        self.message = ''
        self.message_time = 0.0

        self.ai_delay = 0.5
        self.ai_thinking = False
        self.ai_action: int | None = None
        self.ai_player: int | None = None
        self._shutdown = False

    def _compute_ai_action_async(self, player: int):
        action = self.controller.get_ai_action()
        if self._shutdown:
            return
        self.ai_action = action
        self.ai_player = player
        self.ai_thinking = False

    def run(self):
        clock = pygame.time.Clock()
        running = True
        last_ai_move_time = 0.0

        while running:
            current_time = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos)

            # Check if AI finished computing
            if self.ai_action is not None:
                self._apply_ai_action()
                last_ai_move_time = current_time

            # Start AI turn if needed
            if (
                not self.controller.is_game_over()
                and not self.controller.is_human_turn()
                and not self.ai_thinking
                and self.ai_action is None
                and current_time - last_ai_move_time > self.ai_delay
            ):
                self._start_ai_turn()

            self._draw()
            pygame.display.flip()
            clock.tick(60)

        self._shutdown = True
        pygame.quit()
        os._exit(0)

    def _handle_click(self, pos: tuple[int, int]):
        # Check restart button
        if self.restart_button_rect and self.restart_button_rect.collidepoint(pos):
            self.controller.restart()
            self.selected_cards = []
            self.message = ''
            return

        if self.controller.is_game_over():
            return

        if not self.controller.is_human_turn():
            return

        # Check take cards button
        if self.take_button_rect and self.take_button_rect.collidepoint(pos):
            if ACTION_TAKE_CARDS in self.controller.get_human_actions():
                self.controller.execute_action(ACTION_TAKE_CARDS)
                self._set_message('You took 3 cards')
                self.selected_cards = []
            return

        # Check card clicks
        for rect, rank, suit in self.card_rects:
            if rect.collidepoint(pos):
                self._handle_card_click(rank, suit)
                return

    def _handle_card_click(self, rank: int, suit: int):
        legal_actions = self.controller.get_human_actions()

        # Check if this card can be part of a multi-card action
        multi_card_actions = self._get_multi_card_actions_for_card(rank, suit, legal_actions)

        if multi_card_actions:
            # Toggle selection
            card = (rank, suit)
            if card in self.selected_cards:
                self.selected_cards.remove(card)
            else:
                self.selected_cards.append(card)

            # Check if current selection forms a valid multi-card action
            action = self._try_match_selection_to_action(legal_actions)
            if action is not None:
                self.controller.execute_action(action)
                cards = CardRenderer.action_to_cards(action)
                card_names = ' '.join(f'{SUIT_SYMBOLS[s]}{RANKS[r]}' for r, s in cards)
                self._set_message(f'You played {card_names}')
                self.selected_cards = []
            return

        # No multi-card actions possible - try single card
        single_action = CardRenderer.card_to_single_action(rank, suit)
        if single_action in legal_actions:
            self.controller.execute_action(single_action)
            card_name = f'{SUIT_SYMBOLS[suit]}{RANKS[rank]}'
            self._set_message(f'You played {card_name}')
            self.selected_cards = []
            return

        self._set_message('Cannot play this card')

    def _get_multi_card_actions_for_card(self, rank: int, suit: int, legal_actions: list[int]) -> list[int]:
        """Get multi-card actions that include this card."""
        multi_actions = []

        # Check three nines (only for 9s, suits D/C/S - not H which starts)
        if rank == 0 and suit in [1, 2, 3]:  # 9 of D, C, or S
            for action in range(OFFSET_THREE_NINES, OFFSET_FOUR_NINES):
                if action in legal_actions:
                    multi_actions.append(action)

        # Check four nines (only for 9s)
        if rank == 0:
            for action in range(OFFSET_FOUR_NINES, OFFSET_FOUR_CARDS):
                if action in legal_actions:
                    multi_actions.append(action)

        # Check four of a kind (ranks 10, J, Q, K, A = indices 1-5)
        if rank >= 1:
            start = OFFSET_FOUR_CARDS + (rank - 1) * 4
            end = start + 4
            for action in range(start, end):
                if action in legal_actions:
                    multi_actions.append(action)

        return multi_actions

    def _try_match_selection_to_action(self, legal_actions: list[int]) -> int | None:
        """Check if current selection matches any multi-card action."""
        if len(self.selected_cards) < 3:
            return None

        selected_ranks = [r for r, s in self.selected_cards]
        selected_suits = [s for r, s in self.selected_cards]

        # All same rank?
        if len(set(selected_ranks)) != 1:
            return None

        rank = selected_ranks[0]

        # Three nines
        if rank == 0 and len(self.selected_cards) == 3:
            # Find which three-nines action matches (based on spade position)
            sorted_suits = sorted(selected_suits)
            if sorted_suits == [1, 2, 3]:  # D, C, S
                # Determine spade position in play order
                for action in range(OFFSET_THREE_NINES, OFFSET_FOUR_NINES):
                    if action in legal_actions:
                        return action

        # Four nines
        if rank == 0 and len(self.selected_cards) == 4:
            for action in range(OFFSET_FOUR_NINES, OFFSET_FOUR_CARDS):
                if action in legal_actions:
                    return action

        # Four of other ranks
        if rank >= 1 and len(self.selected_cards) == 4:
            start = OFFSET_FOUR_CARDS + (rank - 1) * 4
            end = start + 4
            for action in range(start, end):
                if action in legal_actions:
                    return action

        return None

    def _start_ai_turn(self):
        player = self.controller.get_current_player()
        if self.controller.is_player_done(player):
            return

        self.ai_thinking = True
        self._set_message(f'Player {player} is thinking...')
        thread = threading.Thread(target=self._compute_ai_action_async, args=(player,), daemon=True)
        thread.start()

    def _apply_ai_action(self):
        action = self.ai_action
        player = self.ai_player
        self.ai_action = None
        self.ai_player = None

        if action is None or player is None:
            return

        cards = CardRenderer.action_to_cards(action)
        self.controller.execute_action(action)

        if action == ACTION_TAKE_CARDS:
            self._set_message(f'Player {player} took cards')
        elif cards:
            card_names = ' '.join(f'{SUIT_SYMBOLS[s]}{RANKS[r]}' for r, s in cards)
            self._set_message(f'Player {player} played {card_names}')

    def _set_message(self, msg: str):
        self.message = msg
        self.message_time = time.time()

    def _draw(self):
        self.screen.fill(DARK_GREEN)
        self.card_rects = []

        self._draw_table()
        self._draw_ai_hands()
        self._draw_human_hand()
        self._draw_buttons()
        self._draw_status()
        self._draw_message()

    def _draw_table(self):
        # Draw table area
        table_rect = pygame.Rect(
            self.config.window_width // 2 - 200,
            self.config.window_height // 2 - 100,
            400,
            200,
        )
        pygame.draw.rect(self.screen, GREEN, table_rect, border_radius=20)
        pygame.draw.rect(self.screen, WHITE, table_rect, 3, border_radius=20)

        # Draw cards on table (show last few)
        table_cards = self.controller.get_table_cards()
        visible_cards = table_cards[-6:] if len(table_cards) > 6 else table_cards

        start_x = table_rect.centerx - (len(visible_cards) * 30) // 2
        for i, (rank, suit) in enumerate(visible_cards):
            card_img = self.card_renderer.get_card_image(rank, suit)
            if card_img:
                x = start_x + i * 30
                y = table_rect.centery - self.card_renderer.card_height // 2
                self.screen.blit(card_img, (x, y))

        # Show card count
        count_text = self.small_font.render(f'Cards on table: {len(table_cards)}', True, WHITE)
        self.screen.blit(count_text, (table_rect.centerx - count_text.get_width() // 2, table_rect.bottom + 10))

    def _draw_ai_hands(self):
        # Player positions: 0=bottom (human), 1=left, 2=top, 3=right
        positions = {
            1: (50, self.config.window_height // 2, 'vertical'),
            2: (self.config.window_width // 2, 50, 'horizontal'),
            3: (self.config.window_width - 50, self.config.window_height // 2, 'vertical'),
        }

        for player in range(self.config.player_count):
            if player == self.config.human_player:
                continue

            hand = self.controller.get_player_hand(player)
            is_done = self.controller.is_player_done(player)
            is_current = self.controller.get_current_player() == player

            if player in positions:
                x, y, orientation = positions[player]
                self._draw_opponent_hand(x, y, len(hand), player, orientation, is_done, is_current)

    def _draw_opponent_hand(
        self,
        x: int,
        y: int,
        card_count: int,
        player: int,
        orientation: str,
        is_done: bool,
        is_current: bool,
    ):
        card_back = self.card_renderer.get_card_back()
        small_width = 50
        small_height = 75
        card_back_small = None

        if card_back:
            card_back_small = pygame.transform.scale(card_back, (small_width, small_height))

        # Draw player label
        status = ' ✓' if is_done else ''
        color = YELLOW if is_current else WHITE
        label = self.font.render(f'P{player}{status}', True, color)

        if orientation == 'horizontal':
            label_x = x - label.get_width() // 2
            label_y = y - 30
            self.screen.blit(label, (label_x, label_y))

            # Draw cards horizontally
            total_width = min(card_count * 25, 200)
            start_x = x - total_width // 2
            for i in range(card_count):
                if card_back_small:
                    self.screen.blit(card_back_small, (start_x + i * 25, y))
        else:
            # Vertical layout for left/right players
            if x < self.config.window_width // 2:  # Left player
                label_x = x
                card_x = x
            else:  # Right player
                label_x = x - label.get_width()
                card_x = x - small_width

            label_y = y - small_height - 40
            self.screen.blit(label, (label_x, label_y))

            total_height = min(card_count * 20, 150)
            start_y = y - total_height // 2
            for i in range(card_count):
                if card_back_small:
                    self.screen.blit(card_back_small, (card_x, start_y + i * 20))

    def _draw_human_hand(self):
        hand = self.controller.get_player_hand(self.config.human_player)
        is_current = self.controller.get_current_player() == self.config.human_player
        is_done = self.controller.is_player_done(self.config.human_player)

        # Draw label
        status = ' ✓' if is_done else ''
        color = YELLOW if is_current else WHITE
        label = self.font.render(f'Your Hand{status}', True, color)
        self.screen.blit(label, (self.config.window_width // 2 - label.get_width() // 2, self.config.window_height - 200))

        # Draw cards
        card_spacing = 90
        total_width = len(hand) * card_spacing
        start_x = self.config.window_width // 2 - total_width // 2

        legal_actions = self.controller.get_human_actions() if is_current else []

        for i, (rank, suit) in enumerate(hand):
            x = start_x + i * card_spacing
            y = self.config.window_height - 160

            card_img = self.card_renderer.get_card_image(rank, suit)
            if card_img:
                # Check if card is selected
                is_selected = (rank, suit) in self.selected_cards

                # Highlight playable cards
                single_action = CardRenderer.card_to_single_action(rank, suit)
                is_playable = single_action in legal_actions
                can_be_multi = len(self._get_multi_card_actions_for_card(rank, suit, legal_actions)) > 0

                if is_selected:
                    # Yellow highlight for selected cards
                    highlight_rect = pygame.Rect(x - 4, y - 4, self.card_renderer.card_width + 8, self.card_renderer.card_height + 8)
                    pygame.draw.rect(self.screen, YELLOW, highlight_rect, border_radius=5)
                elif is_playable or can_be_multi:
                    highlight_rect = pygame.Rect(x - 3, y - 3, self.card_renderer.card_width + 6, self.card_renderer.card_height + 6)
                    pygame.draw.rect(self.screen, LIGHT_BLUE, highlight_rect, border_radius=5)

                self.screen.blit(card_img, (x, y))

                rect = pygame.Rect(x, y, self.card_renderer.card_width, self.card_renderer.card_height)
                self.card_rects.append((rect, rank, suit))

    def _draw_buttons(self):
        # Take cards button
        is_human_turn = self.controller.is_human_turn()
        can_take = ACTION_TAKE_CARDS in self.controller.get_human_actions() if is_human_turn else False

        button_color = LIGHT_BLUE if can_take else GRAY
        self.take_button_rect = pygame.Rect(self.config.window_width - 200, self.config.window_height - 100, 150, 50)
        pygame.draw.rect(self.screen, button_color, self.take_button_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, self.take_button_rect, 2, border_radius=10)

        text = self.font.render('Take Cards', True, BLACK if can_take else WHITE)
        text_rect = text.get_rect(center=self.take_button_rect.center)
        self.screen.blit(text, text_rect)

        # Restart button
        self.restart_button_rect = pygame.Rect(50, self.config.window_height - 100, 120, 50)
        pygame.draw.rect(self.screen, RED, self.restart_button_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, self.restart_button_rect, 2, border_radius=10)

        text = self.font.render('Restart', True, WHITE)
        text_rect = text.get_rect(center=self.restart_button_rect.center)
        self.screen.blit(text, text_rect)

    def _draw_status(self):
        if self.controller.is_game_over():
            loser = self.controller.get_loser()
            if loser == self.config.human_player:
                msg = 'YOU LOST! 😢'
                color = RED
            else:
                msg = f'Player {loser} lost! YOU WIN! 🎉'
                color = YELLOW

            text = self.font.render(msg, True, color)
            text_rect = text.get_rect(center=(self.config.window_width // 2, 20))
            self.screen.blit(text, text_rect)
        else:
            current = self.controller.get_current_player()
            if current == self.config.human_player:
                msg = 'Your turn'
                color = YELLOW
            else:
                msg = f"Player {current}'s turn"
                color = WHITE

            text = self.font.render(msg, True, color)
            self.screen.blit(text, (10, 10))

    def _draw_message(self):
        if self.message and time.time() - self.message_time < 2.0:
            text = self.small_font.render(self.message, True, WHITE)
            text_rect = text.get_rect(center=(self.config.window_width // 2, self.config.window_height // 2 + 130))
            self.screen.blit(text, text_rect)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Play Pan against AlphaZero AI')
    parser.add_argument(
        '-c',
        '--config',
        type=Path,
        default=DEFAULT_CONFIG,
        help=f'Path to YAML config file (default: {DEFAULT_CONFIG})',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config.exists():
        config = PlayConfig.from_yaml(args.config)
    else:
        config = PlayConfig()
    app = PanGameApp(config)
    app.run()


if __name__ == '__main__':
    main()
