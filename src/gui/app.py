"""Main GUI application for playing Pan against AlphaZero AI."""

import argparse
import os
import threading
import time
from pathlib import Path

import pygame

from src.game_logic import ACTION_TAKE_CARDS, RANKS, SUIT_SYMBOLS
from src.gui.card_renderer import CardRenderer
from src.gui.config import PlayConfig
from src.gui.game_controller import GameController
from src.gui.selection import CardSelection

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / 'configs' / 'play.yaml'

WHITE = (255, 255, 255)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)
LIGHT_BLUE = (173, 216, 230)

FPS = 60
AI_DELAY_SECONDS = 0.5
MAIN_FONT_SIZE = 36
SMALL_FONT_SIZE = 24
MESSAGE_DURATION_SECONDS = 2.0
MESSAGE_OFFSET_Y = 130
UI_MARGIN = 10
STATUS_TEXT_Y = 20
OPPONENT_POSITION_MARGIN = 50

TABLE_WIDTH = 400
TABLE_HEIGHT = 200
TABLE_CARD_SPACING = 30
TABLE_MAX_VISIBLE_CARDS = 6
TABLE_BORDER_WIDTH = 3
TABLE_CORNER_RADIUS = 20
TABLE_COUNT_OFFSET_Y = 10

HUMAN_HAND_LABEL_OFFSET = 200
HUMAN_HAND_Y_OFFSET = 160
HAND_CARD_SPACING = 90

OPPONENT_CARD_BACK_WIDTH = 50
OPPONENT_CARD_BACK_HEIGHT = 75
OPPONENT_HAND_CARD_SPACING_HORIZONTAL = 25
OPPONENT_HAND_CARD_SPACING_VERTICAL = 20
OPPONENT_HAND_MAX_WIDTH = 200
OPPONENT_HAND_MAX_HEIGHT = 150
OPPONENT_LABEL_OFFSET = 30
OPPONENT_LABEL_PADDING = 40

TAKE_BUTTON_WIDTH = 150
TAKE_BUTTON_HEIGHT = 50
TAKE_BUTTON_X_OFFSET = 200
RESTART_BUTTON_WIDTH = 120
RESTART_BUTTON_HEIGHT = 50
RESTART_BUTTON_X = 50
BUTTON_Y_OFFSET = 100
BUTTON_CORNER_RADIUS = 10
BUTTON_BORDER_WIDTH = 2

HIGHLIGHT_SELECTED_PADDING = 4
HIGHLIGHT_PLAYABLE_PADDING = 3
HIGHLIGHT_CORNER_RADIUS = 5
SELECTION_MARKER_RADIUS = 12


class PanGameApp:
    """Pygame-based GUI application for playing Pan against AlphaZero AI.

    Handles rendering the game board, processing user input, and coordinating
    with the AI through asynchronous MCTS computation.

    Attributes:
        _config: Game and display configuration.
        _controller: Game logic and AI controller.
        _card_renderer: Handles card image loading and rendering.
        _screen: Pygame display surface.
        _font: Main font for text rendering.
        _small_font: Smaller font for messages.
        _selection: Ordered selection for multi-card actions.
        _card_rects: Clickable regions for player's cards.
        _take_button_rect: Clickable region for take cards button.
        _restart_button_rect: Clickable region for restart button.
        _message: Current status message to display.
        _message_time: Timestamp when message was set.
        _ai_delay: Delay before AI moves in seconds.
        _ai_thinking: Whether AI is currently computing.
        _ai_action: Computed AI action waiting to be applied.
        _ai_player: Player index for pending AI action.
    """

    def __init__(self, config: PlayConfig):
        """Initialize the game application.

        Args:
            config: Configuration for the game and display.
        """
        self._config = config
        self._controller = GameController(config)
        self._card_renderer = CardRenderer(config.card_images_dir)

        pygame.init()
        self._screen = pygame.display.set_mode((config.window_width, config.window_height))
        pygame.display.set_caption('Pan - AlphaZero')
        self._font = pygame.font.Font(None, MAIN_FONT_SIZE)
        self._small_font = pygame.font.Font(None, SMALL_FONT_SIZE)

        self._card_renderer.load_images()

        self._selection = CardSelection()
        self._card_rects: list[tuple[pygame.Rect, int, int]] = []
        self._take_button_rect: pygame.Rect | None = None
        self._restart_button_rect: pygame.Rect | None = None

        self._message = ''
        self._message_time = 0.0

        self._ai_delay = AI_DELAY_SECONDS
        self._ai_thinking = False
        self._ai_action: int | None = None
        self._ai_player: int | None = None
        self._shutdown = False

    def _compute_ai_action_async(self, player: int) -> None:
        """Compute AI action in a background thread.

        Args:
            player: Player index for which to compute the action.
        """
        action = self._controller.get_ai_action()
        if self._shutdown:
            return
        self._ai_action = action
        self._ai_player = player
        self._ai_thinking = False

    def run(self) -> None:
        """Run the main game loop until the window is closed."""
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

            if self._ai_action is not None:
                self._apply_ai_action()
                last_ai_move_time = current_time

            if (
                not self._controller.is_game_over()
                and not self._controller.is_human_turn()
                and not self._ai_thinking
                and self._ai_action is None
                and current_time - last_ai_move_time > self._ai_delay
            ):
                self._start_ai_turn()

            self._draw()
            pygame.display.flip()
            clock.tick(FPS)

        self._shutdown = True
        pygame.quit()
        os._exit(0)

    def _handle_click(self, pos: tuple[int, int]) -> None:
        """Handle mouse click events.

        Args:
            pos: Mouse position (x, y) of the click.
        """
        if self._restart_button_rect and self._restart_button_rect.collidepoint(pos):
            self._controller.restart()
            self._selection.clear()
            self._message = ''
            return

        if self._controller.is_game_over():
            return

        if not self._controller.is_human_turn():
            return

        if self._take_button_rect and self._take_button_rect.collidepoint(pos):
            if ACTION_TAKE_CARDS in self._controller.get_human_actions():
                self._controller.execute_action(ACTION_TAKE_CARDS)
                self._set_message('You took 3 cards')
                self._selection.clear()
            return

        for rect, rank, suit in self._card_rects:
            if rect.collidepoint(pos):
                self._handle_card_click(rank, suit)
                return

    def _handle_card_click(self, rank: int, suit: int) -> None:
        """Handle a click on a card in the player's hand.

        Args:
            rank: Card rank index.
            suit: Card suit index.
        """
        legal_actions = self._controller.get_human_actions()
        card = (rank, suit)

        if self._is_multi_selectable(card, legal_actions):
            self._selection.toggle(card)
            if self._try_play_selection(legal_actions):
                return
            return

        if self._try_play_single(card, legal_actions):
            return

        self._set_message('Cannot play this card')

    def _multi_action_cards(self, legal_actions: list[int]) -> set[tuple[int, int]]:
        """Collect cards that are part of any multi-card legal action.

        Args:
            legal_actions: List of currently legal action IDs.

        Returns:
            Set of cards (rank, suit) that appear in multi-card actions.
        """
        cards = set()
        for action in legal_actions:
            action_cards = CardRenderer.action_to_cards(action)
            if len(action_cards) > 1:
                cards.update(action_cards)
        return cards

    def _is_multi_selectable(self, card: tuple[int, int], legal_actions: list[int]) -> bool:
        """Check if a card can be part of a multi-card selection.

        Args:
            card: Card to check as (rank, suit).
            legal_actions: List of currently legal action IDs.

        Returns:
            True if the card appears in any multi-card legal action.
        """
        return card in self._multi_action_cards(legal_actions)

    def _try_play_selection(self, legal_actions: list[int]) -> bool:
        """Try to play the ordered selection as a multi-card action.

        Args:
            legal_actions: List of currently legal action IDs.

        Returns:
            True if a matching action was executed.
        """
        ordered_cards = self._selection.ordered()
        if len(ordered_cards) < 3:
            return False

        action = CardRenderer.action_for_card_sequence(ordered_cards, legal_actions)
        if action is None:
            return False

        self._controller.execute_action(action)
        card_names = ' '.join(f'{SUIT_SYMBOLS[s]}{RANKS[r]}' for r, s in ordered_cards)
        self._set_message(f'You played {card_names}')
        self._selection.clear()
        return True

    def _try_play_single(self, card: tuple[int, int], legal_actions: list[int]) -> bool:
        """Try to play a single card action.

        Args:
            card: Card to play as (rank, suit).
            legal_actions: List of currently legal action IDs.

        Returns:
            True if the single-card action was executed.
        """
        rank, suit = card
        single_action = CardRenderer.card_to_single_action(rank, suit)
        if single_action not in legal_actions:
            return False

        self._controller.execute_action(single_action)
        card_name = f'{SUIT_SYMBOLS[suit]}{RANKS[rank]}'
        self._set_message(f'You played {card_name}')
        self._selection.clear()
        return True

    def _start_ai_turn(self) -> None:
        """Start computing AI action in a background thread."""
        player = self._controller.get_current_player()
        if self._controller.is_player_done(player):
            return

        self._ai_thinking = True
        self._set_message(f'Player {player} is thinking...')
        thread = threading.Thread(target=self._compute_ai_action_async, args=(player,), daemon=True)
        thread.start()

    def _apply_ai_action(self) -> None:
        """Apply the computed AI action to the game state."""
        action = self._ai_action
        player = self._ai_player
        self._ai_action = None
        self._ai_player = None

        if action is None or player is None:
            return

        cards = CardRenderer.action_to_cards(action)
        self._controller.execute_action(action)

        if action == ACTION_TAKE_CARDS:
            self._set_message(f'Player {player} took cards')
        elif cards:
            card_names = ' '.join(f'{SUIT_SYMBOLS[s]}{RANKS[r]}' for r, s in cards)
            self._set_message(f'Player {player} played {card_names}')

    def _set_message(self, msg: str) -> None:
        """Set the status message to display temporarily.

        Args:
            msg: Message text to display.
        """
        self._message = msg
        self._message_time = time.time()

    def _draw(self) -> None:
        """Render the entire game screen."""
        self._screen.fill(DARK_GREEN)
        self._card_rects = []

        self._draw_table()
        self._draw_ai_hands()
        self._draw_human_hand()
        self._draw_buttons()
        self._draw_status()
        self._draw_message()

    def _draw_table(self) -> None:
        """Draw the central table area with cards on it."""
        table_rect = self._draw_table_surface()
        table_cards = self._controller.get_table_cards()
        visible_cards = self._get_visible_table_cards(table_cards)
        self._draw_table_cards(table_rect, visible_cards)
        self._draw_table_count(table_rect, len(table_cards))

    def _draw_table_surface(self) -> pygame.Rect:
        """Draw the table background and return its rectangle.

        Returns:
            Rectangle of the table area.
        """
        table_rect = pygame.Rect(
            self._config.window_width // 2 - TABLE_WIDTH // 2,
            self._config.window_height // 2 - TABLE_HEIGHT // 2,
            TABLE_WIDTH,
            TABLE_HEIGHT,
        )
        pygame.draw.rect(self._screen, GREEN, table_rect, border_radius=TABLE_CORNER_RADIUS)
        pygame.draw.rect(self._screen, WHITE, table_rect, TABLE_BORDER_WIDTH, border_radius=TABLE_CORNER_RADIUS)
        return table_rect

    def _get_visible_table_cards(self, table_cards: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Select the subset of table cards to display.

        Args:
            table_cards: Full list of cards on the table.

        Returns:
            Cards to display.
        """
        if len(table_cards) > TABLE_MAX_VISIBLE_CARDS:
            return table_cards[-TABLE_MAX_VISIBLE_CARDS:]
        return table_cards

    def _draw_table_cards(self, table_rect: pygame.Rect, cards: list[tuple[int, int]]) -> None:
        """Draw cards on the table.

        Args:
            table_rect: Rectangle of the table area.
            cards: Cards to display.
        """
        start_x = table_rect.centerx - (len(cards) * TABLE_CARD_SPACING) // 2
        for i, (rank, suit) in enumerate(cards):
            card_img = self._card_renderer.get_card_image(rank, suit)
            if card_img:
                x = start_x + i * TABLE_CARD_SPACING
                y = table_rect.centery - self._card_renderer.card_height // 2
                self._screen.blit(card_img, (x, y))

    def _draw_table_count(self, table_rect: pygame.Rect, count: int) -> None:
        """Draw the table card count below the table.

        Args:
            table_rect: Rectangle of the table area.
            count: Number of cards on the table.
        """
        count_text = self._small_font.render(f'Cards on table: {count}', True, WHITE)
        self._screen.blit(
            count_text,
            (table_rect.centerx - count_text.get_width() // 2, table_rect.bottom + TABLE_COUNT_OFFSET_Y),
        )

    def _draw_ai_hands(self) -> None:
        """Draw the AI players' hands as face-down cards."""
        positions = {
            1: (OPPONENT_POSITION_MARGIN, self._config.window_height // 2, 'vertical'),
            2: (self._config.window_width // 2, OPPONENT_POSITION_MARGIN, 'horizontal'),
            3: (self._config.window_width - OPPONENT_POSITION_MARGIN, self._config.window_height // 2, 'vertical'),
        }

        for player in range(self._config.player_count):
            if player == self._config.human_player:
                continue

            hand = self._controller.get_player_hand(player)
            is_done = self._controller.is_player_done(player)
            is_current = self._controller.get_current_player() == player

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
    ) -> None:
        """Draw an opponent's hand at the specified position.

        Args:
            x: X coordinate for the hand position.
            y: Y coordinate for the hand position.
            card_count: Number of cards in the hand.
            player: Player index.
            orientation: 'horizontal' or 'vertical' layout.
            is_done: Whether the player has finished.
            is_current: Whether it's this player's turn.
        """
        card_back_small = self._get_opponent_card_back()
        label = self._render_player_label(player, is_done, is_current)
        self._draw_opponent_label(x, y, orientation, label)
        if orientation == 'horizontal':
            self._draw_opponent_cards_horizontal(x, y, card_count, card_back_small)
        else:
            self._draw_opponent_cards_vertical(x, y, card_count, card_back_small, label)

    def _get_opponent_card_back(self) -> pygame.Surface | None:
        """Get the scaled opponent card back surface.

        Returns:
            Scaled card back surface if available.
        """
        card_back = self._card_renderer.get_card_back()
        if not card_back:
            return None
        return pygame.transform.scale(card_back, (OPPONENT_CARD_BACK_WIDTH, OPPONENT_CARD_BACK_HEIGHT))

    def _render_player_label(self, player: int, is_done: bool, is_current: bool) -> pygame.Surface:
        """Render a player label surface.

        Args:
            player: Player index.
            is_done: Whether the player has finished.
            is_current: Whether it's this player's turn.

        Returns:
            Rendered label surface.
        """
        status = ' (done)' if is_done else ''
        color = YELLOW if is_current else WHITE
        return self._font.render(f'P{player}{status}', True, color)

    def _draw_opponent_label(self, x: int, y: int, orientation: str, label: pygame.Surface) -> None:
        """Draw the opponent label at the correct position.

        Args:
            x: X coordinate for the hand position.
            y: Y coordinate for the hand position.
            orientation: 'horizontal' or 'vertical' layout.
            label: Rendered label surface.
        """
        if orientation == 'horizontal':
            label_x = x - label.get_width() // 2
            label_y = y - OPPONENT_LABEL_OFFSET
        else:
            label_x = x if x < self._config.window_width // 2 else x - label.get_width()
            label_y = y - OPPONENT_CARD_BACK_HEIGHT - OPPONENT_LABEL_PADDING
        self._screen.blit(label, (label_x, label_y))

    def _draw_opponent_cards_horizontal(
        self,
        x: int,
        y: int,
        card_count: int,
        card_back_small: pygame.Surface | None,
    ) -> None:
        """Draw a horizontal opponent hand.

        Args:
            x: X coordinate for the hand position.
            y: Y coordinate for the hand position.
            card_count: Number of cards in the hand.
            card_back_small: Scaled card back surface.
        """
        total_width = min(card_count * OPPONENT_HAND_CARD_SPACING_HORIZONTAL, OPPONENT_HAND_MAX_WIDTH)
        start_x = x - total_width // 2
        for i in range(card_count):
            if card_back_small:
                self._screen.blit(card_back_small, (start_x + i * OPPONENT_HAND_CARD_SPACING_HORIZONTAL, y))

    def _draw_opponent_cards_vertical(
        self,
        x: int,
        y: int,
        card_count: int,
        card_back_small: pygame.Surface | None,
        label: pygame.Surface,
    ) -> None:
        """Draw a vertical opponent hand.

        Args:
            x: X coordinate for the hand position.
            y: Y coordinate for the hand position.
            card_count: Number of cards in the hand.
            card_back_small: Scaled card back surface.
            label: Rendered label surface.
        """
        if x < self._config.window_width // 2:
            card_x = x
        else:
            card_x = x - OPPONENT_CARD_BACK_WIDTH

        total_height = min(card_count * OPPONENT_HAND_CARD_SPACING_VERTICAL, OPPONENT_HAND_MAX_HEIGHT)
        start_y = y - total_height // 2
        for i in range(card_count):
            if card_back_small:
                self._screen.blit(card_back_small, (card_x, start_y + i * OPPONENT_HAND_CARD_SPACING_VERTICAL))

    def _draw_human_hand(self) -> None:
        """Draw the human player's hand with playable card highlighting."""
        hand = self._controller.get_player_hand(self._config.human_player)
        is_current = self._controller.get_current_player() == self._config.human_player
        is_done = self._controller.is_player_done(self._config.human_player)
        self._draw_human_label(is_done, is_current)
        legal_actions = self._controller.get_human_actions() if is_current else []
        multi_action_cards = self._multi_action_cards(legal_actions)
        self._draw_human_cards(hand, legal_actions, multi_action_cards)

    def _draw_human_label(self, is_done: bool, is_current: bool) -> None:
        """Draw the label for the human player's hand.

        Args:
            is_done: Whether the human player has finished.
            is_current: Whether it's the human player's turn.
        """
        status = ' (done)' if is_done else ''
        color = YELLOW if is_current else WHITE
        label = self._font.render(f'Your Hand{status}', True, color)
        self._screen.blit(
            label,
            (
                self._config.window_width // 2 - label.get_width() // 2,
                self._config.window_height - HUMAN_HAND_LABEL_OFFSET,
            ),
        )

    def _draw_human_cards(
        self,
        hand: list[tuple[int, int]],
        legal_actions: list[int],
        multi_action_cards: set[tuple[int, int]],
    ) -> None:
        """Draw the human player's cards with selection and playability highlights.

        Args:
            hand: Cards in the human player's hand.
            legal_actions: List of legal action IDs.
            multi_action_cards: Cards that are part of multi-card actions.
        """
        card_spacing = HAND_CARD_SPACING
        total_width = len(hand) * card_spacing
        start_x = self._config.window_width // 2 - total_width // 2
        y = self._config.window_height - HUMAN_HAND_Y_OFFSET
        for i, (rank, suit) in enumerate(hand):
            x = start_x + i * card_spacing
            self._draw_human_card(x, y, rank, suit, legal_actions, multi_action_cards)

    def _draw_human_card(
        self,
        x: int,
        y: int,
        rank: int,
        suit: int,
        legal_actions: list[int],
        multi_action_cards: set[tuple[int, int]],
    ) -> None:
        """Draw a single card in the human player's hand.

        Args:
            x: X coordinate for the card.
            y: Y coordinate for the card.
            rank: Card rank index.
            suit: Card suit index.
            legal_actions: List of legal action IDs.
            multi_action_cards: Cards that are part of multi-card actions.
        """
        card_img = self._card_renderer.get_card_image(rank, suit)
        if not card_img:
            return

        card = (rank, suit)
        selection_index = self._selection.index_of(card)
        single_action = CardRenderer.card_to_single_action(rank, suit)
        is_playable = single_action in legal_actions
        can_be_multi = card in multi_action_cards

        if selection_index is not None:
            highlight_rect = pygame.Rect(
                x - HIGHLIGHT_SELECTED_PADDING,
                y - HIGHLIGHT_SELECTED_PADDING,
                self._card_renderer.card_width + HIGHLIGHT_SELECTED_PADDING * 2,
                self._card_renderer.card_height + HIGHLIGHT_SELECTED_PADDING * 2,
            )
            pygame.draw.rect(self._screen, YELLOW, highlight_rect, border_radius=HIGHLIGHT_CORNER_RADIUS)
            self._draw_selection_index(x, y, selection_index)
        elif is_playable or can_be_multi:
            highlight_rect = pygame.Rect(
                x - HIGHLIGHT_PLAYABLE_PADDING,
                y - HIGHLIGHT_PLAYABLE_PADDING,
                self._card_renderer.card_width + HIGHLIGHT_PLAYABLE_PADDING * 2,
                self._card_renderer.card_height + HIGHLIGHT_PLAYABLE_PADDING * 2,
            )
            pygame.draw.rect(self._screen, LIGHT_BLUE, highlight_rect, border_radius=HIGHLIGHT_CORNER_RADIUS)

        self._screen.blit(card_img, (x, y))

        rect = pygame.Rect(x, y, self._card_renderer.card_width, self._card_renderer.card_height)
        self._card_rects.append((rect, rank, suit))

    def _draw_selection_index(self, x: int, y: int, index: int) -> None:
        """Draw a small selection index marker on a card.

        Args:
            x: X coordinate of the card.
            y: Y coordinate of the card.
            index: 1-based selection index.
        """
        radius = SELECTION_MARKER_RADIUS
        center = (x + radius, y + radius)
        pygame.draw.circle(self._screen, WHITE, center, radius)
        text = self._small_font.render(str(index), True, BLACK)
        text_rect = text.get_rect(center=center)
        self._screen.blit(text, text_rect)

    def _draw_buttons(self) -> None:
        """Draw the Take Cards and Restart buttons."""
        is_human_turn = self._controller.is_human_turn()
        can_take = ACTION_TAKE_CARDS in self._controller.get_human_actions() if is_human_turn else False

        button_color = LIGHT_BLUE if can_take else GRAY
        self._take_button_rect = pygame.Rect(
            self._config.window_width - TAKE_BUTTON_X_OFFSET,
            self._config.window_height - BUTTON_Y_OFFSET,
            TAKE_BUTTON_WIDTH,
            TAKE_BUTTON_HEIGHT,
        )
        pygame.draw.rect(self._screen, button_color, self._take_button_rect, border_radius=BUTTON_CORNER_RADIUS)
        pygame.draw.rect(self._screen, WHITE, self._take_button_rect, BUTTON_BORDER_WIDTH, border_radius=BUTTON_CORNER_RADIUS)

        text = self._font.render('Take Cards', True, BLACK if can_take else WHITE)
        text_rect = text.get_rect(center=self._take_button_rect.center)
        self._screen.blit(text, text_rect)

        self._restart_button_rect = pygame.Rect(
            RESTART_BUTTON_X,
            self._config.window_height - BUTTON_Y_OFFSET,
            RESTART_BUTTON_WIDTH,
            RESTART_BUTTON_HEIGHT,
        )
        pygame.draw.rect(self._screen, RED, self._restart_button_rect, border_radius=BUTTON_CORNER_RADIUS)
        pygame.draw.rect(self._screen, WHITE, self._restart_button_rect, BUTTON_BORDER_WIDTH, border_radius=BUTTON_CORNER_RADIUS)

        text = self._font.render('Restart', True, WHITE)
        text_rect = text.get_rect(center=self._restart_button_rect.center)
        self._screen.blit(text, text_rect)

    def _draw_status(self) -> None:
        """Draw the game status (current turn or game over message)."""
        if self._controller.is_game_over():
            loser = self._controller.get_loser()
            if loser == self._config.human_player:
                msg = 'YOU LOST!'
                color = RED
            else:
                msg = f'Player {loser} lost! YOU WIN!'
                color = YELLOW

            text = self._font.render(msg, True, color)
            text_rect = text.get_rect(center=(self._config.window_width // 2, STATUS_TEXT_Y))
            self._screen.blit(text, text_rect)
        else:
            current = self._controller.get_current_player()
            if current == self._config.human_player:
                msg = 'Your turn'
                color = YELLOW
            else:
                msg = f"Player {current}'s turn"
                color = WHITE

            text = self._font.render(msg, True, color)
            self._screen.blit(text, (UI_MARGIN, UI_MARGIN))

    def _draw_message(self) -> None:
        """Draw the temporary status message if still visible."""
        if self._message and time.time() - self._message_time < MESSAGE_DURATION_SECONDS:
            text = self._small_font.render(self._message, True, WHITE)
            text_rect = text.get_rect(center=(self._config.window_width // 2, self._config.window_height // 2 + MESSAGE_OFFSET_Y))
            self._screen.blit(text, text_rect)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with config path.
    """
    parser = argparse.ArgumentParser(description='Play Pan against AlphaZero AI')
    parser.add_argument(
        '-c',
        '--config',
        type=Path,
        default=DEFAULT_CONFIG,
        help=f'Path to YAML config file (default: {DEFAULT_CONFIG})',
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the Pan game GUI application."""
    args = parse_args()
    if args.config.exists():
        config = PlayConfig.from_yaml(args.config)
    else:
        config = PlayConfig()
    app = PanGameApp(config)
    app.run()


if __name__ == '__main__':
    main()
