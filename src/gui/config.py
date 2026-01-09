"""Configuration for the Pan game GUI application."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class PlayConfig(BaseModel):
    """Configuration settings for playing Pan against the AI.

    Attributes:
        checkpoint_path: Path to the trained model checkpoint.
        num_simulations: MCTS simulations per move.
        num_worlds: Parallel MCTS worlds for variance reduction.
        c_puct_value: PUCT exploration constant.
        policy_temp: Temperature for action selection (0 = greedy).
        player_count: Number of players in the game.
        human_player: Player index controlled by human.
        card_images_dir: Directory containing card image assets.
        window_width: Game window width in pixels.
        window_height: Game window height in pixels.
    """

    checkpoint_path: str = 'checkpoints/run_spring-snowflake-86/22'

    num_simulations: int = 64
    num_worlds: int = 4
    c_puct_value: float = 1.0
    policy_temp: float = 0.0  # Greedy play for AI

    player_count: int = 4
    human_player: int = 0

    card_images_dir: str = 'card_images/'

    window_width: int = 1200
    window_height: int = 800

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'PlayConfig':
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            PlayConfig instance with values from the file.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
