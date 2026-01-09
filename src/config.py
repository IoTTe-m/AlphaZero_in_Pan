"""Configuration classes for AlphaZero training."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class WandbConfig(BaseModel):
    """Weights & Biases logging configuration.

    Attributes:
        entity: W&B entity (team or username).
        project: W&B project name.
    """

    entity: str = 'reinforced-annealer'
    project: str = 'pan-alpha-zero'


class TrainingConfig(BaseModel):
    """Configuration for AlphaZero self-play training.

    Attributes:
        learning_rate: Optimizer learning rate.
        weight_decay: AdamW weight decay coefficient.
        batch_size: Number of samples per training batch.
        batch_count: Number of batches per training step.
        epochs: Total number of training epochs.
        games_per_training: Self-play games before each training step.
        num_simulations: MCTS simulations per move.
        num_worlds: Parallel MCTS worlds for variance reduction.
        c_puct_value: PUCT exploration constant.
        policy_temp: Temperature for action probability scaling.
        max_buffer_size: Maximum replay buffer capacity.
        initial_max_game_length: Starting max moves per game (curriculum).
        capped_max_game_length: Maximum allowed game length.
        game_length_increment: How much to increase max length per epoch.
        player_count: Number of players in the game.
        save_dir: Directory for saving checkpoints.
        wandb: Weights & Biases configuration.
    """

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    batch_size: int = 32
    batch_count: int = 8
    epochs: int = 100

    games_per_training: int = 1
    num_simulations: int = 32
    num_worlds: int = 4

    c_puct_value: float = 1.0
    policy_temp: float = 0.2

    max_buffer_size: int = 1024

    initial_max_game_length: int = 30
    capped_max_game_length: int = 500
    game_length_increment: int = 10

    player_count: int = 4

    save_dir: str = 'checkpoints/'

    wandb: WandbConfig = WandbConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'TrainingConfig':
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            TrainingConfig instance with values from the file.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary, excluding wandb settings.

        Returns:
            Dictionary of configuration values.
        """
        return self.model_dump(exclude={'wandb'})
