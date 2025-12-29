from pathlib import Path

import yaml
from pydantic import BaseModel


class PlayConfig(BaseModel):
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
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
