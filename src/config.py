from pathlib import Path

import yaml
from pydantic import BaseModel


class WandbConfig(BaseModel):
    entity: str = 'reinforced-annealer'
    project: str = 'pan-alpha-zero'


class TrainingConfig(BaseModel):
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
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        return self.model_dump(exclude={'wandb'})
