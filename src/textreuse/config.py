import yaml
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class Config:
    chunk_size: int = 400
    chunk_overlap: int = 200
    min_score: int = 60
    context_size: int = 400
    cache_dir: str = ".cache"
    
    @staticmethod
    def load(path: str = None) -> 'Config':
        if path and Path(path).exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return Config(**data)
        return Config()

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f)
