import yaml
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Singleton config instance
_config = None

def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config