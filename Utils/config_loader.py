import yaml
import os

def load_config(config_path: str = None) -> dict:
    """Load configuration from a YAML file."""
    if config_path is None:
        # Resolve path relative to the project root, not the current working directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(project_root, "config", "config_loader.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        print(config)
    return config