import json

config = None


def get_config(file: str = "ui_config.json"):
    """
    Load a single config instance for UI consistency.

    Args:
        file: Name of the config file to load, default is ui_config.json

    Returns:
        The dictionary instance of the config file.
    """
    global config
    if config is None:
        with open(file, "r") as f:
            config = json.load(f)
    return config
