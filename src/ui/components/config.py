import json

config = None


def get_config(file="ui_config.json"):
    """
    Load a single config instance for UI consistency.
    """
    global config
    if config is None:
        with open(file, "r") as f:
            config = json.load(f)
    return config
