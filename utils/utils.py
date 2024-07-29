import yaml

def load_config(config_path):
    """Load model config files"""

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config

def rm_duplicates(input_list):
    """Removes duplicated elements from a list while preserving the order."""
    seen = set()
    seen_add = seen.add
    return [x for x in input_list if not (x in seen or seen_add(x))]