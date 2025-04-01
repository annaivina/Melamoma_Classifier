import yaml 
import os


def load_config(config_path=""):
    if not config_path:
        raise ValueError("Config is not provided, please provide configuration file.")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' doesnt exist")
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    