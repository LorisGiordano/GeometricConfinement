import os
import argparse
import json

def get_config(config_file: str, dataset_root: str) -> dict:
    
    config_file += '.json'
    
    # transform text file to json if forgotten => not losing queue
    if not os.path.exists(config_file):
        os.rename(f"{os.path.splitext(config_file)[0]}.txt", config_file)
    
    # get configuration as dict
    with open(config_file, 'r') as config:
        config_data = json.load(config)
        
    # always rename config file to txt to apply changes easily
    # os.rename(config_file, f"{os.path.splitext(config_file)[0]}.txt")

    config_data["root"] = dataset_root
    
    return config_data


""" DATASET PARSER """

# Dataset - run.py
def dataset_parser() -> argparse.Namespace:

    default_root_dir = os.environ.get("ROOT_DIR", "NoDefaultRootDir")
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Path to the configuration file')

    parser.add_argument('--dataset-root', '-d', type=str, default=os.path.join(default_root_dir, "Data"))
    
    parser = parser.parse_args()

    if parser.dataset_root == os.path.join("NoDefaultRootDir", "Data"):
        raise Exception("No default root directory found. Please set the ROOT_DIR environment variable or set the dataset_root argument (--dataset_root).")
    
    parser.config = get_config(parser.config, parser.dataset_root)
    parser.dataset_directory = os.path.join(parser.dataset_root, parser.config.get('original_dataset', ''))
    
    return parser