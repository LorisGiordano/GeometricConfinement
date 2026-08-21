""" IMPORTS """

import argparse
import os
import shutil
import json


""" CONFIGURATION FILE """

# Extract config
def get_config(config_file: str, data_dir: str, models_dir: str) -> dict:
    
    # transform text file to json if forgotten => not losing queue
    json_config_file = config_file + '.json'
    text_config_file = config_file + '.txt'
    if not os.path.exists(json_config_file):
        os.rename(text_config_file, json_config_file)
    
    # get configuration as dict
    with open(json_config_file, 'r') as config:
        config_data = json.load(config)
    
    # handle checkpoints
    if not ("continue_training" in list(config_data.keys())):
        config_data["continue_training"] = -1
      
    # complete directories
    config_data["dataset_directory"] = os.path.join(data_dir, config_data.get('dataset_directory'))
    config_data["model_directory"] = os.path.join(models_dir, config_data.get('model_directory'))
    if ("detection_model_path" in list(config_data.get('model').keys())):
        config_data["model"]["detection_model_path"] = os.path.join(models_dir, config_data.get('model').get('detection_model_path'))
    
    return config_data


def save_config(config_file: str, model_run_directory: str):
    
    # transform text file to json if forgotten => not losing queue
    json_config_file = config_file + '.json'
    text_config_file = config_file + '.txt'
    if not os.path.exists(json_config_file):
        os.rename(text_config_file, json_config_file)
    
    # copy config file
    shutil.copy(json_config_file, f"{model_run_directory}/config.json")


""" DETECTION PARSERS """

# Detection - time.py
def det_time_parser(default_config: str = "") -> argparse.Namespace:

    default_root_dir = os.environ.get("ROOT_DIR", "NoDefaultRootDir")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default=default_config, help='Path to the configuration file')

    parser.add_argument('--dataset-root', '-d', type=str, default=os.path.join(default_root_dir, "Data"))
    parser.add_argument('--model-root', '-m', type=str, default=os.path.join(default_root_dir, "Models"))
    
    parser = parser.parse_args()

    if parser.dataset_root == os.path.join("NoDefaultRootDir", "Data"):
        raise Exception("No default root directory found. Please set the ROOT_DIR environment variable or set the dataset_root argument (--dataset_root).")
    if parser.model_root == os.path.join("NoDefaultRootDir", "Models"):
        raise Exception("No default root directory found. Please set the ROOT_DIR environment variable or set the model_root argument (--model_root).")

    parser.config_name = parser.config[7:]
    parser.config = get_config(parser.config, parser.dataset_root, parser.model_root)
    
    parser.dataset_directory = parser.config.get('dataset_directory')
    parser.model_directory = parser.config.get('model_directory')
    
    return parser

# Detection - train
def det_train_parser(default_config: str = "") -> argparse.Namespace:

    default_root_dir = os.environ.get("ROOT_DIR", "NoDefaultRootDir")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default=default_config, help='Path to the configuration file')

    parser.add_argument('--dataset-root', '-d', type=str, default=os.path.join(default_root_dir, "Data"))
    parser.add_argument('--model-root', '-m', type=str, default=os.path.join(default_root_dir, "Models"))
    
    parser = parser.parse_args()
    
    if parser.dataset_root == os.path.join("NoDefaultRootDir", "Data"):
        raise Exception("No default root directory found. Please set the ROOT_DIR environment variable or set the dataset_root argument (--dataset_root).")
    if parser.model_root == os.path.join("NoDefaultRootDir", "Models"):
        raise Exception("No default root directory found. Please set the ROOT_DIR environment variable or set the model_root argument (--model_root).")
    
    parser.config_name = parser.config[7:]
    parser.config_filepath = parser.config
    parser.config = get_config(parser.config, parser.dataset_root, parser.model_root)
    
    parser.short_description = parser.config.get('short_description')
    parser.dataset_directory = parser.config.get('dataset_directory')
    parser.model_directory = parser.config.get('model_directory')
    parser.continue_training = parser.config.get('continue_training')
    
    return parser

# Detection - evaluate.py
def det_evaluate_parser() -> argparse.Namespace:

    default_root_dir = os.environ.get("ROOT_DIR", "NoDefaultRootDir")
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-root', type=str, default=os.path.join(default_root_dir, "Models"))
    parser.add_argument('--task', type=str)
    parser.add_argument('--type', type=str)
    parser.add_argument('--id', type=int)
    
    parser = parser.parse_args()

    if parser.model_root == os.path.join("NoDefaultRootDir", "Models"):
        raise Exception("No default root directory found. Please set the ROOT_DIR environment variable or set the model_root argument (--model_root).")
    
    parser.model_directory = os.path.join(parser.model_root, parser.task, parser.type)
    parser.model_id_directory = os.path.join(parser.model_directory, f"model_{parser.id}")

    return parser