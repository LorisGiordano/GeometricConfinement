""" IMPORTS """

import os
import shutil
import json
from datetime import datetime

from .utils import CLASSIFICATION_DICT
from .utils import zip_dataset, extract_dataset


""" MERGE DATASETS """

# Get metadata of dataset
def _get_metadata(dataset_directory: str) -> tuple[list[int], list[float]]:
    
    # get metadata in dataset path
    metadata_filepath = os.path.join(dataset_directory, f"{os.path.basename(dataset_directory)}_metadata.json")
    with open(metadata_filepath, 'r') as metadata_file:
        metadata_data = json.load(metadata_file)
    
    # get size and spacing
    size = metadata_data["config"]["target_size"]
    spacing = metadata_data["config"]["target_spacing"]
    return size, spacing

# Merge datasets together
def merge_datasets(dataset_directories: list[str], merged_dataset_directory: str) -> None:
    
    # check if size and spacing of all datasets are identical
    extract_dataset(dataset_directories[0])
    size_1, spacing_1 = _get_metadata(dataset_directories[0])
    for dataset_directory in dataset_directories:
        # get spacing of dataset
        extract_dataset(dataset_directory)
        size_i, spacing_i = _get_metadata(dataset_directory)
        # check spacing and size
        if size_i != size_1 or spacing_1 != spacing_i:
            raise Exception("Cannot merge datasets with different sizes or spacings.")
    
    # setup output directory
    if not os.path.isdir(merged_dataset_directory):
        os.makedirs(merged_dataset_directory)
        for classification in list(CLASSIFICATION_DICT.values()):
            os.makedirs(os.path.join(merged_dataset_directory, classification))
            
    # go over datasets
    for dataset_directory in dataset_directories:
        # merge datasets based on classification
        for classification in list(CLASSIFICATION_DICT.values()):
            classification_folder = os.path.join(dataset_directory, classification)
            if not os.path.isdir(classification_folder):
                continue
            for filename in os.listdir(classification_folder):
                # check extension
                if filename.endswith(('_image.nii.gz', '_label.nii.gz', '_bbox.json', '_lm.json')):
                    # copy file to merged dataset
                    shutil.copy(os.path.join(dataset_directory, classification, filename), os.path.join(merged_dataset_directory, classification, filename))
                    
    
    # get metadata of first dataset as base for merged dataset
    merged_metadata_filepath = os.path.join(dataset_directories[0], f"{os.path.basename(dataset_directories[0])}_metadata.json")
    with open(merged_metadata_filepath, 'r') as merged_metadata_file:
        merged_metadata_data = json.load(merged_metadata_file)
    
    # change date of creation
    created_on = datetime.now().strftime("%d/%m/%Y at %H:%M:%S")
    merged_metadata_data["created_on"] = created_on
    # add all configs and special adaptations of the datasets
    special_adaptations = [merged_metadata_data["special_adaptations"]]
    merged_metadata_data["config1"] = merged_metadata_data.pop("config")
    for i in range(1, len(dataset_directories)):
        metadata_file_i = os.path.join(dataset_directories[i], f"{os.path.basename(dataset_directories[i])}_metadata.json")
        with open(metadata_file_i, 'r') as metadata_i:
            metadata_data_i = json.load(metadata_i)
        special_adaptations.append(metadata_data_i["special_adaptations"])
        merged_metadata_data[f"config{i+1}"] = metadata_data_i["config"]
    merged_metadata_data["special_adaptations"] = special_adaptations
    
    # write merged dataset metadata
    merged_metadata_filepath = os.path.join(merged_dataset_directory, f"{os.path.basename(merged_dataset_directory)}_metadata.json")
    with open(merged_metadata_filepath, 'w') as merged_metadata_file:
        merged_metadata_file.write(json.dumps(merged_metadata_data))
        
    for dataset_directory in dataset_directories:
        zip_dataset(dataset_directory)
        
        
""" RUN """

if __name__ == '__main__':
    
    dataset_directories = ["", 
                           ""]
    merged_dataset_directory = ""
    merge_datasets(dataset_directories, merged_dataset_directory)