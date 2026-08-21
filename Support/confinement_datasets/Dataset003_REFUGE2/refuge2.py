import argparse

from PIL import Image
import json
import os
import shutil
from scipy.ndimage import center_of_mass
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    try:
        img = Image.open(path)
        img.load()  # Ensure the image is loaded
        return img
    except IOError as e:
        print(f"Error loading image from {path}: {e}")
        return None
    
def save_landmark(img, path):
    try:
        with open(path, 'w') as f:
            json.dump(img, f, indent=4)
    except IOError as e:
        print(f"Error saving landmark to {path}: {e}")

def extract_center(img):
    np_image = np.array(img)
    inner = np_image == 0
    outer = np_image == 128
    com = center_of_mass(inner)

    com = {"ij_position": [int(com[1]), int(com[0])]}

    return com

def process_folder(input_path: str, output_path: str):
    segmentation_path = os.path.join(output_path, "segmentation")
    landmark_path = os.path.join(output_path, "landmark")

    os.makedirs(segmentation_path, exist_ok=True)
    os.makedirs(landmark_path, exist_ok=True)

    print(f"Filling database {input_path}...")

    N = len(os.listdir(input_path))

    for i, filepath in enumerate(os.listdir(input_path)):
        filepath = os.path.join(input_path, filepath)

        if os.path.isdir(filepath):
            N -= 1
            continue
        if not os.path.isfile(filepath):
            continue
        if filepath.startswith("."):
            continue

        split_file = os.path.basename(filepath).split('.')
        ext = split_file[-1]
        id = split_file[0]

        print(f" Processing file: {id} ({i+1}/{N})", end="\r")

        img = load_image(filepath)
        com = extract_center(img)
        
        save_landmark(com, os.path.join(landmark_path, f"{id}_landmark.json"))
        shutil.copy(filepath, os.path.join(segmentation_path, f"{id}_label.{ext}"))

    print("\nFilling complete.")

def reorganize_database(input_path: str, output_path: str):

    os.makedirs(output_path, exist_ok=True)

    print(f"Reorganizing database {input_path}...")

    N = len(os.listdir(input_path))

    for i, filepath in enumerate(os.listdir(input_path)):
        filepath = os.path.join(input_path, filepath)

        if os.path.isdir(filepath):
            N -= 1
            continue
        if not os.path.isfile(filepath):
            continue
        if filepath.startswith("."):
            continue

        split_file = os.path.basename(filepath).split('.')
        ext = split_file[-1]
        id = split_file[0]

        print(f" Processing file: {id} ({i+1}/{N})", end="\r")

        shutil.copy(filepath, os.path.join(output_path, f"{id}_image.{ext}"))

    print("\nReorganization done.")

def unwrap_landmarks(input_path: str):
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('_landmark.json'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    landmark = json.load(f)
                landmark = landmark["ij_position"]
                with open(filepath, 'w') as f:
                    json.dump(landmark, f, indent=4)



def main():
    
    parser = argparse.ArgumentParser(
        description="Convert REFUGE2 cases to NIfTI and landmark dictionaries."
    )
    parser.add_argument("input", type=str, help="Directory to REFUGE2 dataset.")
    args = parser.parse_args()
    
    input_path = args.input
    output_path = os.path.join(os.path.dirname(input_path), "structured")

    print(f"Reorganizing {input_path}...")

    process_folder(os.path.join(input_path, "val", "mask"), os.path.join(output_path, "LabelsVa"))
    reorganize_database(os.path.join(input_path, "val", "images"), os.path.join(output_path, "ImagesVa"))

    process_folder(os.path.join(input_path, "train", "mask"), os.path.join(output_path, "LabelsTr"))
    reorganize_database(os.path.join(input_path, "train", "images"), os.path.join(output_path, "ImagesTr"))

    process_folder(os.path.join(input_path, "test", "mask"), os.path.join(output_path, "LabelsTs"))
    reorganize_database(os.path.join(input_path, "test", "images"), os.path.join(output_path, "ImagesTs"))

    unwrap_landmarks(output_path)


if __name__ == "__main__":
    main()