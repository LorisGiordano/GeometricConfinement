""" IMPORTS """

import os
import glob
import shutil
import re
import json
import numpy as np
import random

from .loading import get_all_patient_files
from .utils import extract_dataset, zip_dataset
from .utils import CLASSIFICATION_DICT


""" UNSTRUCTURED -> DISEASE CLASSIFICATION """
"""
DatasetName/
* healthy/
  * patientID1_image.nii.gz
  * patientID1_label.nii.gz
  * patientID2_image.nrrd
  * patientID2_label.nrrd
  * ...
* marfan/
  * patientID3_image_marfan.nii.gz
  * patientID3_label_marfan.nii.gz
  * ...
* ad/
  * patientID4_image_ad.nrrd
  * patientID4_label_ad.nrrd
  * ...
* taad/
  * patientID5_image_taad.nii.gz
  * patientID5_label_taad.nii.gz
  * ...
* tbad/
  * patientID6_image_tbad.nrrd
  * patientID6_label_tbad.nrrd
  * ...
* aaa/
  * patientID7_image_aaa.nii.gz
  * patientID7_label_aaa.nii.gz
  * ...
* DatasetName.zip
"""

# Remove subfolders
def _flatten_folder(dataset_directory: str, classification_identification: dict) -> None:
    
    # make flatten folder
    flatten_directory = dataset_directory + "_flatten"
    os.makedirs(flatten_directory)
    
    # take all files out of subfolders
    for root, dirs, files in os.walk(dataset_directory):
        for file in files:
            
            # get relative filepath of image
            source_file_path = os.path.join(root, file)
            
            # classify the patients based on given classifications
            for classification_flag in list(classification_identification.keys()):
                
                # check the classification of the image
                if re.search(classification_flag, source_file_path):
                    classification = classification_identification[classification_flag]
                    
                    # make new relative filepath and rename file
                    filepath, basename = os.path.split(source_file_path)
                    filename = basename[:basename.find('.')]
                    extension = basename[basename.find('.'):]
                    new_source_filepath = filename + "_" + classification + extension
                    os.rename(source_file_path, new_source_filepath)
                    source_file_path = new_source_filepath
                    
            # copy each file to the destination folder
            shutil.move(source_file_path, flatten_directory)
            
    # delete original dataset folder
    shutil.rmtree(dataset_directory)
    os.rename(flatten_directory, dataset_directory)
    
# Rename files so that they all have the same structure: set new filename dependent on dataset
def _rename_files(dataset_directory: str, identifications: list[str] = ["_image","_label"], include_dataste_name: bool = False, report: bool = True) -> None:
    
    # get all files (NRRD or NifTi) in folder
    if report:
        print("Renaming files...")
    filepaths_nrrd = sorted(glob.glob(os.path.join(dataset_directory, '*.nrrd')))
    filepaths_nii = sorted(glob.glob(os.path.join(dataset_directory, '*.nii.gz')))
    filepaths = filepaths_nrrd + filepaths_nii

    # keep amount of images and segmentations
    amount_images = 0
    amount_segmentations = 0
    image_identification, segmentation_identification = identifications
    for filepath in filepaths:
        
        # get filename and extension
        dirname, basename = os.path.split(filepath)
        filename = basename[:basename.find('.')]
        extension = basename[basename.find('.'):]
        
        # include name of dataset
        if include_dataste_name:
            name_dataset = os.path.basename(dataset_directory).lower()
            filename = f"{filename}_{name_dataset}"

        # identify if it is a segmentation
        if re.search(segmentation_identification, filepath):
            
            # modify segmentation filename in order to have 'original identifier' + _label
            new_filepath = re.sub(segmentation_identification, '', filepath)
            dirname, basename = os.path.split(new_filepath)
            filename = basename[:basename.find('.')]
            extension = basename[basename.find('.'):]
            new_segmentation_filepath = os.path.join(dirname, filename + '_label' + extension)
            os.rename(filepath, new_segmentation_filepath)
            amount_segmentations += 1
            
        # identify if it is an image
        elif re.search(image_identification, filepath):
            
            # modify image filename in order to have 'original identifier' + _image
            new_filepath = re.sub(image_identification, '', filepath)
            dirname, basename = os.path.split(new_filepath)
            filename = basename[:basename.find('.')]
            extension = basename[basename.find('.'):]
            new_image_filepath = os.path.join(dirname, filename + '_image' + extension)
            os.rename(filepath, new_image_filepath)
            amount_images += 1
        
        # if image or segmentation could not be identified
        else:
            raise Exception("Incorrect image or segmentation identifier.")
    
    # check if all patients have image and segmentation
    if not amount_images == amount_segmentations:
        raise Exception(f"Diffrence in amount of images and segmentations in dataset: {amount_images} images, {amount_segmentations} segmentations")
        return
    if not len(filepaths) == amount_images+amount_segmentations:
        raise Exception(f"Patients lost: {len(filepaths)} in original dataset, {amount_images+amount_segmentations} in new dataset")
        return
    # report
    if report:
        print("Renaming files done.\n")

# Get classification of patient from filename
def _get_classification(image_filepath: str) -> str:
    
    # check all classification
    for classification in list(CLASSIFICATION_DICT.values()):
        
        # if classification in filename (classification added to filename in _flatten_folder())
        if re.search(classification, os.path.basename(image_filepath)):
            return classification
    
    # if no classification => assume healthy
    return CLASSIFICATION_DICT[0]

# Set files in folders dependent on classification
def _organize_files_per_classification(dataset_directory: str, report: bool = True) -> None:
    
    # get all images (NRRD or NifTi) in folder
    if report:
        print("Organizing files per disease classification...")
    # classification=['','*','*/*'] returns all files in database, subfolders and subsubfolders
    image_files, segmentation_files, _ = get_all_patient_files(dataset_directory, False)
    
    # make folder for each classification
    classification_list = list(CLASSIFICATION_DICT.values())
    for classification in classification_list:
        os.makedirs(os.path.join(dataset_directory, classification))
        
    # move image and segmentation to right folder
    for image_filepath, segmentation_filepath in zip(image_files, segmentation_files):
        classification = _get_classification(image_filepath)
        # move image
        image_filename = os.path.basename(image_filepath)
        new_image_filepath = os.path.join(dataset_directory, classification, image_filename)
        os.rename(image_filepath, new_image_filepath)
        # move segmentation
        segmentation_filename = os.path.basename(segmentation_filepath)
        new_segmentation_filepath = os.path.join(dataset_directory, classification, segmentation_filename)
        os.rename(segmentation_filepath, new_segmentation_filepath)
    
    # report
    if report:
        print("Organizing files per disease classifcation done.\n", end="\r")
        _, _, _ = get_all_patient_files(dataset_directory, True)

# Unstructured to disease classification
def unstructured_to_disease_classification(dataset_directory: str, classification_identification: dict, image_identification: list[str], include_dataste_name: bool = False, report: bool = True, zipping: bool = False, do_backup_check: bool = True) -> None:
    
    # backup check
    backup_check = 'y'
    if do_backup_check:
        backup_check = input("BACK-UP CHECK: Is there a back-up of the original dataset (y/[n])? ")
    if backup_check == 'y':
        
        # extract dataset if not done
        extract_dataset(dataset_directory, report)
        # flatten folder and rename files based on disease classification
        _flatten_folder(dataset_directory, classification_identification)
        # rename file to have distinction between image and label
        _rename_files(dataset_directory, image_identification, include_dataste_name, report)
        # organise per classification
        _organize_files_per_classification(dataset_directory, report)
        
    # zip dataset
    if zipping:
        zip_dataset(dataset_directory, report)


""" DISEASE CLASSIFICATION TO MSD """
"""
DatasetName
* ImageTr
  * patientID1.nii.gz
  * patientID2_marfan.nrrd
  * ...
* ImageTs
  * patientID3.nii.gz
  * ...
* LabelTr
  * patientID4.nii.gz
  * patientID5_ad.nrrd
"""

# Partition dataset in training (train + val) and testing
def _partition_with_classification(classifications: list[str], training_partition: int|float = 85):
    
    training_partition = training_partition/100
    amount_classifications = 6
    train_indices = list()
    test_indices = list()
    
    # for all classifications
    for classification in range(amount_classifications):
        
        # make list of partitioning indices
        indices = [i for i, cls in enumerate(classifications) if cls == classification]
        
        # check if this classification is present in dataset
        if len(indices) == 0: 
            continue
            
        # shuffle the indices and store
        random.shuffle(indices)
        split_index = int(round(training_partition*len(indices)))
        train_indices.extend(indices[:split_index])
        test_indices.extend(indices[split_index:])
        
    return [train_indices, test_indices]
    
# Partition dataset in training, testing and validation set with equal split of different classifications of diseases
def _train_test_split(image_files: list[str], segmentation_files: list[str], classifications: list[str], training_partition: int|float = 85, report: bool = False) -> tuple[list[str], list[str], list[str], list[str]]:
    
    # partition list of indices
    parts = _partition_with_classification(classifications, training_partition)

    # partition data and labels
    image_sets = [list(), list()]
    segmentation_sets = [list(), list()]
    classification_sets = [list(), list()]
    for i, part in enumerate(parts):
        image_sets[i] = [image_files[idx] for idx in part]
        segmentation_sets[i] = [segmentation_files[idx] for idx in part]
        classification_sets[i] = [classifications[idx] for idx in part]

    # organise in training, testing and validation dataset
    train_x, test_x = image_sets
    train_y, test_y = segmentation_sets
    if report:
        print(f"\nTraining count: {len(train_x)}, Testing count: {len(test_x)}")

    return train_x, test_x, train_y, test_y

# Reorganise from my standards to MSD standards
def _organize_files_msd(dataset_directory: str, training_partition: int|float = 85, report: bool = False) -> str:
    
    # get image and segmentation files
    image_files, segmentation_files, classifications = get_all_patient_files(dataset_directory, report)
    
    # partition images and segmentation according to calssifications
    train_x, test_x, train_y, test_y = _train_test_split(image_files, segmentation_files, classifications, training_partition, report)

    # setup new dataset
    msd_dataset_directory = dataset_directory + "_msd"
    # check if already exists
    if os.path.exists(msd_dataset_directory):
        if report:
            print(f"Dataset already reorganised: {msd_dataset_directory}\n")
            return msd_dataset_directory
        
    # if not exist, build directory
    if report:
        print("Reorganizing to MSD standard...", end="\r")
    os.makedirs(msd_dataset_directory)
    folder_list = ["InputsTr", "InputsTs", "LabelsTr"]
    for folder in folder_list:
        os.makedirs(os.path.join(msd_dataset_directory, folder))

    # copy images and segmentations for training
    train_image_folder = os.path.join(msd_dataset_directory, folder_list[0])
    train_label_folder = os.path.join(msd_dataset_directory, folder_list[2])
    for image_filepath, label_filepath in zip(train_x, train_y):
        
        # change patient identiefier so it does not contain '_image' or '_label'
        new_patient_identifier = os.path.basename(re.sub("_image", '', image_filepath))
        
        # segmentation and image filename are now identical
        new_image_filepath = os.path.join(train_image_folder, new_patient_identifier)
        new_label_filepath = os.path.join(train_label_folder, new_patient_identifier)
        
        # copy image and segmentation to different folders
        shutil.copyfile(image_filepath, new_image_filepath)
        shutil.copyfile(label_filepath, new_label_filepath)

    # copy images and segmentations for testing
    test_image_folder = os.path.join(msd_dataset_directory, folder_list[1])
    for image_filepath in test_x:
        
        # change patient identiefier so it does not contain '_image'
        new_patient_identifier = os.path.basename(re.sub("_image", '', image_filepath))
        new_image_filepath = os.path.join(test_image_folder, new_patient_identifier)
        
        # copy image and segmentation to different folders
        shutil.copyfile(image_filepath, new_image_filepath)
        
    # report
    if report:
        print("Reorganizing to MSD standard done.\n", end="\r")

    return msd_dataset_directory

# Disease classification to MSD
def disease_classification_to_msd(dataset_directory: str, training_partition: int|float = 85, report: bool = True, zipping: bool = False) -> str:
    
    # extract dataset if not done
    extract_dataset(dataset_directory, report)
    # organize files
    msd_dataset_directory = _organize_files_msd(dataset_directory, training_partition, report)
    
    # zip dataset
    if zipping:
        zip_dataset(dataset_directory, report)
        
    return msd_dataset_directory


""" DATALIST MSD """

# Find filepaths in subfolders of dataset directory, relative to dataset directory
def _relative_filepath_in_subdirectory(dataset_directory: str, sub_folder: str) -> list[str]:
    
    # get nii.gz files
    files_nii = glob.glob(os.path.join(dataset_directory, sub_folder, '*.nii.gz'))
    # get nrrd files
    files_nrrd = glob.glob(os.path.join(dataset_directory, sub_folder, '*.nrrd'))
    # get all files relative to dataset directory
    filenames = files_nii + files_nrrd
    
    # get relative filenames
    relative_filenames = [os.path.join('.', os.path.relpath(filename, dataset_directory)) for filename in filenames]
    
    return relative_filenames

# Get all filepaths in directory, relative to dataset directory
def _relative_filepaths_in_directory(dataset_directory: str, report: bool = False) -> tuple[list[str], list[str], list[str]]:
    
    # get training images and labels
    inputsTr = _relative_filepath_in_subdirectory(dataset_directory, "InputsTr")
    inputsTs = _relative_filepath_in_subdirectory(dataset_directory, "InputsTs")
    # get testing images
    labelsTr = _relative_filepath_in_subdirectory(dataset_directory, "LabelsTr")
    
    # check amount of data and if all segmentation presents
    amount_images_train = len(inputsTr)
    amount_labels_train = len(labelsTr)
    amount_images_test = len(inputsTs)
    if not (amount_images_train == amount_labels_train):
        print(f" Incomplete data in training folder: {amount_images_train} image files found for {amount_labels_train} segmentation files")
        return [], [], []
    
    # report
    if report:
        print("\nOverview of patients in dataset:")
        print(f" {amount_images_train} patients found for training, all segmentations present")
        print(f" {amount_images_test} patients found for testing\n")
        
    return inputsTr, inputsTs, labelsTr

# Create datalist for particular dataset
def create_msd_datalist(dataset_directory: str, num_folds: int = 0, overwrite: bool = False, report: bool = True) -> None:
    
    # get images and labels for training and testing
    inputsTr, inputsTs, labelsTr = _relative_filepaths_in_directory(dataset_directory, report)
    
    dataset_filename = "dataset.json"
    dataset_filepath = os.path.join(dataset_directory, dataset_filename)
    if os.path.exists(dataset_filepath) and not overwrite:
        print(f"Datalist already exists: {dataset_filepath}")
        return
    
    # check if datalist exists
    datalist_filename = "dataset_trainval.json"
    datalist_filepath = os.path.join(dataset_directory, datalist_filename)
    if os.path.exists(datalist_filepath) and not overwrite:
        print(f"Datalist already exists: {datalist_filepath}")
        return
    
    # make dataset json file
    training_input_paths = glob.glob(os.path.join(dataset_directory, "InputsTr", "*"))
    
    num_training_samples = len(training_input_paths)
    
    sample_path = training_input_paths[0]
    sample_extension = sample_path.split(".")[1:]
    sample_extension = ["." + ext for ext in sample_extension]
    sample_extension = "".join(sample_extension)
    
    dataset_json = {"channel_names": {"0": "CT"}, 
                    "labels": {"background": 0, "aorta": 1},
                    "numTraining": num_training_samples,
                    "file_ending": sample_extension}
    
    # save dataset as json
    with open(dataset_filepath, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, ensure_ascii=False, indent=4)
    print(f"\nDatalist is saved to {dataset_filepath}")

    # create and populate datalist
    datalist_json = {"testing": [], "training": []}
    datalist_json["testing"] = [{"image": image_filename} for image_filename in inputsTs]
    training = []
    for image_filepath, label_filepath in zip(inputsTr, labelsTr):
        dir_name, file_name = os.path.split(image_filepath)
        dir_name = dir_name[2:]
        file_name = file_name.split(".")[0]
        new_image_filepath = f"{dir_name}/{file_name}_0000{sample_extension}"
        new_image_filepath = os.path.join(dataset_directory, new_image_filepath)
        label_filepath = os.path.join(dataset_directory, label_filepath)
        # os.rename(absolute_image_filepath, new_absolute_image_filepath)
        training.append({"image": new_image_filepath, "label": label_filepath})
    
    # if num_folds == 0 => train-val split
    if num_folds > 0:
        datalist_filename = "dataset_crossval.json"
        datalist_filepath = os.path.join(dataset_directory, datalist_filename)
        # set fold number to each image
        fold_size = len(datalist_json["training"]) // num_folds
        for i in range(num_folds):
            for j in range(fold_size):
                datalist_json["training"][i * fold_size + j]["fold"] = i

    # save datalist as json
    with open(datalist_filepath, "w", encoding="utf-8") as f:
        json.dump(datalist_json, f, ensure_ascii=False, indent=4)
    print(f"\nDatalist is saved to {datalist_filepath}")
    
    
""" RUN """

if __name__ == '__main__':
    
    dataset_directory = ""
    classification_identification = {"": "", "": ""}
    image_identification = ["", ""]
    unstructured_to_disease_classification(dataset_directory, classification_identification, image_identification)
    
    training_partition = 85
    msd_dataset_direcotry = disease_classification_to_msd(dataset_directory, training_partition)
    
    create_msd_datalist(msd_dataset_direcotry)