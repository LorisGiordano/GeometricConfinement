""" IMPORTS """

import numpy as np
import shutil
import os
import random
import re
import json


""" PARTITIONING """

def _partition_with_classification(indices_list: list[int], classifications: list[str], partition: list[float, float, float] = [70, 15, 15]) -> list[list[int], list[int], list[int]]:
    
    # check correct partition
    if sum(partition) == 1:
        partition = [100*p for p in partition]
    if sum(partition) != 100:
        raise Exception("Partition values do not add up to 100") 
    
    # split data in train and val (if cross-validation, partition is set to [p, 0, 1-p])
    train_indices = list()
    val_indices = list()
    test_indices = list()
    for classification in np.unique(classifications):
        # get indices
        indices = [i for i, cls in enumerate(classifications) if cls == classification]
        if len(indices) == 0: 
            continue
        # shuffle the indices
        random.shuffle(indices)
        # split indices
        val_cutoff = partition[0]/100
        test_cutoff = (partition[0] + partition[1])/100
        val_split_index = int(round(val_cutoff*len(indices)))
        test_split_index = int(round(test_cutoff*len(indices)))
        train_indices.extend(indices[:val_split_index])
        val_indices.extend(indices[val_split_index:test_split_index])
        test_indices.extend(indices[test_split_index:])
        
    return [train_indices, val_indices, test_indices]
    
# Partition dataset in training, testing and validation set with equal split of different classifications of diseases
def train_val_test_split(image_files: list[str], segmentation_files: list[str], landmark_files: list[str], classifications: list[str], partition: list[float, float, float] = [70, 15, 15], report: bool = True) -> tuple[list[list[str], list[str], list[str]], list[list[str], list[str], list[str]], list[list[str], list[str], list[str]]]:
    
    # partition list of indices
    indices_list = np.arange(len(image_files))
    parts = _partition_with_classification(indices_list, classifications, partition)

    # partition data and labels
    image_sets = [list(), list(), list()]
    label_sets = [list(), list(), list()]
    classification_sets = [list(), list(), list()]
    for i, part in enumerate(parts):
        image_sets[i] = [image_files[idx] for idx in part]
        label_sets[i] = [{"lm": landmark_files[idx], "seg": segmentation_files[idx]} for idx in part]
        classification_sets[i] = [classifications[idx] for idx in part]

    # organise in training, testing and validation dataset
    train_x, val_x, test_x = image_sets
    train_y, val_y, test_y = label_sets
    train_clas, val_clas, test_clas = classification_sets
    if report:
        print(f"\Traing count: {len(train_x)}, validation count: {len(val_x)}, testing count: {len(test_x)}")
    
    return [train_x, train_y, train_clas], [val_x, val_y, val_clas], [test_x, test_y, test_clas]


""" MSD """

# Create datalist for particular dataset
def create_msd_datalist(train_data: list[str], val_data: list[str], test_data: list[str], msd_dataset_directory: str, partition: list[float, float, float] = [70, 15, 15], report: bool = True) -> None:
    
    # get image filepaths
    imagesTs = [test_sample["image"] for test_sample in test_data]
    labelsTs = [test_sample["label"] for test_sample in test_data]
    if not val_data:
        crossvalidation = True
        imagesVa = []
        labelsVa = []
    else:
        crossvalidation = False
        imagesVa = [val_sample["image"] for val_sample in val_data]
        labelsVa = [val_sample["label"] for val_sample in val_data]
    imagesTr = [train_sample["image"] for train_sample in train_data]
    labelsTr = [train_sample["label"] for train_sample in train_data]
    
    # crossvalidation
    if crossvalidation:
        # check if datalist exists
        datalist_filename = "dataset_crossval.json"
        datalist_filepath = os.path.join(msd_dataset_directory, datalist_filename)
        if os.path.exists(datalist_filepath):
            if report:
                print(f"Datalist already exists: {datalist_filepath}")
            return

        # create and populate datalist
        imagesTr = imagesTr + imagesVa
        labelsTr = labelsTr + labelsVa
        datalist_json = {"partition": partition, "testing": [], "validating": [], "training": []}
        datalist_json["testing"] = [{"image": image_filename, "label": label_filenames}
                                     for image_filename, label_filenames in zip(imagesTs, labelsTs)]
        datalist_json["training"] = [{"image": image_filename, "label": label_filenames, "fold": 0}
                                     for image_filename, label_filenames in zip(imagesTr, labelsTr)]
        # set fold number to each image
        num_folds = 5
        fold_size = len(datalist_json["training"]) // num_folds
        for i in range(num_folds):
            for j in range(fold_size):
                datalist_json["training"][i * fold_size + j]["fold"] = i
    
    # train-val split
    else:
        # check if datalist exists
        datalist_filename = "dataset_trainval.json"
        datalist_filepath = os.path.join(msd_dataset_directory, datalist_filename)
        if os.path.exists(datalist_filepath):
            if report:
                print(f"Datalist already exists: {datalist_filepath}")
            return
        
        # create and populate datalist
        datalist_json = {"partition": partition, "testing": [], "validating": [], "training": []}
        datalist_json["testing"] = [{"image": image_filename, "label": label_filenames}
                                     for image_filename, label_filenames in zip(imagesTs, labelsTs)]
        datalist_json["validating"] = [{"image": image_filename, "label": label_filenames}
                                     for image_filename, label_filenames in zip(imagesVa, labelsVa)]
        datalist_json["training"] = [{"image": image_filename, "label": label_filenames}
                                     for image_filename, label_filenames in zip(imagesTr, labelsTr)]

    # save datalist as json
    with open(datalist_filepath, "w", encoding="utf-8") as f:
        json.dump(datalist_json, f, ensure_ascii=False, indent=4)
    if report:
        print(f"Datalist is saved to {datalist_filepath}\n")
    
# Reorganise from my standards to MSD standards
def organize_files_msd(train: list[list[str], list[str], list[str]], val: list[list[dict], list[dict], list[dict]], test: list[list[str], list[str], list[str]], msd_dataset_directory: str, partition: list[float, float, float] = [70, 15, 15], report: bool = True) -> tuple[list[dict], list[dict], list[dict]]:
    
    # get image filepaths
    folder_list = ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]
    trainval_image_folder = os.path.join(msd_dataset_directory, folder_list[0])
    test_image_folder = os.path.join(msd_dataset_directory, folder_list[1])
    trainval_label_folder = os.path.join(msd_dataset_directory, folder_list[2])
    test_label_folder = os.path.join(msd_dataset_directory, folder_list[3])
    
    # prepare directories
    if report:
        print(f"Reorganizing to MSD standard...")
    os.makedirs(msd_dataset_directory)
    for folder in folder_list:
        os.makedirs(os.path.join(msd_dataset_directory, folder))
    
    # copy images and segmentations for training
    train_data = []
    train_x, train_y, train_clas = train
    for image_filepath, label_filepaths, classification in zip(train_x, train_y, train_clas):
        segmentation_filepath = label_filepaths["seg"]
        landmark_filepath = label_filepaths["lm"]
        # change patient identiefier so it does not contain '_image' or '_label'
        new_patient_identifier_image = os.path.basename(re.sub("_image", '', image_filepath))
        new_patient_identifier_segmentation = os.path.basename(re.sub("_image", '_seg', image_filepath))
        new_patient_identifier_landmark = os.path.basename(re.sub("_image", '_lm', image_filepath)).split(".")[0] + os.path.splitext(landmark_filepath)[-1]
        # segmentation and image filename are now identical
        new_image_filepath = os.path.join(trainval_image_folder, new_patient_identifier_image)
        new_segmentation_filepath = os.path.join(trainval_label_folder, new_patient_identifier_segmentation)
        new_landmark_filepath = os.path.join(trainval_label_folder, new_patient_identifier_landmark)
        shutil.copyfile(image_filepath, new_image_filepath)
        shutil.copyfile(segmentation_filepath, new_segmentation_filepath)
        shutil.copyfile(landmark_filepath, new_landmark_filepath)
        train_data.append({"image": new_image_filepath, "label": {"lm": new_landmark_filepath, "seg": new_segmentation_filepath}})
    
    # copy images and segmentations for validation
    val_data = []
    val_x, val_y, val_clas = val
    for image_filepath, label_filepaths, classification in zip(val_x, val_y, val_clas):
        segmentation_filepath = label_filepaths["seg"]
        landmark_filepath = label_filepaths["lm"]
        # change patient identiefier so it does not contain '_image' or '_label'
        new_patient_identifier_image = os.path.basename(re.sub("_image", '', image_filepath))
        new_patient_identifier_segmentation = os.path.basename(re.sub("_image", '_seg', image_filepath))
        new_patient_identifier_landmark = os.path.basename(re.sub("_image", '_lm', image_filepath)).split(".")[0] + os.path.splitext(landmark_filepath)[-1]
        # segmentation and image filename are now identical
        new_image_filepath = os.path.join(trainval_image_folder, new_patient_identifier_image)
        new_segmentation_filepath = os.path.join(trainval_label_folder, new_patient_identifier_segmentation)
        new_landmark_filepath = os.path.join(trainval_label_folder, new_patient_identifier_landmark)
        shutil.copyfile(image_filepath, new_image_filepath)
        shutil.copyfile(segmentation_filepath, new_segmentation_filepath)
        shutil.copyfile(landmark_filepath, new_landmark_filepath)
        val_data.append({"image": new_image_filepath, "label": {"lm": new_landmark_filepath, "seg": new_segmentation_filepath}})
    
    # copy images and segmentations for testing
    test_data = []
    test_x, test_y, test_clas = test
    for image_filepath, label_filepaths, classification in zip(test_x, test_y, test_clas):
        segmentation_filepath = label_filepaths["seg"]
        landmark_filepath = label_filepaths["lm"]
        # change patient identiefier so it does not contain '_image' or '_label'
        new_patient_identifier_image = os.path.basename(re.sub("_image", '', image_filepath))
        new_patient_identifier_segmentation = os.path.basename(re.sub("_image", '_seg', image_filepath))
        new_patient_identifier_landmark = os.path.basename(re.sub("_image", '_lm', image_filepath)).split(".")[0] + os.path.splitext(landmark_filepath)[-1]
        # segmentation and image filename are now identical
        new_image_filepath = os.path.join(test_image_folder, new_patient_identifier_image)
        new_segmentation_filepath = os.path.join(test_label_folder, new_patient_identifier_segmentation)
        new_landmark_filepath = os.path.join(test_label_folder, new_patient_identifier_landmark)
        shutil.copyfile(image_filepath, new_image_filepath)
        shutil.copyfile(segmentation_filepath, new_segmentation_filepath)
        shutil.copyfile(landmark_filepath, new_landmark_filepath)
        test_data.append({"image": new_image_filepath, "label": {"lm": new_landmark_filepath, "seg": new_segmentation_filepath}})
        
    if report:
        print(f"Reorganizing to MSD standard done.\n")
    
    # create corresponding MSD datalist
    create_msd_datalist(train_data, val_data, test_data, msd_dataset_directory, partition, report)
    
    return train_data, val_data, test_data