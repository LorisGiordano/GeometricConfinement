""" IMPORTS """

import os
import glob
import zipfile
import json
from abc import ABCMeta
from monai.data import Dataset, CacheDataset, SmartCacheDataset, DataLoader

from .partition_dataset import create_msd_datalist, train_val_test_split, organize_files_msd
from .normalize_intensity import normalize_intensity_train_data
from .fingerprints_dataset import fingerprints_dataset, show_fingerprints


""" GET ALL FILES IN DICTIONARY """

# If data not extracted yet, extract all to folder with name of dataset
def _extract_dataset(dataset_directory: str, report: bool = True) -> None:
    
    # get potential zipfile name
    zip_filename = dataset_directory + '.zip'
    
    # check if folder exists with name of dataset => already extracted
    if not os.path.isdir(dataset_directory):
        # check if zipfile exists
        if (os.path.exists(zip_filename)):
            # extract all data
            print("Extracting dataset...",end="")
            zip_file = zipfile.ZipFile(zip_filename)
            zip_file.extractall(dataset_directory)
            zip_file.close()
            # move zipfile to the extraction folder
            os.rename(zip_filename, os.path.join(dataset_directory, os.path.basename(dataset_directory)) + '.zip')
            if report:
                print(f"\rDataset extracted to folder '{os.path.basename(dataset_directory)}'\n")
        else:
            raise Exception(f"Zip file with name '{os.path.basename(zip_filename)}' not found")
            return
    else:
        if report:
            print(f"Dataset already extracted to folder '{os.path.basename(dataset_directory)}'\n")


def add_root(path_list: list[dict], root: str):
    for item in path_list:
        for k, v in item.items():
            item[k] = str(os.path.join(root, v))
    return path_list

# Get all images, segmentations, and landmarks in MSD dataset
def retrieve_data(dataset_directory: str, report: bool = True, crossvalidation: bool = False, num_processes: int = 14) -> tuple[list[str], list[str], list[str], dict, dict]:
    
    # get partitioned data
    datalist_filepath = os.path.join(dataset_directory, "datalist_trainval.json")
    if crossvalidation:
        datalist_filepath = os.path.join(dataset_directory, "datalist_crossval.json")
    with open(datalist_filepath, 'r') as json_file:
        datalist_json = json.load(json_file)
    
    # extract infotmation of datalist
    root = datalist_json["root"]
    partition = datalist_json["partition"]
    if "partition" in datalist_json.keys():
        partition = datalist_json["partition"]
    else:
        partition = False
    train_data = add_root(datalist_json["training"], root)
    val_data = add_root(datalist_json["validating"], root)
    test_data = add_root(datalist_json["testing"], root)

    # get fingerprints
    fingerprints_filepath = os.path.join(dataset_directory, "fingerprints.json")
    if not os.path.exists(fingerprints_filepath):
        fingerprints = fingerprints_dataset(train_data, val_data, dataset_directory, report=report)
        normalize_intensity_train_data(train_data, val_data, fingerprints, num_processes=num_processes)
    else:    
        with open(fingerprints_filepath, 'r') as json_file:
            fingerprints = json.load(json_file)
        if report:
            show_fingerprints(fingerprints)

    return train_data, val_data, test_data, fingerprints


""" MAKE MONAI DATSETS """

#
def datasets_trainval(train_data: list[str], train_transforms: ABCMeta, val_data: list[str], val_transforms: ABCMeta, num_processes: int = 14, all_in_cache: bool = True, train_cache_num: int = 75, val_cache_num: int = 10, report: bool = True) -> tuple[CacheDataset, CacheDataset] | tuple[SmartCacheDataset, SmartCacheDataset]:

    # make datasets
    num_processes = int(0.5*num_processes)
    num_workers_train = round(0.7*num_processes)
    num_workers_val = num_processes - num_workers_train
    
    # if all dataset can go in cache
    if all_in_cache:
        if report:
            print("\nTraining dataset:")
        train_ds = CacheDataset(train_data, transform=train_transforms, num_workers=num_workers_train)
        if report:
            print("\nValidation dataset:")
        val_ds = CacheDataset(val_data, transform=val_transforms, num_workers=num_workers_val)
        if report:
            print()
        
    # if dataset too large
    else:
        num_init_workers_train = round(num_workers_train/2)
        num_replace_workers_train = num_workers_train - num_init_workers_train
        num_init_workers_val = round(num_workers_val/2)
        num_replace_workers_val = num_workers_val - num_init_workers_val
        
        # # TODO: check if all number of workers can be used for initializing and replacing (not done at same time)
        if report:
            print("\nTraining dataset:")
        train_ds = SmartCacheDataset(train_data, cache_num=train_cache_num, transform=train_transforms, 
                                          num_init_workers=num_init_workers_train, num_replace_workers=num_replace_workers_train)
        if report:
            print("\nValidation dataset:")
        val_ds = SmartCacheDataset(val_data, cache_num=val_cache_num, transform=val_transforms, 
                                        num_init_workers=num_init_workers_val, num_replace_workers=num_replace_workers_val)
        if report:
            print()
        
    return train_ds, val_ds

#
def dataset_test(test_data: list[str], test_transforms: ABCMeta, num_processes: int = 14) -> tuple[CacheDataset, CacheDataset]:
    
    test_ds = CacheDataset(test_data, transform=test_transforms, num_workers=num_processes)
    
    return test_ds
    

""" MAKE MONAI DATALOADERS """

def dataloaders_trainval(train_ds: CacheDataset|SmartCacheDataset, val_ds: CacheDataset|SmartCacheDataset, batch_size: int, num_processes: int = 14, distributed: bool = False) -> tuple[DataLoader, DataLoader]:
    
    num_processes = int(0.5*num_processes)
    num_workers_train = round(0.7*num_processes)
    num_workers_val = num_processes - num_workers_train

    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        from torch.utils.data import DataLoader

        train_sampler = DistributedSampler(
            train_ds, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_ds, shuffle=False
        )

        train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers_train,
        sampler=train_sampler,
        pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers_val,
            sampler=val_sampler,
            pin_memory=True
        )

    else:
        from monai.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers_train)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers_val)
    
    return train_loader, val_loader

def dataloader_test(test_ds: CacheDataset|SmartCacheDataset, batch_size: int, num_processes: int = 14) -> DataLoader:
    
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_processes)
    
    return test_loader