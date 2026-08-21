""" IMPORTS """

from collections.abc import Callable
import zipfile
import os
import shutil
import json
import time
from datetime import datetime
import multiprocessing
import numpy as np


""" DICTIONARY POSSIBLE DISEASE CLASSIFICATIONS """

CLASSIFICATION_DICT = {0: 'healthy', 1: 'marfan', 2: 'taad', 3: 'tbad', 4: 'aaa', 5: 'ad'}


""" PROGRESSS REPORT """

# Report progress
def report_progress(done: int, total: int, runtime: float = -1.0, bar_len: int = 20) -> None:
    
    # get progress from tasks dome and total tasks
    progress = done/total
    filled_len = int(round(bar_len*progress))
    progress_percent = 100.00*progress
    
    # make progress percentage always 3 chars long
    if progress < 0.1:
        progress_percent = f"{progress_percent:.2f}"
    elif progress == 1:
        progress_percent = f"{progress_percent:.0f}"
    else:
        progress_percent = f"{progress_percent:.1f}"
    
    # print visual progress bar
    report = f"\r  |{'#'*filled_len}{'-'*(bar_len-filled_len)}| {progress_percent}%"
    if not runtime == -1.0:
        mean_time = round(runtime/(done+0.001),2)
        left_time = round(mean_time*(total-done))
        report += f" - {left_time//60}min {left_time%60}s left ({mean_time} s/it)"
    print(report, end="")


""" MULTIPROCESSING """

# Run function in multiprocessing
def run_in_multiprocessing(worker_function: Callable, inputs: list) -> None:
    
    multiprocessing.freeze_support()
    # check that all inputs have the same size
    input_lists_only = [input for input in inputs if isinstance(input, (list,np.ndarray))]
    same_size = all(len(input_list) == len(input_lists_only[0]) for input_list in input_lists_only)
    if not same_size:
        raise Exception("All input lists must have the same size. Use tuple if other size iterator needed.")
        
    # if any single instances given, repeat them to create lists of equal size
    input_size = len(input_lists_only[0])
    for i, input in enumerate(inputs):
        if not isinstance(input, (list,np.ndarray)):
            inputs[i] = input_size*(input,)
            
    # set the number of processes based on the available CPU cores
    try:
        num_cores = int(len(os.sched_getaffinity(0))-1)
    except:
        num_cores = multiprocessing.cpu_count() - 1
    num_processes = min(num_cores, len(inputs[0]))

    # list[list of same variable types] -> list[list of variables for same iteration] (ChatGPT)
    inputs = list(zip(*inputs))
    inputs = [list(t) for t in sorted(enumerate(inputs), key=lambda x: x[0])]
    pool_inputs = list(zip(*inputs))[1]
    
    # run the function with the inputs
    print(f"Running with {num_processes} processes (no progress available):")
    pool = multiprocessing.Pool(processes=num_processes)
    pool.map(worker_function, pool_inputs)
    pool.close()


""" EXTRACT DATASET """

# If data not extracted yet, extract all to folder with name of dataset
def extract_dataset(dataset_directory: str, report: bool = True) -> None:
    
    # get potential zipfile name
    zip_filename = dataset_directory + '.zip'
    
    # if not folder exists with name of dataset => extracted
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
                
        # zipfile not existing
        else:
            raise Exception(f"Zip file with name '{os.path.basename(zip_filename)}' not found")
    
    # already extracted
    else:
        if report:
            print(f"Dataset already extracted to folder '{os.path.basename(dataset_directory)}'\n")


""" DELETE DATASET """

# Delete dataset
def delete_dataset(dataset_directory: str, keep_zip: bool = False) -> None:
    
    # keep zip file or delete it with the extracted dataset
    if keep_zip:
        zip_filename = os.path.join(dataset_directory, os.path.basename(dataset_directory)) + '.zip'
        if os.path.exists(zip_filename):
            os.rename(zip_filename, dataset_directory + '.zip')
        else:
            print(f"Could not delete database '{os.path.basename(dataset_directory)}' because no zip-file found")
            return

    # delete database
    if os.path.isdir(dataset_directory):
        shutil.rmtree(dataset_directory)
    else:
        print(f"Database '{os.path.basename(dataset_directory)}' does not exist")


""" ZIP DATASET """

# Zip dataset
def zip_dataset(dataset_directory: str, report: bool = True) -> None:
    
    # set name of output zipfile
    output_path = dataset_directory + '.zip'
    # remove existing zipfile
    if os.path.exists(output_path):
        os.remove(output_path)
        time.sleep(3)
        
    # zip dataset
    print(f"Zipping dataset {os.path.basename(dataset_directory)}...")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        running_time = 0
        amount_patients = 0
        amount_patients_done = 0
        for root, _, files in os.walk(dataset_directory):
            amount_patients += len(files)
            for file in sorted(files):
                start_time = time.time()
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dataset_directory)
                zipf.write(file_path, arcname)
                running_time += time.time() - start_time
                if report:
                    report_progress(amount_patients_done, amount_patients, running_time)
                    amount_patients_done += 1
                    
    # delete database to keep only zipfile
    print(f"\r\rZipping dataset {os.path.basename(dataset_directory)} done.")
    delete_dataset(dataset_directory)
    

""" CREATE METADATA """

def _copy_confinement_config(original_directory: str, new_directory: str):

    original_filepath = os.path.join(original_directory, "confinement_config.json")
    if not os.path.exists(original_filepath):
        print("Confinement configuration could not be copied because it is missing from origal dataset, please add it manually!")
        return
    new_filepath = os.path.join(new_directory, "confinement_config.json")

    shutil.copyfile(original_filepath, new_filepath)

# Create metadata for new dataset
def create_metadata(dataset_directory: str, config: dict) -> None:
    
    # unchanged information for each file at the top
    created_on = datetime.now().strftime("%d/%m/%Y at %H:%M:%S")
    author = "Loris Giordano (ETRO-VUB)"
    dataset_name_interpretation = {
        "_r": "resolution of all images in dataset changed",
        "_s": "size of all images in dataset changed",
        "_b": "dataset binarized",
        "_o": "dataset oversampled"}
    
    # save metadata file with unchanged information and configuration of dataset
    metadata_data = {'created_on': created_on, 
                     'author': author,
                     'dataset_name_interpretation': dataset_name_interpretation,
                     'config': config}
    metadata_filepath = os.path.join(dataset_directory, f"{os.path.basename(dataset_directory)}_metadata.json")
    with open(metadata_filepath, 'w') as metadata_file:
        metadata_file.write(json.dumps(metadata_data))

    original_directory = os.path.join(config["root"], config["original_dataset"])
    _copy_confinement_config(original_directory, dataset_directory)

        
        
if __name__ == '__main__':
    
    dataset_directory = ""
    
    zip_dataset(dataset_directory)