""" IMPORTS """

import time
import datetime
import os
import shutil
import json


""" TRAINING PIPELINE OVERVIEW """

class TrainOverview():
    
    def __init__(self, dataset_directory: str, model_directory: str, identification: int, model: dict = {}, training: dict = {}) -> None:
        dataset_name = os.path.basename(dataset_directory)
        now = datetime.datetime.now()
        date = now.strftime("%d/%m/%Y %H:%M:%S")
        self.general = {"dataset_directory": dataset_directory, "dataset_name": dataset_name, 
                        "model_directory": model_directory, "identification": identification,
                        "date": date}
        self.model = model
        self.training = training
        self.model_id_directory = os.path.join(model_directory, f"model_{identification}")
        self.trained_filepath = os.path.dirname(model_directory) + "/trained_models.json"
    
    
    # Set general information
    def set_general(self, general: dict) -> None:
        self.general = general
    
    # Get general information
    def get_general(self, key: None|str = None):
        general = self.general
        if key is None:
            return general
        else:
            if key in general.keys():
                return general[key]
            else:
                raise Exception(f"{key} not in general")
    
    # Add value to general information
    def add_general(self, key: str, value) -> None:
        self.general[key] = value
    
    
    # Set model specific information
    def set_model(self, model: dict) -> None:
        self.model = model
    
    # Get model specific information
    def get_model(self, key: None|str = None):
        model = self.model
        if key is None:
            return model
        else:
            if key in model.keys():
                return model[key]
            else:
                raise Exception(f"{key} not in model")
    
    # Add value to model specific information
    def add_model(self, key: str, value) -> None:
        self.model[key] = value
    
    
    # Set training specific information
    def set_training(self, training: dict) -> None:
        self.training = training
    
    # Get training specific information
    def get_training(self, key: None|str = None):
        training = self.training
        if key is None:
            return training
        else:
            if key in training.keys():
                return training[key]
            else:
                raise Exception(f"{key} not in training")
    
    # Add value to training specific information
    def add_training(self, key: str, value) -> None:
        self.training[key] = value
    
    
    # Get complete overview of training pipeline
    def get(self, dict_name: str|None = None, key: str|None = None) -> dict:
        if dict_name == "general":
            d = self.general
        elif dict_name == "model":
            d = self.model
        elif dict_name == "training":
            d = self.training
        else:
            d = {"general": self.general, "model": self.model, "training": self.training}
            
        if key is not None:
            if key in d.keys():
                d = d[key]
        
        return d
    
    def check(self, key: str, return_dict_name: bool = False) -> bool|str:
        if key in list(self.general.keys()):
            if return_dict_name:
                return "general"
            return True
        elif key in list(self.model.keys()):
            if return_dict_name:
                return "model"
            return True
        elif key in list(self.training.keys()):
            if return_dict_name:
                return "training"
            return True
        else:
            if return_dict_name:
                return None
            return False
        
    # Show overview of training pipeline
    def show(self) -> None:

        # general
        if len(self.general) > 0:
            print("\nGeneral:")
            general = self.general
            for key in general.keys():
                print(f"  - {key}: {general[key]}")
        # model
        if len(self.model) > 0:
            print("\nModel:")
            model = self.model
            for key in model.keys():
                if key == "post_processing":
                    print(f"  - {key}: {[post_proc['type'] for post_proc in model[key]]}")
                else:
                    print(f"  - {key}: {model[key]}")
        # training
        if len(self.training) > 0:
            print("\nTraining:")
            training = self.training
            for key in training.keys():
                if key == "data_augmentation":
                    data_augmentation = training[key]
                    print(f"  - {key}: {[data_aug['type'] for data_aug in data_augmentation]}")
                else:
                    print(f"  - {key}: {training[key]}")
            print()
    
    
    # Save training overview 
    def save(self) -> None:
        
        to_save = {"general": self.general, "model": self.model, "training": self.training}
        
        # # read existing data from the file, if any
        # trained_filepath = self.trained_filepath
        # try:
        #     with open(trained_filepath, 'r') as r_file:
        #         model_list = json.load(r_file)
        # except FileNotFoundError:
        #     # If the file doesn't exist, initialize an empty list
        #     model_list = []
        
        # # check if model already exists in trained model database
        # index = None
        # for i, model_info in enumerate(model_list):
        #     general = model_info.get("general")
        #     model_directory = general["model_directory"]
        #     identification = general["identification"]
        #     _model_id_directory = os.path.join(model_directory, f"model_{identification}")
        #     if _model_id_directory == self.model_id_directory:
        #         index = i
        #         break
                
        # # if model not in trained model database, append new model
        # if index is None:     
        #     model_list.append(to_save)
        # # if model already in trained model database, overwrite model information
        # else:
        #     model_list[index] = to_save
            
        # # write the updated data back to the file
        # with open(trained_filepath, 'w') as w_file:
        #     w_file.write(json.dumps(model_list, indent=4))
        
        # save copy in model directory
        with open(os.path.join(self.model_id_directory, 'train_overview.json'), 'w') as w_file:
            w_file.write(json.dumps(to_save, indent=4))
    
    
    # Load training overview
    def load(self, model_id_directory: str, from_trained_models: bool = False) -> None:
        
        if not os.path.isdir(model_id_directory):
            rest, model_id = os.path.split(model_id_directory)
            rest, model_type = os.path.split(rest)
            model_task = os.path.basename(rest)
            raise Exception(f"Model '{model_type} - {model_id}' in task '{model_task}' does not exist.")
            
        if from_trained_models:
            # read all trained models in database
            with open(self.trained_filepath, 'r') as r_file:
                model_list = json.load(r_file)

            # find corresponding model and set general, model and training dictionaries
            for model_info in model_list:
                general = model_info.get("general")
                model_directory = general["model_directory"]
                identification = general["identification"]
                _model_id_directory = os.path.join(model_directory, f"model_{identification}")

                if _model_id_directory == model_id_directory:
                    self.general = general
                    self.model = model_info.get("model")
                    self.training = model_info.get("training")
                    
        else:
            
            # read train overview
            with open(os.path.join(model_id_directory, 'train_overview.json'), 'r') as r_file:
                model_info = json.load(r_file)
            
            # save general, model and training dictionaries
            self.general = model_info.get("general")
            self.model = model_info.get("model")
            self.training = model_info.get("training")
                      
                
    def config(self, python_filepath) -> None:
        config_data = {'dataset_directory': self.general['dataset_directory'],
                       'model_directory': self.general['model_directory'],
                       'training': self.training,
                       'model': self.model}
        
        run_folder = os.path.join(self.model_id_directory, "run")
        if not os.path.isdir(run_folder):
            os.makedirs(run_folder)
        
        run_config_filepath = os.path.join(run_folder, "config.json")
        with open(run_config_filepath, 'w') as w_file:
            w_file.write(json.dumps(config_data, indent=4))
        
        run_python_filepath = os.path.join(run_folder, os.path.basename(python_filepath))
        shutil.copyfile(python_filepath, run_python_filepath)


""" CHRONOMETER FOR RUNTIME OF FUNCTIONS """

#
class Chronometer():
    
    def __init__(self, indices: list|int|str = [0]) -> None:
        
        if not isinstance(indices, list):
            indices = [indices]
            
        self.start_time = dict(zip(indices, [None]*len(indices)))
        self.indices = indices
    
    def go(self, index: int|str = 0) -> None:
        if not (index in self.indices):
            self.indices.append(index)
            self.start_time[index] = time.time()
        elif self.start_time[index] is None:
            self.start_time[index] = time.time()
        else:
            print(f"Clock '{index}' already ticking ({round(time.time()-self.start_time[index])} s).")
        
    def time(self, index: int|str = 0, verbose: bool = False) -> float:
        if not (index in self.indices):
            print(f"Invalid clock '{index}', available clocks: {self.indices}")
        time_time = round(time.time()-self.start_time[index])
        if verbose:
            print(f"Time: {datetime.timedelta(seconds=time_time)} hours")
        return time_time
    
    def stop(self, index: int|str = 0, verbose: bool = False) -> float:
        if not (index in self.indices):
            print(f"Invalid clock '{index}', available clocks: {self.indices}")
        
        elif self.start_time[index] is not None:
            stop_time = round(time.time()-self.start_time[index])
            self.start_time[index] = None
        else:
            print(f"Clock '{index}' already stopped.")
        if verbose:
            print(f"Time: {datetime.timedelta(seconds=stop_time)} hours")
        return stop_time
    

""" MODEL RUN DIRECCTORY """

# Create running directory for model
def create_model_run_directory(model_directory: str, identification: int = 1, create_new_model: bool = True) -> tuple[str, int]:
    
    # if identification number given => not creating new model
    if identification != -1:
        create_new_model = False

    # model directory
    if create_new_model:
        
        identification = 1
        while identification < 1000:
            model_run_directory = os.path.join(model_directory, f"model_{identification}")
            try:
                os.makedirs(model_run_directory, exist_ok=False)  # atomic: fail if already exists
                print(f"REPORT FILE FOR {os.path.basename(model_directory)} {identification}\n")
                return model_run_directory, identification
            except FileExistsError:
                identification += 1

        raise ValueError("Maximum number of model directories (1000) exceeded. Please clean up the model directory.")
        
    else:
        model_run_directory = os.path.join(model_directory, f"model_{identification}")
        return model_run_directory, identification