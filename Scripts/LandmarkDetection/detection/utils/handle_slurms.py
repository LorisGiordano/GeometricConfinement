""" IMPORTS """

import glob
import os


""" GET SLURM FILE """

# Identify corresponding output of train.py (newest slurm_train file)
def get_slurm_train(config: dict, model_root: str) -> str|None:
    
    debug_mode = config.get('model_directory') == 'debug'
    if not debug_mode:
        out_train_filepath = sorted(glob.glob(os.path.join(model_root, '_models_slurms/slurm_train-*.out')))[-1]
    else:
        out_train_filepath = None
        
    return out_train_filepath

# Identify corresponding output of evaluate.py
def get_slurm_evaluate(model_root: str) -> str:
    
    out_evaluate_filepath = sorted(glob.glob(os.path.join(model_root, '_models_slurms/slurm_test-*.out')))[-1]
    
    return out_evaluate_filepath

# Identify corresponding output of evaluate.py
def get_slurm_infer(model_root: str) -> str:
    
    out_infer_filepath = sorted(glob.glob(os.path.join(model_root, '_models_slurms/slurm_infer-*.out')))[-1]
    
    return out_infer_filepath


""" SAVE SLURM FILE """

# Save output files in model directory
def save_slurm(slurm_file: str|None, model_run_directory: str) -> None:
    if slurm_file is not None:
        os.rename(slurm_file, os.path.join(model_run_directory, os.path.basename(slurm_file)))    