#!/bin/bash

#SBATCH --time=07:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16


export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=true
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:516
nvidia-cuda-mps-control -d


export ROOT_DIR=$(dirname "$(realpath $0)")

cd $ROOT_DIR/Scripts/LandmarkDetection

module load MONAI/1.3.0-foss-2023a-CUDA-12.1.1
module load wandb/0.16.1-GCC-12.3.0

python train.py --config config_lm_seg_CTPel