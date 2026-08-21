#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=64


export ROOT_DIR=$(dirname "$(realpath $0)")

cd $ROOT_DIR/Scripts/Dataset

module load SimpleITK/2.3.1-foss-2023a
module load matplotlib/3.7.2-gfbf-2023a

python run.py --config config_CTPel