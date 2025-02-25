#!/bin/bash

#SBATCH --chdir=$EXPERIMENT_DIR
#SBATCH --job-name=exp-lora
#SBATCH --time=15:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --output=%x.out    # Output file name (%x=job_name, %j=jobid)
#SBATCH --error=%x.err     # Error file name

cd $EXPERIMENT_DIR
source .venv/bin/activate

srun python3 -c "
import torch
import sys

if not torch.cuda.is_available():
    print('CUDA is not available!')
    sys.exit(1)
else:
    print('CUDA is available')
    print('CUDA version:', torch.version.cuda)
    print('GPU device:', torch.cuda.get_device_name())
"

if [ $? -eq 0 ]; then
    echo "Environment verification successful."
    srun python experiment.py --dataset=1k --num-experiments=10 --output-directory=1k-exp-results --flat-vec --stats-flat-vec --pca-flat-vec --stats-concat --dense
else
    echo "Environment verification failed!"
    exit 1
fi
