#!/bin/bash

#SBATCH --chdir=$TRAIN_DIR
#SBATCH --job-name=$CLASS_NAME-lora
#SBATCH --time=00:30:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=%x.out    # Output file name (%x=job_name, %j=jobid)
#SBATCH --error=%x.err     # Error file name

cd $TRAIN_DIR
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
    srun train-loras-from-folder.sh $CLASS_NAME
else
    echo "Environment verification failed!"
    exit 1
fi