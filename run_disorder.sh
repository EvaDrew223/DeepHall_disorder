#!/bin/bash
#SBATCH --job-name=deephall_disorder
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jihang@uw.edu

# Load modules
#module purge
module load gcc/13.4.0
#module load cuda/12.9.1

# Activate environment
source ~/.bashrc
conda activate /gpfs/projects/cmt/jihangz/conda/envs/deephall

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Navigate to project
cd /gpfs/projects/cmt/jihangz/DeepHall_disorder

# Set GPU environment
# export CUDA_VISIBLE_DEVICES=0
# export JAX_PLATFORM_NAME=gpu

# Use python -m instead of deephall command 
srun --ntasks=1 --gpus-per-task=4 python -m deephall.train \
        'system.nspins=[8,0]' \
        system.flux=21 \
        optim.iterations=200000 \
        network.psiformer.num_layers=4  \
        batch_size=3360 \
        optim.kfac.lr.rate=0.05 \
        network.psiformer.num_heads=4 \
        mcmc.width=0.1 \
        seed=1758632847 \
        system.disorder.n_dis=5 \
        system.disorder.disorder_separation=0.5 \
        system.disorder.max_disorder_charge=1.0 \
        system.disorder.disorder_strength=1.0 \
        system.disorder.disorder_seed=921
