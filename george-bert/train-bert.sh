#!/bin/bash

#SBATCH --job-name=train_simclr
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --account=COMS031144
#SBATCH --mail-user=ne22902@bristol.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "${SLURM_SUBMIT_DIR}"

module load apptainer/1.1.9-qwwi

apptainer exec --nv --bind /user/work/ne22902:/user/work/ne22902 /user/work/ne22902/pytorch_latest.sif ./bert