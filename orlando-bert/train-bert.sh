#!/bin/bash

#SBATCH --job-name=train_simclr
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --account=COMS035204
#SBATCH --mail-user=sv22482@bristol.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "${SLURM_SUBMIT_DIR}"

module load apptainer/1.1.9-qwwi

apptainer exec --nv --bind /user/work/sv22482:/user/work/sv22482 /user/work/sv22482/pytorch-latest.sif ./bert