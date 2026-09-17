#!/bin/bash
#SBATCH --job-name=p3-tiffs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --constraint=amd
#SBATCH --output=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p3/p3_tiffs-%j.out
#SBATCH --error=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p3/p3_tiffs-%j.out

# Submit from the repository root: sbatch tests/slurm/p3_tiffs.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/time -v python tests/test_voxel_crop.py --tiffs /home/sd2402/torquato-scratch-sd2402/structural_color/data --out /scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p3/crops.json
