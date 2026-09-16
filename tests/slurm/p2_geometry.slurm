#!/bin/bash
#SBATCH --job-name=p2-geom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --constraint=amd
#SBATCH --output=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p2/geometry-%j.out
# orient + MaterialGrid placement tests (tests 1 and 2).  sbatch tests/slurm/p2_geometry.slurm
set -euo pipefail
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
repo=${REPO:-/scratch/gpfs/TORQUATO/sd2402/film-worktrees/p2}
out=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p2
mkdir -p "$out"
python "$repo/tests/test_voxel_geometry.py" --json "$out/geometry_${SLURM_JOB_ID}.json"
