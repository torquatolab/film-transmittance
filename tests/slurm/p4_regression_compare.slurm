#!/bin/bash
# P4 test 1 (comparison only): re-compare an existing p4_regression.slurm output with out_new.
#SBATCH --job-name=p4-regcmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:10:00
#SBATCH --constraint=amd
#SBATCH --output=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p4/%x-%j.out
set -uo pipefail
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
CODE=${CODE:-/scratch/gpfs/TORQUATO/sd2402/film-worktrees/p4}
python "$CODE/tests/test_field_output.py" regression /scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p4/out_reg /scratch/gpfs/TORQUATO/sd2402/film-audit/out_new
