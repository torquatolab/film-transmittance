#!/bin/bash
# P4 test 1: harness regression of this worktree against film-audit/out_new (b71f61b).
#SBATCH --job-name=p4-regression
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:40:00
#SBATCH --constraint=amd
#SBATCH --output=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p4/%x-%j.out
set -uo pipefail
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
CODE=${CODE:-/scratch/gpfs/TORQUATO/sd2402/film-worktrees/p4}
W=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p4
GOLD=/scratch/gpfs/TORQUATO/sd2402/film-audit/out_new
# harness.sh (a copy) resolves ../../packing.txt from OUT/D, i.e. $W/packing.txt
cd $W
bash $W/harness.sh "$CODE" $W/out_reg
python "$CODE/tests/test_field_output.py" regression $W/out_reg $GOLD
