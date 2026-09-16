#!/bin/bash
#SBATCH --job-name=film-trans
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --constraint=amd
#SBATCH --output=film-trans-%j.out
#SBATCH --error=film-trans-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the package directory}"
module purge
module load anaconda3/2026.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate film-meep
set -u
# Use the conda environment's MPICH launcher with its matching Meep build.
# This starter script intentionally stays on one node.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export NP="$SLURM_NTASKS"
export CHECKPOINT=${CHECKPOINT:-0}
# Keep RUN_DIR unchanged when resubmitting a checkpointed job.
bash ./run_spectrum.sh "${RUN_DIR:-$SLURM_SUBMIT_DIR/results}"
