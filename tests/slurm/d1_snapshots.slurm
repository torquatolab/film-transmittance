#!/bin/bash
#SBATCH --job-name=d1-snap
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:50:00
#SBATCH --constraint=amd
#SBATCH --output=%x-%j.out
# D1: -snapshot_dt and the voxel "stopped:" header (tests/test_snapshots.py at 1, 2 and 4
# MPI ranks through the environment's mpirun), then the run_spectrum.sh voxel wrapper
# with the new defaults (MAXT required, -tpml 1.5, 201 frequencies).
# Required env: WORK (output directory).
set -uo pipefail
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
CODE=${CODE:-$SLURM_SUBMIT_DIR}
status=0
python "$CODE/tests/test_snapshots.py" "$WORK/snapshots" ${RANKS:-1 2 4}; rc=$?; echo "===== snapshots rc=$rc"; (( rc )) && status=1
python "$CODE/tests/test_voxel_integration.py" wrapper "$WORK/wrapper"; rc=$?; echo "===== wrapper rc=$rc"; (( rc )) && status=1
exit $status
