#!/bin/bash
#SBATCH --job-name=i1-voxel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:50:00
#SBATCH --constraint=amd
#SBATCH --output=%x-%j.out
# I1 tests 2-4: voxel smoke run, refusals, run_spectrum.sh voxel path.
# Required env: WORK (output directory), VOXEL_MODULES (directory holding voxel_io.py and
# voxel_geometry.py: the Phase-A stubs, or the worktree once P1/P2 are merged).
# Optional: STEPS (default "smoke refusals wrapper").
set -uo pipefail
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
CODE=${CODE:-$SLURM_SUBMIT_DIR}
export PYTHONPATH="$VOXEL_MODULES${PYTHONPATH:+:$PYTHONPATH}"
bash -n "$CODE/run_spectrum.sh" && echo "PASS: bash -n run_spectrum.sh" || fail=1
status=${fail:-0}
for step in ${STEPS:-smoke refusals wrapper}; do
    echo "===== $step"
    python "$CODE/tests/test_voxel_integration.py" "$step" "$WORK/$step"
    rc=$?; echo "===== $step rc=$rc"; (( rc )) && status=1
done
exit $status
