#!/bin/bash
#SBATCH --job-name=p2-grating
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --constraint=amd
#SBATCH --output=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p2/grating-%j.out
# x-periodic voxel grating via film_geometry vs plain mp.Block reference (measurement, not a gate).
# One MPI run (the env's MPICH mpirun, as run_spectrum.sh) per stage; one parity per job:
#   STAGE=ref   PERIOD=60 NY=2 sbatch tests/slurm/p2_grating.slurm                  (even: mp.Block at REF_RES)
#   STAGE=voxel PERIOD=60 NY=2 sbatch --ntasks=4 tests/slurm/p2_grating.slurm       (film_geometry at RES)
#   (odd voxel counts: PERIOD=61 NY=1)
# Film runs stop on DFT decay (DFT_TOL), at MAXT, or (reference) after WALL_LIMIT s of wall clock; voxel runs stop
# at the reference's t_end.  T, R snapshots every SNAP_DT give the convergence check.
set -euo pipefail
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
repo=${REPO:-/scratch/gpfs/TORQUATO/sd2402/film-worktrees/p2}
np=${SLURM_NTASKS:-1}
mpirun -np "$np" python -c "import meep as mp; assert mp.count_processors() == $np, 'MPI rank count mismatch'"
mpirun -np "$np" python "$repo/tests/test_voxel_slab.py" --outdir /scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p2/grating \
	--film grating --stage "${STAGE:-all}" --period "${PERIOD:-60}" --duty "${DUTY:-30}" --ny "${NY:-2}" \
	--nvox "${NVOX:-20}" --res ${RES:-100 200} --ref_res "${REF_RES:-400}" --maxt "${MAXT:-400}" \
	--dft_tol "${DFT_TOL:-1e-8}" --wall_limit "${WALL_LIMIT:-0}" --snap_dt "${SNAP_DT:-10}" ${ARGS:-}
