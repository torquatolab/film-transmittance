#!/bin/bash
#SBATCH --job-name=p2-slab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --constraint=amd
#SBATCH --output=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p2/slab-%j.out
# Uniform voxel slab vs Airy (test 3).  Submit from anywhere:
#   sbatch tests/slurm/p2_slab.slurm            (production z layout)
#   ZOFF=0.5 sbatch tests/slurm/p2_slab.slurm   (film shifted half a pixel; informational)
#   FILM=multilayer sbatch tests/slurm/p2_slab.slurm (internal voxel faces; informational)
set -euo pipefail
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
repo=${REPO:-/scratch/gpfs/TORQUATO/sd2402/film-worktrees/p2}
python "$repo/tests/test_voxel_slab.py" --outdir /scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p2/slab \
	--z_offset_px "${ZOFF:-0}" --film "${FILM:-uniform}" ${BETA:+--beta $BETA} ${GEOM:+--geometry $GEOM} ${RESLIST:+--res $RESLIST}
