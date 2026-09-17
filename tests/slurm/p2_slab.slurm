#!/bin/bash
#SBATCH --job-name=p2-slab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --constraint=amd
#SBATCH --output=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p2/slab-%j.out
# Voxel slab / 7-layer stack vs TMM (z placement via aligned_z_center).  Submit from anywhere:
#   sbatch tests/slurm/p2_slab.slurm                       (gate: uniform + multilayer, both cell z parities)
#   ARGS="--film multilayer --z_align off" sbatch tests/slurm/p2_slab.slurm   (single custom run)
set -euo pipefail
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
repo=${REPO:-/scratch/gpfs/TORQUATO/sd2402/film-worktrees/p2}
outdir=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p2/slab
if [ -n "${ARGS:-}" ]; then
	python "$repo/tests/test_voxel_slab.py" --outdir "$outdir" $ARGS
	exit
fi
rc=0
for film in uniform multilayer; do
	for cellx in 0 1; do
		echo "=== film=$film cell_extra_px=$cellx ==="
		python "$repo/tests/test_voxel_slab.py" --outdir "$outdir" --film "$film" --cell_extra_px "$cellx" || rc=1
	done
done
echo "overall rc=$rc"
exit $rc
