#!/bin/bash
#SBATCH --job-name=i1-xcheck
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --constraint=amd
#SBATCH --output=%x-%j.out
# I1 one_disk voxelisation cross-check (see tests/test_voxel_crosscheck.py).  For each voxel
# size VS: disk path and voxel path, each -ref + sample, at -res round(1/VS), settings of
# run_spectrum.sh's default example with EPS=4 plus -no_fields (spectra only, both paths).
# Required env: WORK (output directory).  Optional: SIZES (default "0.02 0.01 0.005"),
# NP (default SLURM_NTASKS).  Ranks via the env's mpirun inside the job (never srun python).
set -uo pipefail
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
CODE=${CODE:-$SLURM_SUBMIT_DIR}
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
SIZES=${SIZES:-"0.02 0.01 0.005"}
NP=${NP:-$SLURM_NTASKS}
mkdir -p "$WORK"
python "$CODE/tests/test_voxel_crosscheck.py" build "$WORK" $SIZES || exit 1
common=(-eps 4 -eps_ref 1 -polarization x -ks 5.235987756 7.853981634 -nfreqs 31
        -dsrc 0.4 -ddet 0.2 -dpml 0.3 -tpml 0.5 -ScattPower -comp Ex -dft_margin_px 4
        -dft_nconsec 3 -dft_tol 1e-8 -maxt 200 -tempname incident -no_fields)
status=0
for vs in $SIZES; do
    res=$(python -c "print(round(1/$vs))")
    t=$(python -c "print('{0:g}'.format($vs))")
    for path in disk voxel; do
        d="$WORK/${path}_$t"; mkdir -p "$d"; cd "$d"
        if [[ $path == disk ]]; then
            geo=(-load "$CODE/examples/one_disk.txt" -is_point -phi 0.2 -tfilm 0.1)
        else
            geo=(-load "$WORK/disk_vs$t.npz" -voxel_size "$vs")
        fi
        for stage in reference sample; do
            extra=(); [[ $stage == reference ]] && extra=(-ref)
            s0=$(date +%s)
            mpirun -np "$NP" python "$CODE/film_transmittance.py" "${geo[@]}" -res "$res" "${common[@]}" \
                "${extra[@]}" -saveas "$stage" > "$stage.log" 2>&1
            rc=$?; (( rc )) && status=1
            echo "vs $t res $res $path $stage rc=$rc ($(( $(date +%s) - s0 )) s)"
        done
    done
done
python "$CODE/tests/test_voxel_crosscheck.py" compare "$WORK" $SIZES || status=1
exit $status
