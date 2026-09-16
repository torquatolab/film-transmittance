#!/bin/bash
# P4 diagnostic for test 3: separate the -field_wavelengths implementation from the
# stopping time.  Sample runs (Ag, wrapper settings, serial) forced to stop at the same
# step with -dft_tol 1e-30 -maxt 31 (reusing the reference of p4_fields.slurm's ag_full):
#   X full, Y selected (i = 15, 30), Z selected (i = 5, 20; different auto decimation).
# Y vs X isolates the implementation; ag_full vs X measures the truncation of the
# naturally stopped full run.
#SBATCH --job-name=p4-fields-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --constraint=amd
#SBATCH --output=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p4/%x-%j.out
set -uo pipefail
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
CODE=${CODE:-/scratch/gpfs/TORQUATO/sd2402/film-worktrees/p4}
V=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p4
REF=$V/fields/ag_full
W=$V/fields_diag
T="python $CODE/tests/test_field_output.py"
F=$CODE/film_transmittance.py
rm -rf $W; mkdir -p $W
common=(-load $CODE/examples/one_disk.txt -is_point -phi 0.2
        -eps Ag -eps_ref 1 -tfilm 0.1 -polarization x
        -res 80 -ks 5.235987756 7.853981634 -nfreqs 31
        -dsrc 0.4 -ddet 0.2 -dpml 0.3 -tpml 0.5
        -ScattPower -comp Ex -dft_margin_px 4
        -dft_nconsec 3 -dft_tol 1e-30 -maxt 31
        -tempname incident)
run_() {
	local name=$1 d=$W/$1; shift
	mkdir -p $d && cd $d
	cp $REF/film_transmittance-incidentrefl-flux.h5 $REF/film_transmittance-incidentrefl-flux.meta.json $REF/incident_inc_flux.npy .
	python $F "${common[@]}" "$@" -saveas sample > sample.log 2>&1; echo "$name rc=$?: $(grep -h 'finished at' sample.log)"
}
WL=$($T grid_wavelengths 5.235987756 7.853981634 31 15 30 | head -1)
WL2=$($T grid_wavelengths 5.235987756 7.853981634 31 5 20 | head -1)
run_ X
run_ Y -field_wavelengths $WL
run_ Z -field_wavelengths $WL2
echo "== Y (selected 15,30) vs X (full), same stop step"; $T selected $W/X $W/Y $WL
echo "== Z (selected 5,20) vs X (full), same stop step"; $T selected $W/X $W/Z $WL2
echo "== naturally stopped full run (ag_full) vs X at 15,30: truncation of the full run"; $T selected $W/X $REF $WL
echo "== naturally stopped selected run (ag_sel) vs X at 15,30"; $T selected $W/X $V/fields/ag_sel $WL
