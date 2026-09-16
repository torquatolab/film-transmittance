#!/bin/bash
# P4 tests 2-4: -no_fields and -field_wavelengths against full runs, and rejections.
#SBATCH --job-name=p4-fields
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --constraint=amd
#SBATCH --output=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p4/%x-%j.out
set -uo pipefail
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
CODE=${CODE:-/scratch/gpfs/TORQUATO/sd2402/film-worktrees/p4}
W=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p4/fields
T="python $CODE/tests/test_field_output.py"
F=$CODE/film_transmittance.py
rm -rf $W; mkdir -p $W
fail=0
# run_spectrum.sh settings (it cannot pass extra flags, so they are replicated here)
common=(-load $CODE/examples/one_disk.txt -is_point -phi 0.2
        -eps_ref 1 -tfilm 0.1 -polarization x
        -res 80 -ks 5.235987756 7.853981634 -nfreqs 31
        -dsrc 0.4 -ddet 0.2 -dpml 0.3 -tpml 0.5
        -ScattPower -comp Ex -dft_margin_px 4
        -dft_nconsec 3 -dft_tol 1e-8 -maxt 200
        -tempname incident)
# case DIR NP EPS [extra flags...]: reference + sample, both with the extra flags
case_() {
	local d=$W/$1 np=$2 eps=$3; shift 3
	local launch=(); (( np > 1 )) && launch=(mpirun -np $np)
	mkdir -p $d && cd $d
	"${launch[@]}" python $F "${common[@]}" -eps $eps "$@" -ref -saveas reference > reference.log 2>&1 || { echo "$d ref rc=$?"; fail=1; }
	"${launch[@]}" python $F "${common[@]}" -eps $eps "$@" -saveas sample > sample.log 2>&1 || { echo "$d sample rc=$?"; fail=1; }
	echo "$d: $(grep -h 'finished at' reference.log sample.log | tr '\n' ' ')"
}
steps() { grep -h 'finished at' $W/$1/sample.log; }

echo "== test 4: rejections"
cd $W && $T rejections $F || fail=1

echo "== test 2: -no_fields vs full (serial)"
case_ ag_full 1 Ag
case_ ag_nof 1 Ag -no_fields
case_ e4_full 1 4
case_ e4_nof 1 4 -no_fields
$T no_fields $W/ag_full $W/ag_nof || fail=1
$T no_fields $W/e4_full $W/e4_nof || fail=1

echo "== test 3: -field_wavelengths at grid frequencies (i = 15, 30 of 31)"
WL=$($T grid_wavelengths 5.235987756 7.853981634 31 15 30 | head -1)
echo "wavelengths: $WL"
case_ ag_sel 1 Ag -field_wavelengths $WL
$T selected $W/ag_full $W/ag_sel $WL || fail=1
case_ ag_full_np2 2 Ag
case_ ag_sel_np2 2 Ag -field_wavelengths $WL
$T selected $W/ag_full_np2 $W/ag_sel_np2 $WL || fail=1

echo "== informational: selection without the top grid frequency (i = 5, 20)"
WL2=$($T grid_wavelengths 5.235987756 7.853981634 31 5 20 | head -1)
echo "wavelengths: $WL2"
case_ ag_sel_low 1 Ag -field_wavelengths $WL2
$T selected $W/ag_full $W/ag_sel_low $WL2 || echo "(informational, not a pass criterion)"
echo "FIELDS OVERALL: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
