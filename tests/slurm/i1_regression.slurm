#!/bin/bash
#SBATCH --job-name=i1-regress
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:40:00
#SBATCH --constraint=amd
#SBATCH --output=%x-%j.out
# I1 test 1: existing examples bitwise-identical to the b71f61b golden outputs, and no
# voxel module imported.  Required env: OUT (a directory whose parent holds packing.txt,
# as harness.sh loads ../../packing.txt from OUT/D), HARNESS (a COPY of
# film-audit/harness.sh), GOLDEN (film-audit/out_new), GOLDEN_RC (the harness stdout that
# produced GOLDEN), TRIPWIRE (directory of raising voxel_io/voxel_geometry/voxel_crop).
set -uo pipefail
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
CODE=${CODE:-$SLURM_SUBMIT_DIR}
rm -f "$TRIPWIRE"/IMPORTED_*
export PYTHONPATH="$TRIPWIRE${PYTHONPATH:+:$PYTHONPATH}"
bash "$HARNESS" "$CODE" "$OUT" | tee "$OUT.rc"
diff <(grep ' rc=\|^DONE' "$GOLDEN_RC") "$OUT.rc" && echo "PASS: harness return codes identical to golden" || { echo "FAIL: harness return codes differ"; fail=1; }
python "$CODE/tests/test_voxel_integration.py" compare "$OUT" "$GOLDEN" | grep -v '^PASS' ; rc1=${PIPESTATUS[0]}
python "$CODE/tests/test_voxel_integration.py" tripwire "$TRIPWIRE"; rc2=$?
echo "compare rc=$rc1 tripwire rc=$rc2"
exit $(( ${fail:-0} || rc1 || rc2 ))
