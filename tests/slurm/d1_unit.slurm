#!/bin/bash
#SBATCH --job-name=d1-unit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --constraint=amd
#SBATCH --output=%x-%j.out
# D1: select_crops and average_spectra.py unit tests, bash -n of the wrappers, and
# run_crops.sh DRY=1 on DarkGreen.tif (N=5).  Required env: WORK (output directory).
set -uo pipefail
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
CODE=${CODE:-$SLURM_SUBMIT_DIR}
STACK=${STACK:-/home/sd2402/torquato-scratch-sd2402/structural_color/data/DarkGreen.tif}
status=0
for f in run_spectrum.sh run_crops.sh; do
    bash -n "$CODE/$f" && echo "PASS: bash -n $f" || { echo "FAIL: bash -n $f"; status=1; }
done
/usr/bin/time -v python "$CODE/tests/test_voxel_crop.py" 2> "$WORK/test_voxel_crop.time"; rc=$?
grep -E "Elapsed|Maximum resident" "$WORK/test_voxel_crop.time"; echo "===== test_voxel_crop rc=$rc"; (( rc )) && status=1
python "$CODE/tests/test_average_spectra.py" "$WORK/average"; rc=$?; echo "===== test_average_spectra rc=$rc"; (( rc )) && status=1
mkdir -p "$WORK/crops"
cd "$WORK/crops"
env -u MAXT VOXEL_SIZE=0.01 DRY=1 bash "$CODE/run_crops.sh" "$STACK" 5 out > no_maxt.out 2>&1; rc=$?
grep -q "needs MAXT" no_maxt.out && (( rc != 0 )) && echo "PASS: run_crops.sh without MAXT rejected" || { echo "FAIL: run_crops.sh without MAXT (rc=$rc)"; cat no_maxt.out; status=1; }
VOXEL_SIZE=0.01 MAXT=200 BANDS=20 DRY=1 bash "$CODE/run_crops.sh" "$STACK" 5 out > dry.out 2> dry.err; rc=$?
cat dry.out; cat dry.err
n=$(grep -c '^crop_[0-9]*: ' dry.out); nd=$(grep '^crop_[0-9]*: ' dry.out | cut -d' ' -f2 | sort -u | wc -l)
echo "default separation: run_crops.sh DRY=1 rc=$rc printed $n crops ($nd distinct)"
# largest separation that still gives 5 crops on this stack
for sep in 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1; do
    k=$(python "$CODE/voxel_crop.py" "$STACK" --n 5 --min_offset_sep $sep 2>/dev/null | wc -l)
    echo "min_offset_sep $sep: $k crops"
    (( k == 5 )) && break
done
MIN_OFFSET_SEP=$sep VOXEL_SIZE=0.01 MAXT=200 BANDS=20 DRY=1 bash "$CODE/run_crops.sh" "$STACK" 5 out > dry5.out 2> dry5.err; rc=$?
cat dry5.out dry5.err
n=$(grep -c '^crop_[0-9]*: ' dry5.out); nd=$(grep '^crop_[0-9]*: ' dry5.out | cut -d' ' -f2 | sort -u | wc -l)
(( rc == 0 && n == 5 && nd == 5 )) && echo "PASS: run_crops.sh DRY=1 MIN_OFFSET_SEP=$sep printed 5 distinct crops" || { echo "FAIL: run_crops.sh DRY=1 MIN_OFFSET_SEP=$sep rc=$rc crops=$n distinct=$nd"; status=1; }
[[ ! -e out ]] && echo "PASS: DRY=1 created no output directory" || { echo "FAIL: DRY=1 created out/"; status=1; }
# the CLI --json on the same stack, and select_crops(n=1) == best_crop on the real stack
python "$CODE/voxel_crop.py" "$STACK" --n 5 --json crops.json > cli.out 2> cli.err; rc=$?
cmp -s cli.out <(grep '^crop_[0-9]*: ' dry.out | cut -d' ' -f2) && (( rc == 0 )) && echo "PASS: voxel_crop.py CLI output == run_crops.sh crops" || { echo "FAIL: CLI output differs (rc=$rc)"; status=1; }
python - "$CODE" "$STACK" <<'PY' || status=1
import json, sys
sys.path.insert(0, sys.argv[1])
import voxel_io, voxel_crop
v, _ = voxel_io.load_voxels(sys.argv[2])
b = voxel_crop.best_crop(v)
c = voxel_crop.select_crops(v, 1)
j = json.load(open("crops.json"))
ok = c[0] == b and j["crops"][0]["slices"] == b["slices"]
print("%s: DarkGreen select_crops(n=1) == best_crop %s (score %.6f)" % ("PASS" if ok else "FAIL", b["slices"], b["score"]))
for i, k in enumerate(j["crops"]):
    print("crop %d %s score %.6f kept %.4f per_axis %s" % (i, k["crop"], k["score"], k["kept_fraction"],
        {a: round(d["ratio"], 4) for a, d in k["per_axis"].items()}))
sys.exit(0 if ok else 1)
PY
echo "status=$status"
exit $status
