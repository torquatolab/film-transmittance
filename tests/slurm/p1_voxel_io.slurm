#!/bin/bash
#SBATCH --job-name=p1-voxel-io
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --constraint=amd
#SBATCH --output=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p1/p1_voxel_io-%j.out
#SBATCH --error=/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p1/p1_voxel_io-%j.out

# Tests for voxel_io.py (package P1). Submit from anywhere:
#   sbatch tests/slurm/p1_voxel_io.slurm
# Step 1 hashes every data file with structural_color's procfile.load_indicator
# in the beetle env (no film code imported); step 2 runs tests/test_voxel_io.py
# in the film-meep env (no structural_color code imported) and compares.

set -euo pipefail
WT=${WT:-/scratch/gpfs/TORQUATO/sd2402/film-worktrees/p1}
OUT=${OUT:-/scratch/gpfs/TORQUATO/sd2402/film-audit/voxel/p1}
BEETLE=/home/sd2402/torquato-scratch-sd2402/structural_color/beetle/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"

echo "== reference hashes (beetle env, procfile.load_indicator) =="
# REUSE_REF=1 keeps an existing reference_hashes.json if it has 660 error-free entries.
if [ "${REUSE_REF:-0}" = 1 ] && "$BEETLE" -c "import json,sys; d=json.load(open(sys.argv[1])); n=len(d); e=sum('error' in v for v in d.values()); print('reusing reference: entries', n, 'errors', e); sys.exit(0 if n == 660 and e == 0 else 1)" "$OUT/reference_hashes.json"; then
	:
else
"$BEETLE" - "$OUT/reference_hashes.json" <<'PY'
import glob, hashlib, json, os, sys
sys.path.insert(0, "/home/sd2402/torquato-scratch-sd2402/structural_color/src")
import numpy as np
import tifffile
from procfile import load_indicator
D = "/home/sd2402/torquato-scratch-sd2402/structural_color/data"
files = sorted(glob.glob(os.path.join(D, "*.tif"))) + sorted(glob.glob(os.path.join(D, "benchmarks", "*.npz"))) + sorted(glob.glob(os.path.join(D, "shu", "*.npz")))
out = {}
for p in files:
	p = os.path.abspath(p)
	try:
		a = load_indicator(p)
		out[p] = {"shape": [int(n) for n in a.shape], "packbits_sha256": hashlib.sha256(np.packbits(a.astype(bool)).tobytes()).hexdigest()}
	except Exception as e:
		out[p] = {"error": f"{type(e).__name__}: {e}"}
with open(sys.argv[1], "w") as f:
	json.dump(out, f, indent=1)
print(f"tifffile {tifffile.__version__}; numpy {np.__version__}; hashed {len(out)} files; errors {sum('error' in v for v in out.values())}", flush=True)
PY
fi

echo "== film-meep env tests =="
export PATH=/scratch/gpfs/TORQUATO/sd2402/conda_envs/film-meep/bin:$PATH
python -c "import numpy, PIL; print('numpy', numpy.__version__, 'Pillow', PIL.__version__)"
python -c "import tifffile" 2>/dev/null && echo "tifffile present" || echo "tifffile absent"
python "$WT/tests/test_voxel_io.py" -outdir "$OUT" -reference "$OUT/reference_hashes.json"
