#!/usr/bin/env python
# ===========================================
# I1 integration checks for voxel input in film_transmittance.py and run_spectrum.sh.
# Plain script (no pytest); exits nonzero on any failure.  Runs FDTD, so call it from
# a compute node (tests/slurm/i1_*.slurm).  voxel_io / voxel_geometry are imported by
# film_transmittance.py from PYTHONPATH (Phase A: stubs; Phase B: the real modules).
#
#   compare OUT GOLDEN      regression: every .txt/.npy/.h5 under GOLDEN identical in OUT
#   tripwire DIR            no tripwire module in DIR was imported
#   smoke WORK              synthetic .npz through -ref and sample runs
#   refusals WORK           mismatched references and invalid flags are rejected (after smoke)
#   wrapper WORK            run_spectrum.sh with INPUT/VOXEL_SIZE (+CROP, NORMAL_AXIS)
# ===========================================

import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(CODE, "film_transmittance.py")
STEM = "film_transmittance-incidentrefl-flux"
META_KEYS = ("nprocs", "tempname", "resolution", "voxel_sha256", "crop", "voxel_size",
			 "normal_axis", "eps", "eps_ref", "cell", "fields_mode")
# Stored (normal, r0, r1) = (z, y, x) shape, deliberately all different so a wrong
# transpose shows up in the cell: expect Lx = 0.20, Ly = 0.12, h = 0.16.
STORED_SHAPE = (16, 12, 20)
VOXEL_SIZE = 0.01
COMMON = ["-eps", "2.4025", "-eps_ref", "1", "-polarization", "x", "-res", "100",
		  "-ks", "8.0553657784", "16.5346981768", "-nfreqs", "11",
		  "-dsrc", "0.4", "-ddet", "0.2", "-dpml", "0.3", "-tpml", "0.5",
		  "-ScattPower", "-comp", "Ex", "-dft_margin_px", "4",
		  "-dft_nconsec", "3", "-dft_tol", "1e-8", "-maxt", "200", "-tempname", "incident"]

FAILURES = []


def check(cond, what):
	print("{0}: {1}".format("PASS" if cond else "FAIL", what), flush=True)
	if not cond:
		FAILURES.append(what)
	return cond


def run(argv, cwd, log, env=None):
	with open(os.path.join(cwd, log), "w") as fh:
		rc = subprocess.call(argv, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env)
	with open(os.path.join(cwd, log)) as fh:
		return rc, fh.read()


def film(cwd, load, extra, saveas, log):
	return run([sys.executable, SCRIPT, "-load", load] + COMMON + extra + ["-saveas", saveas], cwd, log)


def energy_error(table):
	data = np.loadtxt(table)
	return data, float(np.max(np.abs(data[:, 1] + np.abs(data[:, 2]) - 1.0)))


def make_voxels(path, seed):
	rng = np.random.default_rng(seed)
	g = rng.random(STORED_SHAPE) < 0.4
	np.savez(path, g=g)
	return g


# ---------------------------------------------------------------- regression
def cmd_compare(out, golden):
	n = 0
	for root, _, files in os.walk(golden):
		for name in sorted(files):
			if not name.endswith((".txt", ".npy", ".h5")):
				continue
			gold = os.path.join(root, name)
			rel = os.path.relpath(gold, golden)
			new = os.path.join(out, rel)
			n += 1
			if not os.path.exists(new):
				check(False, "{0} missing".format(rel))
				continue
			if name.endswith(".h5"):
				rc = subprocess.call(["h5diff", gold, new])
				check(rc == 0, "{0} h5diff rc={1}".format(rel, rc))
			elif name.endswith(".txt"):
				with open(gold) as a, open(new) as b:
					la = [l for l in a if "File:" not in l]
					lb = [l for l in b if "File:" not in l]
				check(la == lb, "{0} identical (File: header lines excluded)".format(rel))
			else:
				with open(gold, "rb") as a, open(new, "rb") as b:
					check(a.read() == b.read(), "{0} bytewise identical".format(rel))
	extra = []
	for root, _, files in os.walk(out):
		for name in files:
			rel = os.path.relpath(os.path.join(root, name), out)
			if name.endswith((".txt", ".npy", ".h5")) and not os.path.exists(os.path.join(golden, rel)):
				extra.append(rel)
	check(not extra, "no extra .txt/.npy/.h5 files in OUT ({0})".format(extra))
	check(n > 0, "compared {0} golden files".format(n))


def cmd_tripwire(d):
	hits = glob.glob(os.path.join(d, "IMPORTED_*"))
	check(not hits, "no voxel module imported by non-voxel runs (markers: {0})".format(hits))


# ---------------------------------------------------------------- smoke
def cmd_smoke(work):
	os.makedirs(work, exist_ok=True)
	npz = os.path.join(work, "synthetic.npz")
	g = make_voxels(npz, 1)
	rc, log = film(work, npz, ["-voxel_size", str(VOXEL_SIZE), "-ref"], "reference", "reference.log")
	check(rc == 0, "reference rc={0}".format(rc))
	rc, slog = film(work, npz, ["-voxel_size", str(VOXEL_SIZE)], "sample", "sample.log")
	check(rc == 0, "sample rc={0}".format(rc))
	check("Initializing structure" in slog, "sample log shows structure initialisation (control for refusals)")

	with open(os.path.join(work, STEM + ".meta.json")) as fh:
		meta = json.load(fh)
	print("metadata:", json.dumps(meta))
	check(all(k in meta for k in META_KEYS), "metadata has keys {0}".format(META_KEYS))
	expect = [VOXEL_SIZE*STORED_SHAPE[2], VOXEL_SIZE*STORED_SHAPE[1], VOXEL_SIZE*STORED_SHAPE[0]]
	check(np.allclose(meta["cell"], expect, rtol=0, atol=1e-12),
		  "cell {0} == [Lx, Ly, h] from data {1}".format(meta["cell"], expect))
	check(meta["voxel_size"] == VOXEL_SIZE and meta["crop"] is None and meta["normal_axis"] == 0,
		  "voxel_size/crop/normal_axis recorded")
	check(isinstance(meta["voxel_sha256"], str) and len(meta["voxel_sha256"]) == 64, "voxel_sha256 recorded")
	check(meta["eps"] == "2.4025" and meta["eps_ref"] == "1" and meta["fields_mode"] == "all",
		  "eps/eps_ref strings and fields_mode")
	print("solid fraction of synthetic voxels = {0:.6f}".format(float(g.mean())))

	# the simulated cell itself: x/y metadata of the DFT volume spans Lx, Ly; z spans h + margins
	x = np.load(os.path.join(work, "sample_x.npy"))
	y = np.load(os.path.join(work, "sample_y.npy"))
	print("sample_x: n={0} [{1:.6f}, {2:.6f}]; sample_y: n={3} [{4:.6f}, {5:.6f}]".format(
		len(x), x.min(), x.max(), len(y), y.min(), y.max()))
	check(abs((x.max() - x.min()) - expect[0]) <= 1.01/100 and abs((y.max() - y.min()) - expect[1]) <= 1.01/100,
		  "DFT volume spans the data Lx, Ly (within one pixel)")
	table = os.path.join(work, "sample_trans-x.txt")
	with open(table) as fh:
		header = fh.read()
	check("a voxel array" in header, "table header names a voxel array")
	data, err = energy_error(table)
	print("T:", " ".join("{0:.5f}".format(v) for v in data[:, 1]))
	print("R:", " ".join("{0:.5f}".format(v) for v in data[:, 2]))
	print("max |T+|R|-1| = {0:.6e}".format(err))
	check(data.shape[0] == 11 and err <= 0.01, "energy balance max |T+|R|-1| = {0:.3e} <= 0.01".format(err))


# ---------------------------------------------------------------- refusals
def refused_before_stepping(rc, log, work, saveas, needle, what):
	ok = rc != 0 and needle in log and "Initializing structure" not in log \
		and not os.path.exists(os.path.join(work, saveas + "_trans-x.txt"))
	tail = [l for l in log.splitlines() if "REFUSING" in l or "ERROR" in l or "voxel input" in l or "aborting" in l]
	check(ok, "{0}: rc={1}, message {2!r}".format(what, rc, tail[-2:]))


def ref_copy(src, dst, edit=None):
	os.makedirs(dst, exist_ok=True)
	for name in (STEM + ".h5", STEM + ".meta.json", "incident_inc_flux.npy"):
		shutil.copy(os.path.join(src, name), dst)
	if edit is not None:
		path = os.path.join(dst, STEM + ".meta.json")
		with open(path) as fh:
			meta = json.load(fh)
		edit(meta)
		with open(path, "w") as fh:
			json.dump(meta, fh, indent=1)


def cmd_refusals(work):
	smoke = os.path.join(work, "..", "smoke")
	smoke = os.path.abspath(smoke)
	npz = os.path.join(smoke, "synthetic.npz")
	vs = ["-voxel_size", str(VOXEL_SIZE)]
	os.makedirs(work, exist_ok=True)

	d = os.path.join(work, "sha"); ref_copy(smoke, d)
	other = os.path.join(d, "other.npz"); make_voxels(other, 2)
	rc, log = film(d, other, vs, "sample", "sample.log")
	refused_before_stepping(rc, log, d, "sample", "voxel_sha256", "different voxel_sha256 refused")

	d = os.path.join(work, "crop"); ref_copy(smoke, d)
	rc, log = film(d, npz, vs + ["-crop", "0:16,0:12,0:18"], "sample", "sample.log")
	refused_before_stepping(rc, log, d, "sample", "crop:", "different crop refused")

	d = os.path.join(work, "voxel_size"); ref_copy(smoke, d)
	rc, log = film(d, npz, ["-voxel_size", "0.011"], "sample", "sample.log")
	refused_before_stepping(rc, log, d, "sample", "voxel_size:", "different voxel_size refused")

	d = os.path.join(work, "cell"); ref_copy(smoke, d, lambda m: m.__setitem__("cell", [0.2, 0.12, 0.17]))
	rc, log = film(d, npz, vs, "sample", "sample.log")
	refused_before_stepping(rc, log, d, "sample", "cell:", "different cell (only) refused")

	d = os.path.join(work, "normal_axis"); ref_copy(smoke, d, lambda m: m.__setitem__("normal_axis", 1))
	rc, log = film(d, npz, vs, "sample", "sample.log")
	refused_before_stepping(rc, log, d, "sample", "normal_axis:", "different normal_axis refused")

	def _old(m):
		for k in list(m):
			if k not in ("nprocs", "tempname", "resolution"):
				del m[k]
	d = os.path.join(work, "old_ref"); ref_copy(smoke, d, _old)
	rc, log = film(d, npz, vs, "sample", "sample.log")
	refused_before_stepping(rc, log, d, "sample", "missing from the reference", "old reference without voxel keys refused for voxel input")

	d = os.path.join(work, "flags"); os.makedirs(d, exist_ok=True)
	for extra, needle, what in (
			(["-voxel_size", str(VOXEL_SIZE), "-is_point"], "-is_point", "-is_point rejected"),
			(["-voxel_size", str(VOXEL_SIZE), "-phi", "0.2"], "-phi", "-phi 0.2 rejected"),
			(["-voxel_size", str(VOXEL_SIZE), "-scale2sim", "2"], "-scale2sim", "-scale2sim 2 rejected"),
			(["-voxel_size", str(VOXEL_SIZE), "-tfilm", "0.1"], "-tfilm", "-tfilm 0.1 (data 0.16) rejected"),
			([], "-voxel_size > 0 is required", "missing -voxel_size rejected"),
			(["-voxel_size", "0"], "-voxel_size > 0 is required", "-voxel_size 0 rejected"),
			(["-voxel_size", str(VOXEL_SIZE), "-crop", "0:16,0:12,0:25"], "exceeds stored axis 2", "-crop beyond data rejected"),
			(["-voxel_size", str(VOXEL_SIZE), "-crop", "0:16,0:12"], "-crop:", "malformed -crop rejected")):
		rc, log = film(d, npz, extra + ["-ref"], "flagref", "flag.log")
		refused_before_stepping(rc, log, d, "flagref", needle, what)
	rc, log = film(d, os.path.join(CODE, "examples", "one_disk.txt"),
				   ["-is_point", "-phi", "0.2", "-voxel_size", "0.01", "-ref"], "flagref", "flag.log")
	refused_before_stepping(rc, log, d, "flagref", "only apply to voxel input", "-voxel_size with a .txt pattern rejected")

	# positive controls: matching explicit -tfilm accepted, and a crop builds the cropped cell
	rc, log = film(d, npz, ["-voxel_size", str(VOXEL_SIZE), "-tfilm", "0.16", "-test2", "-ref"], "tf", "tf.log")
	check(rc == 0 and "Initializing structure" in log, "-tfilm 0.16 matching the data accepted (-test2, rc={0})".format(rc))
	rc, log = film(d, npz, ["-voxel_size", str(VOXEL_SIZE), "-crop", "2:14,1:11,3:18", "-test2"], "cr", "cr.log")
	full = np.load(os.path.join(d, "tf_epsilon.npy")).shape if os.path.exists(os.path.join(d, "tf_epsilon.npy")) else None
	eps = np.load(os.path.join(d, "cr_epsilon.npy")) if rc == 0 else None
	shape = None if eps is None else eps.shape
	# oriented [x, y, z]: full [20, 12, 16] voxels, cropped [15, 10, 12] -> 5 fewer x and
	# 2 fewer y pixels at 1 pixel per voxel
	print("epsilon grid: full {0}, cropped {1}".format(full, shape))
	check(rc == 0 and shape is not None and full is not None
		  and (full[0] - shape[0], full[1] - shape[1]) == (5, 2),
		  "-crop 2:14,1:11,3:18 shrinks the lateral grid by (5, 2) pixels (rc={0})".format(rc))
	if eps is not None:
		print("epsilon range in cropped test2: [{0:.4f}, {1:.4f}]".format(float(eps.min()), float(eps.max())))


# ---------------------------------------------------------------- wrapper
def cmd_wrapper(work):
	os.makedirs(work, exist_ok=True)
	npz = os.path.join(work, "synthetic.npz")
	make_voxels(npz, 1)
	wrap = os.path.join(CODE, "run_spectrum.sh")
	env = dict(os.environ, INPUT=npz)
	env.pop("VOXEL_SIZE", None)
	rc, log = run(["bash", wrap, os.path.join(work, "no_size")], work, "no_size.out", env)
	check(rc != 0 and "INPUT needs VOXEL_SIZE" in log, "INPUT without VOXEL_SIZE rejected (rc={0})".format(rc))

	env = dict(os.environ, INPUT=os.path.relpath(npz, work), VOXEL_SIZE=str(VOXEL_SIZE),
			   CROP="0:16,0:12,0:16", NORMAL_AXIS="0")
	out = os.path.join(work, "run")
	rc, log = run(["bash", wrap, out], work, "wrapper.out", env)
	print(log)
	check(rc == 0, "run_spectrum.sh INPUT/VOXEL_SIZE/CROP/NORMAL_AXIS rc={0}".format(rc))
	if rc != 0:
		return
	with open(os.path.join(out, STEM + ".meta.json")) as fh:
		meta = json.load(fh)
	print("metadata:", json.dumps(meta))
	check(meta["voxel_size"] == VOXEL_SIZE and meta["crop"] == [[0, 16], [0, 12], [0, 16]]
		  and meta["normal_axis"] == 0 and meta["resolution"] == 100 and meta["eps"] == "2.4025",
		  "wrapper passed -load/-voxel_size/-crop/-normal_axis/-res 100/-eps 2.4025")
	check(np.allclose(meta["cell"], [0.16, 0.12, 0.16], rtol=0, atol=1e-12), "wrapper cell {0}".format(meta["cell"]))
	data, err = energy_error(os.path.join(out, "sample_trans-x.txt"))
	print("wrapper: {0} frequencies, wavelength [{1:.5f}, {2:.5f}], max |T+|R|-1| = {3:.6e}".format(
		data.shape[0], data[:, 0].min(), data[:, 0].max(), err))
	check(data.shape[0] == 101 and abs(data[:, 0].min() - 0.38) < 1e-6 and abs(data[:, 0].max() - 0.78) < 1e-6,
		  "wrapper spectrum: 101 frequencies over 380-780 nm")
	check(err <= 0.01, "wrapper energy balance {0:.3e} <= 0.01".format(err))


if __name__ == "__main__":
	cmds = {"compare": cmd_compare, "tripwire": cmd_tripwire, "smoke": cmd_smoke,
			"refusals": cmd_refusals, "wrapper": cmd_wrapper}
	if len(sys.argv) < 3 or sys.argv[1] not in cmds:
		raise SystemExit(__doc__ or "usage: test_voxel_integration.py {compare,tripwire,smoke,refusals,wrapper} ARGS")
	cmds[sys.argv[1]](*sys.argv[2:])
	print("FAILURES: {0}".format(len(FAILURES)))
	for f in FAILURES:
		print("  " + f)
	sys.exit(1 if FAILURES else 0)
