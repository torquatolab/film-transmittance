"""Checks for the -no_fields / -field_wavelengths options of film_transmittance.py.

Plain script, no pytest; exits nonzero on failure.  Run on a compute node
(tests/slurm/p4_*.slurm drive it).  Subcommands:

	regression OUT GOLD
		Every non-log output of harness.sh in OUT equals the golden one in GOLD:
		same file list; .npy array_equal; .txt identical except a '# File:' header
		line; .h5 via h5diff (rc 0); checkpoint_meta.json equal except wall_clock;
		anything else byte-identical.
		.log, .out and .meta.json files are excluded.
	no_fields FULL_DIR NOF_DIR
		T/R tables of a full and a -no_fields run agree to <= 0.01, the table format
		is the same, and NOF_DIR holds no __ka-* files but the _x/_y/_z/_w metadata.
	selected FULL_DIR SEL_DIR WL1 [WL2 ...]
		The __ka-* arrays of a -field_wavelengths run equal the full run's arrays
		(same file names) to <= 1e-5 relative (contract amendment A4), SEL_DIR has no other __ka-* files,
		and the T/R table has the same format.
	rejections SCRIPT
		Invalid flag combinations exit nonzero with a clear message before the
		simulation is built.
	grid_wavelengths KMIN KMAX NFREQS I [I ...]
		Print the wavelengths 1/f_i of Meep's DFT frequency grid (for the slurm script).
"""

import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

EXCLUDE_SUFFIXES = (".log", ".out", ".meta.json")
DT_TOL = 0.01
# Contract amendment A4: runs that stop independently differ by ~3e-6 (the full run
# stops earlier); forced to the same stop step the arrays are bitwise equal.
REL_TOL = 1e-5
H5DIFF = os.path.join(os.path.dirname(sys.executable), "h5diff")


def _fail(msg):
	print("FAIL: " + msg, flush=True)
	return 1


def _files(root):
	out = set()
	for dirpath, _, names in os.walk(root):
		for n in names:
			if n.endswith(EXCLUDE_SUFFIXES):
				continue
			out.add(os.path.relpath(os.path.join(dirpath, n), root))
	return out


def regression(out, gold):
	nerr = 0
	a, b = _files(out), _files(gold)
	if a != b:
		nerr += _fail("file lists differ: only in out {0}; only in gold {1}".format(
			sorted(a - b), sorted(b - a)))
	counts = {}
	for rel in sorted(a & b):
		p, q = os.path.join(out, rel), os.path.join(gold, rel)
		kind = os.path.splitext(rel)[1] or "other"
		counts[kind] = counts.get(kind, 0) + 1
		if rel.endswith(".npy"):
			x, y = np.load(p), np.load(q)
			ok = x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
		elif rel.endswith(".txt"):
			with open(p) as fh:
				lp = fh.read().splitlines()
			with open(q) as fh:
				lq = fh.read().splitlines()
			ok = len(lp) == len(lq) and all(
				u == v or (u.startswith("#") and v.startswith("#")
						   and u.strip("# \t").startswith("File:")
						   and v.strip("# \t").startswith("File:"))
				for u, v in zip(lp, lq))
		elif rel.endswith(".h5"):
			ok = subprocess.run([H5DIFF, p, q], stdout=subprocess.DEVNULL,
								stderr=subprocess.DEVNULL).returncode == 0
		elif os.path.basename(rel) == "checkpoint_meta.json":
			# wall_clock is the time the checkpoint was written; everything else must match
			with open(p) as fh:
				jp = json.load(fh)
			with open(q) as fh:
				jq = json.load(fh)
			jp.pop("wall_clock", None)
			jq.pop("wall_clock", None)
			ok = jp == jq
		else:
			with open(p, "rb") as fh:
				bp = fh.read()
			with open(q, "rb") as fh:
				bq = fh.read()
			ok = bp == bq
		if not ok:
			nerr += _fail("differs: " + rel)
	print("regression: compared {0} common files {1}; out {2}, gold {3}; {4} failures".format(
		len(a & b), counts, len(a), len(b), nerr))
	return nerr


def _table(d):
	paths = sorted(glob.glob(os.path.join(d, "*_trans-*.txt")))
	if len(paths) != 1:
		raise SystemExit("expected one *_trans-*.txt in {0}, found {1}".format(d, paths))
	with open(paths[0]) as fh:
		lines = fh.read().splitlines()
	header = [l for l in lines if l.startswith("#")]
	body = [l for l in lines if not l.startswith("#")]
	return os.path.basename(paths[0]), header, body, np.loadtxt(paths[0])


def _same_format(full, other):
	nerr = 0
	nf, hf, bf, tf = _table(full)
	no, ho, bo, to = _table(other)
	if nf != no:
		nerr += _fail("table names differ: {0} vs {1}".format(nf, no))
	if hf != ho:
		nerr += _fail("table headers differ")
	num = re.compile(r"^-?\d\.\d{6}e[+-]\d{2}$")
	for label, body in (("full", bf), ("other", bo)):
		if not all(all(num.match(c) for c in l.split("\t")) for l in body):
			nerr += _fail(label + " table rows are not %1.6e tab-separated")
	if tf.shape != to.shape:
		nerr += _fail("table shapes differ: {0} vs {1}".format(tf.shape, to.shape))
		return nerr, None
	if not np.array_equal(tf[:, 0], to[:, 0]):
		nerr += _fail("wavelength columns differ")
	dT = float(np.max(np.abs(tf[:, 1] - to[:, 1])))
	dR = float(np.max(np.abs(tf[:, 2] - to[:, 2])))
	print("table {0}: shape {1}, max|dT| = {2:.3e}, max|dR| = {3:.3e}, "
		  "max|dInc|/|Inc| = {4:.3e}".format(nf, tf.shape, dT, dR,
		  float(np.max(np.abs(tf[:, -1] - to[:, -1]) / np.abs(tf[:, -1])))))
	return nerr, (dT, dR)


def _ka(d):
	return sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*__ka-*")))


def no_fields(full, nof):
	nerr, d = _same_format(full, nof)
	if d is not None and max(d) > DT_TOL:
		nerr += _fail("-no_fields T/R differ by more than {0}".format(DT_TOL))
	ka = _ka(nof)
	if ka:
		nerr += _fail("-no_fields wrote field files: {0}".format(ka))
	for s in ("x", "y", "z", "w"):
		if not glob.glob(os.path.join(nof, "sample_{0}.npy".format(s))):
			nerr += _fail("missing metadata sample_{0}.npy".format(s))
	print("no_fields: {0} __ka files in full run, {1} in -no_fields run; {2} failures".format(
		len(_ka(full)), len(ka), nerr))
	return nerr


def selected(full, sel, wavelengths):
	nerr, _ = _same_format(full, sel)
	names = set()
	for wl in wavelengths:
		k = round(2.0*np.pi/wl, 4)
		names.update(os.path.basename(p) for p in
					 glob.glob(os.path.join(sel, "*__ka-{0:.4f}-*.npy".format(k))))
		if not glob.glob(os.path.join(sel, "*__ka-{0:.4f}-*.npy".format(k))):
			nerr += _fail("no field file for lambda {0!r} (k {1:.4f})".format(wl, k))
	extra = sorted(set(_ka(sel)) - names)
	if extra:
		nerr += _fail("unexpected field files: {0}".format(extra))
	for n in sorted(names):
		fp = os.path.join(full, n)
		if not os.path.exists(fp):
			nerr += _fail("full run has no " + n)
			continue
		a, b = np.load(os.path.join(sel, n)), np.load(fp)
		if a.shape != b.shape:
			nerr += _fail("{0}: shapes {1} vs {2}".format(n, a.shape, b.shape))
			continue
		rel = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
		print("{0}: shape {1}, max|a-b| = {2:.3e}, max|b| = {3:.3e}, rel = {4:.3e}".format(
			n, a.shape, float(np.max(np.abs(a - b))), float(np.max(np.abs(b))), rel))
		if not rel <= REL_TOL:
			nerr += _fail("{0}: relative difference {1:.3e} > {2:g}".format(n, rel, REL_TOL))
	print("selected: {0} arrays compared; {1} failures".format(len(names), nerr))
	return nerr


def rejections(script):
	base = [sys.executable, script, "-load", "none.txt", "-saveas", "rej"]
	cases = [
		(["-ScattPower", "-no_fields", "-field_wavelengths", "0.9"], "mutually exclusive"),
		(["-ScattPower", "-no_fields", "-JouleHeating"], "-JouleHeating"),
		(["-ScattPower", "-field_wavelengths", "0.9", "-JouleHeating"], "-JouleHeating"),
		(["-ScattPower", "-field_wavelengths", "-0.9"], "finite and positive"),
		(["-ScattPower", "-field_wavelengths", "0.9", "0.9"], "duplicate"),
		(["-no_fields"], "requires -ScattPower"),
	]
	nerr = 0
	for extra, needle in cases:
		r = subprocess.run(base + extra, capture_output=True, text=True)
		msg = (r.stdout + r.stderr).strip().splitlines()
		last = msg[-1] if msg else ""
		built = "Geometry:" in r.stdout
		print("{0}: rc {1}, message {2!r}, simulation started: {3}".format(
			" ".join(extra), r.returncode, last, built))
		if r.returncode == 0 or needle not in last or built:
			nerr += _fail("rejection of " + " ".join(extra))
	print("rejections: {0} cases; {1} failures".format(len(cases), nerr))
	return nerr


def grid_wavelengths(kmin, kmax, nfreqs, idx):
	# Same arithmetic as main() and Meep's fix_dft_args (np.linspace).
	wmin, wmax = 2.0*np.pi/kmax, 2.0*np.pi/kmin
	fmin, fmax = 1.0/wmax, 1.0/wmin
	fcen, fwidth = 0.5*(fmax + fmin), fmax - fmin
	f = np.linspace(fcen - 0.5*fwidth, fcen + 0.5*fwidth, nfreqs)
	print(" ".join(repr(float(1.0/f[i])) for i in idx))
	return 0


if __name__ == "__main__":
	a = sys.argv[1:]
	if not a:
		raise SystemExit(__doc__)
	cmd = a[0]
	if cmd == "regression" and len(a) == 3:
		rc = regression(a[1], a[2])
	elif cmd == "no_fields" and len(a) == 3:
		rc = no_fields(a[1], a[2])
	elif cmd == "selected" and len(a) >= 4:
		rc = selected(a[1], a[2], [float(x) for x in a[3:]])
	elif cmd == "rejections" and len(a) == 2:
		rc = rejections(a[1])
	elif cmd == "grid_wavelengths" and len(a) >= 5:
		rc = grid_wavelengths(float(a[1]), float(a[2]), int(a[3]), [int(x) for x in a[4:]])
	else:
		raise SystemExit(__doc__)
	print("RESULT {0}: {1}".format(cmd, "PASS" if rc == 0 else "FAIL"))
	sys.exit(1 if rc else 0)
