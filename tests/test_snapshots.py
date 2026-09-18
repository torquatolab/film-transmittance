#!/usr/bin/env python
# ===========================================
# D1 checks of -snapshot_dt and the voxel "stopped:" header line in film_transmittance.py.
# Plain script (no pytest); exits nonzero on any failure.  Runs FDTD, so call it from a
# compute node (tests/slurm/d1_snapshots.slurm).
#
#   python tests/test_snapshots.py WORK [RANKS ...]     (default RANKS: 1 2)
#
# Synthetic 20x16x16 voxel film (normal axis 0), res 100, eps 2.4025/1, 201 frequencies
# over 380-780 nm, -no_fields.  For every rank count (mpirun from PATH for > 1 rank):
#   - final table with -snapshot_dt is bytewise identical to the table without it;
#   - snapshot files appear exactly at k*DT <= stop time;
#   - the snapshot at the stop time equals the final table apart from the "# stopped:" line;
#   - -dft_tol 1e-20 -maxt 49.999 stops at the ceiling and says so in the header;
#   - -ref with -snapshot_dt writes no snapshots.
# Once (1 rank): a run stopping on DFT convergence says "dft converged"; argument rejections.
# ===========================================

import glob
import os
import re
import subprocess
import sys

import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(CODE, "film_transmittance.py")
COMMON = ["-voxel_size", "0.01", "-eps", "2.4025", "-eps_ref", "1", "-polarization", "x",
		  "-res", "100", "-ks", "8.0553657784", "16.5346981768", "-nfreqs", "201",
		  "-dsrc", "0.4", "-ddet", "0.2", "-dpml", "0.3", "-tpml", "1.5",
		  "-ScattPower", "-comp", "Ex", "-no_fields", "-dft_nconsec", "3", "-tempname", "incident"]
CEILING = ["-dft_tol", "1e-20", "-maxt", "49.999"]

FAILURES = []


def check(cond, what):
	print("{0}: {1}".format("PASS" if cond else "FAIL", what), flush=True)
	if not cond:
		FAILURES.append(what)
	return cond


def film(cwd, ranks, args, log):
	launcher = [] if ranks == 1 else ["mpirun", "-np", str(ranks)]
	with open(os.path.join(cwd, log), "w") as fh:
		rc = subprocess.call(launcher + [sys.executable, SCRIPT] + args, cwd=cwd,
							 stdout=fh, stderr=subprocess.STDOUT)
	with open(os.path.join(cwd, log)) as fh:
		return rc, fh.read()


def read_bytes(path):
	with open(path, "rb") as fh:
		return fh.read()


def stop_time(log):
	m = re.findall(r"run \d+ finished at t = ([0-9.eE+-]+) \((\d+) timesteps\)", log)
	return float(m[-1][0]) if m else None


def check_snapshots(d, saveas, log, dt, ranks):
	"""Snapshot file set and the stop-time snapshot vs the final table.  Returns the stop time."""
	t_stop = stop_time(log)
	final = os.path.join(d, saveas + "_trans-x.txt")
	snapdir = os.path.join(d, saveas + "_snapshots")
	files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(snapdir, "*")))
	expect = ["trans-x-t{0:.2f}.txt".format(k*dt) for k in range(1, int(np.floor(t_stop/dt + 1e-9)) + 1)]
	print("{0} (ranks {1}): stop t = {2}, snapshots {3}".format(saveas, ranks, t_stop, files))
	check(sorted(files) == sorted(expect), "{0} ranks {1}: snapshot files == k*{2:g} <= {3} ({4})".format(
		saveas, ranks, dt, t_stop, expect))
	final_lines = read_bytes(final).splitlines(keepends=True)
	stripped = b"".join(l for l in final_lines if not l.startswith(b"# stopped:"))
	for name in files:
		data = np.loadtxt(os.path.join(snapdir, name))
		if data.shape != (201, 4):
			check(False, "{0}: shape {1} == (201, 4)".format(name, data.shape))
	at_stop = os.path.join(snapdir, "trans-x-t{0:.2f}.txt".format(t_stop))
	if abs(t_stop/dt - round(t_stop/dt)) < 1e-9:
		check(os.path.exists(at_stop) and read_bytes(at_stop) == stripped,
			  "{0} ranks {1}: snapshot at the stop time t = {2} is bytewise the final table "
			  "without its '# stopped:' line".format(saveas, ranks, t_stop))
		earlier = [os.path.join(snapdir, n) for n in files if os.path.join(snapdir, n) != at_stop]
		if earlier:
			d_max = max(float(np.max(np.abs(np.loadtxt(p)[:, 1] - np.loadtxt(final)[:, 1]))) for p in earlier)
			print("  max |T(snapshot) - T(final)| over earlier snapshots = {0:.3e}".format(d_max))
	return t_stop


def cmd_rank(work, ranks, npz):
	d = os.path.join(work, "np{0}".format(ranks))
	os.makedirs(d, exist_ok=True)
	base = ["-load", npz] + COMMON
	rc, log = film(d, ranks, base + CEILING + ["-snapshot_dt", "10", "-ref", "-saveas", "reference"], "reference.log")
	check(rc == 0, "ranks {0}: reference rc={1}".format(ranks, rc))
	check(not os.path.exists(os.path.join(d, "reference_snapshots")) and "no snapshots for a -ref run" in log,
		  "ranks {0}: -snapshot_dt is a no-op for -ref".format(ranks))
	rc, log_plain = film(d, ranks, base + CEILING + ["-saveas", "plain"], "plain.log")
	check(rc == 0, "ranks {0}: sample without -snapshot_dt rc={1}".format(ranks, rc))
	check("snapshots every" not in log_plain and "-snapshot_dt" not in log_plain
		  and not glob.glob(os.path.join(d, "plain_snapshots*")),
		  "ranks {0}: no snapshot output or log line without -snapshot_dt".format(ranks))
	plain = read_bytes(os.path.join(d, "plain_trans-x.txt"))
	check(b"\n# stopped: maxt ceiling\n# wavelength" in plain and b"# stopped: dft converged" not in plain,
		  "ranks {0}: header says 'stopped: maxt ceiling' above the column names".format(ranks))
	check("WARNING: run reached the -maxt" in log_plain, "ranks {0}: log has the -maxt warning".format(ranks))
	rc, log10 = film(d, ranks, base + CEILING + ["-snapshot_dt", "10", "-saveas", "snap10"], "snap10.log")
	check(rc == 0, "ranks {0}: sample -snapshot_dt 10 rc={1}".format(ranks, rc))
	check(read_bytes(os.path.join(d, "snap10_trans-x.txt")) == plain,
		  "ranks {0}: final table with -snapshot_dt 10 bytewise == without".format(ranks))
	t10 = check_snapshots(d, "snap10", log10, 10.0, ranks)
	check(t10 is not None and abs(t10 - 50.0) < 1e-9, "ranks {0}: stop time {1} == 50 (first step past -maxt 49.999)".format(ranks, t10))
	check(stop_time(log_plain) == t10, "ranks {0}: same stop time with and without snapshots".format(ranks))
	if ranks == 1:
		rc, log7 = film(d, ranks, base + CEILING + ["-snapshot_dt", "7", "-saveas", "snap7"], "snap7.log")
		check(rc == 0 and read_bytes(os.path.join(d, "snap7_trans-x.txt")) == plain,
			  "ranks 1: final table with -snapshot_dt 7 bytewise == without (rc={0})".format(rc))
		check_snapshots(d, "snap7", log7, 7.0, ranks)
	return np.loadtxt(os.path.join(d, "plain_trans-x.txt"))


def cmd_once(work, npz):
	d = os.path.join(work, "np1")
	base = ["-load", npz] + COMMON
	rc, log = film(d, 1, base + ["-dft_tol", "1e-8", "-maxt", "200", "-snapshot_dt", "2", "-saveas", "conv"], "conv.log")
	check(rc == 0, "converging run rc={0}".format(rc))
	table = read_bytes(os.path.join(d, "conv_trans-x.txt"))
	t_stop = stop_time(log)
	check(b"\n# stopped: dft converged\n# wavelength" in table and "WARNING: run reached the -maxt" not in log
		  and t_stop is not None and t_stop < 200,
		  "-dft_tol 1e-8 -maxt 200 stops at t = {0} and the header says 'stopped: dft converged'".format(t_stop))
	check_snapshots(d, "conv", log, 2.0, 1)
	rc, log = film(d, 1, base + ["-dft_tol", "1e-8", "-maxt", "200", "-saveas", "conv_plain"], "conv_plain.log")
	check(rc == 0 and read_bytes(os.path.join(d, "conv_plain_trans-x.txt")) == table,
		  "converging run: final table with -snapshot_dt 2 bytewise == without (rc={0})".format(rc))

	no_sp = [a for a in COMMON if a not in ("-ScattPower", "-no_fields")]
	for extra, needle, what in (
			(COMMON + ["-snapshot_dt", "-1"], "-snapshot_dt must be >= 0", "-snapshot_dt -1 rejected"),
			([a for a in COMMON if a != "-no_fields"] + ["-JouleHeating", "-snapshot_dt", "5"],
			 "-snapshot_dt writes T/R tables", "-snapshot_dt with -JouleHeating rejected"),
			(no_sp + ["-snapshot_dt", "5"], "-snapshot_dt writes T/R tables", "-snapshot_dt without -ScattPower rejected")):
		rc, log = film(d, 1, ["-load", npz] + extra + ["-maxt", "5", "-saveas", "bad"], "bad.log")
		check(rc != 0 and needle in log and "Initializing structure" not in log,
			  "{0} (rc={1}, {2!r})".format(what, rc, log.strip().splitlines()[-1:] if log.strip() else ""))


if __name__ == "__main__":
	if len(sys.argv) < 2:
		raise SystemExit(__doc__ or "usage: test_snapshots.py WORK [RANKS ...]")
	work = os.path.abspath(sys.argv[1])
	ranks_list = [int(r) for r in sys.argv[2:]] or [1, 2]
	os.makedirs(work, exist_ok=True)
	npz = os.path.join(work, "voxels_20x16x16.npz")
	rng = np.random.default_rng(2026)
	# 4-voxel random blocks: a film with lateral structure but a small cell
	g = np.repeat(np.repeat(np.repeat(rng.random((5, 4, 4)) < 0.45, 4, 0), 4, 1), 4, 2)
	np.savez(npz, g=g)
	print("synthetic voxels: stored shape {0}, solid fraction {1:.4f}".format(g.shape, float(g.mean())))
	tables = {}
	for ranks in ranks_list:
		tables[ranks] = cmd_rank(work, ranks, npz)
	if 1 in ranks_list:
		cmd_once(work, npz)
	for ranks in ranks_list[1:]:
		diff = float(np.max(np.abs(tables[ranks] - tables[ranks_list[0]])))
		print("max |table(np{0}) - table(np{1})| = {2:.3e}".format(ranks, ranks_list[0], diff))
	print("FAILURES: {0}".format(len(FAILURES)))
	for f in FAILURES:
		print("  " + f)
	sys.exit(1 if FAILURES else 0)
