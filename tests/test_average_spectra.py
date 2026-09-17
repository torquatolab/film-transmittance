#!/usr/bin/env python3
"""Tests for average_spectra.py (D1) on synthetic tables. Plain script; exits nonzero on failure.

Usage:
	python tests/test_average_spectra.py WORKDIR
"""

import os
import subprocess
import sys

import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import average_spectra  # noqa: E402

FAILS = []
COLUMNS = 'wavelength[1000nm]\ttransmittance\treflectance\tIncidentPower[a.u.]'


def check(cond, msg):
	print(("PASS " if cond else "FAIL ") + msg, flush=True)
	if not cond:
		FAILS.append(msg)


def write_table(path, wl, T, R, stopped=None, columns=COLUMNS, extra_col=False):
	"""A table in film_transmittance.py's format (voxel header when stopped is given)."""
	head = 'Transmittance through a film patterned with a voxel array\n\tFile: {0}\n\targs.dsrc = 4.00e-01'.format(path)
	if stopped:
		head += '\nstopped: ' + stopped
	head += '\n' + columns
	cols = [wl, T, R] + ([1.0 - T - np.abs(R)] if extra_col else []) + [-np.ones_like(wl)]
	np.savetxt(path, np.column_stack(cols), fmt='%1.6e', delimiter='\t', header=head)


def run(argv):
	p = subprocess.run([sys.executable, os.path.join(CODE, 'average_spectra.py')] + argv,
		capture_output=True, text=True)
	return p.returncode, p.stdout + p.stderr


def main(work):
	os.makedirs(work, exist_ok=True)
	rng = np.random.default_rng(5)
	nf = 21
	freqs = np.linspace(1/0.78, 1/0.38, nf)
	wl = np.round(1.0/freqs, 6)[::-1]  # descending, exactly representable in %1.6e
	T = np.round(rng.uniform(0.2, 0.9, (3, nf)), 6)
	R = -np.round(rng.uniform(0.0, 0.1, (3, nf)), 6)
	paths = [os.path.join(work, 't{0}.txt'.format(i)) for i in range(3)]
	for i, p in enumerate(paths):
		write_table(p, wl, T[i], R[i], stopped='maxt ceiling' if i == 1 else 'dft converged')

	# ---- per-wavelength means and stds against direct numpy on the written values
	out = os.path.join(work, 'mean.txt')
	rc, log = run([out] + paths + ['--bands', '4'])
	print(log)
	check(rc == 0, 'average_spectra rc=0')
	got = np.loadtxt(out)
	Tl = np.array([np.loadtxt(p)[:, 1] for p in paths])
	Rl = np.array([np.loadtxt(p)[:, 2] for p in paths])
	check(got.shape == (nf, 7), 'output shape {0} == ({1}, 7)'.format(got.shape, nf))
	check(np.array_equal(got[:, 0], np.loadtxt(paths[0])[:, 0]), 'wavelength column copied in input order')
	exp = [Tl.mean(0), Tl.std(0, ddof=1), Rl.mean(0), Rl.std(0, ddof=1), (Tl + np.abs(Rl) - 1).mean(0)]
	err = max(float(np.max(np.abs(got[:, j + 1] - e))) for j, e in enumerate(exp))
	check(err <= 5e-6, 'mean/std/balance columns match numpy within the %1.6e rounding (max err {0:.2e})'.format(err))
	check(np.all(got[:, 6] == 3), 'n column == 3')
	with open(out) as fh:
		header = [l for l in fh if l.startswith('#')]
	check(sum('input:' in l for l in header) == 3, 'header lists the 3 inputs')
	flagged = [l for l in header if '[stopped: maxt ceiling]' in l]
	check(len(flagged) == 1 and 't1.txt' in flagged[0], 'header flags exactly t1.txt as maxt ceiling: {0}'.format(flagged))
	check(any('maxt ceiling: 1 of 3' in l for l in header), 'header counts 1 of 3 flagged')

	# ---- bands: 21 points, 4 bands -> 6, 5, 5, 5 points ordered by increasing frequency
	bands_path = os.path.join(work, 'mean_bands.txt')
	check(os.path.exists(bands_path), 'band file written next to OUT')
	bands = np.loadtxt(bands_path)
	check(bands.shape == (4, 9), 'band table shape {0} == (4, 9)'.format(bands.shape))
	order = np.argsort(1.0/wl)
	groups = np.array_split(order, 4)
	check(list(bands[:, 2].astype(int)) == [6, 5, 5, 5], 'band point counts {0}'.format(bands[:, 2]))
	ok = True
	for b, idx in enumerate(groups):
		Tb = Tl[:, idx].mean(1)
		Rb = Rl[:, idx].mean(1)
		Bb = (Tl[:, idx] + np.abs(Rl[:, idx]) - 1).mean(1)
		want = [wl[idx].min(), wl[idx].max(), len(idx), Tb.mean(), Tb.std(ddof=1), Rb.mean(), Rb.std(ddof=1), Bb.mean(), 3]
		ok &= bool(np.allclose(bands[b], want, rtol=0, atol=5e-6))
	check(ok, 'band rows equal mean/std over tables of per-table band means')
	check(bands[0, 0] > bands[-1, 1], 'bands run from long to short wavelength (increasing frequency)')
	check(abs(np.dot(bands[:, 2], bands[:, 3]) / nf - got[:, 1].mean()) <= 5e-6,
		'point-weighted band T means reproduce the overall mean T')

	# ---- single table: std 0, n 1
	one = os.path.join(work, 'one.txt')
	rc, _ = run([one, paths[0]])
	g1 = np.loadtxt(one)
	check(rc == 0 and np.all(g1[:, 2] == 0) and np.all(g1[:, 4] == 0) and np.all(g1[:, 6] == 1)
		and np.allclose(g1[:, 1], Tl[0], atol=5e-7), 'single table: mean = input, std 0, n 1')
	check(not os.path.exists(os.path.join(work, 'one_bands.txt')), 'no band file without --bands')

	# ---- rejections
	shifted = os.path.join(work, 'shifted.txt')
	wl2 = wl.copy()
	wl2[3] += 1e-6
	write_table(shifted, wl2, T[0], R[0])
	rc, log = run([os.path.join(work, 'bad.txt'), paths[0], shifted])
	check(rc != 0 and 'wavelengths differ' in log, 'different wavelengths rejected: {0!r}'.format(log.strip()[-80:]))
	short = os.path.join(work, 'short.txt')
	write_table(short, wl[:-1], T[0][:-1], R[0][:-1])
	rc, log = run([os.path.join(work, 'bad.txt'), paths[0], short])
	check(rc != 0 and 'wavelengths differ' in log, 'different wavelength count rejected')
	joule = os.path.join(work, 'joule.txt')
	write_table(joule, wl, T[0], R[0], columns='wavelength[1000nm]\ttransmittance\treflectance\tAbsorbance\tIncidentPower[a.u.]', extra_col=True)
	rc, log = run([os.path.join(work, 'bad.txt'), paths[0], joule])
	check(rc != 0 and 'column layout' in log, 'different column layout rejected')
	rc, log = run([os.path.join(work, 'bad.txt'), paths[0], '--bands', '22'])
	check(rc != 0 and '--bands' in log, '--bands larger than the wavelength count rejected')
	check(not os.path.exists(os.path.join(work, 'bad.txt')), 'no output written on rejection')

	# ---- a real film_transmittance.py voxel table layout parses (header with File: and stopped: lines)
	rows, band_rows, fl = average_spectra.average([paths[1], paths[1]], 2)
	check(fl == [paths[1], paths[1]] and band_rows.shape == (2, 9) and np.all(rows[:, 2] == 0),
		'average() on duplicate inputs: both flagged, zero std')
	check('meep' not in sys.modules, 'average_spectra does not import meep')


if __name__ == '__main__':
	if len(sys.argv) != 2:
		raise SystemExit(__doc__)
	main(sys.argv[1])
	print('{0} failure(s)'.format(len(FAILS)))
	sys.exit(1 if FAILS else 0)
