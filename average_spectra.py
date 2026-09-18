#!/usr/bin/env python3
"""Average several film_transmittance.py spectrum tables (e.g. lateral crops of one stack).

	python average_spectra.py OUT.txt TABLE1 TABLE2 ... [--bands K]

Every table must have the same column layout (column-name header line and column
count) and exactly the same wavelength column. OUT.txt gets, per wavelength:

	wavelength  T_mean  T_std  R_mean  R_std  balance_mean  n

with R the signed reflectance column of the inputs, balance = T + |R| - 1 (averaged
over tables), std the sample standard deviation over tables (ddof = 1; 0 for a single
table) and n the number of tables. The comment header lists the inputs and flags every
input whose own header says ``stopped: maxt ceiling``.

``--bands K`` also writes ``OUT_bands.txt`` (``OUT`` without ``.txt``): the frequency
points are sorted by frequency and split into K contiguous bands of (nearly) equal point
counts (``np.array_split``; Meep's flux frequencies are evenly spaced, so the bands are
equal-width in frequency). For each table the band mean of T, R and T + |R| - 1 is
taken first; the file then holds the mean and std of those band means over tables:

	wl_min  wl_max  n_points  T_mean  T_std  R_mean  R_std  balance_mean  n

Pure NumPy; no Meep import.
"""

import argparse
import sys

import numpy as np

FMT = '%1.6e'
STOPPED_CEILING = 'stopped: maxt ceiling'


def read_table(path):
	"""Return (data, comment lines without '# ') of one spectrum table."""
	with open(path) as fh:
		comments = [line[1:].strip() for line in fh if line.startswith('#')]
	data = np.atleast_2d(np.loadtxt(path))
	if data.shape[1] < 3:
		raise ValueError("{0}: expected at least 3 columns (wavelength, T, R), got {1}".format(
			path, data.shape[1]))
	return data, comments


def _std(values):
	"""Sample standard deviation over axis 0 (ddof = 1), 0 for a single row."""
	if values.shape[0] < 2:
		return np.zeros(values.shape[1:])
	return np.std(values, axis=0, ddof=1)


def average(paths, bands=0):
	"""Return (per-wavelength rows, band rows or None, flagged inputs) for the tables."""
	tables = [read_table(p) for p in paths]
	data0, comments0 = tables[0]
	columns0 = comments0[-1] if comments0 else None
	for path, (data, comments) in zip(paths[1:], tables[1:]):
		columns = comments[-1] if comments else None
		if data.shape[1] != data0.shape[1] or columns != columns0:
			raise ValueError("{0}: column layout ({1} columns, {2!r}) differs from {3} "
				"({4} columns, {5!r})".format(path, data.shape[1], columns, paths[0],
					data0.shape[1], columns0))
		if data.shape[0] != data0.shape[0] or not np.array_equal(data[:, 0], data0[:, 0]):
			raise ValueError("{0}: wavelengths differ from {1}".format(path, paths[0]))
	flagged = [p for p, (_, comments) in zip(paths, tables) if STOPPED_CEILING in comments]

	T = np.array([d[:, 1] for d, _ in tables])  # (tables, wavelengths)
	R = np.array([d[:, 2] for d, _ in tables])
	balance = T + np.abs(R) - 1.0
	n = len(tables)
	wl = data0[:, 0]
	rows = np.column_stack((wl, T.mean(axis=0), _std(T), R.mean(axis=0), _std(R),
		balance.mean(axis=0), np.full(len(wl), n)))

	band_rows = None
	if bands:
		if not 1 <= bands <= len(wl):
			raise ValueError("--bands must be between 1 and the number of wavelengths ({0}), "
				"got {1}".format(len(wl), bands))
		order = np.argsort(1.0/wl, kind='stable')
		band_rows = []
		for idx in np.array_split(order, bands):
			Tb = T[:, idx].mean(axis=1)[:, None]  # (tables, 1)
			Rb = R[:, idx].mean(axis=1)[:, None]
			Bb = balance[:, idx].mean(axis=1)
			band_rows.append([wl[idx].min(), wl[idx].max(), len(idx), Tb.mean(), _std(Tb)[0],
				Rb.mean(), _std(Rb)[0], Bb.mean(), n])
		band_rows = np.array(band_rows)
	return rows, band_rows, flagged


def _header(paths, flagged, columns):
	lines = ["Mean over {0} spectrum tables (std: sample standard deviation over tables, "
		"ddof = 1)".format(len(paths))]
	for p in paths:
		lines.append("\tinput: {0}{1}".format(p, "  [stopped: maxt ceiling]" if p in flagged else ""))
	lines.append("\tinputs stopped at the maxt ceiling: {0} of {1}".format(len(flagged), len(paths)))
	lines.append(columns)
	return "\n".join(lines)


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("out", help="output table")
	ap.add_argument("tables", nargs="+", help="spectrum tables (*_trans-<pol>.txt)")
	ap.add_argument("--bands", type=int, default=0, help="also write K band averages to OUT_bands.txt")
	args = ap.parse_args(argv)
	try:
		rows, band_rows, flagged = average(args.tables, args.bands)
	except (OSError, ValueError) as exc:
		raise SystemExit("ERROR: {0}".format(exc))
	np.savetxt(args.out, rows, fmt=[FMT]*6 + ['%d'], delimiter='\t', header=_header(args.tables,
		flagged, "wavelength[1000nm]\tT_mean\tT_std\tR_mean\tR_std\tbalance_mean(T+|R|-1)\tn"))
	print("wrote {0} ({1} tables, {2} flagged maxt ceiling)".format(args.out, len(args.tables),
		len(flagged)))
	if band_rows is not None:
		stem = args.out[:-4] if args.out.endswith(".txt") else args.out
		band_out = stem + "_bands.txt"
		np.savetxt(band_out, band_rows, fmt=[FMT, FMT, '%d'] + [FMT]*5 + ['%d'], delimiter='\t',
			header=_header(args.tables, flagged, "{0} bands of nearly equal frequency-point "
				"counts; means of per-table band means\nwl_min\twl_max\tn_points\tT_mean\tT_std\t"
				"R_mean\tR_std\tbalance_mean(T+|R|-1)\tn".format(args.bands)))
		print("wrote {0} ({1} bands)".format(band_out, args.bands))
	return 0


if __name__ == "__main__":
	sys.exit(main())
