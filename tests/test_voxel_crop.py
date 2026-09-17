#!/usr/bin/env python3
"""Tests for voxel_crop.py (package P3). Plain script; exits nonzero on failure.

Usage:
	python tests/test_voxel_crop.py                       # synthetic + brute-force tests (best_crop, select_crops)
	python tests/test_voxel_crop.py --tiffs DIR --out J   # best_crop on every DIR/*.tif, write J
"""

import argparse
import glob
import itertools
import json
import math
import os
import resource
import sys
import time
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voxel_crop import best_crop, crop_string, seam_score, select_crops  # noqa: E402

FAILS = []


def check(cond, msg):
	print(("PASS " if cond else "FAIL ") + msg, flush=True)
	if not cond:
		FAILS.append(msg)


# ---- independent brute force (literal definition, exact Fractions) ----

def exact_score(v, n):
	total = Fraction(0)
	for a in [ax for ax in range(3) if ax != n]:
		L = v.shape[a]
		planes = [v.take(i, axis=a) for i in range(L)]
		size = planes[0].size
		seam = Fraction(int(np.sum(planes[0] != planes[-1])), size)
		if L > 1:
			base = sum(Fraction(int(np.sum(planes[i] != planes[i + 1])), size) for i in range(L - 1)) / (L - 1)
		else:
			base = Fraction(0)
		total += seam / base if base > 0 else seam * 10 ** 12
	return total / 2


def brute_crop(v, n, f):
	lat = [ax for ax in range(3) if ax != n]
	opts = []
	for a in lat:
		L = v.shape[a]
		m = math.ceil(f * L)
		opts.append([(s, e) for s in range(L) for e in range(s + m, L + 1)])
	best = None
	for (s1, e1), (s2, e2) in itertools.product(*opts):
		sl = [None] * 3
		sl[n] = (0, v.shape[n])
		sl[lat[0]] = (s1, e1)
		sl[lat[1]] = (s2, e2)
		c = v[tuple(slice(s, e) for s, e in sl)]
		key = (exact_score(c, n), -(e1 - s1) * (e2 - s2), s1, s2)
		if best is None or key < best[0]:
			best = (key, [list(x) for x in sl])
	return best


# ---- synthetic arrays ----

def smooth_periodic(rng, shape, modes=3):
	"""Threshold of a random low-mode Fourier field, periodic in every axis."""
	grids = np.meshgrid(*[np.arange(s) / s for s in shape], indexing="ij")
	field = np.zeros(shape)
	for _ in range(12):
		k = rng.integers(-modes, modes + 1, size=3)
		field += rng.normal() * np.cos(2 * np.pi * sum(ki * g for ki, g in zip(k, grids)) + rng.uniform(0, 2 * np.pi))
	return field > 0


def smooth_nonperiodic(rng, shape, corr=4.0):
	"""Threshold of a random smooth non-periodic field (random incommensurate plane waves)."""
	grids = np.meshgrid(*[np.arange(s, dtype=float) for s in shape], indexing="ij")
	field = np.zeros(shape)
	for _ in range(20):
		k = rng.normal(size=3) / corr
		field += np.cos(sum(ki * g for ki, g in zip(k, grids)) + rng.uniform(0, 2 * np.pi))
	return field > 0


def run_unit():
	rng = np.random.default_rng(1234)

	# 1. periodic in both lateral axes: full array optimal
	for seed in range(3):
		r = np.random.default_rng(seed)
		v = smooth_periodic(r, (10, 12, 12))
		res = best_crop(v, 0, 0.75)
		bf = brute_crop(v, 0, 0.75)
		check(res["slices"] == bf[1], f"periodic seed {seed}: slices {res['slices']} == brute {bf[1]}")
		print(f"  periodic seed {seed}: full_score={res['full_score']:.6f} best score={res['score']:.6f}")
	v = smooth_periodic(np.random.default_rng(7), (24, 64, 48))
	res = best_crop(v, 0, 0.75)
	full = seam_score(v, 0)
	print(f"  periodic 24x64x48: full_score={full['score']:.6f} per_axis={full['per_axis']}")
	check(res["slices"] == [[0, 24], [0, 64], [0, 48]], f"periodic 24x64x48 returns full array: {res['slices']}")
	check(full["score"] <= 1.5, f"periodic 24x64x48 full_score {full['score']:.4f} <= 1.5")
	check(res["kept_fraction"] == 1.0 and res["full_score"] == full["score"], "periodic kept_fraction 1, full_score consistent")

	# 2. planted crop: periodic block embedded in non-periodic padding
	N, A, B = 16, 40, 36
	blk_a, blk_b, s_a, s_b = 32, 28, 5, 3
	v = smooth_nonperiodic(np.random.default_rng(11), (N, A, B), corr=2.0)
	v[:, s_a:s_a + blk_a, s_b:s_b + blk_b] = smooth_periodic(np.random.default_rng(12), (N, blk_a, blk_b))
	res = best_crop(v, 0, 0.75)
	want = [[0, N], [s_a, s_a + blk_a], [s_b, s_b + blk_b]]
	print(f"  planted: slices={res['slices']} score={res['score']:.6f} full_score={res['full_score']:.6f}")
	check(res["slices"] == want, f"planted crop found: {res['slices']} == {want}")
	check(res["full_score"] > res["score"], "planted: full_score > best score")

	# 3. tie-breaks
	z = np.zeros((4, 10, 9), dtype=bool)
	res = best_crop(z, 0, 0.75)
	check(res["slices"] == [[0, 4], [0, 10], [0, 9]] and res["score"] == 0.0, f"all-zero: full array, score 0: {res['slices']}")
	# alternating along axis 1 (length 10), constant along axis 2: (0,9) and (1,10) tie at score 0 -> smaller offset
	v = np.zeros((5, 10, 7), dtype=bool)
	col = np.random.default_rng(3).random((5, 1, 1)) > 0.5
	col[0, 0, 0] = True
	col[1, 0, 0] = False
	v[:, 0::2, :] = col
	v[:, 1::2, :] = ~col
	res = best_crop(v, 0, 0.8)
	check(res["slices"] == [[0, 5], [0, 9], [0, 7]], f"offset tie-break: {res['slices']} == [[0,5],[0,9],[0,7]]")
	check(res["score"] == 0.0 and res["kept_fraction"] == 0.9, f"offset tie-break score {res['score']} kept {res['kept_fraction']}")

	# 4. brute force on random small arrays, all normal axes, several fractions
	n_bf = 0
	for seed in range(8):
		r = np.random.default_rng(100 + seed)
		for n in (0, 1, 2):
			shape = [int(r.integers(5, 13)) for _ in range(3)]
			shape[n] = int(r.integers(2, 7))
			kind = seed % 4
			if kind == 0:
				v = r.random(shape) < r.uniform(0.2, 0.8)
			elif kind == 1:
				v = smooth_nonperiodic(r, tuple(shape), corr=2.5)
			elif kind == 2:
				v = smooth_periodic(r, tuple(shape), modes=2)
			else:
				v = np.tile(r.random((2, 2, 3)) < 0.5, (6, 6, 4))[:shape[0], :shape[1], :shape[2]]
			f = [0.75, 0.5, 0.9][seed % 3]
			res = best_crop(v, n, f)
			bf = brute_crop(v, n, f)
			ok = res["slices"] == bf[1] and abs(res["score"] - float(bf[0][0])) <= 1e-12 * max(1.0, float(bf[0][0]))
			check(ok, f"brute seed {seed} n={n} f={f} shape={v.shape}: {res['slices']} vs {bf[1]} score {res['score']!r} vs {float(bf[0][0])!r}")
			n_bf += 1
	print(f"  brute-force comparisons: {n_bf}")

	# 5. determinism and input validation
	v = smooth_nonperiodic(rng, (6, 12, 11))
	check(best_crop(v) == best_crop(v.copy()), "deterministic repeat")
	check(best_crop(v.astype(np.uint8) * 255)["slices"] == best_crop(v)["slices"], "uint8 input == bool input")
	for bad in (lambda: best_crop(np.zeros((3, 3))), lambda: best_crop(v, 3), lambda: best_crop(v, 0, 0.0)):
		try:
			bad()
			check(False, "invalid input raises ValueError")
		except ValueError:
			check(True, "invalid input raises ValueError")


# ---- select_crops (D1) ----

def brute_select(v, n_normal, f, k, sep):
	"""Literal greedy definition: all crops in exact key order, accept if separated from every accepted one."""
	lat = [ax for ax in range(3) if ax != n_normal]
	opts = []
	for a in lat:
		L = v.shape[a]
		m = math.ceil(f * L)
		opts.append([(s, e) for s in range(L) for e in range(s + m, L + 1)])
	allc = []
	for (s1, e1), (s2, e2) in itertools.product(*opts):
		sl = [None] * 3
		sl[n_normal] = (0, v.shape[n_normal])
		sl[lat[0]] = (s1, e1)
		sl[lat[1]] = (s2, e2)
		c = v[tuple(slice(s, e) for s, e in sl)]
		allc.append(((exact_score(c, n_normal), -(e1 - s1) * (e2 - s2), s1, s2), [list(x) for x in sl]))
	allc.sort(key=lambda t: t[0])
	picked = []
	for key, sl in allc:
		if len(picked) == k:
			break
		if all(max(abs(key[2] - o[0]), abs(key[3] - o[1])) >= sep for o in (p[0] for p in picked)):
			picked.append(((key[2], key[3]), sl))
	return [sl for _, sl in picked]


def offsets(c, n_normal):
	return tuple(c["slices"][a][0] for a in range(3) if a != n_normal)


def run_select():
	# n = 1 equals best_crop (all normal axes, several fractions)
	for seed in range(6):
		r = np.random.default_rng(300 + seed)
		n_normal = seed % 3
		shape = [int(r.integers(6, 14)) for _ in range(3)]
		shape[n_normal] = int(r.integers(2, 6))
		v = smooth_nonperiodic(r, tuple(shape), corr=2.5) if seed % 2 else r.random(shape) < 0.4
		f = [0.75, 0.5, 0.9][seed % 3]
		one = select_crops(v, 1, n_normal, f)
		check(len(one) == 1 and one[0] == best_crop(v, n_normal, f),
			f"select_crops n=1 == best_crop (seed {seed}, normal {n_normal}, f {f}, shape {v.shape})")

	# brute-force greedy definition, default and explicit separations
	n_bf = 0
	for seed in range(8):
		r = np.random.default_rng(400 + seed)
		n_normal = seed % 3
		shape = [int(r.integers(7, 13)) for _ in range(3)]
		shape[n_normal] = int(r.integers(2, 6))
		v = smooth_nonperiodic(r, tuple(shape), corr=2.0) if seed % 2 else r.random(shape) < 0.5
		f = [0.5, 0.6][seed % 2]
		lat = [shape[a] for a in range(3) if a != n_normal]
		for sep in (None, 1, 2, 3):
			sep_eff = max(1, int(0.25 * min(lat))) if sep is None else sep
			got = select_crops(v, 6, n_normal, f, sep)
			want = brute_select(v, n_normal, f, 6, sep_eff)
			check([c["slices"] for c in got] == want,
				f"select_crops == brute greedy (seed {seed}, normal {n_normal}, f {f}, sep {sep} -> {sep_eff}, "
				f"shape {v.shape}): {[c['slices'] for c in got]} vs {want}")
			n_bf += 1
			offs = [offsets(c, n_normal) for c in got]
			ok = all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= sep_eff
				for i, a in enumerate(offs) for b in offs[i + 1:])
			check(ok and len(set(offs)) == len(offs), f"  pairwise offset separation >= {sep_eff}: {offs}")
			for c in got:
				crop = v[tuple(slice(s, e) for s, e in c["slices"])]
				if seam_score(crop, n_normal)["score"] != c["score"]:
					check(False, f"  score of {c['slices']} == seam_score(crop)")
	print(f"  select_crops brute-force comparisons: {n_bf}")

	# determinism and ordering
	v = smooth_nonperiodic(np.random.default_rng(21), (8, 30, 26), corr=3.0)
	a = select_crops(v, 5)
	b = select_crops(v.copy(), 5)
	check(a == b, "select_crops deterministic repeat")
	check(select_crops(v.astype(np.uint8) * 7, 5) == a, "select_crops uint8 input == bool input")
	check(all(a[i]["score"] <= a[i + 1]["score"] for i in range(len(a) - 1)) or len(a) < 2,
		"select_crops scores non-decreasing: {0}".format([round(c["score"], 6) for c in a]))
	check(select_crops(v, 3) == a[:3], "select_crops n=3 is a prefix of n=5")
	# 30x26 lateral, f = 0.75: offsets p0 in [0, 7], q0 in [0, 6]; default sep = 6
	offs = [offsets(c, 0) for c in a]
	print(f"  30x26 default sep: {len(a)} crops, offsets {offs}")
	check(all(max(abs(x[0] - y[0]), abs(x[1] - y[1])) >= 6 for i, x in enumerate(offs) for y in offs[i + 1:]),
		"30x26 default separation 6 holds")
	# separation larger than any offset range: only one crop
	check(len(select_crops(v, 5, min_offset_sep=100)) == 1, "min_offset_sep beyond the offset range gives 1 crop")
	check(crop_string([[0, 8], [2, 25], [0, 20]]) == "0:8,2:25,0:20", "crop_string format")
	for bad in (lambda: select_crops(v, 0), lambda: select_crops(v, 2, min_offset_sep=0),
			lambda: select_crops(v, 2, 3), lambda: select_crops(v, 1.5)):
		try:
			bad()
			check(False, "select_crops invalid input raises ValueError")
		except ValueError:
			check(True, "select_crops invalid input raises ValueError")


# ---- TIFF runs (local reader: no dependency on voxel_io) ----

def read_tiff(path):
	from PIL import Image, ImageSequence
	with Image.open(path) as im:
		frames = [np.array(fr) for fr in ImageSequence.Iterator(im)]
	return np.stack(frames, axis=0).astype(bool)


def run_tiffs(tdir, out):
	files = sorted(glob.glob(os.path.join(tdir, "*.tif")))
	check(len(files) == 5, f"found {len(files)} TIFFs (expect 5)")
	records = []
	for path in files:
		t0 = time.time()
		v = read_tiff(path)
		t_read = time.time() - t0
		t0 = time.time()
		res = best_crop(v, 0, 0.75)
		t_crop = time.time() - t0
		rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
		rec = {
			"file": os.path.abspath(path),
			"shape": list(v.shape),
			"slices": res["slices"],
			"score": res["score"],
			"per_axis": {str(k): d for k, d in res["per_axis"].items()},
			"kept_fraction": res["kept_fraction"],
			"full_score": res["full_score"],
		}
		records.append(rec)
		print(f"TIFF {os.path.basename(path)} shape={v.shape} solid={v.mean():.4f} read={t_read:.1f}s "
			f"best_crop={t_crop:.1f}s maxrss_so_far={rss:.0f}MiB", flush=True)
		print("  " + json.dumps(rec), flush=True)
		sl = [slice(s, e) for s, e in res["slices"]]
		check(abs(seam_score(v[tuple(sl)], 0)["score"] - res["score"]) == 0, f"{os.path.basename(path)}: score == seam_score(crop)")
		check(res["score"] <= res["full_score"], f"{os.path.basename(path)}: score {res['score']:.4f} <= full {res['full_score']:.4f}")
		lat_ok = all(e - s >= math.ceil(0.75 * v.shape[a]) for a, (s, e) in enumerate(res["slices"]) if a != 0)
		check(lat_ok and res["slices"][0] == [0, v.shape[0]], f"{os.path.basename(path)}: crop constraints")
		# brute-force cross-check on a small real sub-block
		sub = v[:min(8, v.shape[0]), :12, :12]
		bf = brute_crop(sub, 0, 0.75)
		check(best_crop(sub, 0, 0.75)["slices"] == bf[1], f"{os.path.basename(path)}: real 12x12 sub-block matches brute force")
	with open(out, "w") as fh:
		json.dump(records, fh, indent=1)
	print(f"wrote {out}")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--tiffs", default=None)
	ap.add_argument("--out", default=None)
	args = ap.parse_args()
	if args.tiffs:
		run_tiffs(args.tiffs, args.out)
	else:
		run_unit()
		run_select()
	print(f"{len(FAILS)} failure(s)")
	sys.exit(1 if FAILS else 0)
