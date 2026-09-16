#!/usr/bin/env python3

"""
Tests for voxel_geometry.orient and voxel_geometry.film_geometry.

Part 1 (pure numpy, exact): axis mapping of orient for normal_axis 0, 1, 2 on an
asymmetric array, crop application, and rejection of bad arguments.

Part 2 (Meep, run on a compute node): a 12 x 10 x 8 asymmetric voxel film (an
L-shaped solid, solids joined across the periodic x and y seams, asymmetric seam
voxels, single voxels on both film faces) is stored in (z, y, x) order with junk
padding, cropped and oriented, and placed with film_geometry in a periodic cell
(k_point = 0, PML in z only) at resolution = k / voxel_size, k = 1, 2, with
do_averaging False and True.  Checks:

  A. MaterialGrid sampling model.  sim.get_epsilon_grid() on the half-pixel
     lattice (every Yee position) must equal a numpy model of a cell-centred,
     trilinear, edge-clamped grid (max diff <= 1e-12) everywhere except on the
     periodic seam planes x = -Lx/2, y = -Ly/2, where the value must equal one of
     the two adjacent voxels (the side is reported).
  B. sim.get_epsilon() over the film, each sample mapped to its voxel by its
     coordinate: bulk samples (voxel and its 26 periodic neighbours of one phase)
     must be exact (<= 1e-9), and such bulk samples must exist next to both
     seams; with do_averaging=False the nearest-phase solid fraction must be
     within FRACTION_TOL of the voxel solid fraction (reported only for True).  The fraction of samples whose epsilon differs from
     the voxel value (> 1e-9) and the nearest-phase mismatch are reported.
  C. Periodic wrap as translation invariance: the film rolled by whole voxels
     in x and y must give the rolled epsilon; the fraction of samples that
     differ is reported, and bulk samples must agree exactly.
  D. (informational) the same array with odd lateral sizes (13 x 11 x 8):
     fraction of get_epsilon() samples lying on a voxel face.

Exits nonzero on any failure.  Usage: test_voxel_geometry.py [--numpy-only]

Sam Dawley
09/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

# ----- numerics -----
import numpy as np

# ----- local modules -----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import voxel_geometry as vg  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

EPS_SOLID = 2.4025
EPS_VOID = 1.0
VOXEL_SIZE = 0.01
FRACTION_TOL = 0.01
FAILURES: List[str] = []

# ============================================================
# HELPERS
# ============================================================

def check(cond: bool, msg: str) -> None:
	"""Record and print one pass/fail line."""
	print(("PASS " if cond else "FAIL ") + msg, flush=True)
	if not cond:
		FAILURES.append(msg)


def synthetic_film(nx: int = 12, ny: int = 10, nz: int = 8) -> np.ndarray:
	"""Asymmetric test film indexed [x, y, z] (see module docstring)."""
	assert nx >= 12 and ny >= 10 and nz == 8, "synthetic_film needs nx >= 12, ny >= 10, nz = 8"
	a = np.zeros((nx, ny, nz), dtype=bool)
	a[1:9, 1:4, 1:7] = True                  # L: arm along x
	a[1:4, 1:9, 1:7] = True                  # L: arm along y
	a[nx-2:nx, 6:9, 2:6] = True              # joined across the x seam ...
	a[0, 6:9, 2:6] = True                    # ... bulk at x = nx-1
	a[nx-1, 4:6, 6:8] = True                 # x seam, asymmetric (a[0, 4:6, 6:8] void)
	a[5:8, ny-2:ny, 3:6] = True              # joined across the y seam ...
	a[5:8, 0, 3:6] = True                    # ... bulk at y = ny-1
	a[9, 0, 0:2] = True                      # y seam, asymmetric (a[9, ny-1, 0:2] void)
	a[6, 6, 0] = True                        # bottom face
	a[9, 5, nz-1] = True                     # top face
	return a


def material_model(w: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
	"""Cell-centred, trilinear, edge-clamped weight u at block-relative voxel coordinates (outside z -> 0)."""
	assert w.ndim == 3 and xs.ndim == ys.ndim == zs.ndim == 1, "material_model: bad shapes"
	per_axis = []
	for n, c in zip(w.shape, (xs, ys, zs)):
		s = np.clip(c - 0.5, 0.0, n - 1.0)
		i0 = np.minimum(np.floor(s).astype(int), n - 1)
		i1 = np.minimum(i0 + 1, n - 1)
		per_axis.append((i0, i1, s - i0))
	(x0, x1, tx), (y0, y1, ty), (z0, z1, tz) = per_axis
	u = np.zeros((xs.size, ys.size, zs.size))
	for xi, wx in ((x0, 1 - tx), (x1, tx)):
		for yi, wy in ((y0, 1 - ty), (y1, ty)):
			for zi, wz in ((z0, 1 - tz), (z1, tz)):
				u += w[np.ix_(xi, yi, zi)]*wx[:, None, None]*wy[None, :, None]*wz[None, None, :]
	inside = (zs >= 0) & (zs <= w.shape[2])
	return u*inside[None, None, :]


def bulk_mask(a: np.ndarray) -> np.ndarray:
	"""Voxels whose 26 neighbours (periodic in x, y; void beyond z) share their phase."""
	assert a.ndim == 3, "bulk_mask: 3D array required"
	p = np.pad(a, ((0, 0), (0, 0), (1, 1)), constant_values=False)
	same = np.ones(a.shape, dtype=bool)
	for dx in (-1, 0, 1):
		for dy in (-1, 0, 1):
			for dz in (-1, 0, 1):
				nb = np.roll(np.roll(p, dx, 0), dy, 1)[:, :, 1 + dz:1 + dz + a.shape[2]]
				same &= (nb == a)
	return same

# ============================================================
# PART 1: ORIENT (NUMPY)
# ============================================================

def test_orient() -> None:
	"""Exact axis mapping and crop checks for orient()."""
	v = np.arange(3*4*5).reshape(3, 4, 5)
	expect = {0: lambda x, y, z: v[z, y, x], 1: lambda x, y, z: v[y, z, x], 2: lambda x, y, z: v[y, x, z]}
	shapes = {0: (5, 4, 3), 1: (5, 3, 4), 2: (4, 3, 5)}
	for n in (0, 1, 2):
		o = vg.orient(v, n)
		ok = o.shape == shapes[n] and all(o[x, y, z] == expect[n](x, y, z)
			for x in range(o.shape[0]) for y in range(o.shape[1]) for z in range(o.shape[2]))
		check(ok, "orient normal_axis={0}: shape {1} and every element".format(n, o.shape))
	crop = [[1, 3], [0, 4], [2, 5]]
	o = vg.orient(v, 0, crop)
	ok = o.shape == (3, 4, 2) and all(o[x, y, z] == v[z + 1, y, x + 2]
		for x in range(3) for y in range(4) for z in range(2))
	check(ok, "orient normal_axis=0 crop {0}: o[x,y,z] == v[z+1, y, x+2]".format(crop))
	o = vg.orient(v, 1, crop)
	ok = o.shape == (3, 2, 4) and all(o[x, y, z] == v[y + 1, z, x + 2]
		for x in range(3) for y in range(2) for z in range(4))
	check(ok, "orient normal_axis=1 crop {0}: o[x,y,z] == v[y+1, z, x+2]".format(crop))
	for bad in ([[0, 3], [0, 4]], [[2, 1], [0, 4], [0, 5]], [[0, 4], [0, 4], [0, 5]]):
		try:
			vg.orient(v, 0, bad)
			check(False, "orient rejects crop {0}".format(bad))
		except ValueError:
			check(True, "orient rejects crop {0}".format(bad))
	for args in ((v, 3), (v[0], 0)):
		try:
			vg.orient(*args)
			check(False, "orient rejects bad input (shape {0}, normal_axis {1})".format(np.shape(args[0]), args[1]))
		except ValueError:
			check(True, "orient rejects bad input (shape {0}, normal_axis {1})".format(np.shape(args[0]), args[1]))

# ============================================================
# PART 2: PLACEMENT (MEEP)
# ============================================================

def build_sim(a: np.ndarray, k: int, avg: bool):
	"""Periodic x/y cell, PML in z, film centred at z = 0 with 6 vacuum voxels each side."""
	import meep as mp
	assert k >= 1 and a.ndim == 3, "build_sim: bad arguments"
	Lx, Ly, h, geom = vg.film_geometry(a, VOXEL_SIZE, mp.Medium(epsilon=EPS_SOLID), mp.Medium(epsilon=EPS_VOID),
									   0.0, do_averaging=avg)
	sim = mp.Simulation(cell_size=mp.Vector3(Lx, Ly, h + 12*VOXEL_SIZE), resolution=k/VOXEL_SIZE, geometry=geom,
						k_point=mp.Vector3(), boundary_layers=[mp.PML(4*VOXEL_SIZE, direction=mp.Z)])
	sim.init_sim()
	return sim


def film_samples(sim, a: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], float]:
	"""get_epsilon() restricted to one period of the film; voxel index per axis; fraction of samples on faces."""
	e = np.real(np.asarray(sim.get_epsilon()))
	coords = [np.asarray(c, dtype=float) for c in sim.get_array_metadata()[:3]]
	assert e.shape == tuple(c.size for c in coords), "get_epsilon / metadata shape mismatch"
	sel, idx, on_face = [], [], []
	for n, c in zip(a.shape, coords):
		u = (c + 0.5*n*VOXEL_SIZE)/VOXEL_SIZE
		keep = np.nonzero((u > -1e-9) & (u < n - 1e-9))[0]
		sel.append(keep)
		idx.append(np.floor(u[keep] + 1e-9).astype(int))
		on_face.append(np.abs(u[keep] - np.round(u[keep])) < 1e-6)
	sub = e[np.ix_(*sel)]
	face = on_face[0][:, None, None] | on_face[1][None, :, None] | on_face[2][None, None, :]
	return sub, tuple(idx), float(np.mean(face))


def test_placement(k: int, avg: bool, out: Dict) -> None:
	"""Checks A, B, C for one resolution multiple and averaging flag."""
	assert vg.PROJECTION_BETA == 0.0, "material_model assumes no projection (beta = 0)"
	a = synthetic_film()
	nx, ny, nz = a.shape
	tag = "k={0} do_averaging={1}".format(k, avg)
	sim = build_sim(a, k, avg)
	rec: Dict = {"voxel_fraction": float(a.mean())}

	# ----- A: MaterialGrid sampling model on every Yee position -----
	w = a.astype(float)
	fx = np.arange(2*k*nx)/(2*k)
	fy = np.arange(2*k*ny)/(2*k)
	fz = np.arange(2*k*nz + 1)/(2*k)
	eg = np.real(np.asarray(sim.get_epsilon_grid(fx*VOXEL_SIZE - nx*VOXEL_SIZE/2, fy*VOXEL_SIZE - ny*VOXEL_SIZE/2,
												  fz*VOXEL_SIZE - nz*VOXEL_SIZE/2)))
	ug = (eg - EPS_VOID)/(EPS_SOLID - EPS_VOID)
	um = material_model(w, fx, fy, fz)
	d = np.abs(ug - um)
	d_in = d[1:, 1:, :]
	check(float(d_in.max()) <= 1e-12, "{0} A: get_epsilon_grid == cell-centred trilinear clamped model off the seams "
		  "(max diff {1:.2e}, {2} points)".format(tag, float(d_in.max()), d_in.size))
	seam_x = ug[0, :, :]
	seam_y = ug[:, 0, :]
	lo_x, hi_x = material_model(w, np.array([1e-9]), fy, fz)[0], material_model(w, np.array([nx - 1e-9]), fy, fz)[0]
	lo_y, hi_y = material_model(w, fx, np.array([1e-9]), fz)[:, 0], material_model(w, fx, np.array([ny - 1e-9]), fz)[:, 0]
	sx0, sx1 = float(np.abs(seam_x - lo_x).max()), float(np.abs(seam_x - hi_x).max())
	sy0, sy1 = float(np.abs(seam_y - lo_y).max()), float(np.abs(seam_y - hi_y).max())
	check(min(sx0, sx1) <= 1e-12 and min(sy0, sy1) <= 1e-12,
		  "{0} A: seam planes take one adjacent voxel's clamped value (x=-Lx/2: vs voxel 0 {1:.3g}, vs voxel nx-1 {2:.3g}; "
		  "y=-Ly/2: vs voxel 0 {3:.3g}, vs voxel ny-1 {4:.3g})".format(tag, sx0, sx1, sy0, sy1))
	rec["seam_x_matches"] = "voxel 0" if sx0 <= 1e-12 else "voxel nx-1" if sx1 <= 1e-12 else "neither"
	rec["seam_y_matches"] = "voxel 0" if sy0 <= 1e-12 else "voxel ny-1" if sy1 <= 1e-12 else "neither"
	rec["yee_lattice_fraction"] = float(ug[:, :, 1:-1].mean())

	# ----- B: get_epsilon() over the film vs the voxel layout -----
	e, (ix, iy, iz), face_frac = film_samples(sim, a)
	check(face_frac == 0.0 and e.shape == (k*nx, k*ny, k*nz),
		  "{0} B: {1} film samples, none on a voxel face (fraction on faces {2})".format(tag, e.shape, face_frac))
	ref_solid = a[np.ix_(ix, iy, iz)]
	ref = np.where(ref_solid, EPS_SOLID, EPS_VOID)
	diff = np.abs(e - ref)
	bulk = bulk_mask(a)[np.ix_(ix, iy, iz)]
	near_x = ((ix == 0) | (ix == nx - 1))[:, None, None] & bulk
	near_y = ((iy == 0) | (iy == ny - 1))[None, :, None] & bulk
	check(bool(near_x.any()) and bool(near_y.any()) and float(diff[bulk].max()) <= 1e-9,
		  "{0} B: bulk samples exact (max diff {1:.2e} over {2} samples; {3} next to the x seam, {4} next to the y seam)".format(
			  tag, float(diff[bulk].max()), int(bulk.sum()), int(near_x.sum()), int(near_y.sum())))
	cls = e > 0.5*(EPS_SOLID + EPS_VOID)
	rec.update({"samples": int(e.size), "exact_mismatch_fraction": float(np.mean(diff > 1e-9)),
				"interface_sample_fraction": float(np.mean(~bulk)),
				"nearest_phase_mismatch_fraction": float(np.mean(cls != ref_solid)),
				"nearest_phase_fraction": float(cls.mean()),
				"linear_eps_fraction": float(((e - EPS_VOID)/(EPS_SOLID - EPS_VOID)).mean()),
				"max_abs_diff": float(diff.max())})
	print("INFO {0} B: exact mismatch (>1e-9) {1:.4f}; nearest-phase mismatch {2:.4f}; interface samples {3:.4f}; "
		  "solid fraction voxel {4:.4f} nearest-phase {5:.4f} linear-eps {6:.4f} Yee-lattice material {7:.4f}".format(
			  tag, rec["exact_mismatch_fraction"], rec["nearest_phase_mismatch_fraction"], rec["interface_sample_fraction"],
			  rec["voxel_fraction"], rec["nearest_phase_fraction"], rec["linear_eps_fraction"], rec["yee_lattice_fraction"]),
		  flush=True)
	frac_ok = abs(rec["nearest_phase_fraction"] - rec["voxel_fraction"]) <= FRACTION_TOL
	frac_msg = "{0} B: nearest-phase solid fraction {1:.4f} within {2} of voxel fraction {3:.4f}".format(
		tag, rec["nearest_phase_fraction"], FRACTION_TOL, rec["voxel_fraction"])
	if avg:
		# do_averaging=True is not the recommended setting; its fraction is reported, not enforced
		print("INFO " + frac_msg + (" -> within" if frac_ok else " -> NOT within"), flush=True)
	else:
		check(frac_ok, frac_msg)

	# ----- C: periodic wrap as translation invariance -----
	rec["roll"] = {}
	for sx, sy in ((5, 0), (0, 3), (7, 4)):
		b = np.roll(np.roll(a, sx, 0), sy, 1)
		e2, _, _ = film_samples(build_sim(b, k, avg), b)
		rolled = np.roll(np.roll(e, sx*k, 0), sy*k, 1)
		dd = np.abs(e2 - rolled)
		bulk2 = np.roll(np.roll(bulk, sx*k, 0), sy*k, 1)
		frac = float(np.mean(dd > 1e-9))
		rec["roll"]["{0},{1}".format(sx, sy)] = {"differ_fraction": frac, "max_abs_diff": float(dd.max()),
												 "nearest_phase_differ_fraction": float(np.mean(
													 (e2 > 0.5*(EPS_SOLID + EPS_VOID)) != (rolled > 0.5*(EPS_SOLID + EPS_VOID))))}
		check(float(dd[bulk2].max()) <= 1e-9,
			  "{0} C: roll ({1},{2}) voxels: bulk samples identical (max {3:.2e}); all samples differ fraction {4:.4f}, "
			  "max diff {5:.3f}, nearest-phase differ {6:.4f}".format(tag, sx, sy, float(dd[bulk2].max()), frac, float(dd.max()),
			  rec["roll"]["{0},{1}".format(sx, sy)]["nearest_phase_differ_fraction"]))
	out[tag] = rec


def test_odd_sizes(out: Dict) -> None:
	"""Informational: odd lateral voxel counts, fraction of get_epsilon() samples on voxel faces."""
	a = synthetic_film(13, 11, 8)
	for k in (1, 2):
		_, _, face_frac = film_samples(build_sim(a, k, False), a)
		print("INFO odd 13x11x8 k={0}: fraction of film samples on a voxel face = {1:.4f}".format(k, face_frac), flush=True)
		out["odd13x11x8 k={0}".format(k)] = {"face_sample_fraction": face_frac}

# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# ----- optional -----
	parser.add_argument("--numpy-only", action="store_true", help="run only the orient tests")
	parser.add_argument("--json", type=str, default="", help="write the numbers to this JSON file")

	args = parser.parse_args()

	# =========================================================
	# ORIENT
	# =========================================================
	test_orient()

	# =========================================================
	# PLACEMENT
	# =========================================================
	results: Dict = {}
	if not args.numpy_only:
		stored = np.transpose(synthetic_film(), (2, 1, 0))        # (z, y, x)
		padded = np.ones(tuple(s + 3 for s in stored.shape), dtype=bool)
		padded[2:-1, 1:-2, 3:] = stored
		crop = [[2, 2 + stored.shape[0]], [1, 1 + stored.shape[1]], [3, 3 + stored.shape[2]]]
		check(np.array_equal(vg.orient(padded, 0, crop), synthetic_film()),
			  "orient(padded stored film, 0, crop) reproduces the [x, y, z] test film")
		for k in (1, 2):
			for avg in (False, True):
				test_placement(k, avg, results)
		test_odd_sizes(results)
		if args.json:
			with open(args.json, "w") as fh:
				json.dump(results, fh, indent=1)

	print("{0} failure(s)".format(len(FAILURES)), flush=True)
	sys.exit(1 if FAILURES else 0)
