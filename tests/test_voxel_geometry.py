#!/usr/bin/env python3

"""
Tests for voxel_geometry.orient and voxel_geometry.film_geometry.

Part 1 (pure numpy, exact): axis mapping of orient for normal_axis 0, 1, 2 on an
asymmetric array, crop application, and rejection of bad arguments.

Part 2 (Meep, run on a compute node): an asymmetric voxel film (an L-shaped
solid, solids joined across the periodic x and y seams, asymmetric seam voxels,
single voxels on both film faces) with even (12 x 10 x 8) and odd (13 x 11 x 8)
lateral counts, placed with film_geometry (amendment A2 centre) in a periodic
cell (k_point = 0) at resolution = k / voxel_size, k = 1, 2, do_averaging False
(and True with --averaging_on).  Checks:

  A. (averaging off) Yee nodes of E_x, E_y, E_z, modelled as (m or m + 1/2) / res
     from the origin: fields.get_chi1inv at every node equals 1/eps of
     get_epsilon_grid and of a cell-centred, trilinear, edge-clamped numpy
     model (<= 1e-9) off the seam planes; no E_x (E_y) node lies on an x (y)
     voxel face.  Reported: fraction of nodes on lateral faces per axis, seam
     plane mismatch, fraction of face nodes carrying a solid/void blend.
  B. get_epsilon() over one lateral period: no sample on a lateral voxel face,
     bulk samples (voxel and 26 periodic neighbours of one phase) exact, and
     (averaging off) nearest-phase solid fraction within FRACTION_TOL.
  C. Periodic wrap: the film rolled by whole voxels in x and y.  get_epsilon
     bulk samples must be identical; per component, every node whose chi1inv
     differs from the rolled one must lie within half a voxel of a seam plane
     (MaterialGrid's clamp zone).  Counts are reported.

Part 3 (Meep): aligned_z_center for k = 1, 2 and cell heights with even, odd
and non-integer pixel counts, three requested centres.  An independent sharp
mp.Block probe locates the E_x z nodes; the shift must be <= 1/2 pixel, every
voxel z face at (m + 1/2) / res, chi1inv at every z node equal to the model,
no E_x / E_y z node on a voxel z face (no u = 1/2 value), and every z face on a
get_epsilon() sample plane.  A control shifted by half a pixel reports the
number of E_x nodes with u = 1/2.

Exits nonzero on any failure.  Usage: test_voxel_geometry.py [--numpy-only] [--averaging_on] [--json FILE]

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
# PART 2: LATERAL PLACEMENT (MEEP)
# ============================================================

COMPONENTS = ("Ex", "Ey", "Ez")


def block_min(n: int) -> float:
	"""Lateral block minimum of film_geometry for n voxels (amendment A2 centre)."""
	return -(n + n % 2)*VOXEL_SIZE/2


def build_sim(a: np.ndarray, k: int, avg: bool, z_center: float = 0.0, cell_z: float = 0.0):
	"""Periodic cell (k_point = 0), film from film_geometry at resolution k / VOXEL_SIZE; no fields are run."""
	import meep as mp
	assert k >= 1 and a.ndim == 3, "build_sim: bad arguments"
	Lx, Ly, h, geom = vg.film_geometry(a, VOXEL_SIZE, mp.Medium(epsilon=EPS_SOLID), mp.Medium(epsilon=EPS_VOID),
									   z_center, do_averaging=avg)
	cz = cell_z if cell_z > 0 else h + 12*VOXEL_SIZE
	sim = mp.Simulation(cell_size=mp.Vector3(Lx, Ly, cz), resolution=k/VOXEL_SIZE, geometry=geom,
						k_point=mp.Vector3())
	sim.init_sim()
	return sim


def lateral_nodes(n: int, k: int, half: bool) -> Tuple[np.ndarray, np.ndarray]:
	"""Node coordinates of one lateral period on the Meep lattice (m or m + 1/2) / res.

	Returns (voxel coordinate in [0, n) measured from the block minimum, query coordinate wrapped to [-L/2, L/2)).
	"""
	res = k/VOXEL_SIZE
	b = block_min(n)
	L = n*VOXEL_SIZE
	m0 = int(np.ceil(b*res - (0.5 if half else 0.0) - 1e-9))
	t = (m0 + np.arange(n*k) + (0.5 if half else 0.0))/res
	u = (t - b)/VOXEL_SIZE
	assert u.min() > -1e-9 and u.max() < n - 1e-9, "lateral_nodes: nodes outside one period"
	q = (t + L/2) % L - L/2
	return u, q


def z_nodes(z_lo: float, z_hi: float, k: int, half: bool) -> np.ndarray:
	"""Meep lattice coordinates (m or m + 1/2) / res in [z_lo, z_hi]."""
	res = k/VOXEL_SIZE
	s = 0.5 if half else 0.0
	m = np.arange(int(np.ceil(z_lo*res - s - 1e-9)), int(np.floor(z_hi*res - s + 1e-9)) + 1)
	return (m + s)/res


def chi1inv_at(sim, comp: str, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
	"""fields.get_chi1inv(comp, comp direction) at every grid point (xs x ys x zs)."""
	import meep as mp
	c, d = {"Ex": (mp.Ex, mp.X), "Ey": (mp.Ey, mp.Y), "Ez": (mp.Ez, mp.Z)}[comp]
	out = np.empty((xs.size, ys.size, zs.size))
	for i, x in enumerate(xs):
		for j, y in enumerate(ys):
			for l, z in enumerate(zs):
				out[i, j, l] = np.real(sim.fields.get_chi1inv(c, d, mp.vec(float(x), float(y), float(z))))
	return out


def node_values(sim, a: np.ndarray, k: int, z_center: float) -> Dict:
	"""Per component: node voxel coordinates, chi1inv at the nodes, 1/eps of get_epsilon_grid there, face masks."""
	nx, ny, nz = a.shape
	h = nz*VOXEL_SIZE
	res = k/VOXEL_SIZE
	out = {}
	for comp in COMPONENTS:
		ux, qx = lateral_nodes(nx, k, comp == "Ex")
		uy, qy = lateral_nodes(ny, k, comp == "Ey")
		zs = z_nodes(z_center - h/2 - 1.0/res, z_center + h/2 + 1.0/res, k, comp == "Ez")
		uz = (zs - (z_center - h/2))/VOXEL_SIZE
		ci = chi1inv_at(sim, comp, qx, qy, zs)
		ox, oy = np.argsort(qx), np.argsort(qy)
		eg = np.real(np.asarray(sim.get_epsilon_grid(qx[ox], qy[oy], zs)))
		eg = eg[np.argsort(ox)][:, np.argsort(oy)]
		out[comp] = {"ux": ux, "uy": uy, "uz": uz, "chi1inv": ci, "inv_eps_grid": 1.0/eg,
					 "face_x": np.abs(ux - np.round(ux)) < 1e-6, "face_y": np.abs(uy - np.round(uy)) < 1e-6,
					 "seam_x": np.abs(ux) < 1e-6, "seam_y": np.abs(uy) < 1e-6,
					 # MaterialGrid clamps within half a voxel of the block edge instead of wrapping
					 "clamp_x": np.minimum(ux, nx - ux) < 0.5 - 1e-9, "clamp_y": np.minimum(uy, ny - uy) < 0.5 - 1e-9}
	return out


def film_samples(sim, a: np.ndarray, k: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], float]:
	"""get_epsilon() over one lateral period of the film (coordinates wrapped into the block span); voxel index per axis."""
	e = np.real(np.asarray(sim.get_epsilon()))
	coords = [np.asarray(c, dtype=float) for c in sim.get_array_metadata()[:3]]
	assert e.shape == tuple(c.size for c in coords), "get_epsilon / metadata shape mismatch"
	sel, idx, on_face = [], [], []
	for ax, (n, c) in enumerate(zip(a.shape, coords)):
		if ax < 2:
			L = n*VOXEL_SIZE
			u = ((c - block_min(n)) % L)/VOXEL_SIZE
			u = np.where(np.abs(u - n) < 1e-9, 0.0, u)
			_, first = np.unique(np.round(u, 6), return_index=True)   # drop the duplicate ghost column
			keep = np.sort(first)
			keep = keep[np.argsort(u[keep])]
		else:
			u = (c + 0.5*n*VOXEL_SIZE)/VOXEL_SIZE
			keep = np.nonzero((u > -1e-9) & (u < n - 1e-9))[0]
		sel.append(keep)
		idx.append(np.floor(u[keep] + 1e-9).astype(int))
		on_face.append(np.abs(u[keep] - np.round(u[keep])) < 1e-6)
	assert len(sel[0]) == k*a.shape[0] and len(sel[1]) == k*a.shape[1], "film_samples: expected k samples per voxel"
	sub = e[np.ix_(*sel)]
	face_lat = on_face[0][:, None, None] | on_face[1][None, :, None]
	return sub, tuple(idx), float(np.mean(np.broadcast_to(face_lat, sub.shape)))


def test_placement(a: np.ndarray, k: int, avg: bool, out: Dict) -> None:
	"""Lateral placement checks for one film shape, resolution multiple and averaging flag."""
	assert vg.PROJECTION_BETA == 0.0, "material_model assumes no projection (beta = 0)"
	nx, ny, nz = a.shape
	tag = "{0}x{1}x{2} k={3} do_averaging={4}".format(nx, ny, nz, k, avg)
	sim = build_sim(a, k, avg)
	rec: Dict = {"voxel_fraction": float(a.mean())}
	w = a.astype(float)
	bulk_vox = bulk_mask(a)

	# ----- A: Yee nodes of every E component (do_averaging off: point sampling) -----
	nodes = None
	if not avg:
		nodes = node_values(sim, a, k, 0.0)
		rec["nodes"] = {}
		for comp, nd in nodes.items():
			ci, ie = nd["chi1inv"], nd["inv_eps_grid"]
			seam = nd["seam_x"][:, None, None] | nd["seam_y"][None, :, None]
			seam = np.broadcast_to(seam, ci.shape)
			um = material_model(w, nd["ux"], nd["uy"], nd["uz"])
			im = 1.0/(EPS_VOID + (EPS_SOLID - EPS_VOID)*um)
			d_grid = np.abs(ci - ie)
			d_model = np.abs(ci - im)
			check(float(d_grid[~seam].max()) <= 1e-9 and float(d_model[~seam].max()) <= 1e-9,
				  "{0} A {1}: chi1inv at modelled Yee nodes == 1/get_epsilon_grid == cell-centred trilinear model off the "
				  "seam planes (max {2:.2e}, {3:.2e}; {4} nodes)".format(tag, comp, float(d_grid[~seam].max()),
				  float(d_model[~seam].max()), int((~seam).sum())))
			seam_grid = float(d_grid[seam].max()) if seam.any() else 0.0
			seam_model = float(d_model[seam].max()) if seam.any() else 0.0
			fx, fy = float(nd["face_x"].mean()), float(nd["face_y"].mean())
			own = fx if comp == "Ex" else fy if comp == "Ey" else None
			if own is not None:
				check(own == 0.0, "{0} A {1}: no node on a voxel face along its own axis (fraction {2})".format(tag, comp, own))
			face_nodes = np.broadcast_to(nd["face_x"][:, None, None] | nd["face_y"][None, :, None], ci.shape)
			blend = face_nodes & (np.abs(um - np.round(um)) > 1e-9)
			rec["nodes"][comp] = {"n_nodes": int(ci.size), "face_fraction_x": fx, "face_fraction_y": fy,
								  "seam_node_fraction": float(seam.mean()), "seam_max_vs_grid": seam_grid,
								  "seam_max_vs_model": seam_model, "blended_face_node_fraction": float(blend.mean())}
			print("INFO {0} A {1}: nodes on lateral voxel faces: along x {2:.3f}, along y {3:.3f}; seam-plane nodes {4:.4f} "
				  "(max |chi1inv - 1/eps_grid| {5:.3g}, vs clamped model {6:.3g}); face nodes with a solid/void blend {7:.4f}".format(
				  tag, comp, fx, fy, float(seam.mean()), seam_grid, seam_model, float(blend.mean())), flush=True)

	# ----- B: get_epsilon() over the film vs the voxel layout -----
	e, (ix, iy, iz), face_frac = film_samples(sim, a, k)
	check(face_frac == 0.0 and e.shape == (k*nx, k*ny, k*nz),
		  "{0} B: {1} get_epsilon samples, none on a lateral voxel face (fraction {2})".format(tag, e.shape, face_frac))
	ref_solid = a[np.ix_(ix, iy, iz)]
	ref = np.where(ref_solid, EPS_SOLID, EPS_VOID)
	diff = np.abs(e - ref)
	bulk = bulk_vox[np.ix_(ix, iy, iz)]
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
		  "solid fraction voxel {4:.4f} nearest-phase {5:.4f} linear-eps {6:.4f}".format(
			  tag, rec["exact_mismatch_fraction"], rec["nearest_phase_mismatch_fraction"], rec["interface_sample_fraction"],
			  rec["voxel_fraction"], rec["nearest_phase_fraction"], rec["linear_eps_fraction"]), flush=True)
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
		sim2 = build_sim(b, k, avg)
		e2, _, _ = film_samples(sim2, b, k)
		rolled = np.roll(np.roll(e, sx*k, 0), sy*k, 1)
		dd = np.abs(e2 - rolled)
		bulk2 = np.roll(np.roll(bulk, sx*k, 0), sy*k, 1)
		r = {"get_epsilon_differ_fraction": float(np.mean(dd > 1e-9)), "get_epsilon_max_abs_diff": float(dd.max()),
			 "nearest_phase_differ_fraction": float(np.mean(
				 (e2 > 0.5*(EPS_SOLID + EPS_VOID)) != (rolled > 0.5*(EPS_SOLID + EPS_VOID))))}
		check(float(dd[bulk2].max()) <= 1e-9,
			  "{0} C: roll ({1},{2}) voxels: get_epsilon bulk samples identical (max {3:.2e}); all samples differ fraction "
			  "{4:.4f}, max diff {5:.3f}, nearest-phase differ {6:.4f}".format(tag, sx, sy, float(dd[bulk2].max()),
			  r["get_epsilon_differ_fraction"], r["get_epsilon_max_abs_diff"], r["nearest_phase_differ_fraction"]))
		if nodes is not None:
			nodes2 = node_values(sim2, b, k, 0.0)
			for comp in COMPONENTS:
				c1 = np.roll(np.roll(nodes[comp]["chi1inv"], sx*k, 0), sy*k, 1)
				c2 = nodes2[comp]["chi1inv"]
				dn = np.abs(c2 - c1) > 1e-9
				shp = dn.shape
				def zone(nd, key):
					return np.broadcast_to(nd[key + "_x"][:, None, None] | nd[key + "_y"][None, :, None], shp)
				def rolled(m):
					return np.roll(np.roll(m, sx*k, 0), sy*k, 1)
				# nodes that are, before or after the roll, on a seam plane / inside the half-voxel clamp zone
				plane = zone(nodes2[comp], "seam") | rolled(zone(nodes[comp], "seam"))
				clamp = zone(nodes2[comp], "clamp") | rolled(zone(nodes[comp], "clamp"))
				outside = dn & ~clamp
				r[comp] = {"differ_fraction": float(dn.mean()), "differ_count": int(dn.sum()),
						   "differ_on_seam_plane": int((dn & plane).sum()), "differ_in_clamp_zone": int((dn & clamp).sum()),
						   "differ_outside_clamp_zone": int(outside.sum())}
				print("INFO {0} C: roll ({1},{2}) {3} nodes differ: {4} of {5} ({6:.4f}); on a seam plane {7}; within half a "
					  "voxel of a seam (clamp zone) {8}; elsewhere {9}".format(tag, sx, sy, comp, int(dn.sum()), dn.size,
					  float(dn.mean()), int((dn & plane).sum()), int((dn & clamp).sum()), int(outside.sum())), flush=True)
				check(int(outside.sum()) == 0, "{0} C: roll ({1},{2}) {3}: every differing node lies within half a voxel "
					  "of a seam plane".format(tag, sx, sy, comp))
		rec["roll"]["{0},{1}".format(sx, sy)] = r
	out[tag] = rec

# ============================================================
# PART 3: Z ALIGNMENT (MEEP)
# ============================================================

def probe_ex_z_nodes(cell_z: float, k: int) -> Tuple[float, float]:
	"""Locate the E_x nodes bracketing a sharp mp.Block face (eps_averaging off) from get_chi1inv interpolation."""
	import meep as mp
	res = k/VOXEL_SIZE
	z0 = 0.0137 + 0.37/res
	sim = mp.Simulation(cell_size=mp.Vector3(2*VOXEL_SIZE, 2*VOXEL_SIZE, cell_z), resolution=res, eps_averaging=False,
						k_point=mp.Vector3(), geometry=[mp.Block(center=mp.Vector3(0, 0, z0 + 0.25*cell_z),
						size=mp.Vector3(mp.inf, mp.inf, 0.5*cell_z), material=mp.Medium(epsilon=EPS_SOLID))])
	sim.init_sim()
	qs = (np.floor(z0*res) + np.arange(-40, 41)/20.0)/res
	v = np.array([np.real(sim.fields.get_chi1inv(mp.Ex, mp.X, mp.vec(0.0, 0.0, float(q)))) for q in qs])
	below = float(qs[np.nonzero(np.abs(v - 1.0) < 1e-12)[0].max()])
	above = float(qs[np.nonzero(np.abs(v - 1.0/EPS_SOLID) < 1e-12)[0].min()])
	return below*res, above*res


def test_z_alignment(k: int, cell_z: float, out: Dict) -> None:
	"""aligned_z_center: faces midway between E_x/E_y nodes, verified with Meep for one cell height."""
	res = k/VOXEL_SIZE
	npx = int(np.floor(cell_z*res + 0.5))
	tag = "z-align k={0} cell_z={1:g} ({2} px, {3})".format(k, cell_z, npx, "odd" if npx % 2 else "even")
	prof = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
	a = np.broadcast_to(prof, (4, 4, prof.size)).copy()
	h = prof.size*VOXEL_SIZE
	rec: Dict = {"cell_pixels": npx}

	# node lattice for this cell (independent probe)
	lo, hi = probe_ex_z_nodes(cell_z, k)
	check(abs(lo - round(lo)) < 1e-9 and abs(hi - lo - 1.0) < 1e-9,
		  "{0}: probe E_x z nodes bracketing a block face at {1:.4f} and {2:.4f} px (integers, one pixel apart)".format(tag, lo, hi))

	for requested in (0.0137, -0.0213, 0.0):
		zc = vg.aligned_z_center(requested, h, res, cell_z)
		shift_px = (zc - requested)*res
		faces_px = (zc - h/2 + np.arange(prof.size + 1)*VOXEL_SIZE)*res
		arith = float(np.abs(faces_px - np.floor(faces_px) - 0.5).max())
		check(abs(shift_px) <= 0.5 + 1e-9 and arith < 1e-6,
			  "{0} request {1:+.4f}: shift {2:+.4f} px (<= 0.5), every z face at m + 1/2 px (max dev {3:.1e})".format(
				  tag, requested, shift_px, arith))
		sim = build_sim(a, k, False, z_center=zc, cell_z=cell_z)
		nd = node_values(sim, a, k, zc)
		for comp in COMPONENTS:
			ci = nd[comp]["chi1inv"]
			uz = nd[comp]["uz"]
			um = material_model(a.astype(float), nd[comp]["ux"], nd[comp]["uy"], uz)
			im = 1.0/(EPS_VOID + (EPS_SOLID - EPS_VOID)*um)
			dm = float(np.abs(ci - im).max())
			zface_vox = np.abs(uz - np.round(uz)) < 1e-6
			outer = (np.abs(uz) < 1e-6) | (np.abs(uz - prof.size) < 1e-6)
			half = np.abs(um - 0.5) < 1e-9
			dm = float(np.abs(ci - im)[:, :, ~outer].max())
			check(dm <= 1e-9, "{0} request {1:+.4f} {2}: chi1inv at z nodes == trilinear model, outer film faces excluded "
				  "(max {3:.2e})".format(tag, requested, comp, dm))
			if outer.any():
				u_meep = (1.0/ci[:, :, outer] - EPS_VOID)/(EPS_SOLID - EPS_VOID)
				print("INFO {0} request {1:+.4f} {2}: nodes exactly on the outer film faces (block boundary) have u = {3} "
					  "(bottom, top; model clamps to the edge voxel: {4})".format(tag, requested, comp,
					  [round(float(v), 6) for v in u_meep[0, 0]], [round(float(v), 6) for v in um[0, 0, outer]]), flush=True)
			if comp in ("Ex", "Ey"):
				check(not zface_vox.any() and not half.any(),
					  "{0} request {1:+.4f} {2}: no z node on a voxel z face and no u = 1/2 value ({3} nodes, u values {4})".format(
						  tag, requested, comp, uz.size, sorted(set(np.round(um.ravel(), 4)))))
			else:
				print("INFO {0} request {1:+.4f} Ez: fraction of z nodes on voxel z faces {2:.3f}".format(
					tag, requested, float(zface_vox.mean())), flush=True)
		# get_epsilon (centred grid) z coordinates: faces coincide with samples (midway between E_x nodes)
		zc_meta = np.asarray(sim.get_array_metadata()[2], dtype=float)
		hit = [bool(np.min(np.abs(zc_meta - f/res)) < 1e-9) for f in faces_px]
		e = np.real(np.asarray(sim.get_epsilon()))
		check(all(hit), "{0} request {1:+.4f}: every z face coincides with a get_epsilon sample plane".format(tag, requested))
		# negative control: faces on the nodes (unaligned) give u = 1/2 at E_x nodes
		zc_bad = zc + 0.5/res
		nd_bad = node_values(build_sim(a, k, False, z_center=zc_bad, cell_z=cell_z), a, k, zc_bad)
		um_bad = material_model(a.astype(float), nd_bad["Ex"]["ux"], nd_bad["Ex"]["uy"], nd_bad["Ex"]["uz"])
		rec["request {0:+.4f}".format(requested)] = {"z_center": zc, "shift_px": shift_px,
			"Ex_nodes_u_half_aligned": 0, "Ex_nodes_u_half_control": int(np.sum(np.abs(um_bad - 0.5) < 1e-9)),
			"get_epsilon_film_mean": float(e.mean())}
		print("INFO {0} request {1:+.4f}: control (film +0.5 px, faces on E_x nodes): {2} E_x nodes with u = 1/2".format(
			tag, requested, rec["request {0:+.4f}".format(requested)]["Ex_nodes_u_half_control"]), flush=True)
	try:
		vg.aligned_z_center(0.0, 2*cell_z, res, cell_z)
		check(False, "{0}: aligned_z_center rejects a film taller than the cell".format(tag))
	except ValueError:
		check(True, "{0}: aligned_z_center rejects a film taller than the cell".format(tag))
	out[tag] = rec

# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# ----- optional -----
	parser.add_argument("--numpy-only", action="store_true", help="run only the orient tests")
	parser.add_argument("--json", type=str, default="", help="write the numbers to this JSON file")
	parser.add_argument("--averaging_on", action="store_true", help="also run the placement checks with do_averaging=True")

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
		for shape in ((12, 10, 8), (13, 11, 8)):
			for k in (1, 2):
				for avg in ((False, True) if args.averaging_on else (False,)):
					test_placement(synthetic_film(*shape), k, avg, results)
		for k in (1, 2):
			for extra in (0.0, 1.0, 0.5):
				test_z_alignment(k, 0.30 + extra*VOXEL_SIZE/k, results)
		if args.json:
			with open(args.json, "w") as fh:
				json.dump(results, fh, indent=1)

	print("{0} failure(s)".format(len(FAILURES)), flush=True)
	sys.exit(1 if FAILURES else 0)
