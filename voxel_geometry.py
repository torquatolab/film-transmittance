#!/usr/bin/env python3

"""
Voxel array -> Meep film geometry (one MaterialGrid block).

orient() crops a stored voxel array and transposes it to Meep [x, y, z] order;
film_geometry() wraps the oriented array in a single mp.Block holding an
mp.MaterialGrid whose weights are the voxel indicator (1 = solid).

How MaterialGrid samples its weights (Meep 1.31, measured by
tests/test_voxel_geometry.py and the P2 exploration jobs):

- CELL-CENTRED.  Weight [i, j, l] of an (Nx, Ny, Nz) grid sits at
  block_min + (i + 1/2, j + 1/2, l + 1/2) * size / N.  A one-hot weight at
  index 3 of 10 over [-0.5, 0.5] peaks at x = -0.15, not at -0.1667 (node grid).
- Between sample centres the weights are interpolated linearly per axis
  (trilinear), so with beta = 0 u(x) goes from 0 to 1 over one voxel and is
  exactly 1/2 on a face shared by a solid and a void voxel.
- Within half a voxel of a block face the value is CLAMPED to the edge voxel
  (no wrap-around); outside the block the default material (vacuum) applies.
  Across the periodic x/y boundary the voxels meet through the copies Meep adds
  with Simulation(ensure_periodicity=True) (the default).
- The flat weight index is C order over [x, y, z] (x slowest), i.e.
  weights.flatten() of the oriented array.

Therefore the weights are the voxel indicator itself, with grid_size equal to
the oriented shape: each voxel's sample lies at its own centre, which is a
Meep pixel centre when resolution = k / voxel_size and the grid lines coincide
with voxel faces.  Measured consequences (P2 report, Slurm jobs cited there):

- Voxel centres and interior faces are exact: fields.get_chi1inv() at every
  E_x, E_y, E_z node equals 1/epsilon of get_epsilon_grid() and of the model
  "cell-centred, trilinear, clamped" to 3e-15 off the seam planes,
  and every get_epsilon() sample whose voxel and 26 neighbours share a phase
  is exact, including next to the periodic seams.
- beta = 0 (no projection) is deliberate.  beta = inf with do_averaging=False
  turns the u = 1/2 face nodes into void, shrinking every solid run by one
  node row; with do_averaging=True the result is identical to beta = 0.
- Nothing is upsampled: an m-fold upsample costs m^3 memory, and a
  473 x 356 x 541 stack is already 0.7 GB of float64 weights per rank.
- Periodic seam: MaterialGrid clamps to the edge voxel within half a voxel of
  the block edge instead of wrapping, so nodes in that zone (at k = 1 only
  the seam plane itself, which E_y/E_z (x seam) and E_x/E_z (y seam) nodes
  occupy; at k = 2 also the E_x (E_y) nodes a quarter voxel from the x (y)
  seam) take one voxel's value instead of the periodic blend.  A film rolled
  by whole voxels therefore differs from the rolled chi1inv at 0-3.5 % of the
  nodes, all inside that zone, never elsewhere (tests/test_voxel_geometry.py
  check C, P2 job 14009948).
- Meep grid (measured, explore/e8, job 14009633): with the cell centred at
  the origin, the Yee nodes sit at fixed multiples of dx = 1/resolution from
  the origin, independent of the parity of the cell's pixel count (an odd
  count moves the cell edges, not the grid).  Along each axis d, component
  E_d sits at (m + 1/2) dx and the other E components at m dx; get_epsilon()
  (the centred grid) sits at (m + 1/2) dx on every axis.
- Lateral placement (contract amendment A2): the block centre is
  (-(nx % 2), -(ny % 2)) * voxel_size / 2, so the lateral voxel faces sit at
  m * voxel_size and the voxel centres at (m + 1/2) * voxel_size for either
  parity: at resolution = k / voxel_size no get_epsilon() sample and no E_x
  (E_y) node along x (y) lies on a lateral voxel face.  The transverse
  components' nodes (E_y, E_z along x; E_x, E_z along y) are on the m dx
  lattice by construction of the Yee cell, so they lie on lateral faces for
  every integer k (all of them at k = 1, half at k = 2) and take the trilinear
  1/2 blend there (verified node by node with fields.get_chi1inv).
- Vertical placement (amendment A3): aligned_z_center() moves the film by at
  most half a pixel so that its z faces sit at (m + 1/2) dx, midway between
  the E_x / E_y nodes (which then sample voxel centres at k = 1).  Measured
  on the 7-layer voxel stack, faces on the nodes gave max |dT| 0.037 (res
  100), faces midway 0.005 (P2 slab jobs 14009198, 14009799; the latter for
  both parities of the cell's z pixel count).  E_z nodes then lie on the z
  faces; exactly on the film's outer faces (the block boundary) Meep may put
  such a node inside or outside the block depending on float rounding.
- do_averaging=True (-interface_averaging) is not recommended: MaterialGrid
  smoothing of 0/1 voxel weights overshoots (epsilon 0.947 < 1 next to a
  face) and a 7-layer voxel stack converged worse than without it.

Sam Dawley
09/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import List, Optional, Sequence, Tuple

# ----- numerics -----
import meep as mp
import numpy as np

# ============================================================
# CONSTANTS
# ============================================================

# MaterialGrid projection (u -> tanh step at ETA); 0 = none (see module docstring).
# Module-level only so the comparison tests can override it.
PROJECTION_BETA: float = 0.0
PROJECTION_ETA: float = 0.5

# ============================================================
# ORIENTATION
# ============================================================

def orient(voxels: np.ndarray, normal_axis: int = 0, crop: Optional[Sequence] = None) -> np.ndarray:
	"""Crop a stored voxel array and transpose it to Meep [x, y, z] order.

	Parameters
	----------
	voxels : np.ndarray
		3D array in stored axis order (TIFF: axis 0 = frames).
	normal_axis : int
		Stored axis that is the film normal (Meep z).
	crop : sequence of three [start, stop] pairs, optional
		Half-open slices in STORED axis order, applied before the transpose
		(same format as voxel_crop.best_crop()["slices"]).  None keeps all.

	Returns
	-------
	np.ndarray
		View indexed [x, y, z]: with remaining stored axes r0 < r1,
		r0 -> y, r1 -> x, normal -> z, i.e. np.transpose(v, (r1, r0, n)).
	"""
	if not isinstance(voxels, np.ndarray) or voxels.ndim != 3:
		raise ValueError("orient: voxels must be a 3D numpy array")
	if normal_axis not in (0, 1, 2):
		raise ValueError("orient: normal_axis must be 0, 1 or 2, got {0!r}".format(normal_axis))
	v = voxels
	if crop is not None:
		if len(crop) != 3:
			raise ValueError("orient: crop needs three [start, stop] pairs (stored axis order)")
		slices = []
		for ax, pair in enumerate(crop):
			if len(pair) != 2:
				raise ValueError("orient: crop entry {0} is not a [start, stop] pair".format(ax))
			s0, s1 = int(pair[0]), int(pair[1])
			if not 0 <= s0 < s1 <= voxels.shape[ax]:
				raise ValueError("orient: crop [{0}, {1}] out of range for axis {2} of length {3}".format(
					s0, s1, ax, voxels.shape[ax]))
			slices.append(slice(s0, s1))
		v = v[tuple(slices)]
	r0, r1 = [a for a in (0, 1, 2) if a != normal_axis]
	return np.transpose(v, (r1, r0, normal_axis))

# ============================================================
# GEOMETRY
# ============================================================

def film_geometry(oriented: np.ndarray, voxel_size: float, medium_solid: mp.Medium, medium_void: mp.Medium,
				  z_center: float, do_averaging: bool = False) -> Tuple[float, float, float, List]:
	"""Build the film as one MaterialGrid block.

	Parameters
	----------
	oriented : np.ndarray
		Voxel indicator indexed [x, y, z] (output of orient); nonzero = solid.
	voxel_size : float
		Voxel edge length in Meep units (um).
	medium_solid, medium_void : mp.Medium
		Materials for weight 1 (solid) and weight 0 (void).
	z_center : float
		Meep z of the film mid-plane (pass aligned_z_center(...) for voxel faces
		midway between grid nodes).  The block centre is
		(-(nx % 2) * voxel_size / 2, -(ny % 2) * voxel_size / 2, z_center).
	do_averaging : bool
		MaterialGrid subpixel smoothing (-interface_averaging).

	Returns
	-------
	Tuple[float, float, float, list]
		Lx, Ly, h (voxel_size * shape) and [mp.Block].
	"""
	if not isinstance(oriented, np.ndarray) or oriented.ndim != 3 or min(oriented.shape) < 1:
		raise ValueError("film_geometry: oriented must be a non-empty 3D numpy array")
	if not (np.isfinite(voxel_size) and voxel_size > 0):
		raise ValueError("film_geometry: voxel_size must be > 0, got {0!r}".format(voxel_size))
	nx, ny, nz = oriented.shape
	Lx, Ly, h = voxel_size*nx, voxel_size*ny, voxel_size*nz
	# Cell-centred samples: one weight per voxel, C order over [x, y, z].
	weights = np.ascontiguousarray(oriented, dtype=bool).astype(np.float64)
	grid = mp.MaterialGrid(mp.Vector3(nx, ny, nz), medium_void, medium_solid,
						   weights=weights, do_averaging=bool(do_averaging),
						   beta=PROJECTION_BETA, eta=PROJECTION_ETA)
	# A2: lateral voxel faces on the m * voxel_size lattice for either parity (rigid periodic translation)
	center = mp.Vector3(-(nx % 2)*voxel_size/2, -(ny % 2)*voxel_size/2, z_center)
	block = mp.Block(center=center, size=mp.Vector3(Lx, Ly, h), material=grid)
	return Lx, Ly, h, [block]


def aligned_z_center(z_center: float, h: float, resolution: float, cell_z: float) -> float:
	"""Film centre moved by at most half a pixel so both z faces lie midway between grid nodes.

	Parameters
	----------
	z_center : float
		Requested Meep z of the film mid-plane (cell centred at the origin).
	h : float
		Film thickness; both faces align only if h * resolution is an integer
		(true when resolution * voxel_size is an integer).
	resolution : float
		Meep resolution (pixels per unit length).
	cell_z : float
		Cell height.  The Meep node lattice m / resolution does not depend on
		the parity of the cell's pixel count (measured, see module docstring);
		cell_z is used only to check that the shifted film lies in the cell.

	Returns
	-------
	float
		z_center + s with |s| <= 0.5 / resolution such that the bottom face
		z_center + s - h / 2 equals (m + 1/2) / resolution for an integer m.
	"""
	for name, val in (("z_center", z_center), ("h", h), ("resolution", resolution), ("cell_z", cell_z)):
		if not np.isfinite(val):
			raise ValueError("aligned_z_center: {0} must be finite, got {1!r}".format(name, val))
	if not (h > 0 and resolution > 0 and cell_z > 0):
		raise ValueError("aligned_z_center: h, resolution and cell_z must be > 0")
	bottom_px = (z_center - 0.5*h)*resolution
	# round away float noise before choosing the nearest half-integer (ties go up: +1/2 pixel)
	bottom_px = np.round(bottom_px, 9)
	target_px = np.floor(bottom_px) + 0.5
	shifted = z_center + (target_px - bottom_px)/resolution
	assert abs(shifted - z_center) <= 0.5/resolution + 1e-12, "aligned_z_center: shift exceeds half a pixel"
	if abs(shifted) + 0.5*h > 0.5*cell_z + 1e-12:
		raise ValueError("aligned_z_center: film [{0:g}, {1:g}] does not fit in the cell of height {2:g}".format(
			shifted - 0.5*h, shifted + 0.5*h, cell_z))
	return float(shifted)
