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

- Voxel centres and interior faces are exact: get_epsilon_grid() equals the
  model "cell-centred, trilinear, clamped" to 2e-15 on every Yee position,
  and every get_epsilon() sample whose voxel and 26 neighbours share a phase
  is exact, including next to the periodic seams.
- beta = 0 (no projection) is deliberate.  beta = inf with do_averaging=False
  turns the u = 1/2 face nodes into void, shrinking every solid run by one
  node row; with do_averaging=True the result is identical to beta = 0.
- Nothing is upsampled: an m-fold upsample costs m^3 memory, and a
  473 x 356 x 541 stack is already 0.7 GB of float64 weights per rank.
- Periodic seam: the plane x = -Lx/2 (y = -Ly/2) takes the value of voxel
  nx-1 (ny-1), not the 1/2 blend of an interior face, so a structure rolled
  by whole voxels differs from the rolled epsilon at a few interface samples
  (never in the bulk).
- Odd voxel counts: Meep shifts the grid of an odd pixel count by half a
  pixel, so at k = 1 every get_epsilon() sample of an odd lateral axis lies
  on a voxel face (k = 2 is aligned).  The block centre is fixed at (0, 0) by
  the contract; a centre of (-(nx % 2), -(ny % 2)) * voxel_size / 2 would
  align every integer k (a rigid translation).
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
		Meep z of the film mid-plane; the block is centred at (0, 0, z_center).
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
	block = mp.Block(center=mp.Vector3(0, 0, z_center), size=mp.Vector3(Lx, Ly, h), material=grid)
	return Lx, Ly, h, [block]
