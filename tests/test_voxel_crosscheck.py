#!/usr/bin/env python
# ===========================================
# I1 cross-check: voxelised examples/one_disk.txt vs the analytic disk path.
# Plain script (no pytest).  The FDTD runs are launched by tests/slurm/i1_crosscheck.slurm.
#
#   build WORK VS [VS ...]    write WORK/disk_vs<VS>.npz for each voxel size VS (um)
#   compare WORK VS [VS ...]  max |dT|, max |dR| between WORK/voxel_<VS> and WORK/disk_<VS>
#
# Geometry (the disk path with -load examples/one_disk.txt -is_point -phi 0.2 -tfilm 0.1):
# square cell 0.2 x 0.2 um, one disk of radius sqrt(0.2 * 0.04 / pi) = 0.050463 um centred
# in the cell (Meep (0, 0)), extruded through a 0.1 um film.  Voxelisation: nx = ny =
# round(0.2 / VS), nz = round(0.1 / VS) (so the voxel cell is exactly the disk path's cell);
# a voxel is solid iff its CENTRE lies inside the disk (distance <= radius).  With even
# nx, ny, film_geometry centres the block at (0, 0), so voxel centres sit at
# (i + 1/2) VS - 0.1, symmetric about the disk centre.  Stored axis order (z, y, x),
# normal axis 0.
# ===========================================

import os
import sys

import numpy as np

CELL = 0.2
TFILM = 0.1
PHI = 0.2
RADIUS = np.sqrt(PHI*CELL*CELL/np.pi)


def tag(vs):
	return "{0:g}".format(float(vs))


def voxelise(vs):
	n = int(round(CELL/vs))
	nz = int(round(TFILM/vs))
	if abs(n*vs - CELL) > 1e-12 or abs(nz*vs - TFILM) > 1e-12:
		raise SystemExit("voxel size {0} does not tile the 0.2 x 0.2 x 0.1 um film".format(vs))
	c = (np.arange(n) + 0.5)*vs - 0.5*CELL
	inside = (c[:, None]**2 + c[None, :]**2) <= RADIUS**2  # [y, x] (symmetric anyway)
	return np.broadcast_to(inside, (nz, n, n)).copy()


def cmd_build(work, *sizes):
	os.makedirs(work, exist_ok=True)
	for vs in sizes:
		g = voxelise(float(vs))
		path = os.path.join(work, "disk_vs{0}.npz".format(tag(vs)))
		np.savez(path, g=g)
		print("{0}: stored shape {1}, solid fraction {2:.6f} (disk path phi {3}), solid voxels per layer {4}".format(
			path, g.shape, float(g.mean()), PHI, int(g[0].sum())))
	return 0


def cmd_compare(work, *sizes):
	status = 0
	for vs in sizes:
		t = tag(vs)
		try:
			v = np.loadtxt(os.path.join(work, "voxel_" + t, "sample_trans-x.txt"))
			d = np.loadtxt(os.path.join(work, "disk_" + t, "sample_trans-x.txt"))
		except OSError as exc:
			print("vs {0}: missing table ({1})".format(t, exc))
			status = 1
			continue
		if v.shape != d.shape or np.max(np.abs(v[:, 0] - d[:, 0])) > 1e-9:
			print("vs {0}: wavelength grids differ".format(t))
			status = 1
			continue
		dT = np.abs(v[:, 1] - d[:, 1])
		dR = np.abs(v[:, 2] - d[:, 2])
		ev = np.max(np.abs(v[:, 1] + np.abs(v[:, 2]) - 1))
		ed = np.max(np.abs(d[:, 1] + np.abs(d[:, 2]) - 1))
		print("vs {0} (res {1}): voxel solid fraction {2:.6f}; max|dT| = {3:.6f} (at wl {4:.4f}), "
			  "max|dR| = {5:.6f} (at wl {6:.4f}); mean|dT| = {7:.6f}; energy balance voxel {8:.2e}, disk {9:.2e}".format(
				t, int(round(1/float(vs))), float(voxelise(float(vs)).mean()), dT.max(), v[np.argmax(dT), 0],
				dR.max(), v[np.argmax(dR), 0], dT.mean(), ev, ed))
		print("  wl     T_disk   T_vox    R_disk   R_vox")
		for row_d, row_v in zip(d, v):
			print("  {0:.4f} {1:.5f} {2:.5f} {3:.5f} {4:.5f}".format(row_d[0], row_d[1], row_v[1], row_d[2], row_v[2]))
	return status


if __name__ == "__main__":
	cmds = {"build": cmd_build, "compare": cmd_compare}
	if len(sys.argv) < 4 or sys.argv[1] not in cmds:
		raise SystemExit("usage: test_voxel_crosscheck.py {build,compare} WORK VS [VS ...]")
	sys.exit(cmds[sys.argv[1]](*sys.argv[2:]))
