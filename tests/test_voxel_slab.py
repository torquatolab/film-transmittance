#!/usr/bin/env python3

"""
Uniform voxel slab vs the analytic normal-incidence (Airy) slab.

Supplementary mode --film multilayer: an asymmetric 7-layer voxel stack
(internal solid/void voxel faces inside the MaterialGrid) against the exact
characteristic-matrix result, to judge -interface_averaging.

An all-solid voxel film (eps 2.4025, n = 1.55) built by voxel_geometry.film_geometry
sits in vacuum in a periodic x/y cell with PML in z and a normally incident,
x-polarised plane wave.  The z layout, the multi-Gaussian source, the flux
monitors and the normalisation reproduce film_transmittance.py (-ScattPower):
a vacuum reference run saves the reflection-monitor flux (save_flux), the film
run subtracts it (load_minus_flux), and T = trans/inc, R = refl/inc.

For each resolution and each do_averaging setting the maxima of |T - T_Airy|,
||R| - R_Airy| and |T + |R| - 1| are printed and written to JSON.
Pass: at every setting max |dT|, |dR| <= 0.01 at the lowest resolution and
strictly smaller at the highest one.  Exits nonzero on failure.

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
import meep as mp
import numpy as np

# ----- local modules -----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import voxel_geometry  # noqa: E402
from voxel_geometry import film_geometry  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

EPS_SOLID = 2.4025
VOXEL_SIZE = 0.01
KS = (8.0553657784, 16.5346981768)   # 380-780 nm, as in the contract
TOL = 0.01

# ============================================================
# PHYSICS
# ============================================================

def airy_slab(wl: np.ndarray, n: float, d: float) -> Tuple[np.ndarray, np.ndarray]:
	"""Return (T, R) of a lossless slab of index n, thickness d, in vacuum at normal incidence."""
	assert n > 0 and d > 0, "airy_slab: n and d must be positive"
	r12 = (1.0 - n)/(1.0 + n)
	ph = np.exp(2j*2.0*np.pi*n*d/wl)
	r = r12*(1.0 - ph)/(1.0 - r12**2*ph)
	t = (1.0 - r12**2)*np.exp(1j*2.0*np.pi*n*d/wl)/(1.0 - r12**2*ph)
	return np.abs(t)**2, np.abs(r)**2


def tmm_stack(wl: np.ndarray, layers: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
	"""Return (T, R) of a lossless layer stack [(n, d), ...] in vacuum at normal incidence (characteristic matrices)."""
	assert len(layers) > 0, "tmm_stack: empty stack"
	T = np.empty(wl.size)
	R = np.empty(wl.size)
	for iw, lam in enumerate(wl):
		M = np.eye(2, dtype=complex)
		for n, d in layers:
			dl = 2.0*np.pi*n*d/lam
			M = M @ np.array([[np.cos(dl), -1j*np.sin(dl)/n], [-1j*n*np.sin(dl), np.cos(dl)]])
		B, C = M[0, 0] + M[0, 1], M[1, 0] + M[1, 1]
		T[iw] = np.abs(2.0/(B + C))**2
		R[iw] = np.abs((B - C)/(B + C))**2
	return T, R


def film_profile(kind: str, nvox: int) -> np.ndarray:
	"""Solid indicator along z (index 0 = bottom): all solid, or a fixed asymmetric multilayer."""
	assert kind in ("uniform", "multilayer"), "unknown film kind"
	if kind == "uniform":
		return np.ones(nvox, dtype=bool)
	runs = [(True, 5), (False, 3), (True, 7), (False, 4), (True, 6), (False, 2), (True, 3)]
	prof = np.concatenate([np.full(m, s) for s, m in runs])
	assert prof.size == nvox, "multilayer profile is 30 voxels; use --nvox 30"
	return prof


def run_stage(args, res: float, avg: bool, ref: bool, tag: str) -> Dict:
	"""Run one reference or film stage with film_transmittance.py's layout; return fluxes."""
	assert res > 0, "resolution must be positive"
	kmin, kmax = KS
	wavelen_max, wavelen_min = 2.0*np.pi/kmin, 2.0*np.pi/kmax
	f_min, f_max = 1.0/wavelen_max, 1.0/wavelen_min
	fcen, fwidth = 0.5*(f_max + f_min), f_max - f_min
	prof = film_profile(args.film, args.nvox)
	vox = np.broadcast_to(prof, (args.lateral, args.lateral, args.nvox)).copy()
	h = VOXEL_SIZE*args.nvox
	PML_thickness = args.tpml*wavelen_max
	z_pos_detector, z_pos_detector2, z_pos_source = -args.ddet, h + args.ddet, h + args.dsrc
	z_min = z_pos_detector - args.dpml - PML_thickness
	z_max = z_pos_source + args.dpml + PML_thickness
	cell_height = z_max - z_min
	z_center_offset = (z_min + z_max)/2
	film_z_center = 0.5*h - z_center_offset + args.z_offset_px/res
	Lx, Ly, hh, geometry = film_geometry(vox, VOXEL_SIZE, mp.Medium(epsilon=EPS_SOLID),
										 mp.Medium(epsilon=1.0), film_z_center, do_averaging=avg)
	assert abs(hh - h) < 1e-12, "film_geometry thickness mismatch"
	if args.geometry == "blocks":
		# comparison only: the same z profile as plain mp.Block layers (eps_averaging = do_averaging)
		geometry = [mp.Block(center=mp.Vector3(0, 0, film_z_center - 0.5*h + (iz + 0.5)*VOXEL_SIZE),
							 size=mp.Vector3(mp.inf, mp.inf, VOXEL_SIZE), material=mp.Medium(epsilon=EPS_SOLID))
					for iz in np.nonzero(prof)[0]]
	# film faces measured in pixels from the bottom cell edge (integer = on the grid)
	faces_px = [(film_z_center - 0.5*h + 0.5*cell_height)*res, (film_z_center + 0.5*h + 0.5*cell_height)*res]
	if ref:
		geometry = []
	# multi-source branch of film_transmittance.py (fwidth >= 1 here), -decay 1e-4
	sources = []
	f_intv = 1.0/16.*np.sqrt(-np.log(5*1e-4))
	f0, f1 = f_min, f_min + f_intv
	while f0 < f_max:
		sources.append(mp.Source(mp.GaussianSource(f1, 2*f_intv), component=mp.Ex,
								 size=mp.Vector3(Lx, Ly, 0), center=mp.Vector3(0, 0, z_pos_source - z_center_offset)))
		f0 += 2*f_intv
		f1 += 2*f_intv
	courant = min(0.5, 0.9*1.0/np.sqrt(3))
	sim = mp.Simulation(cell_size=mp.Vector3(Lx, Ly, cell_height), force_complex_fields=True,
						boundary_layers=[mp.PML(thickness=PML_thickness, direction=mp.Z)],
						geometry=geometry, Courant=courant, sources=sources,
						k_point=mp.Vector3(), resolution=res,
						eps_averaging=(avg if args.geometry == "blocks" else True))
	trans = sim.add_flux(fcen, fwidth, args.nfreqs, mp.FluxRegion(
		center=mp.Vector3(0, 0, z_pos_detector - z_center_offset), size=mp.Vector3(Lx, Ly, 0)))
	refl = sim.add_flux(fcen, fwidth, args.nfreqs, mp.FluxRegion(
		center=mp.Vector3(0, 0, z_pos_detector2 - z_center_offset), size=mp.Vector3(Lx, Ly, 0)))
	fluxname = "slab-ref-res{0:g}{1}".format(res, tag)
	if not ref:
		sim.load_minus_flux(fluxname, refl)
	sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=args.dft_tol, maximum_run_time=args.maxt))
	out = {"t_end": float(sim.round_time()), "faces_px": faces_px,
		   "freqs": list(mp.get_flux_freqs(trans)),
		   "trans": list(mp.get_fluxes(trans)), "refl": list(mp.get_fluxes(refl))}
	if ref:
		sim.save_flux(fluxname, refl)
	return out

# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# ----- run -----
	parser.add_argument("--outdir", type=str, required=True)
	parser.add_argument("--res", type=float, nargs="+", default=[100.0, 200.0])
	parser.add_argument("--averaging", type=str, nargs="+", default=["off", "on"], choices=["off", "on"])
	parser.add_argument("--z_offset_px", type=float, default=0.0,
		help="shift the film by this many pixels (0 = production layout; informational)")

	# ----- film and layout (film_transmittance.py names) -----
	parser.add_argument("--film", type=str, default="uniform", choices=["uniform", "multilayer"],
		help="uniform = contract test; multilayer = supplementary check of internal voxel faces")
	parser.add_argument("--beta", type=float, default=None,
		help="override voxel_geometry.PROJECTION_BETA (comparison runs only; default keeps the module value)")
	parser.add_argument("--geometry", type=str, default="materialgrid", choices=["materialgrid", "blocks"],
		help="blocks = mp.Block layers instead of film_geometry (comparison runs only)")
	parser.add_argument("--nvox", type=int, default=30)
	parser.add_argument("--lateral", type=int, default=4)
	parser.add_argument("--nfreqs", type=int, default=101)
	parser.add_argument("--ddet", type=float, default=0.2)
	parser.add_argument("--dsrc", type=float, default=0.4)
	parser.add_argument("--dpml", type=float, default=0.3)
	parser.add_argument("--tpml", type=float, default=0.5)
	parser.add_argument("--dft_tol", type=float, default=1e-8)
	parser.add_argument("--maxt", type=float, default=400.0)

	args = parser.parse_args()
	assert args.nfreqs >= 41, "need at least 41 frequencies"
	if args.beta is not None:
		voxel_geometry.PROJECTION_BETA = args.beta
	os.makedirs(args.outdir, exist_ok=True)
	os.chdir(args.outdir)
	tag = ("" if args.film == "uniform" else "-" + args.film) + \
		("" if args.beta is None else "-beta{0:g}".format(args.beta)) + \
		("" if args.geometry == "materialgrid" else "-" + args.geometry) + \
		("" if args.z_offset_px == 0 else "-zoff{0:g}".format(args.z_offset_px))
	prof = film_profile(args.film, args.nvox)
	# (n, d) layers from runs of equal voxels, bottom to top (symmetric for T and R at normal incidence
	# only up to reversal of R's phase, which |r|^2 does not see)
	layers: List[Tuple[float, float]] = []
	for s in prof:
		n_s = np.sqrt(EPS_SOLID) if s else 1.0
		if layers and layers[-1][0] == n_s:
			layers[-1] = (n_s, layers[-1][1] + VOXEL_SIZE)
		else:
			layers.append((n_s, VOXEL_SIZE))

	# =========================================================
	# RUN AND COMPARE
	# =========================================================
	summary: Dict = {"beta": voxel_geometry.PROJECTION_BETA, "film": args.film, "z_offset_px": args.z_offset_px, "nvox": args.nvox, "runs": {}}
	for res in args.res:
		ref = run_stage(args, res, False, True, tag)
		inc = np.array(ref["refl"])
		for avg_s in args.averaging:
			film = run_stage(args, res, avg_s == "on", False, tag)
			wl = 1.0/np.array(film["freqs"])
			T = np.array(film["trans"])/inc
			R = np.array(film["refl"])/inc
			Ta, Ra = tmm_stack(wl, layers)
			if args.film == "uniform":
				Tair, Rair = airy_slab(wl, np.sqrt(EPS_SOLID), VOXEL_SIZE*args.nvox)
				assert np.max(np.abs(Tair - Ta)) < 1e-12 and np.max(np.abs(Rair - Ra)) < 1e-12, "TMM != Airy"
			key = "res{0:g}-avg{1}".format(res, avg_s)
			rec = {"max_dT": float(np.max(np.abs(T - Ta))), "max_dR": float(np.max(np.abs(np.abs(R) - Ra))),
				   "max_energy": float(np.max(np.abs(T + np.abs(R) - 1.0))),
				   "t_end_ref": ref["t_end"], "t_end_film": film["t_end"], "faces_px": film["faces_px"]}
			summary["runs"][key] = rec
			np.savetxt("slab_{0}{1}.txt".format(key, tag), np.column_stack((wl, T, R, Ta, Ra)), fmt="%1.8e",
					   header="wl_um T R T_analytic R_analytic")
			if mp.am_master():
				print("RESULT {0}{1}: max|dT|={2:.6f} max|dR|={3:.6f} max|T+|R|-1|={4:.6f} faces_px={5} "
					  "t_end ref/film={6:g}/{7:g}".format(key, tag, rec["max_dT"], rec["max_dR"], rec["max_energy"],
					  ["{0:.4f}".format(f) for f in rec["faces_px"]], rec["t_end_ref"], rec["t_end_film"]), flush=True)

	# =========================================================
	# VERDICT
	# =========================================================
	ok = True
	lo, hi = min(args.res), max(args.res)
	for avg_s in args.averaging:
		a = summary["runs"]["res{0:g}-avg{1}".format(lo, avg_s)]
		b = summary["runs"]["res{0:g}-avg{1}".format(hi, avg_s)]
		ok_lo = a["max_dT"] <= TOL and a["max_dR"] <= TOL
		ok_hi = len(args.res) < 2 or (b["max_dT"] < a["max_dT"] and b["max_dR"] < a["max_dR"])
		print("VERDICT averaging={0}: res{1:g} within {2} -> {3}; res{4:g} smaller -> {5}".format(
			avg_s, lo, TOL, ok_lo, hi, ok_hi), flush=True)
		ok &= ok_lo and ok_hi
	summary["pass"] = bool(ok)
	with open("slab_summary{0}.json".format(tag), "w") as fh:
		json.dump(summary, fh, indent=1)
	print("PASS" if ok else "FAIL", flush=True)
	sys.exit(0 if ok else 1)
