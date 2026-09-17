#!/usr/bin/env python3

"""
Uniform voxel slab vs the analytic normal-incidence (Airy) slab.

Mode --film multilayer: an asymmetric 7-layer voxel stack (5s/3v/7s/4v/6s/2v/3s,
internal solid/void voxel faces inside the MaterialGrid) against the exact
characteristic-matrix result.

Mode --film grating (measurement, not a gate): a binary grating periodic in x
(one period of --period voxels in the cell, solid voxels [0, --duty) so a bar
edge sits on the periodic seam), invariant in y, --nvox voxels thick, built with
film_geometry at each --res and compared with the same grating made of plain
mp.Block objects (Meep subpixel averaging on) at --ref_res.  --stage ref writes the
reference to JSON and --stage voxel reloads it, so each stage is one MPI job; film
runs stop on DFT decay, at --maxt or after --wall_limit s (reference), voxel runs at
the reference's end time, and T, R snapshots every --snap_dt give a convergence check.

Film placement follows the production recipe: film_geometry's lateral centre
(amendment A2) and, unless --z_align off, z_center = aligned_z_center(...)
(amendment A3).  --cell_extra_px adds pixels to the top of the cell to flip the
parity of the cell's z pixel count (the film request moves by half a pixel per
extra pixel; aligned_z_center must absorb it).

An all-solid voxel film (eps 2.4025, n = 1.55) built by voxel_geometry.film_geometry
sits in vacuum in a periodic x/y cell with PML in z and a normally incident,
x-polarised plane wave.  The z layout, the multi-Gaussian source, the flux
monitors and the normalisation reproduce film_transmittance.py (-ScattPower):
a vacuum reference run saves the reflection-monitor flux (save_flux), the film
run subtracts it (load_minus_flux), and T = trans/inc, R = refl/inc.

For each resolution and each do_averaging setting (default: off only) the maxima of |T - T_Airy|,
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
import time
from typing import Dict, List, Optional, Tuple

# ----- numerics -----
import meep as mp
import numpy as np

# ----- local modules -----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import voxel_geometry  # noqa: E402
from voxel_geometry import aligned_z_center, film_geometry  # noqa: E402

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


def film_voxels(args) -> np.ndarray:
	"""Voxel film indexed [x, y, z] for the selected mode."""
	if args.film == "grating":
		assert 0 < args.duty < args.period, "grating needs 0 < duty < period"
		col = np.zeros(args.period, dtype=bool)
		col[:args.duty] = True
		return np.broadcast_to(col[:, None, None], (args.period, args.ny, args.nvox)).copy()
	prof = film_profile(args.film, args.nvox)
	return np.broadcast_to(prof, (args.lateral, args.lateral, args.nvox)).copy()


def reference_blocks(vox: np.ndarray, z_center: float) -> List:
	"""Plain mp.Block objects at the voxel positions film_geometry uses (x runs, y and z uniform per run)."""
	nx, ny, nz = vox.shape
	h = VOXEL_SIZE*nz
	col = vox[:, :, :]
	assert np.all(col == col[:, :1, :1]), "reference_blocks: only films varying in x alone"
	x_min = -(nx + nx % 2)*VOXEL_SIZE/2           # block min of film_geometry (amendment A2)
	blocks, i = [], 0
	solid = col[:, 0, 0]
	while i < nx:
		if solid[i]:
			j = i
			while j < nx and solid[j]:
				j += 1
			blocks.append(mp.Block(center=mp.Vector3(x_min + 0.5*(i + j)*VOXEL_SIZE, 0, z_center),
								   size=mp.Vector3((j - i)*VOXEL_SIZE, mp.inf, h), material=mp.Medium(epsilon=EPS_SOLID)))
			i = j
		else:
			i += 1
	return blocks


def run_stage(args, res: float, avg: bool, ref: bool, tag: str, geometry_kind: str,
			  maxt_override: Optional[float] = None) -> Dict:
	"""Run one reference or film stage with film_transmittance.py's layout; return fluxes."""
	assert res > 0, "resolution must be positive"
	kmin, kmax = KS
	wavelen_max, wavelen_min = 2.0*np.pi/kmin, 2.0*np.pi/kmax
	f_min, f_max = 1.0/wavelen_max, 1.0/wavelen_min
	fcen, fwidth = 0.5*(f_max + f_min), f_max - f_min
	vox = film_voxels(args)
	nz = vox.shape[2]
	h = VOXEL_SIZE*nz
	PML_thickness = args.tpml*wavelen_max
	z_pos_detector, z_pos_detector2, z_pos_source = -args.ddet, h + args.ddet, h + args.dsrc
	z_min = z_pos_detector - args.dpml - PML_thickness
	z_max = z_pos_source + args.dpml + PML_thickness + args.cell_extra_px/res
	cell_height = z_max - z_min
	z_center_offset = (z_min + z_max)/2
	requested = 0.5*h - z_center_offset + args.z_offset_px/res
	film_z_center = aligned_z_center(requested, h, res, cell_height) if args.z_align == "on" else requested
	Lx, Ly, hh, geometry = film_geometry(vox, VOXEL_SIZE, mp.Medium(epsilon=EPS_SOLID),
										 mp.Medium(epsilon=1.0), film_z_center, do_averaging=avg)
	assert abs(hh - h) < 1e-12, "film_geometry thickness mismatch"
	if geometry_kind == "blocks":
		if args.film == "grating":
			geometry = reference_blocks(vox, film_z_center)
		else:
			# comparison only: the same z profile as plain mp.Block layers (eps_averaging = do_averaging)
			prof = film_profile(args.film, args.nvox)
			geometry = [mp.Block(center=mp.Vector3(0, 0, film_z_center - 0.5*h + (iz + 0.5)*VOXEL_SIZE),
								 size=mp.Vector3(mp.inf, mp.inf, VOXEL_SIZE), material=mp.Medium(epsilon=EPS_SOLID))
						for iz in np.nonzero(prof)[0]]
	# film faces in pixels from the origin (Meep node lattice: integer = on an E_x/E_y node) and from the cell bottom
	faces_px = [(film_z_center - 0.5*h)*res, (film_z_center + 0.5*h)*res]
	faces_px_bottom = [(film_z_center - 0.5*h + 0.5*cell_height)*res, (film_z_center + 0.5*h + 0.5*cell_height)*res]
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
	eps_avg = (args.ref_averaging == "on") if args.film == "grating" else avg
	sim = mp.Simulation(cell_size=mp.Vector3(Lx, Ly, cell_height), force_complex_fields=True,
						boundary_layers=[mp.PML(thickness=PML_thickness, direction=mp.Z)],
						geometry=geometry, Courant=courant, sources=sources,
						k_point=mp.Vector3(), resolution=res,
						eps_averaging=(eps_avg if geometry_kind == "blocks" else True))
	trans = sim.add_flux(fcen, fwidth, args.nfreqs, mp.FluxRegion(
		center=mp.Vector3(0, 0, z_pos_detector - z_center_offset), size=mp.Vector3(Lx, Ly, 0)))
	refl = sim.add_flux(fcen, fwidth, args.nfreqs, mp.FluxRegion(
		center=mp.Vector3(0, 0, z_pos_detector2 - z_center_offset), size=mp.Vector3(Lx, Ly, 0)))
	fluxname = "slab-ref-res{0:g}{1}".format(res, tag)
	if not ref:
		sim.load_minus_flux(fluxname, refl)
	# Stop rule: DFT decay (tol) OR simulated time > maxt OR (film runs, --wall_limit > 0) wall clock above the
	# limit.  The wall clock is reduced with max_to_all every 200 steps so all MPI ranks stop at the same step.
	maxt = args.maxt if maxt_override is None else maxt_override
	decay = mp.stop_when_dft_decayed(tol=args.dft_tol, maximum_run_time=maxt)
	state = {"reason": None, "t0": time.time()}

	def stop(s):
		if decay(s):
			state["reason"] = "maxt" if s.round_time() > maxt else "decayed"
			return True
		if not ref and args.wall_limit > 0 and s.timestep() % 200 == 0:
			if mp.max_to_all(time.time() - state["t0"]) > args.wall_limit:
				state["reason"] = "wall_limit"
				return True
		return False

	snaps: Dict[str, List] = {"time": [], "trans": [], "refl": []}

	def snapshot(s):
		snaps["time"].append(float(s.round_time()))
		snaps["trans"].append(list(mp.get_fluxes(trans)))
		snaps["refl"].append(list(mp.get_fluxes(refl)))

	step_funcs = [] if (ref or args.film != "grating" or args.snap_dt <= 0) else [mp.at_every(args.snap_dt, snapshot)]
	sim.run(*step_funcs, until_after_sources=stop)
	out = {"t_end": float(sim.round_time()), "stop_reason": state["reason"], "snapshots": snaps,
		   "wall_s": time.time() - state["t0"], "faces_px": faces_px, "faces_px_from_bottom": faces_px_bottom,
		   "z_shift_px": (film_z_center - requested)*res, "cell_pixels_z": int(np.floor(cell_height*res + 0.5)),
		   "freqs": list(mp.get_flux_freqs(trans)),
		   "trans": list(mp.get_fluxes(trans)), "refl": list(mp.get_fluxes(refl))}
	if ref:
		sim.save_flux(fluxname, refl)
	return out

def snapshot_change(snaps: Dict, T: np.ndarray, R: np.ndarray, t_end: float, snap_dt: float) -> Dict:
	"""Max |change| of T, R between the final spectrum and the last snapshot at least snap_dt before t_end."""
	times = [t for t in snaps["time"] if t <= t_end - snap_dt + 1e-9]
	if not times:
		return {"t_prev": None, "max_dT": None, "max_dR": None}
	i = snaps["time"].index(times[-1])
	return {"t_prev": times[-1], "max_dT": float(np.max(np.abs(T - np.array(snaps["T"][i])))),
			"max_dR": float(np.max(np.abs(R - np.array(snaps["R"][i]))))}


def compare_at(times: List[float], snT: List, snR: List, ref_snaps: Dict, t: Optional[float]) -> Optional[Dict]:
	"""Max |dT|, |dR| between a run and the reference at a common snapshot time t (None if either lacks it)."""
	if t is None:
		return None
	ia = [k for k, tt in enumerate(times) if abs(tt - t) < 1e-6]
	ib = [k for k, tt in enumerate(ref_snaps["time"]) if abs(tt - t) < 1e-6]
	if not ia or not ib:
		return None
	return {"t": t, "max_dT": float(np.max(np.abs(np.array(snT[ia[0]]) - np.array(ref_snaps["T"][ib[0]])))),
			"max_dR": float(np.max(np.abs(np.array(snR[ia[0]]) - np.array(ref_snaps["R"][ib[0]]))))}

# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# ----- run -----
	parser.add_argument("--outdir", type=str, required=True)
	parser.add_argument("--res", type=float, nargs="+", default=[100.0, 200.0])
	parser.add_argument("--averaging", type=str, nargs="+", default=["off"], choices=["off", "on"],
		help="film_geometry do_averaging settings to run (default off, the recommended setting)")
	parser.add_argument("--z_align", type=str, default="on", choices=["on", "off"],
		help="on = z_center from aligned_z_center (production, amendment A3); off = raw centre")
	parser.add_argument("--cell_extra_px", type=float, default=0.0,
		help="extra pixels at the top of the cell (1 flips the parity of the cell's z pixel count)")
	parser.add_argument("--z_offset_px", type=float, default=0.0,
		help="shift the requested film centre by this many pixels before alignment (informational)")

	# ----- film and layout (film_transmittance.py names) -----
	parser.add_argument("--film", type=str, default="uniform", choices=["uniform", "multilayer", "grating"],
		help="uniform = contract test; multilayer = 7-layer voxel stack vs TMM; grating = x-periodic grating vs mp.Block")
	parser.add_argument("--beta", type=float, default=None,
		help="override voxel_geometry.PROJECTION_BETA (comparison runs only; default keeps the module value)")
	parser.add_argument("--geometry", type=str, default="materialgrid", choices=["materialgrid", "blocks"],
		help="blocks = mp.Block layers instead of film_geometry (comparison runs only)")
	parser.add_argument("--nvox", type=int, default=30)
	parser.add_argument("--lateral", type=int, default=4)
	parser.add_argument("--period", type=int, default=60, help="grating: voxels per period (= nx)")
	parser.add_argument("--duty", type=int, default=30, help="grating: solid voxels per period, placed at [0, duty)")
	parser.add_argument("--ny", type=int, default=2, help="grating: voxels along y")
	parser.add_argument("--ref_res", type=float, default=400.0, help="grating: resolution of the mp.Block reference")
	parser.add_argument("--ref_averaging", type=str, default="on", choices=["on", "off"],
		help="grating: Meep eps_averaging of the mp.Block reference")
	parser.add_argument("--nfreqs", type=int, default=101)
	parser.add_argument("--ddet", type=float, default=0.2)
	parser.add_argument("--dsrc", type=float, default=0.4)
	parser.add_argument("--dpml", type=float, default=0.3)
	parser.add_argument("--tpml", type=float, default=0.5)
	parser.add_argument("--dft_tol", type=float, default=1e-8)
	parser.add_argument("--maxt", type=float, default=400.0,
		help="cap on simulated time (grating voxel runs also stop at the reference's t_end)")
	parser.add_argument("--stage", type=str, default="all", choices=["all", "ref", "voxel"],
		help="grating: ref = mp.Block reference only (written to JSON); voxel = load it and run --res")
	parser.add_argument("--wall_limit", type=float, default=0.0,
		help="grating: stop a film run once its wall clock exceeds this many seconds (0 = off)")
	parser.add_argument("--snap_dt", type=float, default=10.0,
		help="grating: record T, R every snap_dt time units (convergence check)")

	args = parser.parse_args()
	assert args.nfreqs >= 41, "need at least 41 frequencies"
	if args.beta is not None:
		voxel_geometry.PROJECTION_BETA = args.beta
	os.makedirs(args.outdir, exist_ok=True)
	os.chdir(args.outdir)
	film_tag = args.film if args.film != "grating" else "grating-P{0}-D{1}-ny{2}-t{3}".format(
		args.period, args.duty, args.ny, args.nvox)
	tag = ("" if args.film == "uniform" else "-" + film_tag) + \
		("" if args.beta is None else "-beta{0:g}".format(args.beta)) + \
		("" if args.geometry == "materialgrid" else "-" + args.geometry) + \
		("" if args.z_offset_px == 0 else "-zoff{0:g}".format(args.z_offset_px)) + \
		("-zalign" if args.z_align == "on" else "") + \
		("" if args.cell_extra_px == 0 else "-cellx{0:g}".format(args.cell_extra_px))
	summary: Dict = {"beta": voxel_geometry.PROJECTION_BETA, "film": film_tag, "z_offset_px": args.z_offset_px,
					 "z_align": args.z_align, "cell_extra_px": args.cell_extra_px, "nvox": args.nvox, "runs": {}}

	# =========================================================
	# REFERENCE SPECTRUM
	# =========================================================
	if args.film == "grating":
		ref_file = os.path.join(args.outdir, "grating_reference{0}-res{1:g}.json".format(tag, args.ref_res))
		if args.stage in ("all", "ref"):
			ref_stage = run_stage(args, args.ref_res, False, True, tag, "blocks")
			blk = run_stage(args, args.ref_res, False, False, tag, "blocks")
			inc = np.array(ref_stage["refl"])
			reference = {"res": args.ref_res, "eps_averaging": args.ref_averaging, "t_end": blk["t_end"],
						 "stop_reason": blk["stop_reason"], "wall_s": blk["wall_s"], "t_end_vacuum": ref_stage["t_end"],
						 "faces_px": blk["faces_px"], "freqs": blk["freqs"],
						 "T": list(np.array(blk["trans"])/inc), "R": list(np.abs(np.array(blk["refl"])/inc)),
						 "snapshots": {"time": blk["snapshots"]["time"],
									   "T": [list(np.array(t)/inc) for t in blk["snapshots"]["trans"]],
									   "R": [list(np.abs(np.array(r)/inc)) for r in blk["snapshots"]["refl"]]},
						 "mpi_ranks": mp.count_processors()}
			if mp.am_master():
				with open(ref_file, "w") as fh:
					json.dump(reference, fh)
		else:
			with open(ref_file) as fh:
				reference = json.load(fh)
		Ta, Ra = np.array(reference["T"]), np.array(reference["R"])
		ref_freqs = np.array(reference["freqs"])
		conv = snapshot_change(reference["snapshots"], Ta, Ra, reference["t_end"], args.snap_dt)
		summary["reference"] = {k: reference[k] for k in ("res", "eps_averaging", "t_end", "stop_reason", "wall_s",
																"t_end_vacuum", "faces_px", "mpi_ranks")}
		summary["reference"].update({"max_energy": float(np.max(np.abs(Ta + Ra - 1.0))), "convergence": conv})
		if mp.am_master():
			print("REFERENCE mp.Block res{0:g} eps_averaging={1} ranks={2}: max|T+|R|-1|={3:.6f} min T={4:.4f} "
				  "max R={5:.4f} t_end={6:g} stop={7} wall={8:.0f}s; change since t={9}: max|dT|={10} max|dR|={11}".format(
				  args.ref_res, args.ref_averaging, reference["mpi_ranks"], summary["reference"]["max_energy"],
				  float(Ta.min()), float(Ra.max()), reference["t_end"], reference["stop_reason"], reference["wall_s"],
				  conv["t_prev"], conv["max_dT"], conv["max_dR"]), flush=True)
		if args.stage == "ref":
			sys.exit(0)
	else:
		prof = film_profile(args.film, args.nvox)
		# (n, d) layers from runs of equal voxels, bottom to top (|r|^2 and T do not depend on the side of
		# incidence for a lossless stack)
		layers: List[Tuple[float, float]] = []
		for sv in prof:
			n_s = np.sqrt(EPS_SOLID) if sv else 1.0
			if layers and layers[-1][0] == n_s:
				layers[-1] = (n_s, layers[-1][1] + VOXEL_SIZE)
			else:
				layers.append((n_s, VOXEL_SIZE))

	# =========================================================
	# RUN AND COMPARE
	# =========================================================
	for res in args.res:
		ref = run_stage(args, res, False, True, tag, args.geometry)
		inc = np.array(ref["refl"])
		for avg_s in args.averaging:
			film = run_stage(args, res, avg_s == "on", False, tag, args.geometry,
							 maxt_override=(min(args.maxt, reference["t_end"]) if args.film == "grating" else None))
			wl = 1.0/np.array(film["freqs"])
			T = np.array(film["trans"])/inc
			R = np.array(film["refl"])/inc
			if args.film == "grating":
				assert np.allclose(np.array(film["freqs"]), ref_freqs, rtol=0, atol=1e-12), "frequency grids differ"
			else:
				Ta, Ra = tmm_stack(wl, layers)
				if args.film == "uniform":
					Tair, Rair = airy_slab(wl, np.sqrt(EPS_SOLID), VOXEL_SIZE*args.nvox)
					assert np.max(np.abs(Tair - Ta)) < 1e-12 and np.max(np.abs(Rair - Ra)) < 1e-12, "TMM != Airy"
			key = "res{0:g}-avg{1}".format(res, avg_s)
			rec = {"max_dT": float(np.max(np.abs(T - Ta))), "max_dR": float(np.max(np.abs(np.abs(R) - Ra))),
				   "max_energy": float(np.max(np.abs(T + np.abs(R) - 1.0))),
				   "t_end_ref": ref["t_end"], "t_end_film": film["t_end"], "faces_px": film["faces_px"],
				   "faces_px_from_bottom": film["faces_px_from_bottom"], "z_shift_px": film["z_shift_px"],
				   "cell_pixels_z": film["cell_pixels_z"], "stop_reason": film["stop_reason"], "wall_s": film["wall_s"]}
			if args.film == "grating":
				snT = [list(np.array(t)/inc) for t in film["snapshots"]["trans"]]
				snR = [list(np.abs(np.array(r)/inc)) for r in film["snapshots"]["refl"]]
				rec["convergence"] = snapshot_change({"time": film["snapshots"]["time"], "T": snT, "R": snR},
													 T, np.abs(R), film["t_end"], args.snap_dt)
				# the same comparison at the reference snapshot time used for its convergence check
				rec["at_t_prev"] = compare_at(film["snapshots"]["time"], snT, snR, reference["snapshots"],
											  rec["convergence"]["t_prev"])
				# fraction of frequencies within TOL, and the 90th percentile of |dT|
				dT = np.abs(T - Ta)
				rec["frac_dT_le_tol"] = float(np.mean(dT <= TOL))
				rec["p90_dT"] = float(np.percentile(dT, 90))
			summary["runs"][key] = rec
			if mp.am_master():
				np.savetxt("slab_{0}{1}.txt".format(key, tag), np.column_stack((wl, T, R, Ta, Ra)), fmt="%1.8e",
						   header="wl_um T R T_reference R_reference")
			if mp.am_master():
				print("RESULT {0}{1}: max|dT|={2:.6f} max|dR|={3:.6f} max|T+|R|-1|={4:.6f} cell_pixels_z={5} "
					  "z_shift_px={6:+.4f} faces_px(origin)={7} faces_px(bottom)={8} t_end ref/film={9:g}/{10:g}".format(
					  key, tag, rec["max_dT"], rec["max_dR"], rec["max_energy"], rec["cell_pixels_z"], rec["z_shift_px"],
					  ["{0:.4f}".format(f) for f in rec["faces_px"]],
					  ["{0:.4f}".format(f) for f in rec["faces_px_from_bottom"]], rec["t_end_ref"], rec["t_end_film"]),
					  flush=True)
				if args.film == "grating":
					print("  stop={0} wall={1:.0f}s p90|dT|={2:.6f} frac(|dT|<={3})={4:.3f}; self change since t={5}: "
						  "max|dT|={6} max|dR|={7}; vs reference at that time: {8}".format(
						  rec["stop_reason"], rec["wall_s"], rec["p90_dT"], TOL, rec["frac_dT_le_tol"],
						  rec["convergence"]["t_prev"], rec["convergence"]["max_dT"], rec["convergence"]["max_dR"],
						  rec["at_t_prev"]), flush=True)

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
		if mp.am_master():
			print("VERDICT averaging={0}: res{1:g} within {2} -> {3}; res{4:g} smaller -> {5}".format(
				avg_s, lo, TOL, ok_lo, hi, ok_hi), flush=True)
		ok &= ok_lo and ok_hi
	summary["pass"] = bool(ok)
	if mp.am_master():
		with open("slab_summary{0}.json".format(tag), "w") as fh:
			json.dump(summary, fh, indent=1)
	if args.film == "grating":
		# measurement only: the verdict above is reported, never enforced
		if mp.am_master():
			print("MEASUREMENT (grating; not a gate): " + ("within/decreasing" if ok else "NOT within/decreasing"), flush=True)
		sys.exit(0)
	print("PASS" if ok else "FAIL", flush=True)
	sys.exit(0 if ok else 1)
