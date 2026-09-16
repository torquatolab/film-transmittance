# ===========================================
# A python code to compute the planar electric (E) and 
# dielectric displacement field (D) at given frequencies 
# and their transmittance spectra  
# via fintie-difference time-difference simulations.
# ===========================================

from __future__ import division

import meep as mp
import numpy as np
import argparse

import os
import time
import json
import shutil
import sys
#sys.path.insert(1, '../')
#import IO_meep as uf
#sys.path.insert(1, '/home/jaeukk/codes/python')
#import PolyDispersion as tp

from shared_FDTD import *

## ---- for using material library ----
# vim ~/miniconda/envs/mp/lib/python3.7/site-packages/meep/materials.py
from meep.materials import Cu
from meep.materials import Au
from meep.materials import Ag


## -------------------
# Setting-up Spectrum!!
# unit length = 1000nm
## -------------------

C0 = 299_792_458.0 # speed of light in vacuum (m/s)
hbar2eV = 6.582119569e-16 # eV * s
a0 = 1e-6 # 1 micron (unit length)
SUM_RULE_TOL = 0.05

def ConvertPacking2Geometry(filename: str, phi2:float, material_particle: mp.Medium, is_packing: bool=True, scale: float=1.0, material_matrix: mp.Medium=mp.Medium(epsilon=1.0), particle_shape = "disk"):
	'''
	read a txt file of sphere packing, 
	and return the corresponding array of geometry objects,
	unit cell size is also returned.
	'''
	return ConvertPacking2Geometry_2D(filename, phi2, material_particle, is_packing, scale, material_matrix, particle_shape)

def _master_save(fn, *a, **kw):
	"""Run a master-only write, returning False instead of raising on failure.

	A bare ``if mp.am_master(): np.save(...)`` lets a quota or disk-full error kill
	rank 0 while every other rank blocks in the next collective, hanging the job
	until walltime rather than failing fast.  Callers accumulate the result and
	settle it collectively with _abort_if_write_failed.
	"""
	if not mp.am_master():
		return True
	try:
		fn(*a, **kw)
	except Exception as exc:
		sys.stderr.write("WRITE FAILED: {0}\n".format(exc))
		return False
	return True


def _abort_if_write_failed(ok, what):
	"""Turn a rank-local I/O failure into a coordinated exit on every rank.

	Collective: every rank must call it, so it goes outside any master-only branch.
	"""
	if not mp.and_to_all(bool(ok)):
		raise SystemExit("aborting: {0} failed".format(what))


def _write_json(path, data):
	with open(path, "w") as fh:
		json.dump(data, fh, indent=1)


def _flux_metadata_path(sim, stem):
	"""Return the sidecar path matching Meep's prefixed flux filename."""
	prefix = sim.get_filename_prefix()
	return "{0}-{1}.meta.json".format(prefix, stem) if prefix else stem + ".meta.json"


def _require_matching_flux_layout(sim, stem):
	"""Refuse a reference flux written with an unknown or different rank count."""
	path = _flux_metadata_path(sim, stem)
	ok = True
	if mp.am_master():
		try:
			with open(path, "r") as fh:
				metadata = json.load(fh)
			written_nprocs = int(metadata["nprocs"])
		except (OSError, ValueError, KeyError, TypeError) as exc:
			ok = False
			sys.stderr.write("REFUSING: cannot verify flux layout from {0}: {1}\n".format(
				path, exc))
		else:
			current_nprocs = mp.count_processors()
			if written_nprocs != current_nprocs:
				ok = False
				sys.stderr.write(
					"REFUSING: {0} was written by {1} ranks, current run has {2}. "
					"Meep silently mis-frames a mismatched flux reference.\n".format(
						path, written_nprocs, current_nprocs))
	ok = mp.broadcast(0, ok)
	if not ok:
		raise SystemExit("aborting: unverified or mismatched reference-flux layout")


def main(args):
	dimension = 3
	eps_pol =  ChooseMaterial(args.eps)	
	eps_ref =  ChooseMaterial(args.eps_ref)	
	output_fields = args.wfield
	
	kmin = args.ks[0]
	kmax = args.ks[1]
	wavelen_min = 2.0*np.pi/kmax
	wavelen_max = 2.0*np.pi/kmin
	f_min = 1.0/ wavelen_max
	f_max = 1.0/ wavelen_min
	fcen = 0.5*(f_max+f_min)
	fwidth = f_max - f_min
	# fwidth = np.min([1.66*(f_max-f_min), 0.95*fcen]) #source range
	# df = f_max - f_min # sampling range
	NumFreqs = args.nfreqs

	print(args.comp)

	h = args.tfilm
	PML_thickness = args.tpml*wavelen_max
	
	# Global Z positions (bottom of film at z=0)
	z_pos_detector = -args.ddet
	z_pos_detector2 = h + args.ddet
	z_pos_source = h + args.dsrc  # assume args.dsrc > args.ddet
		
	z_min = z_pos_detector - args.dpml - PML_thickness
	z_max = z_pos_source + args.dpml + PML_thickness
	cell_height = z_max - z_min
	z_center_offset = (z_min + z_max) / 2

	# Shift positions to be relative to cell center (Meep coordinates)
	pos_detector = z_pos_detector - z_center_offset
	pos_source = z_pos_source - z_center_offset
	pos_detector2 = z_pos_detector2 - z_center_offset
	film_z_center = 0.5*h - z_center_offset

	#generate header and geometry -----#
	header = GenerateHeader (args)

	geometry = []
	Lx, Ly = 1.0, 1.0 # Default if load is empty
	print("Geometry:")
	if args.load != '':
		postfix = args.load.split('.')[-1]
		if (postfix == 'Dispersion'):
			# Polygonal packing
			print("Load a polygonal packing in ", args.load)
			print("ERROR (Not implemented)")
			# [Lx, Ly, geometry] = PolyDispersion2Geometry(args.load, eps_pol)
		else:
			# disk packing
			print("Load a disk packing in ", args.load)
			[Lx, Ly, geometry2D] = ConvertPacking2Geometry(args.load, args.phi, eps_pol, not args.is_point, args.scale2sim, eps_ref, args.particle)
			
			geometry = []
			for obj in geometry2D:
				if isinstance(obj, mp.Block):
					new_size = mp.Vector3(obj.size.x, obj.size.y, h)
					new_center = mp.Vector3(obj.center.x, obj.center.y, film_z_center)
					geometry.append(mp.Block(center=new_center, size=new_size, material=obj.material, e1=obj.e1, e2=obj.e2, e3=obj.e3))
				elif isinstance(obj, mp.Cylinder):
					# Convert Cylinder (disk in 2D) to 3D cylinder with height h
					new_center = mp.Vector3(obj.center.x, obj.center.y, film_z_center)
					geometry.append(mp.Cylinder(radius=obj.radius, height=h, axis=mp.Vector3(0,0,1), center=new_center, material=obj.material))
		
		if args.ref:
			geometry = []			
		
	Courant_parameter = DetermineCourantFactor(args, eps_pol, eps_ref, dimension)
	print("Courant parameter = {0:0.3f}\n".format(Courant_parameter))

	#Source-------------#
	if args.polarization == 'x':
		src_comp = mp.Ex
	elif args.polarization == 'y':
		src_comp = mp.Ey
	else:
		src_comp = mp.Ex 
	print("Source Polarization "+args.polarization)

	if fwidth < 1.0:
		sources = [
			mp.Source(mp.GaussianSource(fcen, fwidth),
				component=src_comp,
				size=mp.Vector3(Lx, Ly, 0),
				center=mp.Vector3(0, 0, pos_source))]
		print("source frequency range = {0:0.3f} +- {1:0.3f}".format(fcen, 0.5*fwidth))
		print("sampling frequency range = {0:0.3f} +- {1:0.3f}".format(fcen, 0.5*fwidth))
	else:
		print("Use multiple sources: ")
		print("sampling frequency range = {0:0.3f} +- {1:0.3f}".format(fcen, 0.5*fwidth))

		fwidth_unit = 1.0
		sources = []
		f_intv = fwidth_unit/16. * np.sqrt(-np.log(5*args.decay))

		f0, f1, f2 = f_min, f_min + f_intv, f_min + 2*f_intv
		while (f0 < f_max):
			sources.append(mp.Source(mp.GaussianSource(f1, 2*f_intv),
				component=src_comp,
				size=mp.Vector3(Lx, Ly, 0),
				center=mp.Vector3(0, 0, pos_source)))	 
			print (f"[{f0:0.5f}, {f2:0.5f}],")
			f0 += 2*f_intv
			f1 += 2*f_intv
			f2 += 2*f_intv
		
	pml_layers = [mp.PML(thickness=PML_thickness, direction=mp.Z)]
	if args.fullprofile:
		LL = cell_height - 2*PML_thickness - 2*args.dpml
		nonpml_vol_center = 0.0
	else:
		# The analysis volume only has to contain the Joule-heating integrand
		# Q = omega*Im(conj(E)*D), whose support is the metal: Im(eps) = 0 in vacuum.
		# The legacy LL = 3*h therefore integrates two film thicknesses of near-zeros,
		# and since get_dft_array replicates the WHOLE array on every rank, that waste
		# is multiplied by ranks-per-node at the end of the run.  It is what OOM-killed
		# res 2500/3100/3200 at tfilm 0.03 on 2026-09-14, in the JouleHeating block of
		# frequency 0.
		#
		# Margin is NOT optional, and not for the reason it looks like.  With
		# yee_grid=False the DFT fields are interpolated to voxel centres, so across a
		# metal/vacuum face Im(conj(E_c)*D_c) picks up cross terms that do NOT vanish:
		# a single pixel row outside the geometric Ag carries ~50% of the peak Q.
		# Measured convergence of A over both grid parities: 0 px is 28% LOW, 1 px is
		# 3.2% low, 2 px is converged.  4 px is 2x that, and costs ~1% of the array.
		LL = 3.*h if args.dft_margin_px <= 0 else h + 2.0*args.dft_margin_px/args.res
		nonpml_vol_center = film_z_center

	nonpml_vol = mp.Volume(mp.Vector3(0, 0, nonpml_vol_center), size=mp.Vector3(Lx, Ly, LL))

	cell = mp.Vector3(Lx, Ly, cell_height)

	sim = mp.Simulation(cell_size=cell,
				force_complex_fields=True,
				boundary_layers=pml_layers,
				geometry=geometry,
#				dimensions=1,
				Courant=Courant_parameter,
				sources=sources,
				k_point=mp.Vector3(),
				resolution=args.res)


	if args.test:
		print("test setups with a short animation\n")
		if args.ScattPower:
			# add scattering power monitor
			trans_freq = mp.FluxRegion(center = mp.Vector3(0, 0, pos_detector), size = mp.Vector3(Lx, Ly, 0))
			trans_refl = mp.FluxRegion(center = mp.Vector3(0, 0, pos_detector2), size = mp.Vector3(Lx, Ly, 0))
			trans = sim.add_flux(fcen, fwidth, NumFreqs, trans_freq)
			refl = sim.add_flux(fcen, fwidth, NumFreqs, trans_refl)
		animate = mp.Animate2D(sim,
					fields=src_comp,
					realtime=True,
					volume=mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(Lx, 0, cell_height)),
					field_parameters={'alpha':0.8, 'cmap':'RdBu', 'interpolation':'none'})

		sim.run(mp.at_every(0.1,animate),until_after_sources=cell_height*(2*np.pi))

	if args.test2:
		print("simplest test of loading configurations\n")
		sim.init_sim()
		# Export the dielectric constant distribution to a NumPy file
		epsilon_data = sim.get_epsilon()
		if mp.am_master():
			np.save(f"{args.saveas}_epsilon.npy", epsilon_data)
		print("Exported dielectric constant distribution.")
		(x,y,z,w) = sim.get_array_metadata(vol=nonpml_vol)
		if mp.am_master():
			np.save(("{0}_{1}.npy").format(args.saveas, 'x'), x)
			if dimension == 2:
				np.save(("{0}_{1}.npy").format(args.saveas, 'y'), y)
			if dimension == 3:
				np.save(("{0}_{1}.npy").format(args.saveas, 'y'), y)
				np.save(("{0}_{1}.npy").format(args.saveas, 'z'), z)
	else:
		if output_fields:
			print('saved epsilon + timelapsed fields\n')

			sim.run(
				mp.at_beginning(mp.output_epsilon),
				mp.in_volume(cell, mp.to_appended("Ex-Slice", mp.at_every(0.3, mp.output_efield_x))),
				mp.in_volume(cell, mp.to_appended("Ey-Slice", mp.at_every(0.3, mp.output_efield_y))),
				until_after_sources = 10)
		else:
			print("regular simulation\n")
			# add dft of spatial fields!!
			# --- checkpoint: structure first, monitors next, fields last -------------
			# meep's supported restart ordering (patches/test_checkpoint_dispersive.py).
			# A one-shot sim.load() here would attach the dumped DFT state before the
			# monitors exist and silently lose it, which is the bug still present in
			# 2D_plasmonic_checkpoints2.py:312.
			ckpt_dir = "{0}_checkpoints".format(args.saveas)
			# Master decides and broadcasts. Each rank stat-ing the filesystem itself can
			# disagree -- a shared filesystem need not make a file visible everywhere at
			# once, and a half-written checkpoint is visible before it is complete. A
			# disagreement here is not a wrong answer, it is a HANG: sim.load() feeds
			# collective HDF5 reads, so ranks that think they are resuming block waiting
			# for ranks that do not.
			def _resumable(d):
				# checkpoint_meta.json is the commit marker: _write_checkpoint puts it
				# inside the staging directory before the rename, so a directory without
				# it is an interrupted dump and must not be loaded.
				return all(os.path.exists(os.path.join(d, f))
						   for f in ("structure.h5", "fields.h5", "checkpoint_meta.json"))

			if mp.am_master() and bool(args.checkpoint):
				if not _resumable(ckpt_dir) and _resumable(ckpt_dir + ".old"):
					# killed between the two renames; the previous checkpoint is intact
					print("recovering checkpoint from {0}.old (a dump was interrupted "
						"mid-rename)".format(ckpt_dir), flush=True)
					shutil.rmtree(ckpt_dir, ignore_errors=True)
					os.rename(ckpt_dir + ".old", ckpt_dir)
				if os.path.exists(ckpt_dir) and not _resumable(ckpt_dir):
					print("WARNING: {0} has no checkpoint_meta.json -- it is an "
						"interrupted dump and will NOT be resumed from".format(ckpt_dir),
						flush=True)
			resuming = bool(args.checkpoint) and _resumable(ckpt_dir) \
				if mp.am_master() else False
			resuming = mp.broadcast(0, resuming)
			if resuming and mp.am_master():
				_meta_path = os.path.join(ckpt_dir, "checkpoint_meta.json")
				if os.path.exists(_meta_path):
					with open(_meta_path, "r") as fh:
						print("checkpoint provenance: {0}".format(json.load(fh)), flush=True)
				else:
					print("WARNING: {0} has no checkpoint_meta.json; its origin is "
						"unknown (written before provenance was recorded?)".format(ckpt_dir),
						flush=True)
			if resuming:
				print("resuming from checkpoint {0} (structure)".format(ckpt_dir), flush=True)
				sim.load(ckpt_dir, load_structure=True, load_fields=False)

			comps_ = args.comp if args.JouleHeating == False else ['Ex','Ey','Ez',"Dx","Dy", "Dz"]
			comps = [component_map[c] for c in comps_]

			dft_field = sim.add_dft_fields(comps, fcen, fwidth, NumFreqs, where=nonpml_vol, yee_grid=False) if args.JouleHeating else sim.add_dft_fields(comps, fcen, fwidth, NumFreqs, where=nonpml_vol) 
			
			# Use provided tempname; otherwise, fall back to basename of saveas
			temp_name = args.tempname if args.tempname != "" else args.saveas.split('/')[-1]
			print(temp_name)

			if args.ScattPower:
				# add scattering power monitor
				trans_freq = mp.FluxRegion(center = mp.Vector3(0, 0, pos_detector), size = mp.Vector3(Lx, Ly, 0))
				trans_refl = mp.FluxRegion(center = mp.Vector3(0, 0, pos_detector2), size = mp.Vector3(Lx, Ly, 0))
				trans = sim.add_flux(fcen, fwidth, NumFreqs, trans_freq)
				refl = sim.add_flux(fcen, fwidth, NumFreqs, trans_refl)
				if not args.ref:
					# Re-applied on a resume too, and that is deliberate. It is part of
					# the monitor's DFT descriptor (stored_weight), so skipping it makes
					# fields::load reject the checkpoint outright: "DFT monitor 0 does
					# not match ... Recreate the DFT monitors exactly as they were when
					# dumped."  It also cannot double-subtract: this runs BEFORE
					# load_fields, which overwrites the accumulators wholesale rather
					# than adding to them.
					if args.salvage:
						# The salvage MUST load the mismatched reference -- reproducing
						# the mis-framing is the whole method -- so the layout check is
						# skipped here and applied to the good reference instead.
						if mp.am_master():
							print("salvage: intentionally replaying the old reference "
								  "without a rank-layout check", flush=True)
					else:
						_require_matching_flux_layout(sim, temp_name+'refl-flux')
					sim.load_minus_flux(temp_name+'refl-flux', refl)
				else:
					print("reference run for scattering power calculation\n")
					# add transmittivity monitor

			# Stop on convergence of the DFT accumulators rather than of the raw field at
			# one point (ported from 2D_plasmonic.py, 2026-08-07): a single probe can sit
			# on a node, and the decay ratio is NOT monotone near the noise floor, so
			# -dft_nconsec > 1 requires the tolerance to hold for that many successive
			# checks.  -decay is retained only for the multi-source spacing.
			stop_kwargs = {'tol': args.dft_tol}
			if args.checkpoint:
				# The convergence history must survive the restart. Without it the checker
				# starts blind on a resumed run and stops it almost immediately -- silently,
				# with a plausible but truncated spectrum. See save_decay_state().
				stop_kwargs['state_file'] = os.path.join(ckpt_dir, "decay_state.json")
			if args.maxt > 0:
				stop_kwargs['maximum_run_time'] = args.maxt
			else:
				print("WARNING: no -maxt ceiling; an undamped mode or a non-converging "
					"residual can make this run forever.\n")
			if args.dft_nconsec > 1:
				print("stop: dft ratio <= {0:g} for {1} consecutive checks\n".format(
					args.dft_tol, args.dft_nconsec))
				stop_fn = stop_when_dft_decayed_stable(
					n_consecutive=args.dft_nconsec, **stop_kwargs)
			else:
				print("stop: dft ratio <= {0:g} once (Meep stock condition)\n".format(
					args.dft_tol))
				stop_fn = mp.stop_when_dft_decayed(**stop_kwargs)
			if resuming:
				print("resuming from checkpoint {0} (fields)".format(ckpt_dir), flush=True)
				sim.load(ckpt_dir, load_structure=False, load_fields=True)

			if args.salvage:
				# ---- flux-reference salvage, single stage, inside THIS simulation ----
				# Repairs a run whose reference flux was read at a rank count other than
				# the one that wrote it.  Meep's flux HDF5 carries no layout metadata:
				# load_dft_hdf5 slices a flat stream by the CURRENT layout's block sizes
				# and validates only a total element count, which is layout-invariant.
				# So a rank-count mismatch loads silently mis-framed and the reflection
				# monitor keeps the raw total instead of the scattered field.  With the
				# detector between source and film that reads
				# (inc + refl)/inc = 1 + R_true + eps.
				#
				#     S - I_true = (S - I@M) + I@M - I_true
				#
				# M, the mis-framing, is never understood -- only REPRODUCED, by
				# reloading the same bad file at the same rank count.  Doing that HERE,
				# in the simulation that already holds the checkpoint, makes the
				# reproduction an identity: one process, one chunk layout, nothing to
				# assume between jobs.
				#
				# An earlier two-stage design dumped S - I@M to disk for a second job.
				# Wrong twice over.  get_flux_data returns a GLOBAL-LENGTH array holding
				# only THIS rank's window (python/meep.i _get_dft_data writes
				# cdata[i + istart] for this rank's chunks alone), so the dump wrote 192
				# near-empty 175 MB files -- 63 GB for 175 MB of data -- and every
				# sampled rank INCLUDING RANK 0 was entirely zero, because the refl
				# plane lives in a handful of chunks.  Collapsing rank 0's copy onto
				# every rank, the "obvious" fix, would have zeroed S - I@M almost
				# everywhere and produced a plausible wrong spectrum with nothing to
				# catch it.  The second job also rebuilt the structure from geometry
				# rather than from structure.h5, so its chunk layout was an assumption,
				# not a guarantee -- the same silent class as the original bug.  Both
				# failure modes are gone by construction here.
				if not resuming:
					raise SystemExit(
						"REFUSING: -salvage needs a loaded checkpoint, but resuming is "
						"False. Without it the accumulator holds -I@M rather than "
						"S - I@M, and the salvage would return -I_true and look "
						"perfectly plausible.")
				if not args.salvage_goodref:
					raise SystemExit("REFUSING: -salvage needs -salvage_goodref")

				_orig = np.loadtxt(args.salvage_origtable)
				_inc = _orig[:, 4]
				# Prove the checkpoint restored the state that produced the original
				# spectrum, BEFORE spending anything on the repair.  Catches a wrong
				# checkpoint, a wrong -tempname, a mismatched cell, and a stock
				# (unpatched) libmeep that loads a dispersive checkpoint and silently
				# zero-inits the polarization state.
				_T_ck = np.array(mp.get_fluxes(trans)) / _inc
				_R_ck = np.array(mp.get_fluxes(refl)) / _inc
				_dchk = max(float(np.max(np.abs(_T_ck - _orig[:, 1]))),
							float(np.max(np.abs(_R_ck - _orig[:, 2]))))
				print("salvage: checkpoint reproduces original T,R to {0:.3e}".format(
					_dchk), flush=True)
				# 1e-5, not 1e-9.  The original table is written with fmt='%1.6e', i.e.
				# 7 significant digits, and BOTH sides of this comparison carry that
				# rounding: the stored T and R, and the _inc column the fresh fluxes are
				# divided by.  Two ~5e-8 contributions put the achievable floor near
				# 1.7e-7, which is exactly what a correct restore measured -- a 1e-9
				# threshold is unsatisfiable by construction, not a real check.  1e-5 sits
				# 60x above that floor and ~5 orders below anything this must catch: a
				# wrong checkpoint, a wrong -tempname, or a stock libmeep zero-initing the
				# polarization state all move T and R by 1e-2 or more.
				if not np.isfinite(_dchk) or _dchk > 1e-5:
					raise SystemExit(
						"REFUSING: checkpoint does not reproduce the original spectrum "
						"({0:.3e} > 1e-5)".format(_dchk))

				_fd = sim.get_flux_data(refl)                      # S - I@M
				_Ec = np.array(_fd.E); _Hc = np.array(_fd.H)
				del _fd
				sim.load_flux(temp_name + 'refl-flux', refl)       # + I@M (the BAD ref)
				_t = sim.get_flux_data(refl)
				_Ec += _t.E; _Hc += _t.H
				del _t
				# The good reference is the one place a layout check still bites: it was
				# written by a different job, so its rank count is not guaranteed.
				_require_matching_flux_layout(sim, args.salvage_goodref)
				sim.load_flux(args.salvage_goodref, refl)          # - I_true
				_t = sim.get_flux_data(refl)
				_Ec -= _t.E; _Hc -= _t.H
				del _t

				from meep.simulation import FluxData as _FluxData
				sim.load_flux_data(refl, _FluxData(E=_Ec, H=_Hc))
				_rf = np.array(mp.get_fluxes(refl))
				_R = _rf / _inc
				_result = np.column_stack((_orig[:, 0], _orig[:, 1], _R,
										   _orig[:, 3], _inc))
				_out = "{0}_trans-{1}-salvaged.txt".format(args.saveas, args.polarization)
				_abort_if_write_failed(
					_master_save(np.savetxt, _out, _result, fmt='%1.6e',
								 delimiter='\t', header=header),
					"salvaged transmittance write")
				print("salvage: wrote {0}".format(_out), flush=True)

				# Acceptance, AFTER the write so the evidence survives a failure.
				# T and A are untouched by this bug -- load_minus_flux only pre-loads a
				# DFT accumulator and never touches field evolution -- so the sum rule
				# is an independent test of the repaired R alone.
				_err = float(np.max(np.abs(_orig[:, 1] + np.abs(_R) + _orig[:, 3] - 1.0)))
				print("salvage: energy sum-rule max error = {0:.6e}".format(_err),
					  flush=True)
				if not np.isfinite(_err) or _err >= SUM_RULE_TOL:
					raise SystemExit(
						"REFUSING: salvaged sum-rule error {0:.6e} is not below {1:g}; "
						"the repair did not work".format(_err, SUM_RULE_TOL))
				return

			def _write_checkpoint(sim_, kind):
				"""Write a checkpoint that is either complete or absent -- never partial.

				sim.dump() is NOT atomic: it truncates and rewrites fields.h5 in place,
				so a signal landing inside a dump destroys the previous good checkpoint.
				Observed on Nurion job 23820725 (res 800, 64 ranks): SIGTERM mid-dump left
				a 1.86 GB fields.h5 beginning with eight zero bytes, and the resume died
				with "file signature not found".  The exposure is (dump duration /
				interval), so it grows with the run -- the production case is the most
				exposed, not the least.

				So the dump goes to a staging directory and the RENAME is the commit.
				checkpoint_meta.json is written inside the staging directory before the
				rename, which makes it a commit marker rather than a description written
				alongside: any ckpt_dir visible to a later run is guaranteed to contain
				it.  Previously the meta survived a failed dump and cheerfully certified
				a checkpoint that no longer existed, so the acceptance guard passed on
				corrupt state.

				Between the two renames ckpt_dir does not exist and ckpt_dir.old does;
				_resumable() below looks for that.
				"""
				staging = ckpt_dir + ".writing"
				previous = ckpt_dir + ".old"
				if mp.am_master():
					shutil.rmtree(staging, ignore_errors=True)
					os.makedirs(staging)
				mp.all_wait()

				sim_.dump(staging)
				save_decay_state(stop_fn, os.path.join(staging, "decay_state.json"),
								 timestep=sim_.fields.t)
				mp.all_wait()   # every rank has closed its HDF5 handles

				if mp.am_master():
					with open(os.path.join(staging, "checkpoint_meta.json"), "w") as fh:
						json.dump({"kind": kind,
								   "meep_time": float(sim_.meep_time()),
								   "timestep": int(sim_.fields.t),
								   "wall_clock": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=1)
					shutil.rmtree(previous, ignore_errors=True)
					if os.path.exists(ckpt_dir):
						os.rename(ckpt_dir, previous)
					os.rename(staging, ckpt_dir)      # <-- the commit
					shutil.rmtree(previous, ignore_errors=True)
				mp.all_wait()

				if mp.am_master():
					print("{0} checkpoint written to {1} at t={2:.6f} (step {3})".format(
						kind, ckpt_dir, sim_.meep_time(), sim_.fields.t), flush=True)

			run_args = [stop_fn]
			if args.checkpoint:
				if mp.am_master() and not os.path.exists(ckpt_dir):
					os.makedirs(ckpt_dir)
				mp.all_wait()
				_last = [time.time()]
				_every = 3600.0 * args.checkpoint_interval

				def _dump(sim_):
					"""Dump a restart checkpoint once every -checkpoint_interval hours.

					Wall-clock paced rather than simulation-time paced: the job dies from
					walltime and node faults, which are wall-clock events. The dump lands
					on whatever step the loop is at, which is always an exact multiple of
					dt, so the resumed run does not gain a spurious extra step.

					Only MASTER's clock decides, and the decision is broadcast. time.time()
					is not synchronised across ranks, so letting each rank test its own
					would eventually have one rank cross the interval a timestep before the
					others: that rank enters the collective sim_.dump() while the rest step
					on, and the job deadlocks until walltime kills it. Every rank must call
					mp.broadcast, so it goes before the early return.
					"""
					due = mp.broadcast(0, (time.time() - _last[0]) >= _every
									   if mp.am_master() else False)
					if not due:
						return
					_write_checkpoint(sim_, "periodic")
					_last[0] = time.time()

				run_args.insert(0, _dump)
				print("checkpointing to {0} every {1:g} h".format(
					ckpt_dir, args.checkpoint_interval), flush=True)

			sim.run(*run_args[:-1], until_after_sources=run_args[-1])

			if args.checkpoint:
				# Final checkpoint, written BEFORE any analysis touches the results.
				# The FDTD run is the expensive part; everything below is cheap and
				# rerunnable, so capturing the end state here means a failure in the
				# analysis (or a change of mind about it) costs minutes instead of the
				# whole job. The periodic _dump only fires every -checkpoint_interval
				# hours, so without this the last hours of the run would be lost.
				#
				# Overwrites ckpt_dir rather than writing a second copy: these files are
				# large, and once the run has finished the periodic checkpoint is
				# superseded by this one. The cost is that an interruption *during* this
				# dump leaves no restartable checkpoint at all.
				_write_checkpoint(sim, "final")

			(x,y,z,w) = sim.get_array_metadata(vol=nonpml_vol)
			print("saving metafiles...\n")
			print("{0}_{1}.npy\n\n".format(args.saveas, '...'))
			# write-race guard (ported from 2D_plasmonic.py, 2026-08-10): only the
			# master rank writes; collective calls above run on all ranks.
			_ok = _master_save(np.save, ("{0}_{1}.npy").format(args.saveas, 'x'), x)
			if dimension == 2:
				_ok &= _master_save(np.save, ("{0}_{1}.npy").format(args.saveas, 'y'), y)
			if dimension == 3:
				_ok &= _master_save(np.save, ("{0}_{1}.npy").format(args.saveas, 'y'), y)
				_ok &= _master_save(np.save, ("{0}_{1}.npy").format(args.saveas, 'z'), z)
			_ok &= _master_save(np.save, ("{0}_{1}.npy").format(args.saveas, 'w'), w)
			_abort_if_write_failed(_ok, "metadata write")
			if args.ref : 
				# write metafiles
				
				if args.ScattPower:
					# Defensive: guarantee a fresh file rather than trusting save_flux to
					# truncate one left by an earlier run under the same -tempname.
					#
					# NOT the explanation for the 2x-size flux files (f25d20, f25d30,
					# f25t25, f25t30, film2500).  That was my first reading on 2026-08-18
					# and it is WRONG: on 2026-08-19 f25d20t8 came out 2x on its very first
					# run, with a fresh -tempname and this guard never firing.  The size is
					# a per-CELL property -- meep stores the flux monitor twice for some
					# cell heights (chunk-boundary duplication; the two halves agree to
					# 8e-16) -- and has nothing to do with re-running.
					_ok = True
					if mp.am_master():
						stale = temp_name + 'refl-flux.h5'
						if os.path.exists(stale):
							print("removing stale flux file {0}\n".format(stale))
							_ok = _master_save(os.remove, stale)
					_abort_if_write_failed(_ok, "stale flux removal")
					mp.all_wait()
					# get incident fluxes
					sim.save_flux(temp_name+'refl-flux', refl)
					inc_flux = mp.get_fluxes(refl)
					_abort_if_write_failed(
						_master_save(np.save, "{0}_inc_flux.npy".format(temp_name), inc_flux),
						"incident-flux write")
					_abort_if_write_failed(
						_master_save(_write_json,
							_flux_metadata_path(sim, temp_name+'refl-flux'),
							{"nprocs": mp.count_processors(), "tempname": temp_name,
							 "resolution": args.res}),
						"reference-flux metadata write")
				
			else: 
				# write dft fields 
				print("saving dft files...\n")
				Absorption = np.zeros(NumFreqs)
				_dft_ok = True
				for i in range(NumFreqs):
					k_i = kmin + (kmax-kmin)*i/(NumFreqs-1)
					k_i = np.round(k_i*10000)/10000
					name = "{0}__ka-{1:.04f}-{2}.npy"

					# store relevant field components
					for idx, cp in enumerate(args.comp):
						field = sim.get_dft_array(dft_field, component_map[cp], i)
						# Record the failure and keep going: every rank must still reach
						# the remaining get_dft_array collectives below, so the abort is
						# deferred to after the loop.
						_dft_ok &= _master_save(np.save, name.format(args.saveas, k_i, cp), field)
					# Drop the last component before the Joule block allocates E, D and the
					# accumulator: otherwise it stays bound for the whole iteration and costs
					# one extra full array per rank at the exact point that OOMs.
					field = None

					# compute and store Joule heating
					if args.JouleHeating:
						omege = k_i #* C0 / 1e-6 # ( unit length = 1e-6 m)
						
						for id, components in enumerate([("Ex","Dx"),("Ey","Dy"),("Ez","Dz")]):
							e_,d_ = components
							E = sim.get_dft_array(dft_field, component_map[e_], i)
							D = sim.get_dft_array(dft_field, component_map[d_], i)
							
							if id == 0:
								absorbed_power_density = omege * np.imag(np.conj(E)*D)		
							else:
								absorbed_power_density += omege * np.imag(np.conj(E)*D)		
						_dft_ok &= _master_save(np.save, name.format(args.saveas, k_i, 'Q'),
											   absorbed_power_density)
						Absorption[i] = float(np.sum(absorbed_power_density*w))
					print (name.format(args.saveas, k_i, '...'))

				# Analyze transmittance and reflectance
				if args.ScattPower:
				# load incident fluxes
					print("computing transmittance\n")
					inc_flux = np.load("{0}_inc_flux.npy".format(temp_name))
					trans_flux = mp.get_fluxes(trans)
					refl_flux = mp.get_fluxes(refl)
					flux_freqs = mp.get_flux_freqs(trans)

					wl = []
					Ts = []
					Inc = []
					Re = []
					for i in range(len(flux_freqs)):
						wl = np.append(wl, 1/flux_freqs[i])
						Ts = np.append(Ts, trans_flux[i]/inc_flux[i])
						Re = np.append(Re, refl_flux[i]/inc_flux[i])
						Inc = np.append(Inc, inc_flux[i])						

					sum_rule_error = None
					if args.JouleHeating:
						# film propagates along -z, so inc_flux < 0; normalize A by |inc|
						# (T and R are same-sign ratios and unaffected). Sign bug found 2026-08-11.
						A = Absorption/np.abs(inc_flux)
						sum_rule_error = float(np.max(np.abs(Ts + np.abs(Re) + A - 1.0)))
						print("energy sum-rule max error = {0:.6e}".format(sum_rule_error),
							  flush=True)
						result = np.column_stack((wl,Ts, Re, A, Inc))
					else:
						result = np.column_stack((wl,Ts, Re, Inc))
					# write transmittivity
					print("saving transmittance\n")
					filename = '{0}_trans-{1}.txt'.format(args.saveas, args.polarization)
					print("at "+filename+"\n")
					_abort_if_write_failed(
						_master_save(np.savetxt, filename, result, fmt='%1.6e',
									 delimiter='\t', header=header),
						"transmittance write")
					# Checked AFTER the write, deliberately.  Refusing before it would
					# destroy a 30 h result to protect against quoting it -- and the T, A
					# and Inc columns stay valid even when R does not, which is exactly
					# what localised the 2026-09-06 reference-layout bug.  The nonzero
					# exit still fails the job and trips the tracker's reject_regex.
					if sum_rule_error is not None and (
							not np.isfinite(sum_rule_error)
							or sum_rule_error >= SUM_RULE_TOL):
						raise SystemExit(
							"REFUSING to certify: energy sum-rule error {0:.6e} is not "
							"below {1:g}; the spectrum WAS written to {2} for "
							"inspection".format(sum_rule_error, SUM_RULE_TOL, filename))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-res', type = int, default= 50, help = 'resolution (default: 50 pixels/um)')
	parser.add_argument('-Z2', action = 'store_true', default = False, help='generate 2D square lattice disk packing')
	parser.add_argument('-num', type=int, default=10, help='the number of unit cells when -Z2 is called' )

	parser.add_argument('-load', type=str, default='', help='load a configuration file')

	parser.add_argument('-polarization', type=str, default='z', help='source polarization')

	parser.add_argument('-phi', type=float, default = -1.0, help = 'volume fraction of phase 2')
	parser.add_argument('-tfilm', type=float, default = 0.1, help = 'thickness of the film')
	parser.add_argument('-ref', action = 'store_true', default=False, help='remove the film')
	# Recovery of runs whose reference flux was loaded at a rank count other than the
	# one that wrote it (silent mis-framing; R comes out as ~1 + R_true).  Two stages:
	#   dump   with -checkpoint, off the finished run's checkpoint -> S - I(M) per rank
	#   apply  in a fresh sim, with a rank-matched reference        -> corrected R
	# Both stages must run at the SAME rank count; layout depends on rank count alone,
	# not on how those ranks are spread over nodes.
	parser.add_argument('-salvage', action='store_true', default=False,
		help='repair R in a checkpoint whose reference flux was read at the wrong '
			 'rank count; single stage, requires -checkpoint and -salvage_goodref')
	parser.add_argument('-salvage_goodref', type=str, default='',
		help='salvage: the rank-MATCHED reference, as the BARE STEM meep uses '
			 "(e.g. 'fres3500grefl-flux') -- load_flux adds the prefix and '.h5'")
	parser.add_argument('-salvage_origtable', type=str, default='',
		help='salvage: the original *_trans-x.txt from the affected run, used to '
			 'verify the checkpoint restored and to supply the untouched T, A, Inc')
	parser.add_argument('-wfield', action='store_true', default=False, help='write field slices')
	parser.add_argument('-dpml', type=float, default=0.5, help='distance from PML to the source')
	parser.add_argument('-tpml', type=float, default=0.5, help='relative thickness of PML to the longest wavelength')
	parser.add_argument('-dsrc', type=float, default=0.3, help='distance from source to the film')
	parser.add_argument('-ddet', type = float, default = 0.3, help='distance from the film to the detector')
	parser.add_argument('-decay', type=float, default=1e-4, help='decay rate (multi-source spacing only; no longer the stopping criterion)')
	parser.add_argument('-dft_tol', type=float, default=1e-11, help='stop_when_dft_decayed tolerance on the DFT accumulators (Meep default 1e-11; NOT comparable to -decay)')
	parser.add_argument('-maxt', type=float, default=0.0, help='maximum_run_time ceiling for stop_when_dft_decayed (0 = unbounded, warns)')
	parser.add_argument('-dft_nconsec', type=int, default=1, help='require -dft_tol to hold for this many consecutive checks before stopping (1 = Meep stock behaviour; 3 recommended, the ratio is not monotone)')
	parser.add_argument('-dft_margin_px', type=int, default=0,
		help='half-width of the DFT analysis volume beyond the film, in PIXELS. '
			 '0 (default) keeps the legacy volume LL = 3*tfilm, so existing results stay '
			 'reproducible. Any N > 0 gives LL = tfilm + 2*N/res; N >= 2 is converged, 4 recommended.')
	parser.add_argument('-eps', type=str,  default=2, help='dielectric constant of the polarized phase (phase2)')
	parser.add_argument('-eps_ref', type=str,  default=1, help='dielectric constant of the reference phase (phase1)')
	parser.add_argument('-saveas', type=str, default='./temp', help='directory for saving spectra')

	parser.add_argument('-comp', type=str, nargs='+', default=["Ex","Ey"], help='the list of components to save')

	parser.add_argument('-ks', type=float, nargs='+', default=[0.05, 6.05], help='the min and max of wavenumbers')
	parser.add_argument('-nfreqs', type=int, default=50, help='the number of frequencies')
	parser.add_argument('-fullprofile', action = 'store_true', default=False, help='perform FFT including the reference space')

	parser.add_argument('-tempname', type=str, default='', help='file name for the spectra in the -ref mode')
	parser.add_argument('-is_point', action = 'store_true', default=False, help='load point configuration. convert it to a packing')

	parser.add_argument('-JouleHeating', action = 'store_true', default=False, help='Add Joule heating calculation')
	parser.add_argument('-ScattPower', action = 'store_true', default=False, help='Add Scattering power calculation')
	parser.add_argument('-checkpoint', action='store_true', default=False, help='write restart checkpoints, and resume from one if present. Requires the dispersive-media dump/load patch (30_Codes/patches/); a STOCK libmeep loads a patched checkpoint silently with P=0, so preflight with install_patched_libmeep.sh --check')
	parser.add_argument('-checkpoint_interval', type=float, default=6.0, help='hours between checkpoints (default: 6)')

	parser.add_argument('-scale2sim', type=float, default=1.0, help='scaling factor from the input geometry to the sample in simulation')

	parser.add_argument('-test', action = 'store_true', default=False, help='testing the setup with a short animation')
	parser.add_argument('-test2', action = 'store_true', default=False, help='testing the setup with a short animation')

	parser.add_argument('-particle', type=str, default='disk', help='particle shape')

	args = parser.parse_args()
	# Validate salvage arguments BEFORE main() builds the simulation: at res 3500
	# _set_materials plus the checkpoint load cost well over an hour, and a missing
	# flag that surfaces as a NameError after that wastes a six-node allocation.
	if args.salvage:
		_missing = [f for f in ('ScattPower', 'checkpoint', 'salvage_goodref',
								'salvage_origtable') if not getattr(args, f)]
		if _missing or args.ref:
			raise SystemExit("-salvage requires {0}{1}".format(
				_missing, " and is incompatible with -ref" if args.ref else ""))
	main(args)
