# Helpers for film_transmittance.py: pattern-file readers and 2D-pattern geometry,
# material selection, output header, Courant factor, and a restart-safe DFT
# convergence stopping condition.

import meep as mp
import numpy as np
import json
import os

## ---- Meep material library (meep/materials.py) ----
from meep.materials import Cu
from meep.materials import Au
from meep.materials import Ag

C0 = 299_792_458.0 # speed of light in vacuum (m/s)
hbar2eV = 6.582119569e-16 # eV * s
a0 = 1e-6 # 1 micron (unit length)

component_map = {
    'Ex': mp.Ex,
    'Ey': mp.Ey,
    'Ez': mp.Ez,
    'Dx': mp.Dx,
    'Dy': mp.Dy,
    'Dz': mp.Dz,
    'Hx': mp.Hx,
    'Hy': mp.Hy,
    'Hz': mp.Hz,
    'Sx': mp.Sx,
    'Sy': mp.Sy,
    'Sz': mp.Sz,
}


def read_packing(load_prefix):
    with open(load_prefix) as f: 
        lines = [line.rstrip('\n') for line in f]
    # space dimension
    d = int(lines[0].rstrip('\t'))
   
    del lines[0]
    
    # basis vectors
    basis = np.zeros((d,d))
    for i in range(d):
        temp = lines[i].split('\t')
        for j  in range(d):
            basis[i,j] = float(temp[j])
    
    V = np.abs(np.linalg.det(basis))
    del lines[0:d]

    ##positions in a unit cell
    pos_unitcell = np.zeros((0,d))
    for pos in lines:
        temp = np.array([float(x) for x in pos.split('\t')[0:d]])
        pos_unitcell = np.vstack((pos_unitcell, temp))
    radii = np.zeros((0,1))
    idx = 0
    for pos in lines:
        #print(pos)
        temp = np.array([float(pos.split('\t')[4])])
        radii = np.vstack((radii, temp))
            
    return list([d, basis, pos_unitcell, radii])


def read_config(load_prefix):
    
    with open(load_prefix) as f: 
        lines = [line.rstrip('\n') for line in f]
    # space dimension
    d = int(lines[0].rstrip('\t'))
   
    del lines[0]
    
    # basis vectors
    basis = np.zeros((d,d))
    for i in range(d):
        temp = lines[i].split('\t')
        for j  in range(d):
            basis[i,j] = float(temp[j])
    
    V = np.abs(np.linalg.det(basis))
    del lines[0:d]

    ##positions in a unit cell
    pos_unitcell = np.zeros((0,d))
    for pos in lines:
        temp = np.array([float(x) for x in pos.split('\t')[0:d]])
        pos_unitcell = np.vstack((pos_unitcell, temp))
            
    return list([d, basis, pos_unitcell])


# def PolyDispersion2Geometry (filename: str, dispersion: mp.Medium, scale: float=1.0):
#     dis = tp.PolyDispersion()
#     dis.load(filename)
#     Lx = dis.basis[0,0]; Ly = dis.basis[1,1]
#     Geometry = []
    
#     # base
#     #Geometry.append(
#     #    mp.Block(center = mp.Vector3(),size = mp.Vector3(Lx,Ly,mp.inf),material=matrix))

#     for vs in dis.prts_list:
#         list_vertices = []
#         for v in vs.vertices:
#             v_ = v + vs.centroid
#             list_vertices.append(mp.Vector3(v_[0],v_[1]))
        
#         #centroid = mp.Vector3(vs.centroid[0], vs.centroid[1])

#         Geometry.append(
#             mp.Prism(
#                 vertices = list_vertices,
#                 height = mp.inf,  \
#                 material = dispersion
#                 #, center = centroid,
#             ))
#     return [Lx,Ly,Geometry]


def ConvertPacking2Geometry_2D(filename: str, phi2:float, material_particle: mp.Medium, is_packing: bool=True, scale: float=1.0, material_matrix: mp.Medium=mp.Medium(epsilon=1.0), prt_shape = "disk"):
	'''
	read a txt file of disk packing, 
	and return the corresponding array of geometry objects,
	unit cell size is also returned.
	'''
	vd = np.pi 
	if prt_shape == "disk":
		vd = np.pi # Area of a disk with radius 1
	elif prt_shape == "square" or prt_shape == "diamond":
		vd = 4.0 # Area of a square with half-side length 1 (side length 2)
	else:
		raise ValueError(f"Unsupported particle shape: {prt_shape}")

	geometry = []
	
	# Define a lambda function to create the particle geometry based on prt_shape
	particle_creator = None
	if prt_shape == "disk":
		particle_creator = lambda r, center_vec, material: mp.Cylinder(
			radius = r, height = mp.inf, axis = mp.Vector3(0,0,1), center = center_vec, material=material
		)
	elif prt_shape == "square":
		particle_creator = lambda r, center_vec, material: mp.Block(
			size = mp.Vector3(2 * r, 2 * r, mp.inf), center = center_vec, material=material
		)
	elif prt_shape == "diamond":
		# A diamond is a square of side 2*r rotated by 45 degrees (area 4*r**2,
		# matching vd above); its vertices lie r*sqrt(2) from the center.
		particle_creator = lambda r, center_vec, material: mp.Block(
			size = mp.Vector3(2 * r, 2 * r, mp.inf), center = center_vec, e1 = mp.Vector3(1, 1, 0), e2 = mp.Vector3(-1,1,0), material=material
		)

	if is_packing:		
		pos = read_packing(filename)
		d = pos[0]
		assert(d==2) 
		Lx = np.linalg.norm(pos[1][0])
		Ly = pos[1][1][1]

		V = np.abs(np.linalg.det(pos[1]))
		N = np.shape(pos[2])[0]
		phi_curr = vd * np.sum(pos[3]**d) / V # d=2 for area
		# read_packing returns an (N, 1) column; flatten it so radii[i] is a scalar.
		radii = np.copy(pos[3][:, 0])
		if phi2 >0.:
			radii *= (phi2 / phi_curr)**(1./d)
	else: 
		pos = read_config(filename)
		d = pos[0] 
		assert(d==2)
		Lx = np.linalg.norm(pos[1][0])
		Ly = pos[1][1][1]

		V = np.abs(np.linalg.det(pos[1]))
		N = np.shape(pos[2])[0]
		rho = N/V
		
		if phi2 >0.:
			radii = (phi2 / (vd * rho))**(1./d) * np.ones(N)
		else: 
			print("use default packing fraction: 0.20")
			radii = (0.2 / (vd * rho))**(1./d) * np.ones(N)
	
	Lx *= scale
	Ly *= scale
	radii *= scale
	pos[2]	*= scale

	geometry.append(mp.Block(center = mp.Vector3(),size = mp.Vector3(Lx,Ly,mp.inf),material=material_matrix))
	for i in range(N):
		x0 = pos[2][i][0] - 0.5*Lx
		y0 = pos[2][i][1] - 0.5*Ly
		center_vec = mp.Vector3(x0, y0)
		current_radius = radii[i]

		# Print statements remain conditional as they describe the specific shape being added
		if prt_shape == "disk":
			print(f"Adding a disk at ({x0:.3f}, {y0:.3f}) with radius {current_radius:.3f}")
		elif prt_shape == "square":
			print(f"Adding a square at ({x0:.3f}, {y0:.3f}) with side {2 * current_radius:.3f}")
		elif prt_shape == "diamond":
			print(f"Adding a diamond (rotated square) at ({x0:.3f}, {y0:.3f}) with side {2 * current_radius:.3f}")

		geometry.append(
			particle_creator(current_radius, center_vec, material_particle)
		)
	return [Lx, Ly, geometry]


def ConvertPacking2Geometry_3D(filename: str, phi2:float, material_particle: mp.Medium, is_packing: bool=True, scale: float=1.0, material_matrix: mp.Medium=mp.Medium(epsilon=1.0)):
	'''
	read a txt file of sphere packing, 
	and return the corresponding array of geometry objects,
	unit cell size is also returned.
	'''
	vd = 4.0 * np.pi /3.0
	geometry = []
	if is_packing:		
		pos = read_packing(filename)
		d = pos[0]
		assert( d == 3 ) 
		Lx = np.linalg.norm(pos[1][0])
		Ly = pos[1][1][1]
		Lz = pos[1][2][2]

		V = np.abs(np.linalg.det(pos[1]))
		N = np.shape(pos[2])[0]
		phi_curr = vd * np.sum(pos[3]**d) / V
		# read_packing returns an (N, 1) column; flatten it so radii[i] is a scalar.
		radii = np.copy(pos[3][:, 0])
		if phi2 >0.:
			radii *= (phi2 / phi_curr)**(1./d)
	else: 
		pos = read_config(filename)
		d = pos[0] 
		assert( d == 3 )
		Lx = np.linalg.norm(pos[1][0])
		Ly = pos[1][1][1]
		Lz = pos[1][2][2]

		V = np.abs(np.linalg.det(pos[1]))
		N = np.shape(pos[2])[0]
		rho = N/V
		
		if phi2 >0.:
			radii = (phi2 / (vd * rho))**(1./d) * np.ones(N)
		else: 
			print("use default packing fraction: 0.20")
			radii = (0.2 / (vd * rho))**(1./d) * np.ones(N)
	
	Lx *= scale
	Ly *= scale
	Lz *= scale
	radii *= scale
	pos[2]	*= scale

	geometry.append(mp.Block(center = mp.Vector3(),size = mp.Vector3(Lx,Ly,mp.inf),material=material_matrix))
	for i in range(N):
		x0 = pos[2][i][0] - 0.5*Lx
		y0 = pos[2][i][1] - 0.5*Ly
		z0 = pos[2][i][2] - 0.5*Lz
		print(x0, y0, z0)
		geometry.append(
			mp.Sphere(
				radius = radii[i], \
				center = mp.Vector3(x0, y0, z0),\
				material=material_particle
			)
		)
	return [Lx, Ly, Lz, geometry]


def ChooseMaterial(material_name: str):
	if material_name == 'Cu':
		Material = Cu
		print('Use copper')
	elif material_name == 'Au':
		Material = Au
		print('Use gold')
	elif material_name == 'Ag':
		Material = Ag
		print('Use silver')
	elif material_name == 'Ag_Drude':
		factor = (1/hbar2eV * a0 / C0)
		gamma_D = 36.7e-3 *factor / (2*np.pi)
		omega_p = 8.8 *factor #2*np.pi
		Material = mp.Medium(
			epsilon=3,
			E_susceptibilities=[
				mp.DrudeSusceptibility(
					frequency=omega_p/(2*np.pi),
					gamma=gamma_D,
					sigma=1  # tune to match data
				)
				# ,
				# mp.LorentzianSusceptibility(
				# 	frequency=f0_L,
				# 	gamma=gamma_L,
				# 	sigma=sigma_L
				# )
			]
		)
		print('Use silver, Drude model')

	else:
		Material =  mp.Medium(epsilon=float(material_name))
		print("Use a dielectric material {0}".format(material_name))

	return Material


def GenerateHeader(args):
	header = 'Transmittance through '

	## add information about the geometry ##

	if args.load != '':
		header += 'a film patterned with a custom disk/square packing'
		if args.ref:
			header += '(reference)\n'
			print("(reference)\n")	
		
		else:
			header += '\n\tFile: {0}\n'.format(args.load)
			print("Load a saved configuration {0}".format(args.load))

		header += '\tby FDTD simulations (MEEP)\n'
		header += '\targs.dsrc = {0:0.2e}\n'.format(args.dsrc)
		header += '\targs.ddet = {0:0.2e}\n'.format(args.ddet)
		header += '\targs.dpml = {0:0.2e}\n'.format(args.dpml)
		if hasattr(args, 'dft_tol'):
			# The run stops on the DFT decay criterion; -decay only spaces multi-source pulses.
			header += '\tdft_tol = {0:.1e}, dft_nconsec = {1}, maxt = {2:g}\n'.format(
				args.dft_tol, args.dft_nconsec, args.maxt)
		else:
			header += '\tdecayed ratio = {0:.1e}\n'.format(args.decay)
		if args.JouleHeating:
			header += 'wavelength[1000nm]\ttransmittance\treflectance\tAbsorbance\tIncidentPower[a.u.]'	
		else:
			header += 'wavelength[1000nm]\ttransmittance\treflectance\tIncidentPower[a.u.]'
	return header

def DetermineCourantFactor(args, material1: mp.Medium, material2: mp.Medium, dimension):
	r"""!
	@brief Courant factor S from the instantaneous (infinite-frequency) index.

	FDTD/CFL stability with dispersive media is governed by the instantaneous
	permittivity @f$\varepsilon_\infty@f$ that multiplies E in the update -- not
	by @f$\mathrm{Re}\sqrt{\varepsilon(\omega)}@f$ over the source band, which
	for a metal is the loss-induced residual of an evanescent mode, not a phase
	velocity (Petropoulos, IEEE Trans. Antennas Propag. 42, 62 (1994);
	Bidegaray-Fesquet, SIAM J. Numer. Anal. 46, 2551 (2008)).  Meep's condition
	is @f$S < n_\min/\sqrt{d}@f$; 0.9 is a safety margin, capped at Meep's
	default 0.5.

	@note This bound ignores the dispersive poles.  With meep.materials.Ag in the 3D
	      film example, S = 0.5 diverged at resolution <= 60 and S = 0.4 was stable
	      at resolution 40, so coarse metal runs need a smaller S than returned here.

	@param args       Unused; kept for caller compatibility.
	@param material1  Particle medium (mp.Medium).
	@param material2  Matrix/reference medium (mp.Medium).
	@param dimension  Spatial dimension d (2 or 3).
	@return Courant factor S (dimensionless), <= 0.5.
	"""
	n_min = min(1.0, *(np.sqrt(m.epsilon_diag.x) for m in (material1, material2)))
	print ("n_min = ", n_min)
	Courant_parameter = min(0.5, 0.9 * n_min / np.sqrt(dimension))
	return Courant_parameter


def stop_when_dft_decayed_stable(tol=1e-11, n_consecutive=3, minimum_run_time=0, maximum_run_time=None, state_file=None):
	r"""!
	@brief Like mp.stop_when_dft_decayed, but requires the decay ratio to stay below
	       @p tol for @p n_consecutive successive checks before stopping.

	Meep's stock condition stops the moment the ratio
	@f$ \Delta / \Delta_{max} @f$ first dips below @p tol, where
	@f$ \Delta @f$ is the change in fields.dft_norm() between checks and
	@f$ \Delta_{max} @f$ is the largest such change seen so far.  That ratio is
	*not* monotone near the noise floor: in the phi=0.75 Ag-chain run at
	resolution 2500 it sat at 1e-6..1e-5 from t~8 to t~16 with a single isolated
	excursion to 2.32e-8 at t=11.24 -- 300x below both neighbours -- and that one
	sample ended three separate runs (tol = 5e-6, 1e-6 and 1e-7 all stopped at the
	identical timestep 682721).  Requiring @p n_consecutive hits makes an isolated
	dip harmless: the counter resets whenever the ratio pops back above @p tol.

	Replayed against the measured 95-point trace of that run, @p tol = 1e-6 with
	@p n_consecutive = 3 stops at t = 15.21 (1.35x the timesteps of the stock
	condition) -- cheaper *and* more robust than buying the same confidence by
	tightening the tolerance to 1e-8, which costs 1.82x and is still decided by a
	single sample.

	@param tol            Tolerance on the DFT decay ratio (Meep default 1e-11).
	@param n_consecutive  Number of successive checks that must satisfy the
	                      tolerance.  1 reproduces mp.stop_when_dft_decayed.
	@param minimum_run_time  Minimum simulated time before stopping is allowed.
	@param maximum_run_time  Ceiling on simulated time; returns True regardless.
	@return A condition function suitable for Simulation.run(until_after_sources=...).

	@param state_file  Optional JSON file holding the convergence history.  Loaded at
	                   construction if it exists; write it with save_decay_state()
	                   whenever a checkpoint is dumped.

	@note The check interval is Meep's own: max(1/dft_maxfreq/dt, max_decimation)
	      timesteps.  Meep's stock version calibrates it only at fields.t == 0, which
	      makes it ill-defined on a continuation run; this version calibrates whenever
	      the interval is still 0, so a resumed run recalibrates instead of sampling
	      every step.  Combined with @p state_file and with the seed sample excluded
	      from maxchange, the criterion survives a restart.  Measured before the fix:
	      a run converging at t = 13.58 resumed from t = 9.58 stopped at t = 9.71 with
	      a reported ratio of exactly 0.0000e+00.
	"""
	closure = {"previous_fields": 0, "t0": 0, "dt": 0, "maxchange": 0, "hits": 0}

	# A resumed run starts with an empty closure, and every field of it matters.
	# Restoring it is what keeps the stopping criterion meaningful across a restart;
	# see save_decay_state() and the three failure modes in the note above.
	if state_file and os.path.exists(state_file):
		try:
			with open(state_file, "r") as fh:
				saved = json.load(fh)
			for k in ("previous_fields", "t0", "dt", "maxchange", "hits"):
				if k in saved:
					closure[k] = saved[k]

			# previous_fields is the DFT norm at the LAST CHECK, at timestep t0, while
			# the restored fields are at the timestep the dump was taken.  Normally the
			# dump lands within one check interval of t0, so the next comparison spans a
			# proper interval and the resumed run reproduces a continued one exactly.
			#
			# It can be staler than that -- e.g. a -maxt ceiling makes the condition
			# return True while Meep keeps stepping until the sources finish, so the
			# checker stops updating while t runs on.  Then the first post-resume
			# difference spans the whole gap, becomes the new maxchange, and every later
			# ratio is driven toward 0: measured 5.5e-1 error against straight-through.
			# In that case only previous_fields is discarded; maxchange and hits, which
			# are what keep a resumed run from having to re-earn its convergence, are
			# kept.
			t_saved = saved.get("timestep", closure["t0"])
			if closure["dt"] and (t_saved - closure["t0"]) > closure["dt"]:
				print("decay checker: last check at step {0:g} is stale relative to the "
					  "checkpoint at step {1:g}; re-seeding previous_fields only".format(
						  closure["t0"], t_saved), flush=True)
				closure["previous_fields"] = 0
			print("decay checker resumed: maxchange={0:.6e} hits={1:g} dt={2:g} t0={3:g}".format(
				closure["maxchange"], closure["hits"], closure["dt"], closure["t0"]), flush=True)
		except Exception as exc:
			print("WARNING: could not read decay state {0}: {1}".format(state_file, exc),
				  flush=True)

	def _stop(_sim):
		# Calibrate whenever uncalibrated, NOT only at fields.t == 0.  On a resumed run
		# t is restored non-zero, so the original `if _sim.fields.t == 0` never fired and
		# the check interval stayed 0.  With interval 0 the checker samples every
		# timestep while the DFT accumulators only advance every decimation_factor
		# steps, so two consecutive checks read an identical dft_norm(), change is
		# EXACTLY zero, the ratio is exactly zero, and the run stops after
		# n_consecutive steps believing it has converged.
		if closure["dt"] == 0:
			closure["dt"] = max(
				1 / _sim.fields.dft_maxfreq() / _sim.fields.dt,
				_sim.fields.max_decimation(),
			)
		if maximum_run_time and _sim.round_time() > maximum_run_time:
			return True
		elif _sim.fields.t <= closure["dt"] + closure["t0"]:
			return False
		else:
			previous_fields = closure["previous_fields"]
			current_fields = _sim.fields.dft_norm()

			if previous_fields == 0:
				# Seed only.  The first sample is an absolute norm, not a change, so it
				# must NOT enter maxchange: doing so seeds the denominator with the whole
				# accumulated norm and every later ratio is divided by a number orders of
				# magnitude too large.  t0 is set here so the interval throttling starts
				# from this sample rather than from 0.
				closure["previous_fields"] = current_fields
				closure["t0"] = _sim.fields.t
				return False

			change = np.abs(previous_fields - current_fields)
			closure["maxchange"] = max(closure["maxchange"], change)
			closure["previous_fields"] = current_fields
			closure["t0"] = _sim.fields.t

			if closure["maxchange"] == 0:
				return False

			ratio = np.real(change / closure["maxchange"])
			# the guard: an isolated dip cannot end the run
			closure["hits"] = closure["hits"] + 1 if ratio <= tol else 0
			if mp.verbosity.meep > 1:
				print("DFT fields decay(t = {0:0.2f}): {1:0.4e}  [{2}/{3}]".format(
					_sim.meep_time(), ratio, closure["hits"], n_consecutive))
			return closure["hits"] >= n_consecutive and _sim.round_time() >= minimum_run_time

	_stop.state = closure
	return _stop


def save_decay_state(stop_fn, path, timestep=None):
	"""!
	@brief Persist a stop_when_dft_decayed_stable checker's convergence history.

	Written next to the checkpoint it belongs to, so a resumed run restores the same
	maxchange reference, check interval and consecutive-hit count.  Without it the
	checker restarts blind and terminates the resumed run almost immediately, which is
	silent: no error, and a plausible-looking but truncated spectrum.

	Master-only and atomic (write to .tmp, then os.replace), so a kill during the dump
	cannot leave a half-written state file for the next run to load.

	@param stop_fn   The condition returned by stop_when_dft_decayed_stable.
	@param path      Destination JSON file.
	@param timestep  fields.t at the moment of the dump, so a resumed run can tell
	                 whether the saved last-check state is still adjacent to it.
	@return True if a state was written (or this rank is not master), False if stop_fn
	        carries no state.
	"""
	state = getattr(stop_fn, "state", None)
	if state is None:
		return False
	if mp.am_master():
		out = {k: float(v) for k, v in state.items()}
		if timestep is not None:
			out["timestep"] = float(timestep)
		tmp = path + ".tmp"
		with open(tmp, "w") as fh:
			json.dump(out, fh)
		os.replace(tmp, path)
	return True
