"""Verify meep checkpoint/restart fidelity for dissipative (metallic) media.

Runs a simulation straight through, then repeats it with a mid-flight
``Simulation.dump`` and a restart in a fresh ``Simulation``, and checks that the
resumed run reproduces the uninterrupted one.  Restart follows meep's supported
ordering: load the structure, add the DFT monitors, then load the fields.

Two things are checked independently:

* **field state** -- must agree to round-off.  This is what
  ``meep-1.31.0-checkpoint-dispersive.patch`` fixes; without it ``dump`` aborts
  outright with "non-null polarization_state in fields::dump (unsupported)".
* **accumulated DFT flux** -- also asserted.  This used to be excluded: stock meep
  dumps the DFT accumulators but restores only part of them, because fields::dump
  walks the wrong dft_chunk link.  The patch fixes that, so a resumed run now
  reproduces the uninterrupted spectrum exactly.

``T1``/``T2`` must be exact multiples of ``dt = Courant / resolution``; otherwise
the split run takes one extra timestep and nothing matches.

Usage
-----
    python test_checkpoint_dispersive.py [tmpdir] [drude|ag|diel]
    mpirun -np 2 python test_checkpoint_dispersive.py ck_mpi ag
"""

import os
import shutil
import sys

import numpy as np

import meep as mp

mp.verbosity(0)

TMP = sys.argv[1] if len(sys.argv) > 1 else "ck_tmp"
KIND = sys.argv[2] if len(sys.argv) > 2 else "ag"

if KIND == "ag":  # multi-pole Drude+Lorentz fit
    from meep.materials import Ag as MAT
elif KIND == "drude":  # single Drude pole
    MAT = mp.Medium(
        epsilon=1,
        E_susceptibilities=[mp.DrudeSusceptibility(frequency=1.0, gamma=0.1, sigma=5.0)],
    )
elif KIND == "diel":  # non-dispersive control
    MAT = mp.Medium(epsilon=12)
else:
    raise SystemExit(f"unknown material {KIND!r}")

PT = mp.Vector3(0.35, 0.11)  # sample point inside the particle
T1, T2 = 9.0, 9.0  # both exact multiples of dt = Courant / resolution
FCEN, DF, NFREQ = 1.4, 0.6, 11  # 1/um; inside Ag's fitted band
# Courant=0.3 at resolution 40 is the stability floor for meep's Ag fit here;
# the default 0.5 diverges (negative eps needs the extra margin).
RESOLUTION, COURANT = 40, 0.3


def build() -> mp.Simulation:
    """Build the test simulation (no monitors attached).

    Returns
    -------
    mp.Simulation
        A 2D TM simulation of a metal cylinder driven by a Gaussian dipole.
    """
    return mp.Simulation(
        cell_size=mp.Vector3(6, 5),
        resolution=RESOLUTION,
        Courant=COURANT,
        geometry=[mp.Cylinder(radius=0.5, center=mp.Vector3(0.3), material=MAT)],
        sources=[
            mp.Source(
                mp.GaussianSource(FCEN, fwidth=DF),
                component=mp.Ez,
                center=mp.Vector3(-2),
            )
        ],
        boundary_layers=[mp.PML(1.0)],
    )


def add_mon(sim: mp.Simulation):
    """Attach the transmission flux monitor.

    Parameters
    ----------
    sim : mp.Simulation
        Simulation to attach the monitor to (this triggers ``init_sim``).

    Returns
    -------
    meep.simulation.DftFlux
        The flux object, for ``mp.get_fluxes``.
    """
    return sim.add_flux(
        FCEN, DF, NFREQ, mp.FluxRegion(center=mp.Vector3(1.8), size=mp.Vector3(0, 2))
    )


# --- reference: one uninterrupted run -------------------------------------
ref = build()
ref_fx = add_mon(ref)
ref.run(until=T1 + T2)
ref_pt = ref.get_field_point(mp.Ez, PT).real
ref_flux = np.array(mp.get_fluxes(ref_fx))

# --- same run, checkpointed at T1 and resumed in a fresh Simulation --------
a = build()
add_mon(a)
a.run(until=T1)
if mp.am_master():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
mp.all_wait()
a.dump(TMP)

b = build()
b.load(TMP, load_structure=True, load_fields=False)
b_fx = add_mon(b)
b.load(TMP, load_structure=False, load_fields=True)
b.run(until=T2)
ck_pt = b.get_field_point(mp.Ez, PT).real
ck_flux = np.array(mp.get_fluxes(b_fx))

scale = max(np.abs(ref_flux).max(), 1e-30)
d_pt = abs(ck_pt - ref_pt) / max(abs(ref_pt), 1e-30)
d_flux = np.abs(ck_flux - ref_flux).max() / scale
ok = d_pt <= 1e-9 and d_flux <= 1e-9

if mp.am_master():
    print(f"[{KIND}] Ez(t=T1+T2) ref={ref_pt: .8e} ckpt={ck_pt: .8e} rel={d_pt:.2e}")
    print(f"[{KIND}] DFT flux    rel={d_flux:.2e}")
    # RESULT: is the machine-readable marker install_patched_libmeep.sh requires.
    # Deliberately not an `assert`: `python -O` / PYTHONOPTIMIZE strips asserts, which
    # would let a failing test exit 0 and pass the installer's gate.
    print(f"[{KIND}] RESULT: {'PASS' if ok else 'FAIL'}")

if not ok:
    raise SystemExit(
        f"{KIND}: not restored (field rel err {d_pt:.3e}, DFT flux rel err {d_flux:.3e})"
    )
