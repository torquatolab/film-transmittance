# Transmittance through a patterned film

This package uses 3D Meep FDTD to calculate broadband power transmittance (plus
signed reflectance, and optionally absorption) of a finite-thickness film
patterned with a periodic 2D array of disks or squares. The pattern is extruded
along z through the film thickness. Boundaries are periodic in x/y, with PML in
z; a normally incident plane wave travels toward negative z. Particles may be
metals (Ag, Au, Cu) or dielectrics. It models an infinite periodic film, not an
isolated object. A 3D voxel array (`.npz` or `.tif`) can also be used as the
film; see "Voxel input" below.

| File | Purpose |
|---|---|
| `film_transmittance.py` | The simulation: reference or sample run, spectrum and field output |
| `film_fdtd_utils.py` | Helpers: pattern-file readers, geometry, materials, Courant factor, stopping condition |
| `run_spectrum.sh` | Runs the reference and sample stages with shared settings |
| `della.slurm` | Princeton Della batch template for `run_spectrum.sh` |
| `voxel_io.py` | Voxel input: reads `.npz`/`.tif` arrays, checks JSON sidecar fractions |
| `voxel_geometry.py` | Voxel input: axis orientation, crop, MaterialGrid film, grid alignment |
| `voxel_crop.py` | Chooses the lateral crop of a non-periodic stack with the best-matching faces |
| `tests/` | Validation scripts and Della job files used to check the voxel input (see `tests/README.md`) |
| `examples/one_disk.txt` | Example pattern: one disk per square unit cell |
| `patches/` | Optional Meep patch and installer needed for checkpoint/restart |
| `environment.yml` | Conda environment with MPI-enabled Meep 1.31.0 |

These two Python files were previously named `2D_plasmonic_film.py` and
`shared_FDTD.py`. Meep prefixes flux files with the script name, so reference
flux files written by the old script (`2D_plasmonic_film-*refl-flux.h5` and its
`.meta.json`) are not found by the renamed script. Rerun the reference stage,
or rename those two files to the `film_transmittance-` prefix.

## Install and run

Use Linux or WSL and Conda. Meep is **pymeep**, not the unrelated pip package
named `meep`. The environment pins a version covered by the included patch.

```bash
conda env create -f environment.yml
conda activate film-meep
bash run_spectrum.sh results
# Optional parallel run in a NEW directory:
NP=4 bash run_spectrum.sh results-mpi
```

The example is one silver cylinder per 0.2 by 0.2 micrometer unit cell, area
fraction 0.2, thickness 0.1 micrometer, x polarization, wavelengths 0.8–1.2
micrometers. `film_fdtd_utils.py` is the only local Python dependency. NumPy and Meep
(including its material library) are required; matplotlib supports optional plots.

## Why two runs?

1. **Reference run (`-ref`):** removes the entire film, including its matrix,
   and measures the incident spectrum in vacuum. It also saves the incident
   reflection-monitor DFT fields for subtraction in the sample run.
2. **Sample run:** loads that reference, simulates the film, and divides the
   transmitted flux by the incident flux. Subtracting incident fields at the
   upper monitor separates reflection from illumination.

The wrapper runs both stages in order with identical settings. Preserve the
reference files together in the run directory: `incident_inc_flux.npy`,
`film_transmittance-incidentrefl-flux.h5`, and its `.meta.json` (the stem is
`-tempname`, here `incident`).
Use the same geometry input (even for `-ref`), cell, resolution, source,
frequencies, monitor layout, Meep build, and MPI rank count for both stages.
The code checks the saved rank count but does not validate every parameter.
Use a new directory whenever parameters change; do not run two jobs in one directory.

## Output and customization

`results/sample_trans-x.txt` has four columns (comment lines begin with `#`):

| Column | Meaning |
|---|---|
| 1 | Vacuum wavelength in micrometers |
| 2 | Power transmittance T |
| 3 | Signed reflected/incident flux ratio; physical reflectance is its negative |
| 4 | Signed incident flux (normally negative because illumination travels −z) |

Wavelengths appear in descending order. Multiply T by 100 for percent.
The script also saves field arrays and coordinate/weight arrays; it is not a
flux-only solver. For `-saveas sample` these are `sample_{x,y,z,w}.npy` and, per
frequency, `sample__ka-<k>-<component>.npy`, where `<k>` is the angular
wavenumber rounded to four decimals. `-JouleHeating` adds absorption A as column 4, moving incident
flux to column 5, and saves additional fields; the run exits nonzero (after
writing the table) if max |T + |R| + A − 1| reaches 0.05. Without it, A can be
estimated as 1 − T + (column 3), since column 3 is the negative of the physical
reflectance, after verifying numerical convergence. The comment header records
the file, `dsrc`, `ddet`, `dpml`, and the stopping settings.

Edit the shared `common` array in `run_spectrum.sh` to change the calculation.
Lengths use 1 micrometer as the unit; resolution is pixels per micrometer.
`-ks` takes two angular wavenumbers: k = 2π / wavelength (micrometers), with
kmin first. Use at least two frequencies. Keep `dsrc > ddet > 0` so the
reflection monitor is between the source and film. Invalid frequency settings
are rejected at startup; a violated monitor order only prints a warning,
because the script's own defaults (`-dsrc 0.3 -ddet 0.3`) violate it. Select
polarization x or y; any other value (including the script's default, `z`) runs
an Ex source with a warning.
`-eps` accepts Ag, Au, Cu, Ag_Drude, or a numeric dielectric constant;
`-eps_ref` sets the film matrix, while the exterior remains vacuum.

`examples/one_disk.txt` is a tab-separated point file: dimension 2, two lattice
basis rows, then x/y positions in the uncentered cell. Use an axis-aligned
rectangular cell. With `-is_point`, `-phi` sets particle area fraction. Packing
files without `-is_point` instead require the radius in the fifth column of
each particle row. `-particle` selects `disk` (default; becomes a cylinder),
`square`, or `diamond` (a square rotated 45°); squares become blocks.
An empty `-load` gives vacuum, not a uniform film. `-Z2` and polygonal
`.Dispersion` packings are not implemented and are rejected.

`RES=120` changes example resolution; `EPS=4` selects a dielectric example.
The supplied resolution is a starting point, not a convergence claim. Do not
lower it for silver: the script uses Courant factor 0.5 for these materials,
and the Ag example diverged (`simulation fields are NaN or Inf`) at `RES` 40,
50, and 60, while 70 and 80 ran. At `RES=40`, Courant 0.4 was stable. Refine
resolution, PML/separations, and DFT tolerance until the spectrum is stable.
`MAXT=200` is a **simulation-time ceiling**, not seconds of wall time; reaching
it produces output without proving convergence. The log then contains a
`WARNING: run reached the -maxt ... ceiling` line; increase the ceiling if so.

## Voxel input

`-load` with a `.npz` (array under `-voxel_key`, default `g`), `.tif`, or `.tiff`
file uses a 3D binary array as the whole film instead of an extruded pattern.
Nonzero voxels are solid (`-eps`); zero voxels are the matrix (`-eps_ref`);
outside the film is vacuum. `-voxel_size` (micrometers per voxel) is required.
The film normal is stored axis `-normal_axis` (default 0, the frame axis of a
TIFF stack); the other two axes become y and x in stored order. Lx, Ly, and the
thickness come from the array, so `-tfilm`, `-phi`, `-is_point`, and
`-scale2sim` do not apply. `-crop z0:z1,y0:y1,x0:x1` (stored axis order,
half-open) simulates a sub-block. A JSON sidecar next to an `.npz` is checked:
the solid fraction must match `phi1_realized`, or else `phi2_realized`.

```bash
INPUT=path/structure.npz VOXEL_SIZE=0.01 bash run_spectrum.sh results-voxel
INPUT=stack.tif VOXEL_SIZE=0.01 CROP=0:258,0:57,16:134 bash run_spectrum.sh results-tif
```

The array is placed with `mp.MaterialGrid`, one sample per voxel. Use
`-res` equal to an integer multiple of `1/voxel_size`: the lateral block centre
and the film z position are shifted by at most half a pixel so voxel faces lie
midway between grid nodes (the log prints the shift). Keep
`-interface_averaging` off; it was less accurate in every test. The x/y
boundaries are periodic, so the array is treated as one period of an infinite
film. For stacks that are not periodic, `voxel_crop.best_crop()` finds the
lateral crop whose opposite faces match best. `-no_fields` skips the volume
field arrays (spectra only); `-field_wavelengths` saves them only at listed
wavelengths. Reference and sample runs must use the same file, crop, voxel
size, and normal axis; the sample run refuses a reference that differs.

**Known limitations (validation of 2026-09-17; the defaults are being revised).**
For cells whose lateral period exceeds the wavelength, the default z padding
(`-ddet 0.2 -dpml 0.3 -tpml 0.5`) is not adequate: a periodic unit cell gave an
unphysical T = -0.052 near a diffraction cutoff, and a TIFF crop grew without
bound after t ~ 250. `-tpml 1.5` removed both. Runs of these structures do not
meet the DFT stopping criterion and stop at `-maxt`, so the ceiling must be
chosen and checked per structure family. Spectra of cropped non-periodic
stacks depend on the lateral boundary treatment by about 0.02 in T on average
(up to about 0.2 at sharp resonances), and 101 frequencies under-resolve their
narrow features.

## Checkpoints: continue an interrupted calculation

Checkpoints save the simulation state and accumulated Fourier fields so a long
run can resume without starting over. They are different from reference files:
a checkpoint resumes time evolution; a reference normalizes the spectrum.

Checkpointing is off by default. For this script, use the included Meep
checkpoint patch before enabling it, including for dielectric runs because it
also repairs DFT restoration. Dispersive metals require their auxiliary
polarization state to be saved. See `patches/README.md` for patch details.
The included installer targets a Linux Conda MPI environment and replaces its
libmeep after making a backup; it needs compilers, Autotools, make, patch,
libtool, and the Meep dependency headers. Build on an allocated compute node
on Della, not as a long-running login-node task.

```bash
# In the activated environment, after installing build prerequisites:
bash patches/install_patched_libmeep.sh --prefix "$CONDA_PREFIX"
bash patches/install_patched_libmeep.sh --verify-only
CHECKPOINT=1 CHECKPOINT_HOURS=0.5 bash run_spectrum.sh results-checkpoint
# If interrupted, repeat exactly the same command to resume.
```

Use `-dft_nconsec 3` (already set by the wrapper): the stock single-check mode
cannot preserve this helper's convergence history, so `-checkpoint` with
`-dft_nconsec 1` is rejected at startup. Periodic dumps occur every
0.5 wall-clock hours in this example, plus a final dump before analysis.
Separate `reference_checkpoints/` and `sample_checkpoints/` contain structure,
fields, convergence history, and a completion marker. Keep each directory
intact, along with reference flux files, and retain identical parameters and
MPI layout when resuming. A partial `.writing` directory is not a checkpoint;
`.old` can recover an interrupted rename. Resubmission is manual; no Slurm
requeue or last-second signal dump is installed. Recent progress since the
last complete checkpoint may be lost. Completed checkpoints are retained.

## Princeton Della

Create `film-meep` on Della using the environment file, then submit from this
package directory on a shared filesystem:

```bash
sbatch della.slurm
# Only after installing and verifying the checkpoint patch:
sbatch --export=ALL,CHECKPOINT=1,RUN_DIR="$PWD/results-checkpoint" della.slurm
```

The script requests one AMD CPU node, eight MPI ranks, 4 GB per rank, and four
hours; no GPU or explicit QOS is needed. Adjust memory/time after a small run.
It uses the Conda environment's matching MPICH launcher. Do not load a different
MPI module on top of it. For optimized multi-node production, ask Princeton
Research Computing to build Meep against the site's MPI and adapt the launcher.
Check the available Anaconda module if the pinned module name is unavailable.
This job template has not been executed on Della.

Setup references (consulted September 16, 2026):
- [Della](https://researchcomputing.princeton.edu/systems/della)
- [Princeton Python guide](https://researchcomputing.princeton.edu/support/knowledge-base/python)
- [Princeton MPI guidance](https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py)
- [Meep installation](https://meep.readthedocs.io/en/latest/Installation/)

## Validation

Earlier local checks with Meep 1.31.0 (patched libmeep): shell/Python syntax, a
serial dielectric run, a two-rank silver run, and two-rank checkpoint
write/reload. Both reference and sample resumed; the maximum change in the
saved spectrum table after resuming was 3e-7.

Della compute nodes, Meep 1.31.0 from `environment.yml` with stock libmeep:
the wrapper with Ag (1 and 2 ranks) and with `EPS=4`, a `-JouleHeating` Ag run,
a packing-file run, and a dielectric checkpoint write/resume, each compared with
the preceding version of the code at `RES=80`. Transmittance, reflectance,
incident flux, field arrays, and flux HDF5 datasets were identical; the
absorption column changed by 2.3e-6 relative when ω stopped being rounded to four
decimals, and the sum-rule error was 5.0e-3. After the file renames, the same
runs plus square and diamond particles gave numerically identical outputs; only
the flux-file prefix and the header text changed. These are workflow checks, not a resolution-convergence study;
`della.slurm` itself was not submitted.

Voxel input (Della, Meep 1.31.0, stock libmeep; job files in `tests/slurm/`):
all 660 files in the target data set load and match an independent loader voxel
for voxel; the epsilon grid reproduces the voxel layout for odd and even sizes;
a uniform slab and a 7-layer voxel stack match the analytic results to 0.005 at
`-res 100` and 0.0012/0.0005 at `-res 200`; a voxelized `one_disk.txt` matches the
disk path to 0.003, 0.0008, and 0.00013 in T at 20, 10, and 5 nm voxels. With no
voxel input or new flag, every output of the existing examples is identical to
the previous version. MPI rank count, x/y polarization of symmetric cells, and
2x2 lateral tiling give identical spectra.
