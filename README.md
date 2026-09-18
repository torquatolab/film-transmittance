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
| `voxel_crop.py` | Chooses lateral crops of a non-periodic stack with well-matching faces |
| `run_crops.sh` | Voxel input: runs `run_spectrum.sh` on several crops of one stack and averages them |
| `average_spectra.py` | Mean and crop-to-crop spread of several spectrum tables, plus band averages |
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
`MAXT=200` (the non-voxel example's default) is a **simulation-time ceiling**, not seconds of wall time; reaching
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
INPUT=path/structure.npz VOXEL_SIZE=0.01 MAXT=<family value> bash run_spectrum.sh results-voxel
INPUT=stack.tif VOXEL_SIZE=0.01 MAXT=<family value> CROP=0:258,0:57,16:134 bash run_spectrum.sh results-tif
```

The voxel example in `run_spectrum.sh` uses the production settings: `-res 100`,
`-tpml 1.5` (PML 1.17 micrometers thick), 201 frequencies over 380–780 nm
(`NFREQS` overrides), `-dsrc 0.4 -ddet 0.2 -dpml 0.3`, and `-dft_tol 1e-8
-dft_nconsec 3`. `MAXT` has no default for voxel input: these structures
generally do not meet the DFT stopping criterion and stop at the ceiling, so the
ceiling is a fixed value per structure family, taken from convergence runs.
The script's own argparse defaults, and the non-voxel example, are unchanged.

| Family (voxel size 10 nm, n = 1.55, res 100) | `MAXT` | Basis |
|---|---|---|
| SHU packings (`data/shu`, 256³) | 200 | settled by t = 150 (dense φ₂ = 0.45) and 50 (dilute φ₂ = 0.05) |
| Disordered benchmarks (GRF; DHU/DRM untested) | 200 | GRF settled by t = 125 |
| Periodic benchmarks (TPMS, channels) | 1125 | SquareChannel settled by t = 900; SchoenG never settles (see below) |
| TIFF stacks, crop-averaged spectrum | 750 | 4-crop DarkGreen average settled by t = 600; single crops do not |

"Settled" means that from that time to the end of a much longer run, every
20-band average of T and R changed by at most 0.01, the median per-wavelength
change of T was at most 0.002, and the band-averaged T + |R| − 1 was within
0.01; `MAXT` is that time rounded up to a multiple of 50 and multiplied by 1.25.
Individual wavelengths near diffraction thresholds of the lateral period
(λ = L/√(m² + n²)) can still move by 0.01–0.03 (up to 0.07 for SquareChannel near
689 nm, a slowly beating guided resonance). These values come from two SHU
files, three benchmarks, and one TIFF stack; check a new family with
`-snapshot_dt` before a campaign.
The voxel spectrum header records how the run ended: `# stopped: maxt ceiling`
or `# stopped: dft converged`.

`-snapshot_dt DT` (sample runs with `-ScattPower`; ignored for `-ref`; not with
`-JouleHeating`) writes the spectrum table every DT simulation time units to
`<saveas>_snapshots/trans-<pol>-t<time>.txt`, from the flux data accumulated so
far, with the same columns, format, and computation as the final table (the
snapshot header has no `stopped:` line). The final table is unchanged by the
option. Comparing snapshots shows when T and R stop changing, which is how a
family's `MAXT` is chosen.

The array is placed with `mp.MaterialGrid`, one sample per voxel. Use
`-res` equal to an integer multiple of `1/voxel_size`: the lateral block centre
and the film z position are shifted by at most half a pixel so voxel faces lie
midway between grid nodes (the log prints the shift). Keep
`-interface_averaging` off; it was less accurate in every test. The x/y
boundaries are periodic, so the array is treated as one period of an infinite
film. For stacks that are not periodic, `voxel_crop.best_crop()` finds the
lateral crop whose opposite faces match best, and `voxel_crop.select_crops()`
returns several low-seam crops whose lateral offsets differ by at least 25% of
the smaller lateral extent on one axis (they may overlap). `-no_fields` skips the volume
field arrays (spectra only); `-field_wavelengths` saves them only at listed
wavelengths. Reference and sample runs must use the same file, crop, voxel
size, and normal axis; the sample run refuses a reference that differs.

For a TIFF stack, report the spectrum averaged over several crops rather than
one crop:

```bash
VOXEL_SIZE=0.01 MAXT=<family value> BANDS=20 bash run_crops.sh stack.tif 5 results-crops
DRY=1 VOXEL_SIZE=0.01 MAXT=<family value> bash run_crops.sh stack.tif 5 results-crops  # print only
```

`run_crops.sh STACK N OUTDIR` selects up to N crops with
`python voxel_crop.py STACK --n N` (one `-crop` string per line; `--json` saves
the scores; `MIN_OFFSET_SEP` / `--min_offset_sep` lowers the required offset
difference when a small stack yields fewer than N crops), runs `run_spectrum.sh` for each into `OUTDIR/crop_<i>` one after
another with the same environment, then runs
`python average_spectra.py OUTDIR/mean_trans-x.txt OUTDIR/crop_*/sample_trans-x.txt`.
The mean table has, per wavelength, mean and standard deviation over crops of T
and of the signed reflectance, the mean of T + |R| − 1, and the number of
crops; its header lists the inputs and flags those that stopped at the `MAXT`
ceiling. `BANDS=K` (`--bands K`) adds `mean_trans-x_bands.txt`: K contiguous
frequency bands, each with the mean and spread over crops of the band-averaged
T and R, for single-number or colour use. The crops overlap, so their spread
indicates the sensitivity to the crop choice, not an independent error bar.

**Known limitations (validation of 2026-09-17 and 2026-09-18).**
For cells whose lateral period exceeds the wavelength, the former z padding
(`-tpml 0.5`) was not adequate: a periodic unit cell gave an unphysical
T = -0.052 near a diffraction cutoff, and a TIFF crop grew without bound after
t ~ 250. `-tpml 1.5`, now the voxel default, removed both. Runs of these
structures do not meet the DFT stopping criterion and stop at `MAXT`, so the
ceiling must be chosen and checked per structure family. Spectra of one cropped
non-periodic stack depend on the lateral boundary treatment by about 0.02 in T
on average (up to about 0.2 at sharp resonances), which is why TIFF spectra are
averaged over crops; 101 frequencies under-resolved their narrow features,
hence 201.

Longer is not always better. With `-tpml 1.5`, SchoenG (gyroid, 0.64 µm unit
cell) shows an error that grows exponentially after t ≈ 1150 just above the
640 nm (1,0) diffraction threshold: T at 641.6 nm is 0.95 at t = 1125, 0.68 at
1300, and −2.16 at 2000, and band-averaged T never settles (window spread about
0.02–0.03 over t = 600–1100). GRF shows a weak broadband version from
t ≈ 600, and a DarkGreen crop one at 592.8 nm. The cause is inferred to be PML
reflection of grazing diffracted orders; it is not fixed. Use the `MAXT`
values above rather than longer runs, and treat SchoenG spectra near 640–740 nm
as uncertain by about 0.03.

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

Production defaults, snapshots and crop averaging (Della, `tests/slurm/d1_*.slurm`):
with no voxel input and no `-snapshot_dt`, every output of the existing examples is
again identical to the previous version. On a 20x16x16 voxel film at 1, 2 and 4
MPI ranks, the final table with `-snapshot_dt` is byte-identical to the table
without it, snapshot files appear at every multiple of DT up to the stop time, and
the snapshot at the stop time equals the final table; the three rank counts give
identical tables. `select_crops` and `average_spectra.py` match independent
reference implementations on synthetic data. On `DarkGreen.tif` (lateral 75 x 145
voxels) the default offset separation (18 voxels) admits 4 crops;
`MIN_OFFSET_SEP=17` gives 5.

Convergence runs for `MAXT` (Della, 32–96 ranks, `-snapshot_dt 25`, all other
settings as the voxel example): SHU φ₂ = 0.45 and 0.05 to t = 1000, GRF to 1000,
SquareChannel and SchoenG to 2000 (one full 256³ cell of about 4×10⁷ grid cells
each), and four DarkGreen crops to 2000. None met the DFT stopping criterion. At
the recommended `MAXT` one 256³ structure costs about 55–76 core-hours (SHU, GRF;
1.7–2.4 h on 32 cores, about 40 GB) or about 370–450 core-hours (periodic
benchmarks); 32, 64, and 96 ranks give identical spectra, and 96 ranks cut wall
time about 2.8× at similar core-hours. Snapshots showed which settled times
the table above uses.
