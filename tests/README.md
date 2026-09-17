# Voxel-input validation

These scripts are the checks run on Princeton Della when voxel input was added.
They are validation harnesses, not a unit-test suite: several need the
`structural_color` data set, the Meep environment from `../environment.yml`,
and reference outputs from the previous version of the code. Paths in the job
files are the original authors' scratch directories; override them through the
environment variables each job file reads (`CODE`, `REPO`, `WT`, `OUT`, `WORK`,
`HARNESS`, `GOLDEN`) or edit them before use. Submit through Slurm; compute nodes
cannot see `/tmp`. Launch MPI with the environment's `mpirun`, not `srun python`.

| Script | Checks |
|---|---|
| `test_voxel_io.py` | Loading `.npz`/`.tif`, sidecar fraction checks, agreement with an independent loader, error handling |
| `test_voxel_geometry.py` | Axis orientation, crop, MaterialGrid placement and grid alignment (`sim.get_epsilon`) |
| `test_voxel_slab.py` | Uniform slab and layered stack vs analytic results; voxel grating vs `mp.Block` |
| `test_voxel_crop.py` | Seam score, exact best-crop search and `select_crops` vs brute force |
| `test_field_output.py` | `-no_fields`, `-field_wavelengths`, argument rejections, regression of existing outputs |
| `test_voxel_integration.py` | Voxel runs end to end, reference refusals, wrapper, `-normal_axis`, TIFF crops, regression comparison |
| `test_voxel_crosscheck.py` | Voxelized `examples/one_disk.txt` vs the disk path |
| `test_snapshots.py` | `-snapshot_dt` leaves the final table unchanged, snapshot times and contents, `stopped:` header line (1, 2, 4 ranks) |
| `test_average_spectra.py` | `average_spectra.py` means, spreads, bands, maxt-ceiling flags and rejections on synthetic tables |

Every script exits nonzero on failure.
