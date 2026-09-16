# Meep checkpointing for dispersive (metallic) media

This directory supports the optional `-checkpoint` mode of `../film_transmittance.py`
(`CHECKPOINT=1` in `../run_spectrum.sh`). It is not needed for ordinary runs.

`meep-checkpoint-dispersive.patch` extends `Simulation.dump`/`load` so a
run containing **dispersive media** — materials carrying Lorentz/Drude
`E_susceptibilities` / `H_susceptibilities`, i.e. every metal — can be checkpointed and
resumed. Gyrotropic media are covered too. `multilevel_susceptibility` is **not**: it
still aborts, as it does in stock meep, because its *structure* does not round-trip
either (it has no `dump_params`).

Applies cleanly to Meep **1.31.0** (the version pinned in `../environment.yml`;
re-checked with `patch --dry-run` on 2026-09-16) and was also built against **1.33.0**.
Stock meep refuses outright:

```
RuntimeError: meep: non-null polarization_state in fields::dump (unsupported)
```

## Why it was dielectric-only

Under the ADE scheme each Lorentz/Drude pole carries its own auxiliary
polarization field `P_n`, time-stepped in lockstep with `E` and `H`.
`P_n` is *independent dynamical state*: it cannot be recomputed from `E`, `H` and
the material at a single instant. `fields::dump` only wrote `f`, `f_u`, `f_w`,
`f_cond`, `f_bfast`, `f_w_prev` and the DFT chunks, so upstream took the honest
route and aborted rather than silently restarting with `P = 0`.

Conductivity loss (`D_conductivity`) was never affected — `f_cond` was already
dumped, and a checkpoint of a lossy non-dispersive medium round-trips exactly on
stock meep. The gap was specifically the ADE state.

## What the patch does

1. **`susceptibility` gains two hooks** (`num_internal_data`, `internal_data_ptr`)
   exposing the contiguous `realnum` block that every internal-data layout ends
   with. Only that block is state — the `P`/`P_prev` pointers into it are rebuilt
   by `init_internal_data` when the fields are reconstructed from the dumped
   structure. Implemented for `lorentzian_susceptibility` (hence Drude too) and
   `gyrotropic_susceptibility`. **`noisy_lorentzian_susceptibility` inherits these
   hooks but is only partly checkpointable** — see the caveats. The base
   implementation aborts, so an unimplemented subclass fails loudly rather than
   checkpointing state it cannot see. `multilevel_susceptibility` is deliberately
   left out: it has no `dump_params`, so its *structure* does not round-trip
   either, and it still hits the same abort.
2. **`fields::dump_polarizations` / `load_polarizations`** write and read those
   blocks, mirroring the collective pattern of `dump_fields_chunk_field`
   (`sum_to_master` / `partial_sum_to_all` / `sum_to_all` / `broadcast`).
   Iteration order — chunk, then field type, then the `pol` linked list — is the
   order `fields_chunk` builds `pol[]` from `structure_chunk::chiP`, so it is
   reproduced exactly when the same structure is loaded back.
3. **Forces the lazy allocation.** `p->data` is allocated on the *first* call to
   `fields_chunk::update_pols`, so at `fields::load` time there is nothing to
   read into. Both routines now allocate it first (via the same
   `new_internal_data` + `init_internal_data` pair, which zeroes it — the correct
   `t = 0` state) and invalidate the chunk connections, as `fields::update_pols`
   does. Doing it symmetrically in dump and load keeps the two sides agreeing on
   what is in the file. **This was the whole bug on the second attempt**: without
   it the dump/load pair silently wrote and read nothing.
4. **`structure_dump.cpp`: per-pixel susceptibility `sigma` is now stored in
   double.** It was created with `create_data`'s default `single_precision=true`
   while `chi1inv` beside it is written in full precision. `realnum` is `double`
   in this build, so every dispersive material was being quantised to float32 on
   dump. This is why an Ag run reloaded from a checkpoint drifted ~1e-6 relative
   even with `P` correctly restored.

5. **`susceptibility::supports_checkpoint()`** — a const, allocation-independent
   capability query, defaulting to **false**. `fields::dump` and `fields::load` call
   `assert_checkpointable()` as their first statement, before the HDF5 file is opened
   or any field is overwritten. Checking there rather than at the first
   `num_internal_data()` matters: `dump` truncates the target and writes `t` and the
   field arrays before it reaches the polarization pass, so a late refusal would
   destroy a previously good checkpoint in that directory. The check is collective
   (`or_to_all`), so all ranks refuse together.
6. **`noisy_lorentzian_susceptibility` refuses.** It inherits the Lorentzian hooks, so
   without this it would checkpoint `P` while its MT19937 noise state — `static` in
   `src/support/mt19937ar.c` — stayed behind, resuming from a different noise stream
   while looking like a faithful restart. Refusing matches what stock meep does for
   every dispersive material. Supporting it properly means dumping each rank's 624 MT
   words plus `mti`; not done.
7. **A per-pole descriptor** (`pol_desc`): `{global chunk index, block length}` per
   pole, in the same order as the data, compared before any polarization data is read.
   It pins per-pole boundaries that a per-chunk total misses — dumping
   `[Lorentzian 2N, gyrotropic 6N]` and loading `[gyrotropic 6N, Lorentzian 2N]` has
   the same chunk total `8N` but a different descriptor, and is rejected.
8. **DFT monitors are now restored.** `fields::dump` always wrote the DFT
   accumulators, but `fields::load` brought back only part of them, so a resumed run's
   spectrum was wrong by ~0.3–3 %. Cause: a `dft_chunk` is threaded onto **two** lists
   — `next_in_dft` chains one DFT *object's* chunks across fields chunks, while
   `next_in_chunk` chains every DFT chunk of one *fields* chunk. `fields::dump`/`load`
   iterate fields chunks but called `save_dft_hdf5`/`load_dft_hdf5`, which walk
   `next_in_dft`, so everything after the head of each fields chunk's list was skipped.
   For an ordinary flux monitor that saved E but not H, and `flux = Re(E* × H)` came
   back identically zero while the file looked populated. `fields_dump.cpp` now has its
   own `save_dft_chunk_list`/`load_dft_chunk_list` walking `next_in_chunk`; `dft.cpp` is
   untouched, so `sim.save_flux`/`load_flux` keep their existing behaviour. This is a
   **pre-existing upstream bug** that also affects pure dielectrics.
9. **A per-DFT-monitor descriptor** (`<chunk>_dftdesc`): component, `N`, frequency count,
   and signatures over the frequency list and over everything that changes how
   `update_dft` accumulates — grid bounds, `s0/s1/e0/e1`, `shift`, `dV0/dV1`,
   decimation, the volume/interp weight flags, `stored_weight`, `extra_weight`,
   `scale`, `avg1/avg2`. Checked before any DFT data is read, so a monitor recreated
   with different frequencies, position, count, or even a different `Courant` is
   rejected instead of inheriting incompatible samples. A checkpoint with DFT data but
   no descriptor predates this fix and is refused explicitly.
10. **`size_t` metadata is written at full precision.** `h5file::create_data` defaults
   to a **float32** dataset while `write_chunk(size_t*)` writes through it, so any
   value above 2²⁴ is silently rounded. That is an upstream latent bug reachable by a
   3D chunk with `ntot > 16.7e6`: `num_f*`, `t`, `num_chi1inv`, `gv_nums` and `num_sus`
   are now created with `single_precision=false`, as are this patch's own `num_pol`,
   `pol_desc` and `_dftdesc`.

If a simulation has no susceptibilities, nothing is written and the file is
byte-identical to what stock meep produces, so existing dielectric checkpoints
still load.

## Verification

`test_checkpoint_dispersive.py` runs a 2D metal-cylinder simulation straight
through, then repeats it with a mid-flight dump and a restart in a fresh
`Simulation`, and compares.

| material | stock 1.31.0 | patched |
|---|---|---|
| `diel` — `Medium(epsilon=12)` (control) | FAIL: fields exact, DFT flux rel err `3.6e-03` | PASS |
| `drude` — single Drude pole | **abort** | PASS, rel err `0.00e+00` |
| `ag` — `meep.materials.Ag`, multi-pole | **abort** | PASS, rel err `0.00e+00` |

For deterministic Lorentz/Drude materials the restored **field trajectory** is
bit-exact, not merely close. The test passes only if one `Ez` sample after `T2`
agrees to ≤1e-9 relative **and** the accumulated DFT flux spectrum agrees to ≤1e-9
relative; it is a gross-error tripwire, not a component-by-component comparison of
every array. It excludes `noisy_lorentzian_susceptibility`, which the patch refuses
(see item 6). The test is non-vacuous — during development, when the dump/load pair
silently wrote and read nothing (P zero on restart), the `drude` case failed at
`rel = 1.262e+01`.

```bash
python test_checkpoint_dispersive.py /tmp/ck ag
```

### Caveats

* **MPI** was verified on another cluster (2026-09-01, Meep 1.33.0): `mpirun -np 2`
  and `-np 4` both restart bit-exact (`rel=0.00e+00`) for `meep.materials.Ag`.
* **DFT monitors must be recreated exactly as they were** — same components,
  frequencies, positions, count, weights and `Courant`. Anything else is rejected with
  a message naming the mismatching monitor. Checkpoints written before this fix carry
  incomplete DFT data and are refused; re-dump them. A run with no DFT monitors at all
  is unaffected, including old checkpoints of such runs.
* **The descriptor signatures are non-cryptographic.** Each is a full 64-bit hash
  stored as two 32-bit words (both exact in the binary64 dataset `h5file` writes
  `size_t` through, and both fitting a 32-bit `size_t`). They detect an accidental
  mismatch, not a chosen one.
* **A checkpoint is not guaranteed portable across builds.** The signatures hash exact
  double bit patterns, including `scale`, which is recomputed from the symmetry/Bloch
  phase and the timestep. That is deterministic for the same build and inputs — the
  supported case — but a different compiler or libm could differ in a final bit and be
  reported as a monitor mismatch.
* **Equal-size pole reordering is not detected** on a *fields-only* load. The
  descriptor carries no susceptibility identity: `get_id()` is a static counter, so two
  poles built in swapped order receive the same ids positionally (it would not catch
  the reordering) while an independently rebuilt structure gets *different* ids (it
  would abort a legitimate load). Real identity needs a content-derived fingerprint of
  the pole parameters, which the `const susceptibility` interface cannot supply today.
  Loading the structure from the same checkpoint — the supported flow — reconstructs
  the original pole order and makes this unreachable.
* **All MPI ranks must call `dump`/`load`**, including with
  `single_parallel_file=false`. The preflight is an `MPI_Allreduce`; a rank-subset call
  can deadlock. A complete sharded checkpoint needs every rank's shard anyway.
* **Identical chunk layout, process count and symmetry are required.** This is an
  *inherited* upstream constraint, not something this patch introduces —
  `src/fields_dump.cpp` has always carried "Only works if the number of
  processors/chunks is the same", and the field arrays are already checked the same
  way. The polarization data is checked per chunk by total size only, which is
  coarser than the field arrays' per-component shape metadata, so a topology change
  that happens to preserve per-chunk totals would misassign state rather than abort.
  Do not resume a checkpoint under a different process count.
* **`T1` must be an exact multiple of `dt = Courant / resolution`.** Otherwise
  the resumed run takes one extra timestep and nothing matches. This bites any
  checkpointed script, dispersive or not.
* **Restart ordering matters** (this is stock meep behaviour, not the patch).
  A monitor added before the load forces `init_sim` too early. Use meep's
  supported sequence:

  ```python
  sim.load(d, load_structure=True,  load_fields=False)
  flux = sim.add_flux(...)                    # monitors here
  sim.load(d, load_structure=False, load_fields=True)
  ```

  `../film_transmittance.py` follows this order; a one-shot
  `sim.load(checkpoint_dir)` would not restore the flux monitors.
* **Ag is stiff.** `Courant=0.3` at `resolution=40` was the stability floor for
  meep's Ag fit in the test; the default `0.5` diverges. Unrelated to
  checkpointing; the 3D film example shows the same effect at low resolution
  (see `../README.md`).

## Installing

`install_patched_libmeep.sh` does the whole thing: fetch the source matching the
env's meep version, patch, build, back the original library up, install, and
verify — rolling the install back automatically if verification fails. Only
`libmeep` is replaced; the SWIG bindings (`_meep*.so`) are untouched, because the
new virtuals are appended at the end of `susceptibility`'s vtable.

```bash
./install_patched_libmeep.sh --prefix /path/to/env      # build, install, verify
./install_patched_libmeep.sh --prefix ... --no-install  # build + LD_PRELOAD hint
./install_patched_libmeep.sh --prefix ... --verify-only # full test of the library
./install_patched_libmeep.sh --prefix ... --check       # fast: is the patch in there?
./install_patched_libmeep.sh --prefix ... --restore     # undo, back to stock
```

Two backups are kept beside the library, because they answer different questions:
`…orig-<version>` is the pristine stock library, written once and never overwritten;
`….prev` is whatever was installed immediately before this run, and is the rollback
target when verification fails. Rolling back to `orig-` would silently discard an
already-working patched library when a *second* install fails.

Each backup gets a `.sha256` beside it at the moment it is written, and both
`--restore` and the failed-install rollback check it before copying anything over the
live library. A **mismatch refuses**: installing a backup that changed since it was
written is exactly the failure the post-restore `verify` cannot catch, because a stock
library is *expected* to fail `drude`/`ag` there. A **missing** record only warns — it
means the backup predates this check, not that it is bad.

Backup and record are published together — both are staged as temp files, the record
is moved into place first, then the backup — so a backup this script wrote never exists
without one. A kill in between leaves the new record beside the old backup, which reads
as a mismatch and is refused: wrong in the safe direction. Install and restore also take
an `flock` on the env, because two concurrent runs could otherwise leave `.prev`
checksummed against a library the *other* run installed.

The two backups installed on 2026-09-01 were backfilled by validating each against the
copy in its own conda **pkgs cache** (a hash computed from a possibly-corrupt file
would prove nothing); both matched, so their recorded checksums carry real provenance
rather than self-attestation.

**What the checksum does not do.** It detects accidental change to a backup — bit rot,
a truncated copy, a half-finished write. It is not authentication: anyone who can
replace a backup can replace its `.sha256` too. It also says nothing about whether the
backed-up library was *correct* to begin with; that is what the pkgs-cache comparison
above establishes, and it is a manual step.

The install writes a new file and renames over the target rather than editing in
place — the library is usually **hardlinked into the conda pkgs cache**, and
editing it in place would corrupt the cache.

Verification requires **both** a zero exit status and the test's explicit
`RESULT: PASS` marker, and the test signals failure with `SystemExit` rather than
`assert`: under `python -O` / `PYTHONOPTIMIZE` an assert is stripped, so an
assert-based test would exit 0 while printing FAIL and a bad library would pass the
gate. `verify` also checks that the library actually exports
`fields::dump_polarizations` before running anything.

### Building on a cluster

The installer downloads the Meep source unless `--tarball` is given, so on a
cluster with offline compute nodes (such as Princeton Della) fetch the tarball
on a login node first, then build on an allocated compute node:

```bash
# login node
curl -fL -o meep-1.31.0.tar.gz \
  https://codeload.github.com/NanoComp/meep/tar.gz/refs/tags/v1.31.0
# compute node, env activated
bash patches/install_patched_libmeep.sh --prefix "$CONDA_PREFIX" \
  --tarball "$PWD/meep-1.31.0.tar.gz" --build-dir "$PWD/meep-build" --jobs 8
```

Where Autotools are too old to run `autoreconf`, apply the patch and run
`autoreconf --install --force` elsewhere, write
`sha256sum meep-checkpoint-dispersive.patch > .patch-applied` in the source tree
(the installer checks which patch a reused tree was built from), and pass the
repacked tree as `--tarball`. Notes:

* Meep needs only **C++11**; an older GCC is fine, since a library built against
  an older libstdc++ ABI still runs against the environment's newer one.
* Conda's `mpicc` wraps a backend compiler that may be missing. The script probes it
  with a real compile and falls back to `MPICH_CC=gcc MPICH_CXX=g++`.
* **Rebuilding or recreating the environment reinstalls stock libmeep.** Rerun the
  installer afterwards; `run_spectrum.sh` runs `--check` before any checkpointed job.

## Review history

The patch and installer were reviewed in several rounds on 2026-09-01, with the
code run between rounds. Findings that shaped the current version:

* **C++ patch:** `noisy_lorentzian_susceptibility` was only half checkpointed (fixed
  by refusing it); per-chunk totals did not pin per-pole boundaries (fixed by
  `pol_desc`); a late refusal truncated an existing checkpoint (fixed by the
  `supports_checkpoint()` preflight); keying the descriptor on `get_id()` was
  removed because the id cannot identify poles.
* **DFT restore:** the monitor descriptor initially ignored decimation, weights,
  `scale`, and `avg1/avg2`, and an early hash kept only 52 bits; the signature is now
  a full 64-bit hash over those fields.
* **Installer:** `--check` reads the libraries actually mapped by Python rather than
  globbing `$PREFIX/lib`; the symbol check uses `nm`/`readelf`/`objdump` and no longer
  races with SIGPIPE; a reconfigure runs `make clean`; `--restore` takes
  `.orig-$MEEP_VER` exactly.

Runtime evidence at the time: `diel`/`drude`/`ag` restored bit-exact serially and at
`mpirun -np 2` and `-np 4`, DFT flux agreed to ≤1e-9, and differently recreated
monitors (frequency count, position, monitor count, frequencies, `Courant`) were each
rejected.

### Known gaps

1. **`.prev` keeps a single generation.** Every install overwrites it. If the live
   library were already wrong, that wrong library becomes the new checksum-valid
   rollback target. `.orig-<version>` is the escape hatch, and `--restore` targets it.
2. **`multilevel_susceptibility` remains unimplemented** — it aborts, as in stock Meep.
3. **Equal-size pole reordering is not detected** on a fields-only load (see caveats);
   loading the structure from the same checkpoint, as the film script does, avoids it.

## Upstreaming

Items 1–7 replace an explicit "unsupported" abort and form one upstream
contribution. Items 8–10 fix pre-existing upstream bugs (DFT restore and float32
`size_t` metadata) that also affect pure dielectrics and are worth separate PRs.
