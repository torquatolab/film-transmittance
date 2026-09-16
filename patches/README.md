# meep checkpointing for dissipative (metallic) media

`meep-checkpoint-dispersive.patch` extends `Simulation.dump`/`load` so a
run containing **dispersive media** — materials carrying Lorentz/Drude
`E_susceptibilities` / `H_susceptibilities`, i.e. every metal — can be checkpointed and
resumed. Gyrotropic media are covered too. `multilevel_susceptibility` is **not**: it
still aborts, as it does in stock meep, because its *structure* does not round-trip
either (it has no `dump_params`).

Applies cleanly to **1.31.0** (local `pmp`) and **1.33.0** (Nurion `pmp`).
Stock meep refuses outright:

```
RuntimeError: meep: non-null polarization_state in fields::dump (unsupported)
```

## Why it was dielectric-only

Under the ADE scheme each Lorentz/Drude pole carries its own auxiliary
polarization field `P_n`, time-stepped in lockstep with `E` and `H` (see
`<vault>/20_Topics/10_Classical_Electrodynamics/99_FDTD/Dispersive_Media_in_FDTD.md`).
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
7. **A per-pole descriptor** (`pol_desc`): `{global chunk index, block length}` per
   pole, in the same order as the data, compared before any polarization data is read.
   It pins per-pole boundaries that a per-chunk total misses — dumping
   `[Lorentzian 2N, gyrotropic 6N]` and loading `[gyrotropic 6N, Lorentzian 2N]` has
   the same chunk total `8N` but a different descriptor, and is rejected.

If a simulation has no susceptibilities, nothing is written and the file is
byte-identical to what stock meep produces, so existing dielectric checkpoints
still load.

## Verification

`test_checkpoint_dispersive.py` runs a 2D metal-cylinder simulation straight
through, then repeats it with a mid-flight dump and a restart in a fresh
`Simulation`, and compares.

| material | stock 1.31.0 | patched |
|---|---|---|
| `diel` — `Medium(epsilon=12)` (control) | PASS | PASS |
| `drude` — single Drude pole | **abort** | PASS, rel err `0.00e+00` |
| `ag` — `meep.materials.Ag`, multi-pole | **abort** | PASS, rel err `0.00e+00` |

For deterministic Lorentz/Drude materials the restored **field trajectory** is
bit-exact, not merely close. That is not the same as the whole simulation being
bit-exact, and it excludes `noisy_lorentzian_susceptibility` entirely (see caveats): the accumulated DFT flux still differs
(see caveats), and the pass criterion is one `Ez` sample after `T2`, not a full-array
comparison. The test is non-vacuous — during development, when the dump/load pair
silently wrote and read nothing (P zero on restart), the `drude` case failed at
`rel = 1.262e+01` — but it demonstrates a gross-error tripwire at one point, not
component-by-component equivalence.

```bash
python test_checkpoint_dispersive.py /tmp/ck ag
```

### Caveats

* ~~MPI is unverified.~~ **Verified on Nurion** (2026-09-01, meep 1.33.0):
  `mpirun -np 2` and `-np 4` both restart bit-exact (`rel=0.00e+00`) for
  `meep.materials.Ag`. It could not be tested locally — `mpirun` does not run in
  the sandbox this was developed in.
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

  `2D_plasmonic_checkpoints2.py:312` calls `sim.load(checkpoint_dir)` in one
  shot; it needs splitting for the flux monitors to be restored.
* **Ag is stiff.** `Courant=0.3` at `resolution=40` was the stability floor for
  meep's Ag fit in the test; the default `0.5` diverges. Unrelated to
  checkpointing, but it will bite anyone running the test.

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

### Done so far

| env | meep | soname | status |
|---|---|---|---|
| `~/miniconda3/envs/pmp` (local) | 1.31.0 | `libmeep.so.35.0.0` | installed, verified |
| `/scratch/e1837a01/.conda/envs/pmp` (Nurion) | 1.33.0 | `libmeep.so.37.0.0` | installed, verified serial + MPI np=2,4 |

### On Nurion

Nurion's autotools are old, so ship a source tree that is already patched **and**
`autoreconf`'d rather than regenerating there:

```bash
# locally
tar xzf meep-1.33.0.tar.gz && cd meep-1.33.0
patch -p1 < meep-checkpoint-dispersive.patch
# the installer checks WHICH patch a reused build dir was built from
sha256sum ../meep-checkpoint-dispersive.patch > .patch-applied
autoreconf --install --force
cd .. && tar czf meep-1.33.0-patched.tar.gz meep-1.33.0

# stage through nurion-dm (file I/O never goes through `nurion`)
scp -O install_patched_libmeep.sh meep-checkpoint-dispersive.patch \
       test_checkpoint_dispersive.py meep-1.33.0-patched.tar.gz \
       nurion-dm:/scratch/e1837a01/meep-patch/

# build detached on `nurion`, so a dropped ssh master cannot SIGHUP it
ssh nurion 'cd /scratch/e1837a01/meep-patch && setsid bash install_patched_libmeep.sh \
  --prefix /scratch/e1837a01/.conda/envs/pmp \
  --tarball /scratch/e1837a01/meep-patch/meep-1.33.0-patched.tar.gz \
  --build-dir /scratch/e1837a01/meep-patch/build --jobs 8 \
  > install.log 2>&1 </dev/null &'
```

Notes for Nurion specifically:

* meep only needs **C++11**, so the default `gcc/7.2.0` is fine — and preferable
  to a newer module, since a lib built against an older libstdc++ ABI still runs
  against the env's newer one, not the reverse.
* conda's `mpicc` is a wrapper whose backend compiler is not installed. The
  script probes it with a real compile (`mpicc -show` succeeds regardless) and
  falls back to `MPICH_CC=gcc MPICH_CXX=g++`.
* The build survives the login node's 1200 s **per-process** CPU cap: `make -j`
  spawns many short compiler processes, none near the limit.
* **Scratch purge will eventually eat the env.** After any rebuild of `pmp`,
  rerun this script — a stock env silently reverts to aborting at the first
  checkpoint, hours into a job.

## Review

Reviewed 2026-09-01 by two independent Codex-family workers (critic `gpt-5.6-sol`,
verifier `gpt-5.6-terra`) under the `orchestration` policy, since the patch was
Claude-authored. Independently confirmed correct: the `internal_data_num` size
arithmetic for `num == 0` and `num == 1`; that **no P/P_prev pointer swap** exists
(`SWAP` only exchanges local anisotropic-direction variables — a swap would have
silently transposed the two halves on restore); that the forced allocation's use of
`fc->f` matches `fields_chunk::update_pols`; that the added collective ordering is
consistent under unchanged topology including the sharded path; that appending the
two virtuals leaves class size and existing vtable slots untouched, so the SWIG ABI
claim holds for the same toolchain; that HDF5 converts old float32 `sigma` to double
automatically; and that `multilevel_susceptibility` has no `dump_params` and so
correctly inherits the aborting base hook.

The review found **no defect in the C++ patch**. Everything it did find was in the
claims, the test's failure path, or the installer. One finding was **refuted** on
verification and deliberately not acted on: the identical-chunk-layout requirement is
a documented upstream constraint, not a regression introduced here.

### C++ audit, three rounds (2026-09-01)

A later three-round audit targeted the **C++ patch alone**, with the conductor running
the code between rounds (the workers are sandboxed read-only and can execute nothing).
It found real defects the earlier script-focused passes had missed:

* **Round 1** — `noisy_lorentzian` silently half-checkpointed (P restored, noise stream
  not); per-chunk totals established no per-pole boundaries. Both fixed. It also cleared
  the complex/Bloch, Mirror-symmetry, cylindrical, magnetic-dispersion and gyrotropic
  paths, and confirmed repeated dump/load does not reallocate.
* **Round 2** — caught a **regression I had just introduced**: keying the descriptor on
  `get_id()` was false assurance (a static counter cannot distinguish reordered poles)
  *and* could spuriously abort a legitimate fields-only load onto a rebuilt structure.
  The id was removed. It also found the noisy refusal fired *too late* — after `dump`
  had already truncated the target file, destroying any existing checkpoint there.
  Hence `supports_checkpoint()` and the preflight.
* **Round 3** — `ship-with-documented-limitations`; no remaining reachable defect for
  2D/3D Ag FDTD with PML, real fields, MPI and DFT monitors. One stale comment fixed.

Runtime evidence gathered between rounds: `diel`/`drude`/`ag` stay bit-exact
(`rel = 0.00e+00`) after every change, serial and at `mpirun -np 2` and `-np 4`; a noisy
sim now refuses with a pre-existing good checkpoint left byte-identical (313512 bytes
before and after, where the earlier build truncated it); and a fields-only load onto a
rebuilt structure with deliberately desynchronised ids is accepted.

### DFT audit, three rounds (2026-09-01)

A third three-round audit covered the DFT-restore fix. Each round found something real:

* **Round 1** — `sound-with-caveats`: the DFT data carried no per-monitor identity, so
  a monitor recreated differently could inherit another's samples. Confirmed
  `next_in_chunk` was the right list, and that the collective structure was unchanged.
* **Round 2** — `defective`, and it caught a genuine flaw in the descriptor I had just
  added: masking the hash at **every** mixing step meant only the low 52 bits of each
  input survived, which for an IEEE-754 double is the mantissa — so `ω` and `2ω`
  collided. It also flagged that the descriptor ignored `decimation_factor` and the
  weights, and that upstream's `num_f*`/`t`/`num_chi1inv`/`gv_nums` share the float32
  metadata bug. All fixed.
* **Round 3** — `do-not-ship`: `scale` and `avg1/avg2` were still unsigned (so the same
  monitor under a different `Courant` was accepted), `fold52` truncated on a 32-bit
  `size_t`, and `num_sus` was still float32. All three fixed.
* **Round 4**, a re-audit of those round-3 fixes — `sound-with-caveats`: they were
  correct, with the avalanche verified bit by bit. But narrowing to 31 bits had not
  been *necessary* (only the 32-bit `size_t` cast was at issue, not the HDF5 path), and
  it cost detection strength. The signature is now the full 64-bit hash stored as two
  32-bit words. It also noted that hashing a recomputed `scale` ties a checkpoint to
  its build; that is now documented rather than changed.

Two bugs surfaced only by instrumenting the running library, not by reading:
`ivec` holds five slots of which only the active directions are meaningful, and
`h5file` keeps a *current* dataset that `read_size` selects — reading the descriptor
between `read_size(data)` and the bulk reads silently redirected them.

Runtime evidence: identical monitor restores `rel = 0.00e+00`; different frequency
count, position, monitor count, frequencies×2, and `Courant` are each rejected;
`diel`/`drude`/`ag` pass with DFT flux now **asserted** at ≤1e-9, serial and at
`mpirun -np 2` and `-np 4`; monitor-free runs still restore bit-exact.

### Script audit rounds

A second critic pass over the *fixes* returned `fixes-defective` and was largely
right. Round two therefore changed:

* `--check` resolved the library by globbing `$PREFIX/lib` instead of asking what the
  interpreter actually maps — `LD_PRELOAD` / `LD_LIBRARY_PATH` diverge those. It now
  reads `/proc/self/maps` and requires every mapped `libmeep` image to carry the symbol.
* the `grep -a` byte-search fallback was not a fail-closed symbol test (it would accept
  debug or string-table residue). Now `nm` → `readelf` → `objdump`, and it reports
  "cannot verify" rather than guessing.
* **a SIGPIPE race in the symbol check.** `nm … | grep -q` lets grep exit on the first
  hit, which SIGPIPEs `nm`; under `set -o pipefail` that surfaces as status 141 and a
  spurious "not patched". It failed only *sometimes* — the worst way for a preflight to
  be wrong, and the earlier "symbols: PASS" runs on both machines were luck. The symbol
  table is now captured before matching (verified 8/8 locally, 5/5 on Nurion).
* `make` does not rebuild objects merely because the compile commands changed, so a
  reconfigure of an existing tree now runs `make clean` first.
* `--restore` chose the newest `.orig-*` by mtime; it now takes `.orig-$MEEP_VER` exactly.

### Known gaps — reviewed, not actioned

Recorded rather than fixed, so nobody has to rediscover them:

1. **`.prev` keeps a single generation.** Every install overwrites it. If the live
   library were already wrong, that wrong library becomes the new checksum-valid
   rollback target — the record proves it did not change, not that it was ever good.
   `.orig-<version>` is the escape hatch, and `--restore` targets it.
2. **`multilevel_susceptibility` remains unimplemented** — it aborts, as in stock meep.
   Its structure does not round-trip either, so implementing the field side alone
   would not help.
3. **The DFT-across-checkpoint gap is untouched** (see caveats). It is a third,
   pre-existing defect that also affects pure dielectrics, and it is the one thing
   here that still silently perturbs a published number — the ε_eff retrieval reads
   `add_flux` results across a restart.

## Upstreaming

Items 1–3 are a clean upstream contribution (they replace an explicit
"unsupported" abort). Item 4 is an independent bug fix worth its own PR. The
DFT-across-checkpoint gap in the caveats is a third, separate issue.
