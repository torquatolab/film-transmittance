#!/usr/bin/env bash
#
# Build meep from source with the dispersive-checkpoint patch applied and
# install the resulting libmeep over a conda/micromamba env's copy.
#
# Only libmeep is replaced -- the SWIG bindings (_meep*.so) are untouched, so
# the env's pymeep keeps working.  The original library is copied aside first,
# and a failed verification rolls the install back automatically.
#
#   ./install_patched_libmeep.sh                      # build, install, verify
#   ./install_patched_libmeep.sh --prefix /path/env   # a specific env
#   ./install_patched_libmeep.sh --tarball meep-1.33.0.tar.gz   # no network
#   ./install_patched_libmeep.sh --no-install         # build + stage only
#   ./install_patched_libmeep.sh --restore            # undo, back to the stock library
#   ./install_patched_libmeep.sh --verify-only        # full test of the current env
#   ./install_patched_libmeep.sh --check              # fast: is the patch installed?
#
# Use --check as a preflight in any job that RESUMES a dispersive checkpoint.  A stock
# libmeep loads such a checkpoint without complaining and then zero-initialises the
# polarization state, so the run continues with wrong physics and says nothing.
#
# On an HPC login node run it detached, so a dropped ssh master cannot SIGHUP
# the build:
#
#   setsid bash install_patched_libmeep.sh >build.log 2>&1 </dev/null &
#
set -euo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH=$SELF_DIR/meep-checkpoint-dispersive.patch
TESTPY=$SELF_DIR/test_checkpoint_dispersive.py
PREFIX=${CONDA_PREFIX:-}
TARBALL=""
BUILD_DIR=""
JOBS=8
CC_OVERRIDE=""
CXX_OVERRIDE=""
DO_INSTALL=1
MODE=install

die() { echo "ERROR: $*" >&2; exit 1; }
say() { echo "==> $*"; }

while [[ $# -gt 0 ]]; do
  case $1 in
    --prefix)    PREFIX=$2; shift 2 ;;
    --patch)     PATCH=$2; shift 2 ;;
    --tarball)   TARBALL=$2; shift 2 ;;
    --build-dir) BUILD_DIR=$2; shift 2 ;;
    --jobs|-j)   JOBS=$2; shift 2 ;;
    --cc)        CC_OVERRIDE=$2; shift 2 ;;
    --cxx)       CXX_OVERRIDE=$2; shift 2 ;;
    --no-install) DO_INSTALL=0; shift ;;
    --restore)   MODE=restore; shift ;;
    --verify-only) MODE=verify; shift ;;
    --check)     MODE=check; shift ;;
    -h|--help)   sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *)           die "unknown option: $1" ;;
  esac
done

# --------------------------------------------------------------- environment
[[ -n $PREFIX ]] || die "no env prefix: pass --prefix or activate the env"
[[ -x $PREFIX/bin/python ]] || die "$PREFIX/bin/python not found"
PY=$PREFIX/bin/python

# meep prints an MPI banner on import and a timing line at exit, so fish the
# version out with a marker rather than trusting stdout to be clean.
MEEP_VER=$("$PY" -c 'import meep; print("MEEPVER<"+meep.__version__+">")' 2>&1 \
           | sed -n 's/.*MEEPVER<\(.*\)>.*/\1/p' | head -1)
[[ -n $MEEP_VER ]] || die "cannot read meep.__version__ from $PREFIX"

# The real library file, e.g. libmeep.so.37.0.0; libmeep.so and libmeep.so.37
# are symlinks to it and are left alone.
# Match exactly libmeep.so.N.N.N: the glob alone also matches this script's own
# .orig-*/.prev/.lock files, which would be picked if a stale set sorts first.
LIB=$(ls "$PREFIX"/lib/libmeep.so.*.*.* 2>/dev/null | grep -E '/libmeep\.so\.[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || true)
[[ -n $LIB && -f $LIB ]] || die "no libmeep.so.N.N.N under $PREFIX/lib"
SONAME=$(basename "$LIB")

say "env      : $PREFIX"
say "meep     : $MEEP_VER"
say "library  : $SONAME"

# The pristine backup for THIS meep version, chosen exactly -- not the newest by
# mtime, which could belong to a different version left over in the same directory.
pristine_backup() { echo "$LIB.orig-$MEEP_VER"; }

# Provenance for the backups.  Both --restore and the failed-install rollback copy a
# file back over the live library, so they must be able to tell it is still the file
# that was written there.  A corrupted backup would otherwise be installed silently:
# the post-restore verify() cannot catch it, because a stock library is *expected* to
# fail drude/ag.  Bare sha256, stored beside the backup.
# Publish a backup and its checksum so the backup NEVER exists without a record: both
# are staged as temp files, the record is moved into place first, then the backup.  A
# kill in between leaves the new record beside the old backup, which reads as a
# mismatch and is refused -- wrong in the safe direction.
publish_backup() {
  local src=${1:?} dst=${2:?}
  cp -p "$src" "$dst.tmp.$$"
  sha256sum <"$dst.tmp.$$" | cut -d' ' -f1 >"$dst.sha256.tmp.$$"
  mv -f "$dst.sha256.tmp.$$" "$dst.sha256"
  mv -f "$dst.tmp.$$" "$dst"
}

# 0 = matches, 1 = MISMATCH, 2 = no record (predates this check), 3 = cannot verify.
# Both digests are range-checked: an unreadable file or empty record would otherwise
# leave both substitutions empty, and `[[ "" == "" ]]` is a PASS -- commands inside
# `[[ ]]` do not trip `set -e`, so that failure would look like a match.
check_checksum() {
  local f=${1:?} want have
  [[ -f $f.sha256 ]] || return 2
  want=$(cat "$f.sha256") || return 3
  have=$(sha256sum <"$f" | cut -d' ' -f1) || return 3
  [[ $want =~ ^[0-9a-f]{64}$ ]] || return 3
  [[ $have =~ ^[0-9a-f]{64}$ ]] || return 3
  [[ $have == "$want" ]]
}

# Refuse to install a backup that has changed since it was written.  A missing record
# only warns: publish_backup makes that impossible for backups this script wrote, so
# it can only mean a backup predating the check -- refusing would strand those.
assert_restorable() {
  local f=${1:?} rc=0
  [[ -f $f ]] || die "missing backup $(basename "$f")"
  check_checksum "$f" || rc=$?
  case $rc in
    0) ;;
    2) echo "WARNING: no .sha256 beside $(basename "$f") -- cannot verify it is intact" >&2 ;;
    3) die "cannot read or parse the checksum record for $(basename "$f"); refusing to
     install a backup that cannot be verified." ;;
    *) die "$(basename "$f") does not match its recorded sha256; refusing to install a
     backup that changed since it was written. Inspect it, or reinstall from source." ;;
  esac
}

# Serialise against another run on the same env.  install and restore both mutate the
# live library and .prev; interleaving two runs can leave .prev checksummed against a
# library the *other* run installed, i.e. a valid record for the wrong content.
acquire_env_lock() {
  if ! command -v flock >/dev/null 2>&1; then
    echo "WARNING: no flock -- concurrent runs on $PREFIX are unprotected" >&2
    return 0
  fi
  exec 9>"$LIB.lock" || return 0
  flock -w 300 9 || die "another install/restore is holding the lock on $PREFIX"
}

# Path of the libmeep the interpreter ACTUALLY loads -- not merely the one sitting in
# $PREFIX/lib.  LD_PRELOAD or LD_LIBRARY_PATH can make those differ, and it is the
# loaded one that decides whether a checkpoint resumes correctly.
# Prints EVERY distinct libmeep image mapped into the interpreter, one per line.
# More than one appears when something is LD_PRELOADed alongside the env's own copy;
# all of them must be patched for the answer to be safe.
loaded_libmeep() {
  "$PY" -c "import meep; print(''.join(sorted({'LOADEDLIB<'+l.split()[-1]+'>\n' for l in open('/proc/self/maps') if 'libmeep.so' in l})), end='')" 2>&1 |
    sed -n 's/.*LOADEDLIB<\(.\+\)>.*/\1/p'
}

# Does this .so export the patch's symbol?  Fail CLOSED (status 2) when no symbol
# reader is available: a byte search of the file would accept debug or string-table
# residue, which is worse than admitting we cannot tell.
# Capture the symbol table first and match it in memory.  Piping into `grep -q` makes
# grep exit on the first hit, which SIGPIPEs the reader; under `set -o pipefail` that
# surfaces as status 141 and a spurious "not patched" -- a race, so it fails only
# sometimes, which is the worst way for a preflight to be wrong.
has_patch_symbol() {
  local lib=${1:?} syms
  if command -v nm >/dev/null 2>&1; then
    syms=$(nm -D --defined-only "$lib" 2>/dev/null) || true
  elif command -v readelf >/dev/null 2>&1; then
    syms=$(readelf --dyn-syms -W "$lib" 2>/dev/null) || true
  elif command -v objdump >/dev/null 2>&1; then
    syms=$(objdump -T "$lib" 2>/dev/null) || true
  else
    echo "no nm/readelf/objdump: cannot verify $lib" >&2
    return 2
  fi
  [[ -n $syms ]] || { echo "empty symbol table for $lib" >&2; return 2; }
  grep -q 'dump_polarizations' <<<"$syms"
}

# Cheap enough to call from a job script before resuming a dispersive checkpoint -- a
# STOCK library loads such a checkpoint without complaint and then zero-initialises the
# polarization state, i.e. silently wrong physics.  fields::dump_polarizations only
# exists in the patched build.  Sets LOADED_LIB for the caller's message.
LOADED_LIB=""
lib_is_patched() {
  local libs lib
  libs=$(loaded_libmeep) || true
  LOADED_LIB=$(echo "$libs" | tr '\n' ' ' | sed 's/ $//')
  if [[ -z $libs ]]; then
    echo "cannot determine which libmeep python loads" >&2
    return 2
  fi
  while read -r lib; do
    [[ -n $lib ]] || continue
    has_patch_symbol "$lib" || return $?
  done <<<"$libs"
  return 0
}

verify() {
  local tmp
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' RETURN
  local kind rc=0

  if lib_is_patched; then
    echo "    symbols: PASS ($LOADED_LIB exports fields::dump_polarizations)"
  else
    echo "    symbols: FAIL -- python loads ${LOADED_LIB:-<unresolved>}, not a patched build"
    return 1
  fi

  for kind in diel drude ag; do
    # Require BOTH a zero exit status and the test's explicit marker.  Exit status
    # alone is not enough: under `python -O` an assert-based test would exit 0 while
    # reporting failure, so a bad library could pass this gate.
    if "$PY" "$TESTPY" "$tmp/ck_$kind" "$kind" >"$tmp/$kind.log" 2>&1 &&
       grep -q "RESULT: PASS" "$tmp/$kind.log" &&
       ! grep -q "RESULT: FAIL" "$tmp/$kind.log"; then
      echo "    $kind: PASS  $(grep -o 'rel=[0-9.e+-]*' "$tmp/$kind.log" | head -1)"
    else
      echo "    $kind: FAIL"
      sed 's/^/      /' "$tmp/$kind.log" | tail -5
      rc=1
    fi
  done
  return $rc
}

# ------------------------------------------------------------------- restore
if [[ $MODE == restore ]]; then
  acquire_env_lock
  BK=$(pristine_backup)
  [[ -f $BK ]] || die "no pristine backup $(basename "$BK") for meep $MEEP_VER"
  assert_restorable "$BK"
  say "restoring $(basename "$BK") -> $SONAME"
  cp "$BK" "$LIB.new.$$" && mv -f "$LIB.new.$$" "$LIB"
  say "restored. verifying (drude/ag are expected to FAIL on a stock library)"
  verify || true
  exit 0
fi

if [[ $MODE == verify ]]; then
  say "verifying current library"
  verify
  exit $?
fi

# Fast preflight for a job script: no meep run, just "is the patch in there?".
if [[ $MODE == check ]]; then
  if lib_is_patched; then
    say "patched ($LOADED_LIB exports fields::dump_polarizations)"
    exit 0
  fi
  echo "python in $PREFIX loads ${LOADED_LIB:-<unresolved>}, which is NOT patched --" >&2
  echo "dispersive checkpoints would resume with P=0 and no warning" >&2
  exit 1
fi

# --------------------------------------------------------------- get sources
[[ -f $PATCH ]] || die "patch not found: $PATCH"
[[ -f $TESTPY ]] || die "test not found: $TESTPY"

BUILD_DIR=${BUILD_DIR:-${TMPDIR:-/tmp}/meep-patchbuild-$MEEP_VER-$USER}
mkdir -p "$BUILD_DIR"
SRC=$BUILD_DIR/meep-$MEEP_VER

if [[ ! -d $SRC ]]; then
  if [[ -z $TARBALL ]]; then
    TARBALL=$BUILD_DIR/meep-$MEEP_VER.tar.gz
    if [[ ! -f $TARBALL ]]; then
      say "downloading meep $MEEP_VER"
      curl -fsSL --max-time 300 -o "$TARBALL" \
        "https://codeload.github.com/NanoComp/meep/tar.gz/refs/tags/v$MEEP_VER" \
        || die "download failed -- stage the tarball yourself and pass --tarball"
    fi
  fi
  say "extracting $(basename "$TARBALL")"
  tar xzf "$TARBALL" -C "$BUILD_DIR"
  [[ -d $SRC ]] || die "expected $SRC after extracting $TARBALL"
fi

# ----------------------------------------------------------------- patch it
cd "$SRC"
# Record WHICH patch was applied, not merely that one was: a reused build dir must
# not silently ship a different patch than the one named on the command line.
PATCH_SHA=$(sha256sum "$PATCH" | cut -d' ' -f1)
if [[ ! -f .patch-applied ]]; then
  say "applying $(basename "$PATCH")"
  patch -p1 --dry-run <"$PATCH" >/dev/null || die "patch does not apply to meep $MEEP_VER"
  patch -p1 <"$PATCH"
  printf '%s  %s\n' "$PATCH_SHA" "$PATCH" > .patch-applied
elif [[ $(cut -d' ' -f1 <.patch-applied) == "$PATCH_SHA" ]]; then
  say "patch already applied in $SRC"
else
  die "$SRC was built from a different patch ($(cut -c1-12 <.patch-applied)...);
     use a fresh --build-dir, or delete $SRC"
fi

# ------------------------------------------------------------------- build
MPICC=$PREFIX/bin/mpicc
MPICXX=$PREFIX/bin/mpicxx
[[ -x $MPICC && -x $MPICXX ]] || die "no mpicc/mpicxx in $PREFIX/bin"

# conda's mpicc is a wrapper around a backend compiler that is often not
# installed; MPICH_CC/CXX redirect it to whatever this machine actually has.
# "mpicc -show" only prints the command line, so probe with a real compile.
mpicc_works() {
  local d
  d=$(mktemp -d)
  echo 'int main(void){return 0;}' >"$d/t.c"
  local ok=1
  "$MPICC" -o "$d/t" "$d/t.c" >/dev/null 2>&1 && ok=0
  rm -rf "$d"
  return $ok
}

if [[ -n $CC_OVERRIDE ]]; then
  export MPICH_CC=$CC_OVERRIDE MPICH_CXX=${CXX_OVERRIDE:-g++}
  say "using MPICH_CC=$MPICH_CC MPICH_CXX=$MPICH_CXX"
elif ! mpicc_works; then
  export MPICH_CC=gcc MPICH_CXX=g++
  say "mpicc backend missing; falling back to MPICH_CC=gcc MPICH_CXX=g++"
  mpicc_works || die "mpicc still cannot compile; pass --cc/--cxx (module load gcc?)"
fi

if [[ ! -f configure ]]; then
  say "autoreconf"
  autoreconf --install --force >"$BUILD_DIR/autoreconf.log" 2>&1 \
    || { tail -20 "$BUILD_DIR/autoreconf.log"; die "autoreconf failed"; }
fi

# Reconfigure when the inputs change; an existing src/Makefile alone would otherwise
# pin the build to a stale prefix or compiler.
CONF_STAMP="$PREFIX|$MPICC|$MPICXX|${MPICH_CC:-}|${MPICH_CXX:-}"
if [[ ! -f src/Makefile || $(cat .configured-with 2>/dev/null) != "$CONF_STAMP" ]]; then
  # make does not rebuild objects merely because the compile commands changed, so a
  # reconfigure of an existing tree must discard them or stale objects get shipped.
  if [[ -f src/Makefile ]]; then
    say "configuration changed -- cleaning previous objects"
    make -C src clean >/dev/null 2>&1 || true
  fi
  say "configure"
  ./configure --prefix="$SRC/inst" --without-python --without-scheme --with-mpi \
      --enable-shared --disable-static --with-libctl="$PREFIX/share/libctl" \
      CC="$MPICC" CXX="$MPICXX" \
      CPPFLAGS="-I$PREFIX/include" \
      LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib" \
      >"$BUILD_DIR/configure.log" 2>&1 \
    || { tail -30 "$BUILD_DIR/configure.log"; die "configure failed"; }
  printf '%s' "$CONF_STAMP" > .configured-with
fi

say "make -j$JOBS (this is the long step)"
make -C src -j"$JOBS" >"$BUILD_DIR/make.log" 2>&1 \
  || { grep -iE " error" "$BUILD_DIR/make.log" | head -20; die "build failed, see $BUILD_DIR/make.log"; }

BUILT=$SRC/src/.libs/$SONAME
[[ -f $BUILT ]] || die "built library is not $SONAME -- source version does not match the env"
say "built $BUILT"

if [[ $DO_INSTALL -eq 0 ]]; then
  say "--no-install: stopping. Try it without touching the env:"
  echo "    LD_PRELOAD=$BUILT $PY $TESTPY /tmp/ck ag"
  exit 0
fi

# ------------------------------------------------------------------ install
acquire_env_lock
# Two distinct backups, because they answer different questions:
#   .orig-<ver>  the pristine stock library, written once and never overwritten
#   .prev        whatever was installed a moment ago -- the correct rollback target
# Rolling back to .orig-<ver> would silently discard an already-working patched
# library when a *second* install fails verification.
BK=$LIB.orig-$MEEP_VER
if [[ -e $BK ]]; then
  say "keeping existing pristine backup $(basename "$BK")"
else
  say "backing up -> $(basename "$BK")"
  publish_backup "$LIB" "$BK"
fi
PREV=$LIB.prev
publish_backup "$LIB" "$PREV"

# $LIB is usually hardlinked into the conda pkgs cache, so write a new file and
# rename over it rather than editing in place -- otherwise the cache is corrupted.
say "installing"
cp "$BUILT" "$LIB.new.$$"
chmod 755 "$LIB.new.$$"
mv -f "$LIB.new.$$" "$LIB"

# ------------------------------------------------------------------- verify
say "verifying"
if verify; then
  say "OK -- patched libmeep installed in $PREFIX"
  say "to undo:  $0 --prefix $PREFIX --restore"
else
  say "verification FAILED -- rolling back to what was installed before this run"
  assert_restorable "$PREV"
  cp "$PREV" "$LIB.new.$$" && mv -f "$LIB.new.$$" "$LIB"
  die "rolled back from $(basename "$PREV"); stock copy is still $(basename "$BK"); build log: $BUILD_DIR/make.log"
fi
