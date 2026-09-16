#!/usr/bin/env bash
set -euo pipefail
package_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
run_dir=${1:-"$PWD/results"}
# Optional voxel example, enabled only by INPUT=<.npz/.tif/.tiff> VOXEL_SIZE=<micrometers>
# (optional CROP=z0:z1,y0:y1,x0:x1 in stored axis order, NORMAL_AXIS). Resolved before cd.
voxel_input=
if [[ -n ${INPUT:-} ]]; then
    : "${VOXEL_SIZE:?INPUT needs VOXEL_SIZE (voxel edge length in micrometers)}"
    voxel_input=$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")
    [[ -f $voxel_input ]] || { echo "INPUT $INPUT not found" >&2; exit 1; }
fi
mkdir -p "$run_dir"
cd "$run_dir"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
ranks=${NP:-1}
launcher=()
if (( ranks > 1 )); then
    launcher=(mpirun -np "$ranks")
fi
# Check the actual MPI size before allowing multiple ranks to write results.
"${launcher[@]}" python -c "import meep as mp; assert mp.count_processors() == $ranks, 'MPI rank count mismatch'"
checkpoint=()
if [[ ${CHECKPOINT:-0} == 1 ]]; then
    bash "$package_dir/patches/install_patched_libmeep.sh" --check
    checkpoint=(-checkpoint -checkpoint_interval "${CHECKPOINT_HOURS:-0.5}")
fi
# Both stages MUST share these settings and the same MPI rank count.
common=(-load "$package_dir/examples/one_disk.txt" -is_point -phi 0.2
        -eps "${EPS:-Ag}" -eps_ref 1 -tfilm 0.1 -polarization x
        -res "${RES:-80}" -ks 5.235987756 7.853981634 -nfreqs 31
        -dsrc 0.4 -ddet 0.2 -dpml 0.3 -tpml 0.5
        -ScattPower -comp Ex -dft_margin_px 4
        -dft_nconsec 3 -dft_tol 1e-8 -maxt "${MAXT:-200}"
        -tempname incident)
if [[ -n $voxel_input ]]; then
    # Solid n = 1.55 in vacuum, 380-780 nm; thickness and lateral cell come from the data.
    common=(-load "$voxel_input" -voxel_size "$VOXEL_SIZE"
            -eps "${EPS:-2.4025}" -eps_ref 1 -polarization x
            -res "${RES:-100}" -ks 8.0553657784 16.5346981768 -nfreqs 101
            -dsrc 0.4 -ddet 0.2 -dpml 0.3 -tpml 0.5
            -ScattPower -comp Ex -dft_margin_px 4
            -dft_nconsec 3 -dft_tol 1e-8 -maxt "${MAXT:-200}"
            -tempname incident)
    if [[ -n ${CROP:-} ]]; then common+=(-crop "$CROP"); fi
    if [[ -n ${NORMAL_AXIS:-} ]]; then common+=(-normal_axis "$NORMAL_AXIS"); fi
fi
"${launcher[@]}" python "$package_dir/film_transmittance.py" \
    "${common[@]}" "${checkpoint[@]}" -ref -saveas reference > reference.log 2>&1
"${launcher[@]}" python "$package_dir/film_transmittance.py" \
    "${common[@]}" "${checkpoint[@]}" -saveas sample > sample.log 2>&1
printf 'Spectrum: %s/sample_trans-x.txt\n' "$PWD"
