#!/usr/bin/env bash
# Spectrum of a non-periodic voxel stack averaged over N lateral crops.
# usage: VOXEL_SIZE=<um> MAXT=<ceiling> [other run_spectrum.sh env] bash run_crops.sh STACK N OUTDIR
# Crops come from voxel_crop.select_crops (normal axis 0, MIN_FRACTION default 0.75,
# MIN_OFFSET_SEP default 25% of the smaller lateral extent). Each
# crop runs run_spectrum.sh into OUTDIR/crop_<i>, sequentially with the same environment;
# then average_spectra.py writes OUTDIR/mean_trans-x.txt (BANDS=K adds mean_trans-x_bands.txt).
# DRY=1 prints the crops and the commands without running them.
set -euo pipefail
package_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if (( $# != 3 )); then
    echo "usage: VOXEL_SIZE=<um> MAXT=<ceiling> bash run_crops.sh STACK N OUTDIR" >&2
    exit 2
fi
stack=$1 n=$2 outdir=$3
: "${VOXEL_SIZE:?run_crops.sh needs VOXEL_SIZE (voxel edge length in micrometers)}"
: "${MAXT:?run_crops.sh needs MAXT (simulation-time ceiling): use the value for the structure family from README.md, Voxel input}"
[[ -f $stack ]] || { echo "STACK $stack not found" >&2; exit 1; }
[[ -z ${CROP:-} ]] || { echo "CROP is set by run_crops.sh; unset it" >&2; exit 1; }
[[ ${NORMAL_AXIS:-0} == 0 ]] || { echo "run_crops.sh supports normal axis 0 only" >&2; exit 1; }
stack=$(cd "$(dirname "$stack")" && pwd)/$(basename "$stack")
dry=${DRY:-0}
chooser=(python "$package_dir/voxel_crop.py" "$stack" --n "$n" --min_fraction "${MIN_FRACTION:-0.75}")
if [[ -n ${MIN_OFFSET_SEP:-} ]]; then chooser+=(--min_offset_sep "$MIN_OFFSET_SEP"); fi
if [[ $dry != 1 ]]; then
    mkdir -p "$outdir"
    chooser+=(--json "$outdir/crops.json")
fi
crop_list=$("${chooser[@]}")
mapfile -t crops <<< "$crop_list"
(( ${#crops[@]} > 0 )) && [[ -n ${crops[0]} ]] || { echo "no crops selected" >&2; exit 1; }
tables=()
for i in "${!crops[@]}"; do
    printf 'crop_%d: %s\n' "$i" "${crops[$i]}"
    cmd=(env INPUT="$stack" CROP="${crops[$i]}" bash "$package_dir/run_spectrum.sh" "$outdir/crop_$i")
    tables+=("$outdir/crop_$i/sample_trans-x.txt")
    if [[ $dry == 1 ]]; then
        printf '%q ' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi
done
average=(python "$package_dir/average_spectra.py" "$outdir/mean_trans-x.txt" "${tables[@]}")
if [[ -n ${BANDS:-} ]]; then average+=(--bands "$BANDS"); fi
if [[ $dry == 1 ]]; then
    printf '%q ' "${average[@]}"; echo
else
    "${average[@]}"
fi
