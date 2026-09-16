# Voxel-array reader for film_transmittance.py: loads a 3D two-phase
# microstructure from an .npz archive or a multi-page TIFF stack as a boolean
# array (True = solid) in stored axis order, with provenance for the run
# metadata. Semantics match structural_color/src/procfile.py::load_indicator.

import hashlib
import json
import os

import numpy as np

NPZ_DEFAULT_KEY = "g"
TIFF_SUFFIXES = (".tif", ".tiff")
SIDECAR_FRACTION_TOL = 1e-6
# Checked in this order: benchmark families (true phase = phase 1) carry
# phi1_realized; SHU and GRF files carry only phi2_realized.
SIDECAR_FRACTION_KEYS = ("phi1_realized", "phi2_realized")


def _sha256_file(path):
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for block in iter(lambda: f.read(1 << 20), b""):
			h.update(block)
	return h.hexdigest()


def _read_npz(path, key):
	try:
		with np.load(path) as archive:
			if key not in archive.files:
				raise ValueError(f"key {key!r} not in {path}; available keys: {archive.files}")
			return archive[key]
	except ValueError:
		raise
	except Exception as e:
		raise ValueError(f"cannot read .npz file {path}: {e}") from e


def _read_tiff_pillow(path):
	# Pillow returns palette indices (not the expanded colour map) for P-mode
	# frames, and 0/255 for L-mode frames; astype(bool) treats both alike.
	try:
		from PIL import Image, ImageSequence
		with Image.open(path) as im:
			frames = [np.asarray(frame) for frame in ImageSequence.Iterator(im)]
	except Exception as e:
		raise ValueError(f"cannot read TIFF file {path} with Pillow: {e}") from e
	if len(frames) == 0:
		raise ValueError(f"no frames decoded from {path}")
	try:
		return np.stack(frames)
	except ValueError as e:
		raise ValueError(f"TIFF frames of {path} have inconsistent shapes: {e}") from e


def _read_tiff(path):
	# tifffile first; LZW stacks without imagecodecs raise ValueError there,
	# and the film-meep environment has no tifffile at all.
	try:
		from tifffile import TiffFile
		with TiffFile(path) as tif:
			return tif.asarray(), "tifffile"
	except (ImportError, ValueError):
		return _read_tiff_pillow(path), "pillow"


def load_voxels(path, key=NPZ_DEFAULT_KEY):
	"""Load a 3D voxel array (True = solid) from .npz or .tif/.tiff.

	Returns (voxels, provenance). voxels is a bool ndarray with ndim 3 in the
	stored axis order (TIFF: axis 0 = frames). provenance has keys path,
	sha256, shape, solid_fraction, reader ("npz"|"tifffile"|"pillow") and
	sidecar (parsed <stem>.json next to the file, or None) and fraction_key.
	If the sidecar has phi1_realized, the solid fraction must match it within
	1e-6; otherwise, if it has phi2_realized, it must match that; otherwise no
	check. fraction_key records the key checked ("phi1_realized",
	"phi2_realized" or None).
	Every failure raises ValueError.
	"""
	path = os.path.abspath(str(path))
	if not os.path.isfile(path):
		raise ValueError(f"voxel file not found: {path}")
	stem, suffix = os.path.splitext(path)
	suffix = suffix.lower()

	if suffix == ".npz":
		raw = _read_npz(path, key)
		reader = "npz"
	elif suffix in TIFF_SUFFIXES:
		raw, reader = _read_tiff(path)
	else:
		raise ValueError(f"unsupported voxel file type {suffix!r} for {path} (expected .npz, .tif or .tiff)")

	voxels = np.asarray(raw).astype(bool)
	if voxels.ndim != 3:
		raise ValueError(f"expected a 3D voxel array in {path}, got ndim={voxels.ndim} with shape {voxels.shape}")
	if voxels.size == 0:
		raise ValueError(f"empty voxel array in {path}: shape {voxels.shape}")

	solid_fraction = float(voxels.mean())

	sidecar = None
	fraction_key = None
	sidecar_path = stem + ".json"
	if os.path.isfile(sidecar_path):
		try:
			with open(sidecar_path) as f:
				sidecar = json.load(f)
		except Exception as e:
			raise ValueError(f"cannot parse sidecar {sidecar_path}: {e}") from e
		if isinstance(sidecar, dict):
			for k in SIDECAR_FRACTION_KEYS:
				if k in sidecar:
					fraction_key = k
					break
		if fraction_key is not None:
			phi = float(sidecar[fraction_key])
			if abs(solid_fraction - phi) > SIDECAR_FRACTION_TOL:
				raise ValueError(f"solid fraction {solid_fraction:.9f} of {path} differs from sidecar {fraction_key} {phi:.9f} by more than {SIDECAR_FRACTION_TOL:g}")

	provenance = {
		"path": path,
		"sha256": _sha256_file(path),
		"shape": [int(n) for n in voxels.shape],
		"solid_fraction": solid_fraction,
		"reader": reader,
		"sidecar": sidecar,
		"fraction_key": fraction_key,
	}
	return voxels, provenance
