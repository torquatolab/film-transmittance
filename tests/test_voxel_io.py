# Tests for voxel_io.load_voxels (package P1). Plain script, exits nonzero on
# failure. Run in the film-meep environment on a compute node:
#   python tests/test_voxel_io.py -outdir DIR [-reference REF.json]
# Writes DIR/inventory.json (one entry per data file) and DIR/film_hashes.json
# (per-file shape and SHA-256 of np.packbits(voxels)). With -reference, the
# hashes are compared voxel-for-voxel against a JSON computed independently
# with structural_color/src/procfile.py::load_indicator.

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import traceback

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voxel_io import load_voxels

DATA_DIR = "/home/sd2402/torquato-scratch-sd2402/structural_color/data"

failures = []


def check(cond, msg):
	if not cond:
		failures.append(msg)
		print(f"FAIL: {msg}", flush=True)
	return cond


def packbits_sha(voxels):
	return hashlib.sha256(np.packbits(voxels.astype(bool)).tobytes()).hexdigest()


def data_files():
	npz = sorted(glob.glob(os.path.join(DATA_DIR, "shu", "*.npz"))) + sorted(glob.glob(os.path.join(DATA_DIR, "benchmarks", "*.npz")))
	tif = sorted(glob.glob(os.path.join(DATA_DIR, "*.tif")))
	return npz, tif


def expected_shape(path):
	name = os.path.basename(path)
	if name.endswith(".npz"):
		m = re.search(r"_l=(\d+)_", name)
		if m is None:
			return None
		n = int(m.group(1))
		return [n, n, n]
	with Image.open(path) as im:
		return [int(im.n_frames), int(im.size[1]), int(im.size[0])]


## ---- test 1: every data file loads ----
def test_all_files(outdir):
	npz, tif = data_files()
	inventory = []
	hashes = {}
	key_counts = {}
	family_keys = {}
	max_dphi = 0.0
	for path in npz + tif:
		try:
			v, prov = load_voxels(path)
		except Exception as e:
			check(False, f"load {path}: {type(e).__name__}: {e}")
			continue
		exp = expected_shape(path)
		check(v.ndim == 3, f"{path}: ndim {v.ndim}")
		check(v.dtype == np.bool_, f"{path}: dtype {v.dtype}")
		check(exp is None or list(v.shape) == exp, f"{path}: shape {list(v.shape)} != expected {exp}")
		check(prov["shape"] == list(v.shape), f"{path}: provenance shape")
		check(prov["path"] == os.path.abspath(path), f"{path}: provenance path")
		check(abs(prov["solid_fraction"] - float(v.mean())) == 0.0, f"{path}: provenance solid_fraction")
		sc = prov["sidecar"]
		exp_key = None
		if isinstance(sc, dict):
			exp_key = "phi1_realized" if "phi1_realized" in sc else "phi2_realized" if "phi2_realized" in sc else None
		check(prov.get("fraction_key", "absent") == exp_key, f"{path}: fraction_key {prov.get('fraction_key', 'absent')} != expected {exp_key}")
		key_counts[str(exp_key)] = key_counts.get(str(exp_key), 0) + 1
		family = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path).split("_")[0]
		family_keys.setdefault(family, {})
		family_keys[family][str(exp_key)] = family_keys[family].get(str(exp_key), 0) + 1
		if exp_key is not None:
			d = abs(prov["solid_fraction"] - float(sc[exp_key]))
			max_dphi = max(max_dphi, d)
			check(d <= 1e-6, f"{path}: |phi - {exp_key}| = {d}")
		with open(path, "rb") as f:
			check(prov["sha256"] == hashlib.sha256(f.read()).hexdigest(), f"{path}: provenance sha256")
		inventory.append({
			"path": prov["path"],
			"shape": prov["shape"],
			"solid_fraction": prov["solid_fraction"],
			"sha256": prov["sha256"],
			"reader": prov["reader"],
			"fraction_key": prov["fraction_key"],
		})
		hashes[prov["path"]] = {"shape": prov["shape"], "packbits_sha256": packbits_sha(v)}
	with open(os.path.join(outdir, "inventory.json"), "w") as f:
		json.dump(inventory, f, indent=1)
	with open(os.path.join(outdir, "film_hashes.json"), "w") as f:
		json.dump(hashes, f, indent=1)
	readers = sorted(set(e["reader"] for e in inventory))
	n_files = len(npz) + len(tif)
	check(len(inventory) == n_files, f"loaded {len(inventory)} of {n_files} files")
	print(f"test1: files={n_files} (npz={len(npz)}, tif={len(tif)}) loaded={len(inventory)} fraction_key_counts={key_counts} max_abs_dphi={max_dphi:.3e} readers={readers}", flush=True)
	for fam in sorted(family_keys):
		print(f"  family {fam}: {family_keys[fam]}", flush=True)
	for e in inventory:
		if e["path"].endswith(".tif"):
			print(f"  {os.path.basename(e['path'])}: shape={e['shape']} solid_fraction={e['solid_fraction']:.6f} reader={e['reader']}", flush=True)
	return hashes


## ---- test 2: TIFF encodings ----
def test_tiff_encodings(outdir):
	t1 = os.path.join(DATA_DIR, "T1.binary.closing.opening.tif")
	with Image.open(t1) as im:
		mode = im.mode
		raw0 = np.unique(np.asarray(im))
	check(mode == "L", f"T1 mode {mode}")
	check(set(raw0.tolist()) == {0, 255}, f"T1 frame-0 raw values {raw0.tolist()}")
	v, _ = load_voxels(t1)
	u = np.unique(v).tolist()
	check(u == [False, True], f"T1 unique {u}")
	print(f"test2: T1 mode={mode} raw_frame0_values={raw0.tolist()} loaded_unique={u}", flush=True)

	db = os.path.join(DATA_DIR, "DB.tif")
	with Image.open(db) as im:
		mode = im.mode
		raw0 = np.unique(np.asarray(im))
	v, _ = load_voxels(db)
	u = np.unique(v).tolist()
	check(mode == "P", f"DB mode {mode}")
	check(u == [False, True], f"DB unique {u}")
	print(f"test2: DB.tif mode={mode} raw_frame0_values={raw0.tolist()} loaded_unique={u}", flush=True)

	# synthetic palette stack with indices 0/1 and a palette mapping 1 -> white
	rng = np.random.default_rng(0)
	ref = rng.random((4, 7, 9)) < 0.4
	frames = []
	for k in range(ref.shape[0]):
		fr = Image.fromarray(ref[k].astype(np.uint8), mode="P")
		fr.putpalette([0, 0, 0, 255, 255, 255] + [0] * (3 * 254))
		frames.append(fr)
	p = os.path.join(outdir, "synthetic_palette.tif")
	frames[0].save(p, save_all=True, append_images=frames[1:], compression="tiff_lzw")
	with Image.open(p) as im:
		smode = im.mode
	v, prov = load_voxels(p)
	u = np.unique(v).tolist()
	check(smode == "P", f"synthetic mode {smode}")
	check(u == [False, True], f"synthetic palette unique {u}")
	check(v.shape == ref.shape and bool(np.array_equal(v, ref)), "synthetic palette voxels differ from source")
	print(f"test2: synthetic palette mode={smode} shape={list(v.shape)} unique={u} equal_to_source={bool(np.array_equal(v, ref))} reader={prov['reader']}", flush=True)


## ---- test 3: agreement with procfile.load_indicator ----
def test_reference(hashes, reference):
	with open(reference) as f:
		ref = json.load(f)
	n_cmp = 0
	mism = []
	for path, r in ref.items():
		if "error" in r:
			check(False, f"reference failed for {path}: {r['error']}")
			continue
		h = hashes.get(path)
		if h is None:
			check(False, f"no film hash for {path}")
			continue
		n_cmp += 1
		if h["shape"] != r["shape"] or h["packbits_sha256"] != r["packbits_sha256"]:
			mism.append(path)
	check(len(mism) == 0, f"voxel mismatches vs load_indicator: {mism}")
	check(n_cmp > 0, "no reference files compared")
	check(n_cmp == len(hashes) and set(ref) == set(hashes), f"reference covers {len(ref)} files, compared {n_cmp}, film hashed {len(hashes)}")
	n_tif = sum(1 for p in ref if p.endswith(".tif"))
	print(f"test3: compared={n_cmp} (tif={n_tif}, npz={n_cmp - n_tif}) mismatches={len(mism)}", flush=True)


## ---- test 4: error handling ----
def expect_value_error(name, fn):
	try:
		fn()
	except ValueError as e:
		print(f"test4: {name}: ValueError: {e}", flush=True)
		return
	except Exception as e:
		check(False, f"{name}: raised {type(e).__name__} instead of ValueError: {e}")
		return
	check(False, f"{name}: no exception")


def test_errors(outdir):
	d = os.path.join(outdir, "errors")
	os.makedirs(d, exist_ok=True)
	ok = np.zeros((3, 4, 5), dtype=np.uint8)
	ok[0, 0, :3] = 1
	np.savez(os.path.join(d, "ok.npz"), g=ok)
	np.savez(os.path.join(d, "twod.npz"), g=np.ones((4, 5), dtype=np.uint8))
	np.save(os.path.join(d, "bad.npy"), ok)
	np.savez(os.path.join(d, "mismatch.npz"), g=ok)
	with open(os.path.join(d, "mismatch.json"), "w") as f:
		json.dump({"phi2_realized": 0.5}, f)
	np.savez(os.path.join(d, "match.npz"), g=ok)
	with open(os.path.join(d, "match.json"), "w") as f:
		json.dump({"phi2_realized": 3.0 / 60.0}, f)
	# phi1 sidecars (benchmark style): phi1 takes precedence over phi2
	np.savez(os.path.join(d, "phi1_match.npz"), g=ok)
	with open(os.path.join(d, "phi1_match.json"), "w") as f:
		json.dump({"phi1_realized": 3.0 / 60.0, "phi2_realized": 57.0 / 60.0}, f)
	np.savez(os.path.join(d, "phi1_mismatch.npz"), g=ok)
	with open(os.path.join(d, "phi1_mismatch.json"), "w") as f:
		json.dump({"phi1_realized": 57.0 / 60.0, "phi2_realized": 3.0 / 60.0}, f)
	np.savez(os.path.join(d, "nokey.npz"), g=ok)
	with open(os.path.join(d, "nokey.json"), "w") as f:
		json.dump({"phi2": 0.5}, f)

	v, prov = load_voxels(os.path.join(d, "ok.npz"))
	check(v.dtype == np.bool_ and v.shape == (3, 4, 5) and prov["sidecar"] is None and prov["reader"] == "npz", "ok.npz control")
	v, prov = load_voxels(os.path.join(d, "match.npz"))
	check(prov["sidecar"] == {"phi2_realized": 0.05} and prov["fraction_key"] == "phi2_realized", "match.npz sidecar control")
	_, prov = load_voxels(os.path.join(d, "ok.npz"))
	check(prov["fraction_key"] is None, "ok.npz fraction_key")
	_, prov = load_voxels(os.path.join(d, "phi1_match.npz"))
	check(prov["fraction_key"] == "phi1_realized", f"phi1_match.npz fraction_key {prov['fraction_key']}")
	_, prov = load_voxels(os.path.join(d, "nokey.npz"))
	check(prov["fraction_key"] is None and prov["sidecar"] == {"phi2": 0.5}, f"nokey.npz fraction_key {prov['fraction_key']}")
	print("test4: phi1_match.npz loaded with fraction_key=phi1_realized (phi2_realized in sidecar ignored); nokey.npz fraction_key=None", flush=True)
	print(f"test4: controls ok.npz solid_fraction={load_voxels(os.path.join(d, 'ok.npz'))[1]['solid_fraction']} match.npz loaded", flush=True)

	expect_value_error("missing file", lambda: load_voxels(os.path.join(d, "does_not_exist.npz")))
	expect_value_error("bad suffix", lambda: load_voxels(os.path.join(d, "bad.npy")))
	expect_value_error("missing key", lambda: load_voxels(os.path.join(d, "ok.npz"), key="nope"))
	expect_value_error("2D array", lambda: load_voxels(os.path.join(d, "twod.npz")))
	expect_value_error("sidecar phi2 mismatch", lambda: load_voxels(os.path.join(d, "mismatch.npz")))
	expect_value_error("sidecar phi1 mismatch (phi2 matches)", lambda: load_voxels(os.path.join(d, "phi1_mismatch.npz")))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-outdir", required=True)
	parser.add_argument("-reference", default=None)
	args = parser.parse_args()
	os.makedirs(args.outdir, exist_ok=True)

	for name, fn in [("errors", lambda: test_errors(args.outdir)), ("tiff_encodings", lambda: test_tiff_encodings(args.outdir))]:
		try:
			fn()
		except Exception:
			check(False, f"{name} crashed: {traceback.format_exc()}")
	try:
		hashes = test_all_files(args.outdir)
		if args.reference is not None:
			test_reference(hashes, args.reference)
	except Exception:
		check(False, f"all_files/reference crashed: {traceback.format_exc()}")

	if failures:
		print(f"FAILED: {len(failures)} check(s)", flush=True)
		sys.exit(1)
	print("ALL PASS", flush=True)


if __name__ == "__main__":
	main()
