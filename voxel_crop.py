#!/usr/bin/env python3
"""Seam score and exact best lateral crop of a voxel film (package P3).

A voxel film is repeated periodically in the two lateral axes (the axes other
than ``normal_axis``). The wrap joins the first and last plane of each lateral
axis; if they differ much more than neighbouring planes do, the periodic copy
has a visible seam. For a lateral axis ``a`` with planes
``P_i = voxels.take(i, axis=a)`` (length ``L_a``):

	seam_a     = mean(P_first != P_last)
	baseline_a = mean over i in [0, L_a - 2] of mean(P_i != P_{i+1})
	ratio_a    = seam_a / baseline_a
	score      = (ratio_p + ratio_q) / 2      (p < q the lateral axes)

Degenerate cases (documented choices):
- ``baseline_a == 0`` (all neighbouring planes equal, or ``L_a == 1`` so there
  are no neighbour pairs): ``ratio_a = seam_a / 1e-12`` (0 if the seam is 0 too).
  Any non-zero baseline is >= 1 / voxel count >> 1e-12, so this equals
  ``seam_a / max(baseline_a, 1e-12)``.
- Non-bool input is converted with ``astype(bool)`` (solid = nonzero).

``best_crop`` searches every lateral crop ``[p0, p1) x [q0, q1)`` with
``p1 - p0 >= ceil(min_fraction * n_p)`` and ``q1 - q0 >= ceil(min_fraction * n_q)``
(the normal axis is kept whole), minimises ``score``, and breaks ties by larger
kept fraction, then smaller offsets ``(p0, q0)`` (stored axis order).

Algorithm (exact, deterministic). Let the array be ``(N, A, B)`` after moving
the normal axis first (A = n_p, B = n_q), and ``m_A = A - ceil(f A) + 1``,
``m_B = B - ceil(f B) + 1`` the number of allowed lengths. The candidate
``p``-ranges number ``K_A = m_A (m_A + 1) / 2`` (likewise ``K_B``). All sums
below are integer counts of differing voxels.

1. Seam of axis p depends on the q-range: for each allowed p-range
   ``(p0, p1)`` store ``D_p[k] = #{z : v[z,p0,k] != v[z,p1-1,k]}`` and its
   prefix sum over ``k``; then the seam count for any ``[q0, q1)`` is a
   difference of two prefix entries. Cost ``O(K_A N B)`` time,
   ``O(K_A B)`` memory. Symmetric for axis q: ``O(K_B N A)``, ``O(K_B A)``.
2. Baseline of axis p: ``E_p[i, k] = #{z : v[z,i,k] != v[z,i+1,k]}`` with a 2D
   prefix sum, so the baseline count of any rectangle is four look-ups.
   ``O(N A B)`` time, ``O(A B)`` memory. Symmetric for axis q.
3. For each p-range, evaluate all ``K_B`` q-ranges vectorised with the
   look-ups: ``O(K_A K_B)`` time in total (two passes: global float minimum,
   then collection of near-ties).
4. The ratios are formed from exact integers, so the float score has relative
   error ~1e-15. All candidates within relative 1e-10 of the float minimum
   (or exactly 0 when the minimum is 0, which only integer zero seams give)
   are re-scored with ``fractions.Fraction`` and resolved exactly, then
   tie-broken. This makes the result independent of floating-point rounding.

Peak memory is dominated by step 1 (int64 arrays of ``K_A (B+1)`` and
``K_B (A+1)`` entries) plus one boolean plane-pair block per row. For the
largest stack (541 x 356 x 473, f = 0.75): K_A = 4095, K_B = 7140,
K_A K_B ~ 2.9e7 crops.
"""

import math
from fractions import Fraction

import numpy as np

_EPS_BASELINE = 1e-12
_REL_TIE = 1e-10


def _check(voxels, normal_axis):
	v = np.asarray(voxels)
	if v.ndim != 3:
		raise ValueError(f"voxels must be 3D, got ndim={v.ndim}")
	if normal_axis not in (0, 1, 2):
		raise ValueError(f"normal_axis must be 0, 1 or 2, got {normal_axis}")
	if min(v.shape) < 1:
		raise ValueError(f"voxels must be non-empty, got shape {v.shape}")
	if v.dtype != bool:
		v = v.astype(bool)
	return v


def _ratio_float(seam, baseline):
	return seam / baseline if baseline > 0 else seam / _EPS_BASELINE


def seam_score(voxels: np.ndarray, normal_axis: int = 0) -> dict:
	"""Seam score of the whole array (see module docstring).

	Returns ``{"score": float, "per_axis": {a: {"seam", "baseline", "ratio"}}}``
	with ``a`` the stored lateral axis index.
	"""
	v = _check(voxels, normal_axis)
	per_axis = {}
	for a in [ax for ax in range(3) if ax != normal_axis]:
		L = v.shape[a]
		first = v.take(0, axis=a)
		last = v.take(L - 1, axis=a)
		plane_size = first.size
		seam = int(np.count_nonzero(first != last)) / plane_size
		if L > 1:
			diff = np.diff(v.view(np.uint8), axis=a) != 0
			baseline = int(np.count_nonzero(diff)) / ((L - 1) * plane_size)
		else:
			baseline = 0.0
		per_axis[a] = {"seam": seam, "baseline": baseline, "ratio": _ratio_float(seam, baseline)}
	score = float(np.mean([d["ratio"] for d in per_axis.values()]))
	return {"score": score, "per_axis": per_axis}


def _ranges(n, min_len):
	"""All half-open ranges [s, e) in [0, n) with e - s >= min_len, ordered by (s, e)."""
	s, e = [], []
	for s0 in range(0, n - min_len + 1):
		for e0 in range(s0 + min_len, n + 1):
			s.append(s0)
			e.append(e0)
	return np.array(s, dtype=np.int64), np.array(e, dtype=np.int64)


def _pair_prefix(w, starts, ends):
	"""w: (N, A, B) bool. For each range r, prefix over k of #{z: w[z,s,k] != w[z,e-1,k]}.

	Returns int64 (len(starts), B + 1).
	"""
	N, A, B = w.shape
	out = np.zeros((len(starts), B + 1), dtype=np.int64)
	for s0 in np.unique(starts):
		idx = np.nonzero(starts == s0)[0]
		lasts = ends[idx] - 1
		diff = w[:, lasts, :] != w[:, s0:s0 + 1, :]  # (N, len(idx), B)
		counts = np.count_nonzero(diff, axis=0)  # (len(idx), B)
		out[idx, 1:] = np.cumsum(counts, axis=1)
	return out


def _neighbour_prefix(w):
	"""w: (N, A, B) bool. P[i, k] = sum_{i' < i, k' < k} #{z: w[z,i',k'] != w[z,i'+1,k']}.

	Returns int64 (A, B + 1); P[0, :] = 0 (row i sums neighbour pairs 0..i-1).
	"""
	N, A, B = w.shape
	P = np.zeros((A, B + 1), dtype=np.int64)
	if A > 1:
		E = np.count_nonzero(w[:, 1:, :] != w[:, :-1, :], axis=0).astype(np.int64)  # (A-1, B)
		P[1:, 1:] = np.cumsum(np.cumsum(E, axis=0), axis=1)
	return P


def _exact_ratio(S, Bsum, L, plane):
	"""Exact ratio for seam count S, baseline count Bsum, axis length L, plane size."""
	if Bsum > 0:
		return Fraction((L - 1) * S, Bsum)
	return Fraction(S, plane) * 10 ** 12


def best_crop(voxels: np.ndarray, normal_axis: int = 0, min_fraction: float = 0.75) -> dict:
	"""Exact minimum-seam-score lateral crop (see module docstring).

	Returns ``{"slices": [[s0,e0],[s1,e1],[s2,e2]], "score", "per_axis",
	"kept_fraction", "full_score"}`` in stored axis order (half-open slices).
	"""
	v = _check(voxels, normal_axis)
	if not (0 < min_fraction <= 1):
		raise ValueError(f"min_fraction must be in (0, 1], got {min_fraction}")
	p, q = [ax for ax in range(3) if ax != normal_axis]
	w = np.ascontiguousarray(np.transpose(v, (normal_axis, p, q)))  # (N, A, B)
	N, A, B = w.shape
	minA = max(1, math.ceil(min_fraction * A))
	minB = max(1, math.ceil(min_fraction * B))

	a0s, a1s = _ranges(A, minA)
	b0s, b1s = _ranges(B, minB)
	LA = a1s - a0s
	LB = b1s - b0s

	# axis p: seam prefix over q-index; baseline prefix over (p, q)
	CDp = _pair_prefix(w, a0s, a1s)  # (K_A, B+1)
	Pp = _neighbour_prefix(w)  # (A, B+1)
	# axis q: same with p and q swapped
	wq = np.ascontiguousarray(np.transpose(w, (0, 2, 1)))  # (N, B, A)
	CDq = _pair_prefix(wq, b0s, b1s)  # (K_B, A+1)
	Pq = _neighbour_prefix(wq)  # (B, A+1)
	del wq

	def row(ia):
		a0 = a0s[ia]
		a1 = a1s[ia]
		la = a1 - a0
		S1 = CDp[ia, b1s] - CDp[ia, b0s]
		B1 = Pp[a1 - 1, b1s] - Pp[a0, b1s] - Pp[a1 - 1, b0s] + Pp[a0, b0s]
		S2 = CDq[:, a1] - CDq[:, a0]
		B2 = Pq[b1s - 1, a1] - Pq[b0s, a1] - Pq[b1s - 1, a0] + Pq[b0s, a0]
		with np.errstate(divide="ignore", invalid="ignore"):
			r1 = np.where(B1 > 0, (la - 1) * S1 / np.where(B1 > 0, B1, 1),
				S1 / (N * LB) / _EPS_BASELINE)
			r2 = np.where(B2 > 0, (LB - 1) * S2 / np.where(B2 > 0, B2, 1),
				S2 / (N * la) / _EPS_BASELINE)
		return 0.5 * (r1 + r2), S1, B1, S2, B2

	best = np.inf
	for ia in range(len(a0s)):
		best = min(best, float(row(ia)[0].min()))

	thresh = best * (1 + _REL_TIE) if best > 0 else 0.0
	cands = []
	for ia in range(len(a0s)):
		sc, S1, B1, S2, B2 = row(ia)
		hits = np.nonzero(sc <= thresh)[0]
		if best == 0 and len(hits) > 1:
			# every hit is an exact 0 score: keep only this row's tie-break winner
			hits = hits[np.lexsort((b0s[hits], -LB[hits]))[:1]]
		for ib in hits:
			cands.append((ia, int(ib), int(S1[ib]), int(B1[ib]), int(S2[ib]), int(B2[ib])))

	def key(c):
		ia, ib, S1, B1, S2, B2 = c
		la = int(LA[ia])
		lb = int(LB[ib])
		exact = _exact_ratio(S1, B1, la, N * lb) + _exact_ratio(S2, B2, lb, N * la)
		return (exact, -la * lb, int(a0s[ia]), int(b0s[ib]))

	ia, ib = min(cands, key=key)[:2]
	slices = [[0, 0], [0, 0], [0, 0]]
	slices[normal_axis] = [0, N]
	slices[p] = [int(a0s[ia]), int(a1s[ia])]
	slices[q] = [int(b0s[ib]), int(b1s[ib])]
	crop = v[tuple(slice(s, e) for s, e in slices)]
	res = seam_score(crop, normal_axis)
	full = seam_score(v, normal_axis)
	return {
		"slices": slices,
		"score": res["score"],
		"per_axis": res["per_axis"],
		"kept_fraction": float(LA[ia] * LB[ib]) / float(A * B),
		"full_score": full["score"],
	}
