"""Quasi-static soft-tissue solver: projective dynamics (Bouaziz et al. 2014, "Projective
Dynamics: Fusing Constraint Projections for Fast Simulation"), for the bake step (NumPy + SciPy;
runs outside Blender).

The energy is the same as tissuelab.softbody (validated there against analytic cases):

* Tetrahedra: co-rotational, Σ V [ μ‖F − R‖² + (λ/2)(J − 1)² ], realised as two projective
  constraints per tetrahedron: the strain constraint (projection of F onto rotations, weight
  2μV) and the volume constraint (projection of F onto det = 1, weight 3λV, which matches
  (λ/2)(J − 1)² at small strain).
* Membrane triangles (fasciae thinner than a voxel): co-rotational in the plane (weight 2μ t A),
  an area constraint (weight λ_m t A), and tension-only collagen fibres along a rest direction a
  (weight k_f t A; projection |F a| → min(|F a|, 1)).
* Dirichlet vertices (pinned, or driven to a target such as a retractor's grip) are eliminated.
* Contact for vertices that rest on the viscera: a projective constraint with constant weight
  whose projection is supplied by the caller (`project(points) -> points`, e.g. pushing points
  out of organ surfaces along their normals), so the system matrix stays constant.

Each iteration: local step (per-element projections, vectorised) then global step (one sparse
solve with a matrix factorised once). Quasi-static: no inertia; external forces (gravity) enter
the right-hand side. Units: SI.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def _project_rotation(F: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    flip = np.linalg.det(R) < 0
    if flip.any():
        U = U.copy()
        U[flip, :, -1] *= -1
        R[flip] = U[flip] @ Vt[flip]
    return R


def _cof3(F):
    """Cofactor matrices (rows b×c, c×a, a×b for rows a, b, c) and determinants of 3×3 matrices:
    F⁻ᵀ = cof / det."""
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    bc = np.cross(b, c)
    cof = np.stack([bc, np.cross(c, a), np.cross(a, b)], 1)
    return cof, np.einsum("ij,ij->i", a, bc)


def _project_rotation_fast(F: np.ndarray, iters: int = 5) -> np.ndarray:
    """Rotation of the polar decomposition by determinant-scaled Newton iterations (Higham
    1986), for live interaction; elements with det F ≤ 0 fall back to the SVD."""
    _, d0 = _cof3(F)
    ok = d0 > 1e-12
    allok = bool(ok.all())
    R = F.copy() if allok else F[ok].copy()
    for _ in range(iters):
        cof, d = _cof3(R)
        g = np.abs(d) ** (-1.0 / 3.0)
        R = 0.5 * (g[:, None, None] * R + cof / (g * d)[:, None, None])
    if allok:
        return R
    out = np.empty_like(F)
    out[ok] = R
    out[~ok] = _project_rotation(F[~ok])
    return out


def _project_volume_fast(F: np.ndarray, iters: int = 3) -> np.ndarray:
    """Projection onto det F = 1 by Newton steps along the constraint's gradient (the cofactor
    matrix): P ← P − (det P − 1) cof P / ‖cof P‖², without an SVD; for live interaction.
    Elements with det F ≤ 0 fall back to the SVD projection."""
    _, d0 = _cof3(F)
    ok = d0 > 1e-12
    allok = bool(ok.all())
    P = F.copy() if allok else F[ok].copy()
    for _ in range(iters):
        cof, d = _cof3(P)
        n2 = np.maximum(np.einsum("kij,kij->k", cof, cof), 1e-30)
        P = P - ((d - 1.0) / n2)[:, None, None] * cof
    if allok:
        return P
    out = np.empty_like(F)
    out[ok] = P
    out[~ok] = _project_volume(F[~ok])
    return out


def _project_volume(F: np.ndarray, iters: int = 4) -> np.ndarray:
    """Nearest F with det F = 1, via its singular values (Newton on Σ ⇒ Π σ = 1)."""
    U, S, Vt = np.linalg.svd(F)
    d = np.zeros_like(S)
    for _ in range(iters):
        s = S + d
        C = s.prod(axis=1) - 1.0
        g = np.stack([s[:, 1] * s[:, 2], s[:, 0] * s[:, 2], s[:, 0] * s[:, 1]], -1)
        d = ((g * d).sum(1) - C)[:, None] * g / np.maximum((g * g).sum(1), 1e-30)[:, None]
    s = S + d
    neg = np.linalg.det(U) * np.linalg.det(Vt) < 0
    if neg.any():                                         # Inverted element: flip one axis
        s = s.copy()
        s[neg, 2] *= -1
    return U @ (s[:, :, None] * Vt)


class Solver:
    def __init__(self, x0, tets=None, mu_t=None, lam_t=None, tris=None, mu_m=None, lam_m=None, thick_m=None,
                 fibre_m=None, kf_m=None, fixed=None, floor_ids=None, floor_weight=None, fast=False,
                 stretch_limit_m=None, stretch_stiffness_m=None, attachment_ids=None, attachment_weights=None,
                 tether_nodes=None, tether_coefficients=None, tether_stiffness=None, tether_slack=None):
        """`fast`: approximate local projections (Newton polar, uniform volume scaling, closed-form
        membrane terms) for live interaction; the default (exact SVD projections) is the
        validated path used by the bakes. Optional membrane stretch limits are dimensionless;
        stretch_stiffness_m is the penalty in Pa (zero disables it). Attachment weights are
        spring stiffnesses in N/m on unique free vertices. Neither option changes legacy calls."""
        self.fast = fast
        self.x0 = np.asarray(x0, float)
        n = len(self.x0)
        self.n = n
        self.tets = np.zeros((0, 4), int) if tets is None else np.asarray(tets)
        self.tris = np.zeros((0, 3), int) if tris is None else np.asarray(tris)
        self.fixed = np.zeros(n, bool) if fixed is None else np.asarray(fixed, bool)
        rows, cols, vals = [], [], []
        self.blocks = []                                   # (kind, data) for the local step
        # --- tetrahedra: F = Ds Dm⁻¹; selection matrix maps x (n) to the 3 columns of Ds Dm⁻¹
        if len(self.tets):
            T = self.tets
            Dm = (self.x0[T[:, 1:]] - self.x0[T[:, :1]]).transpose(0, 2, 1)
            self.vol = np.linalg.det(Dm) / 6.0
            if (self.vol <= 0).any():
                raise ValueError(f"{int((self.vol <= 0).sum())} inverted or degenerate tetrahedra")
            Dinv = np.linalg.inv(Dm)
            # G: (m, 3 rows = F columns, 4 vertices): F[:, :, c] = Σ_v G[c, v] x_v
            G = np.zeros((len(T), 3, 4))
            G[:, :, 1:] = Dinv.transpose(0, 2, 1)
            G[:, :, 0] = -G[:, :, 1:].sum(axis=2)
            self.tet_G = G
            # The volume constraint's energy is the squared distance from F to {det F = 1}, which
            # for small strain is (J − 1)²/3: weight 3λV reproduces (λ/2)(J − 1)² V exactly.
            for kind, w in (("tet_strain", 2 * np.asarray(mu_t) * self.vol), ("tet_volume", 3 * np.asarray(lam_t) * self.vol)):
                self.blocks.append((kind, w))
                self._add_AtA(rows, cols, vals, T, G, w)
        # --- membranes: F (3×2) = Ds Tm⁻¹ in the triangle's rest frame
        if len(self.tris):
            T = self.tris
            e1 = self.x0[T[:, 1]] - self.x0[T[:, 0]]
            e2 = self.x0[T[:, 2]] - self.x0[T[:, 0]]
            nrm = np.cross(e1, e2)
            self.area = 0.5 * np.linalg.norm(nrm, axis=1)
            if (self.area <= 0).any():
                raise ValueError("degenerate membrane triangles")
            u = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
            w_ = np.cross(nrm / (2 * self.area[:, None]), u)
            Dm = np.stack([np.stack([(e1 * u).sum(1), (e1 * w_).sum(1)], -1),
                           np.stack([(e2 * u).sum(1), (e2 * w_).sum(1)], -1)], -1)
            Tinv = np.linalg.inv(Dm)
            G = np.zeros((len(T), 2, 3))
            G[:, :, 1:] = Tinv.transpose(0, 2, 1)
            G[:, :, 0] = -G[:, :, 1:].sum(axis=2)
            self.tri_G = G
            tA = np.asarray(thick_m) * self.area
            self.blocks.append(("tri_strain", 2 * np.asarray(mu_m) * tA))
            self._add_AtA(rows, cols, vals, T, G, 2 * np.asarray(mu_m) * tA)
            self.blocks.append(("tri_area", np.asarray(lam_m) * tA))
            self._add_AtA(rows, cols, vals, T, G, np.asarray(lam_m) * tA)
            kf = np.zeros(len(T)) if kf_m is None else np.asarray(kf_m)
            if (kf > 0).any():
                a = np.asarray(fibre_m)                       # (k, 2) in the rest frame
                Gf = np.einsum("kcv,kc->kv", G, a)[:, None, :]  # F a = Σ_v Gf[v] x_v
                self.fib_G, self.fib_ids = Gf[kf > 0], np.flatnonzero(kf > 0)
                self.blocks.append(("tri_fibre", kf[kf > 0] * tA[kf > 0]))
                self._add_AtA(rows, cols, vals, T[kf > 0], Gf[kf > 0], kf[kf > 0] * tA[kf > 0])
        # Embedded tension-only connections: q = sum_i c_i*x_i = endpoint B - A.
        # Repeated node indices are legal (endpoints may share an element); sparse assembly
        # and scatter sum their contributions. Weights are N/m, slack lengths are metres.
        if tether_nodes is not None and len(tether_nodes):
            nodes = np.asarray(tether_nodes, int)
            c = np.asarray(tether_coefficients, float)
            k = np.asarray(tether_stiffness, float)
            slack = np.asarray(tether_slack, float)
            if (nodes.ndim != 2 or c.shape != nodes.shape or k.shape != (len(nodes),)
                    or slack.shape != k.shape or (nodes < 0).any() or (nodes >= n).any()
                    or not np.isfinite(c).all() or not np.allclose(c.sum(1), 0, atol=1e-12)
                    or not np.isfinite(k).all() or (k < 0).any()
                    or not np.isfinite(slack).all() or (slack <= 0).any()):
                raise ValueError("invalid embedded tether arrays")
            keep = k > 0
            self.tether_nodes, self.tether_G = nodes[keep], c[keep, None, :]
            self.tether_slack = slack[keep]
            if keep.any():
                self.blocks.append(("tether", k[keep]))
                self._add_AtA(rows, cols, vals, self.tether_nodes, self.tether_G, k[keep])
        # Optional tension-only membrane stiffening: distance to the set of deformation
        # gradients whose principal stretches are <= limit. Zero energy below the limit.
        if stretch_stiffness_m is not None:
            k = np.broadcast_to(stretch_stiffness_m, (len(self.tris),)).astype(float)
            limit = np.broadcast_to(stretch_limit_m, k.shape).astype(float)
            if not np.isfinite(k).all() or (k < 0).any() or not np.isfinite(limit).all() or (limit < 1).any():
                raise ValueError("membrane stiffness must be finite/nonnegative and stretch limit >= 1")
            self.limit_ids = np.flatnonzero(k > 0)
            self.stretch_limit = limit[self.limit_ids]
            if len(self.limit_ids):
                w = k[self.limit_ids] * tA[self.limit_ids]
                self.blocks.append(("tri_limit", w))
                self._add_AtA(rows, cols, vals, self.tris[self.limit_ids], self.tri_G[self.limit_ids], w)
        # Spring attachments: weights in N/m, targets supplied at solve time.
        self.attachment_ids = np.asarray([] if attachment_ids is None else attachment_ids, dtype=int)
        self.attachment_weights = np.broadcast_to(
            1.0 if attachment_weights is None else attachment_weights, self.attachment_ids.shape).copy()
        if len(self.attachment_ids):
            if ((self.attachment_ids < 0).any() or (self.attachment_ids >= n).any()
                    or self.fixed[self.attachment_ids].any()
                    or len(np.unique(self.attachment_ids)) != len(self.attachment_ids)
                    or not np.isfinite(self.attachment_weights).all() or (self.attachment_weights <= 0).any()):
                raise ValueError("attachments need unique free vertices and positive finite weights")
            rows.append(self.attachment_ids)
            cols.append(self.attachment_ids)
            vals.append(self.attachment_weights)
        # --- floor
        self.floor_ids = np.zeros(0, int) if floor_ids is None else np.asarray(floor_ids)
        if len(self.floor_ids):
            fw = np.full(len(self.floor_ids), 1.0) if floor_weight is None else np.broadcast_to(floor_weight, self.floor_ids.shape)
            self.floor_w = np.asarray(fw, float)
            rows.append(self.floor_ids)
            cols.append(self.floor_ids)
            vals.append(self.floor_w)
        L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        self.L = L
        free = np.flatnonzero(~self.fixed)
        self.free = free
        self.L_ff = L[free][:, free].tocsc()
        self.L_fc = L[free][:, np.flatnonzero(self.fixed)]
        self.lu = splu(self.L_ff)

    @staticmethod
    def _add_AtA(rows, cols, vals, T, G, w):
        """Σ_e w_e Gᵀ G over each element's vertices (identical for x, y, z)."""
        K = np.einsum("e,ecv,ecu->evu", w, G, G)          # (m, nv, nv)
        nv = T.shape[1]
        rows.append(np.repeat(T, nv, axis=1).ravel())
        cols.append(np.tile(T, (1, nv)).ravel())
        vals.append(K.reshape(len(T), -1).ravel())

    def _rhs(self, x, floor_fn):
        b = np.zeros((self.n, 3))

        def scatter(T, G, w, P):                           # b += Σ w Gᵀ P
            contrib = np.einsum("e,ecv,eic->evi", w, G, P)   # P: (m, 3, c) target F columns
            np.add.at(b, T.ravel(), contrib.reshape(-1, 3))

        for kind, w in self.blocks:
            if kind.startswith("tet_"):
                T, G = self.tets, self.tet_G
                if kind == "tet_strain" or not self.fast:
                    F = np.einsum("ecv,evi->eic", G, x[T])
                if self.fast:
                    P = _project_rotation_fast(F) if kind == "tet_strain" else _project_volume_fast(F)
                else:
                    P = _project_rotation(F) if kind == "tet_strain" else _project_volume(F)
                scatter(T, G, w, P)
            elif kind in ("tri_strain", "tri_area") and self.fast:
                T, G = self.tris, self.tri_G
                F = np.einsum("ecv,evi->eic", G, x[T])     # (k, 3, 2)
                C = np.einsum("kia,kib->kab", F, F)        # Fᵀ F
                dC = np.maximum(C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0], 1e-24)
                if kind == "tri_strain":                    # F (FᵀF)^(-1/2), 2×2 square root in closed form
                    sd = np.sqrt(dC)
                    t = np.sqrt(np.maximum(C[:, 0, 0] + C[:, 1, 1] + 2 * sd, 1e-24))
                    Sq = (C + sd[:, None, None] * np.eye(2)) / t[:, None, None]
                    dS = Sq[:, 0, 0] * Sq[:, 1, 1] - Sq[:, 0, 1] * Sq[:, 1, 0]
                    Si = np.stack([np.stack([Sq[:, 1, 1], -Sq[:, 0, 1]], -1),
                                   np.stack([-Sq[:, 1, 0], Sq[:, 0, 0]], -1)], 1) / dS[:, None, None]
                    P = F @ Si
                else:                                      # Area ratio 1: exact by uniform scaling
                    P = F * (dC ** -0.25)[:, None, None]
                scatter(T, G, w, P)
            elif kind in ("tri_strain", "tri_area"):
                T, G = self.tris, self.tri_G
                F = np.einsum("ecv,evi->eic", G, x[T])     # (k, 3, 2)
                U, S, Vt = np.linalg.svd(F, full_matrices=False)
                if kind == "tri_strain":
                    P = U @ Vt
                else:                                      # Nearest F with area ratio 1
                    s = S * (1.0 / np.sqrt(np.maximum(S.prod(axis=1), 1e-12)))[:, None]
                    P = U @ (s[:, :, None] * Vt)
                scatter(T, G, w, P)
            elif kind == "tri_limit":
                T, G = self.tris[self.limit_ids], self.tri_G[self.limit_ids]
                F = np.einsum("ecv,evi->eic", G, x[T])
                U, S, Vt = np.linalg.svd(F, full_matrices=False)
                P = U @ (np.minimum(S, self.stretch_limit[:, None])[:, :, None] * Vt)
                scatter(T, G, w, P)
            elif kind == "tether":
                T, G = self.tether_nodes, self.tether_G
                q = np.einsum("ecv,evi->eic", G, x[T])
                length = np.linalg.norm(q[:, :, 0], axis=1)
                factor = np.minimum(1.0, self.tether_slack / np.maximum(length, 1e-30))
                scatter(T, G, w, q * factor[:, None, None])
            elif kind == "tri_fibre":
                T, G = self.tris[self.fib_ids], self.fib_G
                fa = np.einsum("ecv,evi->eic", G, x[T])[:, :, 0]   # (k, 3)
                L = np.linalg.norm(fa, axis=1)
                P = (fa * (np.minimum(L, 1.0) / np.maximum(L, 1e-12))[:, None])[:, :, None]
                scatter(T, G, w, P)
        if len(self.floor_ids):
            p = x[self.floor_ids].copy()
            p = floor_fn(p) if floor_fn is not None else p
            np.add.at(b, self.floor_ids, self.floor_w[:, None] * p)
        return b

    def _deformation_gradients(self, x):
        return [(kind, np.einsum("ecv,evi->eic", G, x[T]))
                for kind, T, G in (("volume", self.tets, getattr(self, "tet_G", None)),
                                   ("surface", self.tris, getattr(self, "tri_G", None))) if len(T)]

    @staticmethod
    def _within_stretch_limit(gradients, limit):
        for kind, F in gradients:
            if not np.isfinite(F).all():
                return False
            singular = np.linalg.svd(F, compute_uv=False)
            if (singular[:, 0] > limit).any() or (singular[:, -1] <= 1e-8).any():
                return False
            if kind == "volume" and (np.linalg.det(F) <= 1e-8).any():
                return False
        return True

    def _limit_step(self, x, new, limit):
        """Shorten the entire proposed step to a feasible state, never reset the rest shape.

        This conservative global line search can also slow distant tissue when one element
        reaches its bound. It is an interaction guard, not a constitutive failure model.
        """
        if not np.isfinite(new).all():
            raise ValueError("non-finite tissue solver update")
        candidate = x.copy()
        candidate[self.free] = new
        end = self._deformation_gradients(candidate)
        if self._within_stretch_limit(end, limit):
            return new, False
        start = self._deformation_gradients(x)
        # At a hard stop, avoid twenty full-mesh checks on every subsequent timer tick.
        tiny = [(kind, A + 2**-20 * (B - A)) for (kind, A), (_, B) in zip(start, end)]
        if not self._within_stretch_limit(tiny, limit):
            return x[self.free].copy(), True
        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = (lo + hi) / 2
            gradients = [(kind, A + mid * (B - A)) for (kind, A), (_, B) in zip(start, end)]
            if self._within_stretch_limit(gradients, limit):
                lo = mid
            else:
                hi = mid
        return x[self.free] + lo * (new - x[self.free]), True

    def solve(self, x, targets=None, f_ext=None, floor_fn=None, iters=50, tol=1e-7, rho=None, gamma=0.9, attachment_targets=None, max_stretch=None):
        """Quasi-static equilibrium from `x`. `targets`: (n, 3) for fixed vertices (NaN keeps x).
        `f_ext`: (n, 3) external forces (N). `floor_fn`: projection for the contact vertices
        (points (k, 3) -> points). `rho`: estimated spectral radius for Chebyshev acceleration
        (None: plain iterations). `attachment_targets`: (number of attachments, 3) in metres,
        ordered like attachment_ids. Optional `max_stretch` caps the largest principal stretch
        of every volume and membrane relative to x0; it also rejects collapsed/inverted elements.
        `limit_reached` records clipping, NOT equilibrium or measured failure.
        Returns (x, iterations, last max displacement)."""
        x = np.asarray(x, float).copy()
        if targets is not None:
            have = self.fixed & np.isfinite(targets[:, 0])
            x[have] = targets[have]
        if len(self.attachment_ids):
            attachment_targets = np.asarray(attachment_targets, float)
            if attachment_targets.shape != (len(self.attachment_ids), 3) or not np.isfinite(attachment_targets).all():
                raise ValueError("finite attachment_targets of shape (attachments, 3) required")
        self.limit_reached = False
        if max_stretch is not None:
            if not np.isfinite(max_stretch) or max_stretch <= 1:
                raise ValueError("max_stretch must be finite and greater than 1")
            # A previous accepted result is already checked. Compare values, not identity:
            # callers can mutate x or fixed targets between interactive updates.
            cached = getattr(self, "_guard_state", None)
            known_valid = cached is not None and cached[0] == max_stretch and np.array_equal(cached[1], x)
            if not known_valid and not self._within_stretch_limit(self._deformation_gradients(x), max_stretch):
                raise ValueError("initial state or fixed targets exceed the tissue stretch limit")
        xc = x[self.fixed]
        it, dmax = 0, np.inf
        prev = x[self.free].copy()                         # x_{k−1} (Chebyshev)
        omega = 1.0
        for it in range(1, iters + 1):
            b = self._rhs(x, floor_fn)
            if len(self.attachment_ids):
                b[self.attachment_ids] += self.attachment_weights[:, None] * attachment_targets
            if f_ext is not None:
                b += f_ext
            rhs = b[self.free] - self.L_fc @ xc
            new = np.column_stack([self.lu.solve(np.ascontiguousarray(rhs[:, k])) for k in range(3)])
            cur = x[self.free]
            if rho is not None:
                # Chebyshev semi-iterative acceleration (Wang 2015), from the 10th iteration on,
                # with under-relaxation γ: x_{k+1} = ω(γ(x̂ − x_k) + x_k − x_{k−1}) + x_{k−1}.
                if it < 10:
                    omega = 1.0
                elif it == 10:
                    omega = 2.0 / (2.0 - rho * rho)
                else:
                    omega = 4.0 / (4.0 - rho * rho * omega)
                new = omega * (gamma * (new - cur) + cur - prev) + prev
            if max_stretch is not None:
                new, clipped = self._limit_step(x, new, max_stretch)
                self.limit_reached |= clipped
            dmax = float(np.abs(new - cur).max())
            prev = cur.copy()
            x[self.free] = new
            if dmax < tol:
                break
        self._guard_state = (max_stretch, x.copy()) if max_stretch is not None else None
        return x, it, dmax


def tet_masses(x0, tets, rho):
    D = (x0[tets[:, 1:]] - x0[tets[:, :1]]).transpose(0, 2, 1)
    V = np.linalg.det(D) / 6.0
    m = np.zeros(len(x0))
    np.add.at(m, tets.ravel(), np.repeat(np.asarray(rho) * V / 4.0, 4))
    return m


class SurfaceContact:
    """Keep points outside a set of closed surfaces (organs): a point closer than `gap` to the
    nearest surface vertex, measured along that vertex's outward normal, is pushed out along
    it. Approximate signed distance; accurate for surfaces sampled finer than the motion."""

    def __init__(self, verts: np.ndarray, normals: np.ndarray, gap: float = 0.0005):
        from scipy.spatial import cKDTree
        self.tree, self.v, self.n, self.gap = cKDTree(verts), verts, normals, gap

    def __call__(self, p: np.ndarray) -> np.ndarray:
        _, i = self.tree.query(p)
        s = ((p - self.v[i]) * self.n[i]).sum(1)
        push = np.maximum(0.0, self.gap - s)
        return p + push[:, None] * self.n[i]

    def violations(self, p: np.ndarray) -> np.ndarray:
        _, i = self.tree.query(p)
        return ((p - self.v[i]) * self.n[i]).sum(1) < -1e-4

    def no_new_contact(self, p0: np.ndarray) -> "RelativeContact":
        """A contact that allows each point (rows of p0, its rest position) the clearance it
        has at rest and no less: existing near-contacts (where the surfaces' sampled
        distance is unreliable) are left alone, new penetration is prevented."""
        _, i = self.tree.query(p0)
        s0 = ((p0 - self.v[i]) * self.n[i]).sum(1)
        return RelativeContact(self, np.minimum(s0, self.gap))


class RelativeContact:
    def __init__(self, contact: SurfaceContact, allow: np.ndarray):
        self.c, self.allow = contact, allow

    def __call__(self, p: np.ndarray) -> np.ndarray:
        _, i = self.c.tree.query(p)
        s = ((p - self.c.v[i]) * self.c.n[i]).sum(1)
        push = np.maximum(0.0, self.allow - s)
        return p + push[:, None] * self.c.n[i]

    def violations(self, p: np.ndarray) -> np.ndarray:
        _, i = self.c.tree.query(p)
        return ((p - self.c.v[i]) * self.c.n[i]).sum(1) < self.allow - 1e-4
