"""Soft-tissue solver (NumPy only; runs inside Blender and in tests).

Extended position-based dynamics (XPBD; Macklin, Müller & Chentanez 2016) with small substeps
(Macklin et al. 2019) and these constraints:

* Tetrahedra: co-rotational elasticity, Ψ = μ‖F − R‖² + λ/2 (J − 1)² (R: rotation of the polar
  decomposition of the deformation gradient F), as two constraints per tetrahedron:
  C_D = ‖F − R‖ (compliance 1/(2μV)) and C_H = J − 1 (compliance 1/(λV)). Both vanish at
  rest (stress-free), the small-strain response is linear elasticity with Lamé μ, λ, and large
  rotations cost nothing. (A stable neo-Hookean split, Macklin & Müller 2021, was tried first:
  solved sequentially with a stiff volume term its two parts do not cancel at rest.)
* Membrane triangles (fasciae thinner than a voxel): the same co-rotational term in the plane
  (F is 3×2; compliance 1/(2μ t A)), an area term C = a/a₀ − 1 (compliance 1/(λ_m t A)), and an
  optional fibre term C_f = |F a| − 1, active only in tension (compliance 1/(k_f t A)), for
  collagen fibres along a (e.g. the transverse fibres of the linea alba and posterior sheath).
* Kinematic vertices (pinned, or driven along a path, e.g. a retractor's grip) and a
  height-field floor (z ≥ floor(x, y)) for vertices that rest on the viscera.

Elements are graph-coloured once so each colour is solved as a vectorised Gauss–Seidel batch
(no two elements in a batch share a vertex), which is deterministic.

Units: SI (m, kg, s, Pa).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def colour(elements: np.ndarray, n_vertices: int) -> list[np.ndarray]:
    """Greedy graph colouring: lists of element indices, no shared vertex within a list."""
    order = np.arange(len(elements))
    used = [set() for _ in range(n_vertices)]           # Colours touching each vertex
    colour_of = np.empty(len(elements), np.int64)
    for e in order:
        taken = set()
        for v in elements[e]:
            taken |= used[v]
        c = 0
        while c in taken:
            c += 1
        colour_of[e] = c
        for v in elements[e]:
            used[v].add(c)
    return [np.flatnonzero(colour_of == c) for c in range(colour_of.max() + 1 if len(order) else 0)]


@dataclass
class Body:
    x: np.ndarray                               # (n, 3) current positions
    tets: np.ndarray = None                     # (m, 4)
    mu_t: np.ndarray = None                     # (m,) shear modulus per tet (Pa)
    lam_t: np.ndarray = None                    # (m,) Lamé λ per tet (Pa)
    rho_t: np.ndarray = None                    # (m,) density (kg/m³)
    tris: np.ndarray = None                     # (k, 3) membranes
    mu_m: np.ndarray = None                     # (k,) membrane shear modulus (Pa)
    lam_m: np.ndarray = None                    # (k,) membrane area modulus (Pa)
    thick_m: np.ndarray = None                  # (k,) thickness (m)
    rho_m: np.ndarray = None                    # (k,)
    fibre_m: np.ndarray = None                  # (k, 2) fibre direction in the triangle's rest frame (or zeros)
    kf_m: np.ndarray = None                     # (k,) fibre stiffness (Pa), 0 = none
    kinematic: np.ndarray = None                # (n,) bool: position prescribed
    floor_ids: np.ndarray = None                # vertices subject to the floor
    gravity: tuple = (0.0, 0.0, -9.81)
    damping: float = 0.0                        # Velocity damping per second
    _prep: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.x)
        self.x = np.asarray(self.x, float).copy()
        self.x0 = self.x.copy()
        self.v = np.zeros_like(self.x)
        if self.tets is None:
            self.tets = np.zeros((0, 4), np.int64)
        if self.tris is None:
            self.tris = np.zeros((0, 3), np.int64)
        if self.kinematic is None:
            self.kinematic = np.zeros(n, bool)
        m, k = len(self.tets), len(self.tris)
        # Rest shapes
        if m:
            D = (self.x0[self.tets[:, 1:]] - self.x0[self.tets[:, :1]]).transpose(0, 2, 1)   # columns
            self.vol = np.linalg.det(D) / 6.0
            if (self.vol <= 0).any():
                raise ValueError(f"{int((self.vol <= 0).sum())} inverted or degenerate tetrahedra")
            self.Dm_inv = np.linalg.inv(D)
        if k:
            e1 = self.x0[self.tris[:, 1]] - self.x0[self.tris[:, 0]]
            e2 = self.x0[self.tris[:, 2]] - self.x0[self.tris[:, 0]]
            nrm = np.cross(e1, e2)
            self.area = 0.5 * np.linalg.norm(nrm, axis=1)
            if (self.area <= 0).any():
                raise ValueError("degenerate membrane triangles")
            u = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
            w = np.cross(nrm / (2 * self.area[:, None]), u)
            Dm = np.stack([np.stack([(e1 * u).sum(1), (e1 * w).sum(1)], -1),
                           np.stack([(e2 * u).sum(1), (e2 * w).sum(1)], -1)], -1)          # (k, 2, 2) columns
            self.Tm_inv = np.linalg.inv(Dm)
            if self.fibre_m is None:
                self.fibre_m = np.zeros((k, 2))
            if self.kf_m is None:
                self.kf_m = np.zeros(k)
        # Lumped masses
        mass = np.zeros(n)
        if m:
            np.add.at(mass, self.tets.ravel(), np.repeat(self.rho_t * self.vol / 4.0, 4))
        if k:
            np.add.at(mass, self.tris.ravel(), np.repeat(self.rho_m * self.thick_m * self.area / 3.0, 3))
        if (mass[~self.kinematic] <= 0).any():
            raise ValueError("free vertices without mass (not in any element)")
        self.w = np.where(self.kinematic | (mass <= 0), 0.0, 1.0 / np.maximum(mass, 1e-30))
        self.tet_colours = colour(self.tets, n) if m else []
        self.tri_colours = colour(self.tris, n) if k else []

    # ------------------------------------------------------------------ constraints
    def _solve_tets(self, ids, inv_dt2):
        T = self.tets[ids]
        x = self.x
        Ds = (x[T[:, 1:]] - x[T[:, :1]]).transpose(0, 2, 1)
        F = Ds @ self.Dm_inv[ids]
        V = self.vol[ids]
        w = self.w[T]                                                  # (b, 4)
        DmT = self.Dm_inv[ids].transpose(0, 2, 1)

        def project(C, dCdF, compliance):
            g = dCdF @ DmT                                             # (b, 3, 3): columns for verts 1..3
            g0 = -g.sum(axis=2)
            grads = np.concatenate([g0[:, :, None], g], axis=2)        # (b, 3, 4)
            denom = (w[:, None, :] * grads ** 2).sum(axis=(1, 2)) + compliance * inv_dt2
            dl = -C / np.maximum(denom, 1e-30)
            corr = (w[:, None, :] * grads * dl[:, None, None]).transpose(0, 2, 1)   # (b, 4, 3)
            np.add.at(self.x, T.ravel(), corr.reshape(-1, 3))
            return grads

        # Deviatoric (co-rotational): C = ‖F − R‖
        U, _, Vt = np.linalg.svd(F)
        R = U @ Vt
        flip = np.linalg.det(R) < 0
        if flip.any():
            U[flip, :, 2] *= -1
            R[flip] = U[flip] @ Vt[flip]
        D = F - R
        C = np.sqrt((D ** 2).sum(axis=(1, 2)))
        ok = C > 1e-12
        if ok.any():
            project(np.where(ok, C, 0.0), D / np.where(ok, C, 1.0)[:, None, None], 1.0 / (2 * self.mu_t[ids] * V))
        # Volume: C = J − 1 (recompute F after the deviatoric update)
        Ds = (self.x[T[:, 1:]] - self.x[T[:, :1]]).transpose(0, 2, 1)
        F = Ds @ self.Dm_inv[ids]
        f0, f1, f2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        J = np.einsum("bi,bi->b", f0, np.cross(f1, f2))
        cof = np.stack([np.cross(f1, f2), np.cross(f2, f0), np.cross(f0, f1)], axis=2)
        project(J - 1.0, cof, 1.0 / (self.lam_t[ids] * V))

    def _solve_tris(self, ids, inv_dt2):
        T = self.tris[ids]
        x = self.x
        A = self.area[ids] * self.thick_m[ids]                        # Volume of the membrane element
        w = self.w[T]
        Tinv = self.Tm_inv[ids]
        TT = Tinv.transpose(0, 2, 1)

        def F_now():
            Ds = np.stack([x[T[:, 1]] - x[T[:, 0]], x[T[:, 2]] - x[T[:, 0]]], axis=2)   # (b, 3, 2)
            return Ds @ Tinv

        def project(C, dCdF, compliance, active=None):
            g = dCdF @ TT                                              # (b, 3, 2)
            g0 = -g.sum(axis=2)
            grads = np.concatenate([g0[:, :, None], g], axis=2)        # (b, 3, 3)
            denom = (w[:, None, :] * grads ** 2).sum(axis=(1, 2)) + compliance * inv_dt2
            dl = -C / np.maximum(denom, 1e-30)
            if active is not None:
                dl = np.where(active, dl, 0.0)
            corr = (w[:, None, :] * grads * dl[:, None, None]).transpose(0, 2, 1)
            np.add.at(self.x, T.ravel(), corr.reshape(-1, 3))

        F = F_now()
        U, _, Vt = np.linalg.svd(F, full_matrices=False)               # (b,3,2), (b,2,2)
        D = F - U @ Vt
        C = np.sqrt((D ** 2).sum(axis=(1, 2)))
        ok = C > 1e-12
        if ok.any():
            project(np.where(ok, C, 0.0), D / np.where(ok, C, 1.0)[:, None, None], 1.0 / (2 * self.mu_m[ids] * A))
        # Area term: C = a/a₀ − 1 (a = |f0 × f1| is the area ratio)
        F = F_now()
        c = np.cross(F[:, :, 0], F[:, :, 1])
        a = np.linalg.norm(c, axis=1)
        n = c / np.maximum(a, 1e-12)[:, None]
        dA = np.stack([np.cross(F[:, :, 1], n), np.cross(n, F[:, :, 0])], axis=2)
        project(a - 1.0, dA, 1.0 / (self.lam_m[ids] * A))
        # Fibres (tension only)
        kf = self.kf_m[ids]
        has = kf > 0
        if has.any():
            F = F_now()
            a0 = self.fibre_m[ids]
            fa = np.einsum("bij,bj->bi", F, a0)
            s = np.linalg.norm(fa, axis=1)
            dF = (fa / np.maximum(s, 1e-12)[:, None])[:, :, None] * a0[:, None, :]
            project(s - 1.0, dF, 1.0 / (np.where(has, kf, 1.0) * A), active=has & (s > 1.0))

    # ------------------------------------------------------------------ stepping
    def step(self, dt: float, targets: np.ndarray | None = None, floor=None):
        """One substep. `targets`: (n, 3) prescribed positions for kinematic vertices
        (NaN rows keep their current position). `floor`: callable(xy) -> min z."""
        free = ~self.kinematic
        x_prev = self.x.copy()
        self.v[free] += dt * np.asarray(self.gravity)
        if self.damping:
            self.v *= max(0.0, 1.0 - self.damping * dt)
        self.x[free] += dt * self.v[free]
        if targets is not None:
            have = self.kinematic & np.isfinite(targets[:, 0])
            self.x[have] = targets[have]
        inv_dt2 = 1.0 / (dt * dt)
        for ids in self.tet_colours:
            self._solve_tets(ids, inv_dt2)
        for ids in self.tri_colours:
            self._solve_tris(ids, inv_dt2)
        if floor is not None and self.floor_ids is not None:
            ids = self.floor_ids
            zmin = floor(self.x[ids, :2])
            self.x[ids, 2] = np.maximum(self.x[ids, 2], zmin)
        self.v = (self.x - x_prev) / dt
        self.v[self.kinematic] = 0.0

    def tet_volume_ratio(self) -> np.ndarray:
        D = (self.x[self.tets[:, 1:]] - self.x[self.tets[:, :1]]).transpose(0, 2, 1)
        return np.linalg.det(D) / 6.0 / self.vol


def box_tets(nx: int, ny: int, nz: int, size=(1.0, 1.0, 1.0)):
    """A box meshed with 6 positively oriented tetrahedra per cell (Kuhn subdivision)."""
    xs = [np.linspace(0, s, k + 1) for s, k in zip(size, (nx, ny, nz))]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
    idx = np.arange(len(verts)).reshape(nx + 1, ny + 1, nz + 1)
    # Kuhn paths: 0 -> a -> a|b -> 7 for each ordering of the axes
    paths = []
    for first in (4, 2, 1):
        for second in (4, 2, 1):
            if second != first:
                paths.append((first, first | second))
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [idx[i + a, j + b, k + d] for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                for p1, p2 in paths:
                    tets.append([c[0], c[p1], c[p2], c[7]])
    tets = np.array(tets)
    # Orient positively
    D = (verts[tets[:, 1:]] - verts[tets[:, :1]]).transpose(0, 2, 1)
    neg = np.linalg.det(D) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return verts, tets
