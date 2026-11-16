"""Simulation session for the TissueLab Blender add-on: the body wall (surgvid body-wall) solved
live with tissuelab.tissue_pd while handles are dragged. Pure NumPy/SciPy (no bpy), so it is
testable outside Blender.

Coordinates are the wall file's: metres, +X patient left, +Y cranial, +Z anterior (Blender's
world, with the patient supine: Z up).
"""
from __future__ import annotations

import json

import numpy as np

try:                                       # Packaged add-on: a copy sits next to this file
    from . import tissue_pd as PD
except ImportError:                        # Developing from the repository
    from tissuelab import tissue_pd as PD


class Handle:
    def __init__(self, name: str, ids: np.ndarray, rest: np.ndarray, matrix0: np.ndarray):
        self.name, self.ids, self.rest = name, ids, rest
        self.force_n = 0.0
        self.inv0 = np.linalg.inv(matrix0)

    def targets(self, matrix: np.ndarray) -> np.ndarray:
        """The grabbed points moved rigidly with the handle since it was placed."""
        M = matrix @ self.inv0
        return self.rest @ M[:3, :3].T + M[:3, 3]


class Session:
    def __init__(self, path: str, handle_stiffness=1000.0, skin_stretch_limit=1.2,
                 skin_stiffness=0.0, max_stretch=1.5, tethering=False,
                 tether_modulus=10000.0, tether_slack_fraction=0.1):
        # ASSUMPTION: interaction spring, not a measured clamp material. Total N/m per handle.
        if not np.isfinite(handle_stiffness) or handle_stiffness <= 0:
            raise ValueError("handle stiffness must be positive and finite")
        if not np.isfinite(skin_stiffness) or skin_stiffness < 0 or not np.isfinite(skin_stretch_limit) or skin_stretch_limit < 1:
            raise ValueError("skin stiffness must be nonnegative and stretch limit >= 1")
        # ASSUMPTION: adjustable interaction ceiling, not calibrated failure strain.
        if not np.isfinite(max_stretch) or max_stretch <= 1:
            raise ValueError("max_stretch must be finite and greater than 1")
        if not np.isfinite(tether_modulus) or tether_modulus < 0:
            raise ValueError("tether modulus must be finite and nonnegative")
        if not np.isfinite(tether_slack_fraction) or tether_slack_fraction < 0:
            raise ValueError("tether slack fraction must be finite and nonnegative")
        self.tethering = tethering
        self.tether_modulus, self.tether_slack_fraction = tether_modulus, tether_slack_fraction
        self.max_stretch = max_stretch
        self.limit_reached = False
        self.handle_stiffness = handle_stiffness
        self.skin_stretch_limit, self.skin_stiffness = skin_stretch_limit, skin_stiffness
        d = np.load(path)
        # Versionless SurgVid files are the supported legacy v0 layout.
        version = np.asarray(d["format_version"]) if "format_version" in d else np.array([0, 0])
        if version.shape != (2,) or version.dtype.kind not in "iu" or tuple(version) not in ((0, 0), (1, 0)):
            raise ValueError(f"Unsupported body-wall format {version.tolist()}; update TissueLab")
        self.format_version = tuple(int(v) for v in version)
        self.path = path
        self.info = json.loads(str(np.asarray(d["info"]).ravel()[0]))
        self.x0 = d["x"].astype(float)
        self.x = self.x0.copy()
        self.tets = d["tets"].astype(np.int64)
        self.tissue_names = [str(t) for t in d["tissue_names"]]
        self.tet_tissue = d["tet_tissue"].astype(int)
        self.role_names = [str(r) for r in d["role_names"]]
        self.tris, self.tri_role, self.tri_tet = d["tris"], d["tri_role"], d["tri_tet"]
        self.mech = {k: d[k] for k in ("mu_t", "lam_t", "mem_tris", "mu_m", "lam_m", "th_m", "fib_m", "kf_m",
                                       "fixed")}
        self.surfaces = {}
        for name in (str(s) for s in d["surface_names"]):
            self.surfaces[name] = {k: d[f"{name}__{k}"] for k in ("verts", "faces", "tet", "w", "offset")}
        # Old wall files remain usable. Empty/missing connections never fabricate anatomy.
        self.tether_tets = d["tether_tets"] if "tether_tets" in d else np.empty((0, 2), int)
        self.tether_weights = d["tether_weights"] if "tether_weights" in d else np.empty((0, 2, 4))
        self.tether_area = d["tether_area_m2"] if "tether_area_m2" in d else np.empty(0)
        self.tether_file_present = "tether_tets" in d
        self.handles: list[Handle] = []
        self.solver = None
        self.last_move = 0.0
        Dm = (self.x0[self.tets[:, 1:]] - self.x0[self.tets[:, :1]]).transpose(0, 2, 1)
        self._Dm_inv = np.linalg.inv(Dm)

    # ------------------------------------------------------------------ handles
    def grab(self, centre: np.ndarray, radius: float) -> np.ndarray:
        """Free vertices of the surface nearest `centre` (skin from outside; peritoneum or
        pleura from inside) within `radius`: an instrument holds a surface, and the tissue under
        it follows elastically. A rigidly held ball of tissue would tear the fat at its rim."""
        d = np.linalg.norm(self.x - centre, axis=1)
        free = ~self.mech["fixed"]
        best = None
        for r in ("skin", "peritoneum", "pleura"):
            if r not in self.role_names:
                continue
            ids = np.unique(self.tris[self.tri_role == self.role_names.index(r)])
            ids = ids[(d[ids] <= radius) & free[ids]]
            if len(ids) and (best is None or d[ids].min() < d[best].min()):
                best = ids
        return best if best is not None else np.zeros(0, np.int64)

    def add_handle(self, name: str, centre: np.ndarray, radius: float, matrix: np.ndarray) -> int:
        ids = self.grab(np.asarray(centre, float), radius)
        taken = set(int(i) for h in self.handles for i in h.ids)
        ids = np.array([i for i in ids if int(i) not in taken], np.int64)
        if len(ids):
            self.handles.append(Handle(name, ids, self.x[ids].copy(), np.asarray(matrix, float)))
            self.solver = None
        return len(ids)

    def remove_handles(self, names=None):
        self.handles = [h for h in self.handles if names is not None and h.name not in names]
        self.solver = None

    def reset(self):
        self.x = self.x0.copy()
        self.handles = []
        self.solver = None
        self.limit_reached = False

    # ------------------------------------------------------------------ solving
    def _build(self):
        fixed = self.mech["fixed"].copy()
        ids = np.concatenate([h.ids for h in self.handles]) if self.handles else np.zeros(0, int)
        weights = np.concatenate([np.full(len(h.ids), self.handle_stiffness / len(h.ids))
                                  for h in self.handles]) if self.handles else np.zeros(0)
        m = self.mech
        # Match membrane triangles to their boundary role without assuming file ordering.
        skin_faces = {tuple(sorted(t)) for t in self.tris[self.tri_role == self.role_names.index("skin")]}
        stiffness = np.array([self.skin_stiffness if tuple(sorted(t)) in skin_faces else 0.0
                              for t in m["mem_tris"]])
        tether_args = {}
        if self.tethering and len(self.tether_tets):
            endpoints = self.tether_positions(rest=True)
            length = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1)
            if not np.isfinite(length).all() or (length <= 1e-8).any():
                raise ValueError("invalid tether rest lengths")
            tether_args = dict(tether_nodes=self.tets[self.tether_tets].reshape(-1, 8),
                               tether_coefficients=(self.tether_weights * np.array([-1., 1.])[None, :, None]).reshape(-1, 8),
                               tether_stiffness=self.tether_modulus * self.tether_area / length,
                               tether_slack=length * (1 + self.tether_slack_fraction))
        self.solver = PD.Solver(self.x0, self.tets, m["mu_t"], m["lam_t"], m["mem_tris"].astype(np.int64),
                                m["mu_m"], m["lam_m"], m["th_m"], m["fib_m"], m["kf_m"], fixed=fixed, fast=True,
                                attachment_ids=ids, attachment_weights=weights,
                                stretch_limit_m=self.skin_stretch_limit, stretch_stiffness_m=stiffness, **tether_args)

    def step(self, matrices: dict, iters: int = 3) -> float:
        """Advance toward equilibrium with each handle at its current transform
        (`matrices`: name -> 4x4 world matrix). Returns the largest vertex move (m)."""
        # Deleting a handle releases its tissue; no stale spring remains in the matrix.
        gone = [h.name for h in self.handles if h.name not in matrices]
        if gone:
            self.remove_handles(gone)
        if self.solver is None:
            self._build()
        targets = np.concatenate([h.targets(np.asarray(matrices[h.name], float))
                                  for h in self.handles]) if self.handles else np.empty((0, 3))
        before = self.x.copy()
        self.x, _, _ = self.solver.solve(self.x, attachment_targets=targets,
                                                      iters=iters, tol=1e-7, rho=None,
                                                      max_stretch=self.max_stretch)
        # The last internal iteration may be zero at a hard stop even though earlier
        # iterations moved. Return the whole update so Blender displays the accepted shape.
        self.last_move = float(np.linalg.norm(self.x - before, axis=1).max())
        self.limit_reached = self.solver.limit_reached
        offset = 0
        for h in self.handles:
            force = self.handle_stiffness / len(h.ids) * (targets[offset:offset + len(h.ids)] - self.x[h.ids])
            h.force_n = float(np.linalg.norm(force.sum(axis=0)))
            offset += len(h.ids)
        return self.last_move

    # ------------------------------------------------------------------ outputs
    def tether_positions(self, rest=False):
        """Embedded endpoints (connections, 2, 3), also usable for future fascial layers."""
        x = self.x0 if rest else self.x
        return np.einsum("nei,neij->nej", self.tether_weights, x[self.tets[self.tether_tets]])

    def surface_positions(self, name: str) -> np.ndarray:
        """A render surface carried by the tissue: barycentric position in its tetrahedron
        plus its rest offset turned by the tetrahedron's rotation."""
        s = self.surfaces[name]
        T = self.tets[s["tet"]]
        p = np.einsum("ni,nij->nj", s["w"], self.x[T])
        if np.abs(s["offset"]).max() > 0:
            used, inv = np.unique(s["tet"], return_inverse=True)
            Ds = (self.x[self.tets[used, 1:]] - self.x[self.tets[used, :1]]).transpose(0, 2, 1)
            R = PD._project_rotation_fast(Ds @ self._Dm_inv[used])
            p += np.einsum("nij,nj->ni", R[inv], s["offset"])
        return p

    def scan_edge(self):
        """The wall's cut faces where the scan ends (vertex ids into x, tissue per face)."""
        k = self.role_names.index("scan_edge")
        sel = self.tri_role == k
        return self.tris[sel], self.tet_tissue[self.tri_tet[sel]]

    def stats(self) -> dict:
        Ds = (self.x[self.tets[:, 1:]] - self.x[self.tets[:, :1]]).transpose(0, 2, 1)
        J = np.linalg.det(Ds @ self._Dm_inv)
        return {"inverted": int((J <= 0).sum()), "volume_ratio_min": float(J.min()),
                "max_displacement_m": float(np.linalg.norm(self.x - self.x0, axis=1).max()),
                "last_move_m": float(self.last_move)}
