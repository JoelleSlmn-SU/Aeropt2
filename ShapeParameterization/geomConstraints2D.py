"""
geomConstraints2D.py -- geometric constraint metrics of a 2D edge-mode design, before any CFD.

Use in a constraint line exactly like a CFD metric (bo_settings / objective editor):

    geom_envelope >= 0          ADODG Case 1: upper surface may only rise, lower only drop,
                                at every chord station (z_u >= z_u0 and z_l <= z_l0)
    geom_thickness_ratio >= 1   thickness t(x)/t0(x) >= 1 at every chord station
    geom_area_ratio >= 1        enclosed area / baseline area (closed T loop only)
    geom_normal_disp >= 0       every T node moves outwards (or not at all)

Station-wise metrics return one value per station; the constraint holds at all of
them. Stations: `n_stations` cosine-spaced chord positions strictly between the
leading and trailing edge of the BASELINE wall (x_min, x_max of the T nodes); the
thickness at a station is the vertical distance between the highest and lowest
crossing of the morphed wall with the line x = station.

"Chord" is the x direction of the mesh: an aerofoil at incidence in the mesh is
measured along x, not along its chord line (the benchmark cases are at alpha = 0
in the mesh and set incidence in the solver).

The morphed wall is xy + (phi c) n, exactly what Morph2D puts on the T nodes, so
these are the constraint values of the mesh the solver will see.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

METRICS_2D = ("geom_envelope", "geom_thickness_ratio", "geom_area_ratio", "geom_normal_disp")


def _segments(modes):
    """Index pairs (a, b) of consecutive T nodes along every chain (closed chains wrap)."""
    A, B = [], []
    for ch in modes.chains:
        idx = np.arange(int(ch["start"]), int(ch["stop"]))
        A.append(idx[:-1]); B.append(idx[1:])
        if ch.get("closed"):
            A.append(idx[-1:]); B.append(idx[:1])
    return np.concatenate(A), np.concatenate(B)


def _crossings(xy, a, b, stations):
    """Highest and lowest z where the polyline crosses each vertical line (n_st,)."""
    xa, xb = xy[a, 0][:, None], xy[b, 0][:, None]
    za, zb = xy[a, 1][:, None], xy[b, 1][:, None]
    s = stations[None, :]
    lo, hi = np.minimum(xa, xb), np.maximum(xa, xb)
    hit = (s >= lo) & (s <= hi) & (hi > lo)
    t = np.where(hit, (s - xa) / np.where(hi > lo, xb - xa, 1.0), 0.0)
    z = za + t * (zb - za)
    up = np.where(hit, z, -np.inf).max(axis=0)
    dn = np.where(hit, z, np.inf).min(axis=0)
    return up, dn


def _area(xy, modes):
    tot = 0.0
    for ch in modes.chains:
        if not ch.get("closed"):
            return np.nan
        p = xy[int(ch["start"]):int(ch["stop"])]
        tot += 0.5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
    return tot


class WallGeometry2D:
    """Callable for BayesianOptimiser(geom_func=...): X (m, k) -> {metric: (m, p)}."""

    def __init__(self, modes, metrics: Optional[Iterable[str]] = None, n_stations: int = 101,
                 edge_margin: float = 0.0):
        self.modes = modes
        self.metrics = list(metrics) if metrics is not None else list(METRICS_2D)
        bad = [m for m in self.metrics if m not in METRICS_2D]
        if bad:
            raise ValueError(f"Unknown 2D geometric metric(s) {bad}; available: {list(METRICS_2D)}")
        self.xy0 = np.asarray(modes.xy, float)
        self.a, self.b = _segments(modes)
        x0, x1 = float(self.xy0[:, 0].min()), float(self.xy0[:, 0].max())
        t = np.linspace(0.0, 1.0, int(n_stations) + 2)[1:-1]
        st = x0 + (x1 - x0) * 0.5 * (1.0 - np.cos(np.pi * t))
        span = x1 - x0
        self.stations = st[(st > x0 + edge_margin * span) & (st < x1 - edge_margin * span)]
        self.up0, self.dn0 = _crossings(self.xy0, self.a, self.b, self.stations)
        ok = np.isfinite(self.up0) & np.isfinite(self.dn0)
        if not np.all(ok):
            self.stations = self.stations[ok]
            self.up0, self.dn0 = self.up0[ok], self.dn0[ok]
        self.t0 = self.up0 - self.dn0
        self.area0 = _area(self.xy0, modes)
        # outward = away from the enclosed region (sign of the loop orientation)
        sgn = 1.0
        if np.isfinite(self.area0) and self.area0 != 0.0:
            # for a counter-clockwise loop the outward normal is the tangent turned clockwise
            tang = np.gradient(self.xy0, axis=0)
            outward = np.column_stack([tang[:, 1], -tang[:, 0]]) * (1.0 if self.area0 > 0 else -1.0)
            sgn = np.sign(np.sum(np.einsum("ij,ij->i", outward, modes.normals)))
        self.out_sign = float(sgn) if sgn != 0 else 1.0
        if "geom_area_ratio" in self.metrics and not np.isfinite(self.area0):
            raise ValueError("geom_area_ratio needs the T edges to form closed loop(s).")

    def wall(self, x) -> np.ndarray:
        return self.xy0 + np.asarray(self.modes.displacement(np.asarray(x, float)), float)[:, :2]

    def __call__(self, X) -> Dict[str, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, float))
        out = {m: [] for m in self.metrics}
        span = float(np.ptp(self.xy0[:, 0]))
        for x in X:
            xy = self.wall(x)
            need_st = any(m in self.metrics for m in ("geom_envelope", "geom_thickness_ratio"))
            if need_st:
                up, dn = _crossings(xy, self.a, self.b, self.stations)
                # the wall no longer reaches this station (e.g. the leading edge moved
                # back past it): the section has lost thickness there -> a finite,
                # clearly violating value instead of +-inf
                gone = ~(np.isfinite(up) & np.isfinite(dn))
                up = np.where(gone, self.up0 - span, up)
                dn = np.where(gone, self.dn0 + span, dn)
            if "geom_envelope" in self.metrics:
                out["geom_envelope"].append(np.concatenate([up - self.up0, self.dn0 - dn]))
            if "geom_thickness_ratio" in self.metrics:
                out["geom_thickness_ratio"].append((up - dn) / np.maximum(self.t0, 1e-300))
            if "geom_area_ratio" in self.metrics:
                out["geom_area_ratio"].append([_area(xy, self.modes) / self.area0])
            if "geom_normal_disp" in self.metrics:
                out["geom_normal_disp"].append(self.out_sign * (self.modes.phi @ x))
        return {m: np.asarray(v, float) for m, v in out.items()}
