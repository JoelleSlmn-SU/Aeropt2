from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import statistics as stats
import math
import numpy as np
from collections import defaultdict

class DatFile:
    def __init__(self, path: str):
        self.path = Path(path)
        self.lines: List[str] = self.path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        self.idx = self._find_sections(self.lines)

    def clean(
        self,
        tol: float = 1e-3,            # remove "point curves" (first↔last nearly same)
        mode: str = "axis",           # 'axis' or 'euclid'
        dedupe: bool = True,          # dedupe identical-endpoint curves across surfaces
        sig_decimals: int = 6,        # rounding for endpoint signature (dedupe)
        short_tol: float = 0.0,       # absolute drop threshold for polyline length (0 disables)
        rel_factor: float = 0.0,      # relative drop: length < rel_factor * median(surface) (0 disables)
        # NEW: targeted re-linking
        relink_surfaces: Optional[Dict[int, List[int]]] = None,  # {target_surface: [preferred_neighbours]}
        endpoint_tol: float = 1e-3,   # endpoint proximity for matching (after re-link)
        len_rel_tol: float = 0.05,    # relative length tolerance for matching (±5%)
        out_path: Optional[str] = None
    ) -> Dict:
        """
        Cleans the DAT by:
          1) removing degenerate "point-curves",
          2) (optional) targeted re-linking of curves on specified surfaces to originals on neighbours,
          3) (optional) deduplicating curves with identical endpoints (global),
          4) (optional) removing very short curves (absolute and/or relative per-surface),
          5) renumbering curves and rewriting references.
        """
        
        
        def _cumlens(P):
            P = np.asarray(P, float)
            d = np.sqrt(((P[1:] - P[:-1])**2).sum(axis=1))
            s = np.concatenate([[0.0], np.cumsum(d)])
            return s, s[-1]

        def _resample_poly(P, n=25):
            """Arc-length resample to n points (including endpoints)."""
            P = np.asarray(P, float)
            s, L = _cumlens(P)
            if L <= 0.0 or len(P) == 1:
                return np.repeat(P[:1], n, axis=0)
            t = np.linspace(0.0, L, n)
            # piecewise-linear interpolation
            idx = np.searchsorted(s, t, side="right") - 1
            idx = np.clip(idx, 0, len(P)-2)
            s0, s1 = s[idx], s[idx+1]
            w = np.divide(t - s0, np.clip(s1 - s0, 1e-12, None))
            return P[idx] * (1.0 - w[:,None]) + P[idx+1] * w[:,None]

        def _shape_distance(P, Q, n=25):
            """Orientation-agnostic max deviation after arc-length resampling."""
            Pr = _resample_poly(P, n)
            Qr = _resample_poly(Q, n)
            d1 = np.linalg.norm(Pr - Qr, axis=1).max()
            d2 = np.linalg.norm(Pr - Qr[::-1], axis=1).max()
            return min(d1, d2)

        def _poly_len(P):
            P = np.asarray(P, float)
            if len(P) < 2: return 0.0
            return np.sqrt(((P[1:] - P[:-1])**2).sum(axis=1)).sum()
        
        # --- header counts ---
        hdr_line = self.idx["Geometry"] + 1
        ncur_old, nsurf_old = self._parse_ints(self.lines[hdr_line])[:2]

        # --- parse curves & detect point-curves ---
        curves, removed_point = self._parse_curves_with_detection(
            self.lines, self.idx["Curves"], self.idx["Surfaces1"], tol, mode
        )

        # numeric points for all curves
        all_pts = {c["orig_no"]: [self._xyz(ln) for ln in c["points"]] for c in curves}
        poly_len = {cid: self._polyline_len(all_pts[cid]) for cid in all_pts}
        ends = {cid: (all_pts[cid][0], all_pts[cid][-1]) for cid in all_pts}

        # --- parse final surfaces & segments early ---
        s4_blocks = self._parse_surfaces4(self.lines, self.idx["Surfaces4"])
        seg_entries, seg_end_idx = self._parse_segments(self.lines, self.idx["Segments"], self.idx["Surfaces4"])

        # surface_id -> set(curve_ids)
        surf_to_curves: Dict[int, Set[int]] = {}
        for blk in s4_blocks:
            sid = blk["header_vals"][0]
            surf_to_curves[sid] = set(blk["ids"])

        # --- initial canonicalization map (identity) ---
        canon_map: Dict[int, int] = {c["orig_no"]: c["orig_no"] for c in curves}

        # --- (2) targeted re-link pass (surface-specific) ---
        if relink_surfaces:
            # build inverse: curve_id -> set(surface_ids that use it)
            curve_to_surfs: Dict[int, Set[int]] = defaultdict(set)
            for sid, ids in surf_to_curves.items():
                for cid in ids:
                    curve_to_surfs[cid].add(sid)

            # helper: try match a curve to an existing "original" on neighbour surfaces
            def try_match(curve_id: int, candidate_surfaces: Optional[Set[int]]) -> Optional[int]:
                p0, p1 = ends[curve_id]
                L = poly_len[curve_id]
                best = None
                best_eps = float('inf')

                # iterate other curves that are not from this curve's own surface
                for cand_id, (q0, q1) in ends.items():
                    if cand_id == curve_id:
                        continue
                    # restrict to neighbour surfaces if provided
                    if candidate_surfaces is not None:
                        if curve_id in curve_to_surfs and len(curve_to_surfs[curve_id]) > 0:
                            # any of the curve's current surfaces
                            home_surfs = curve_to_surfs[curve_id]
                            # candidate must be used by at least one of the preferred neighbour surfaces
                            if not (curve_to_surfs[cand_id] & candidate_surfaces):
                                continue
                    # length gate
                    Lc = poly_len[cand_id]
                    if Lc == 0.0:
                        continue
                    if abs(L - Lc) / Lc > len_rel_tol:
                        continue
                    # endpoint gate (orientation-agnostic)
                    e1 = self._l2(p0, q0) + self._l2(p1, q1)
                    e2 = self._l2(p0, q1) + self._l2(p1, q0)
                    eps = min(e1, e2)
                    if eps/2.0 <= endpoint_tol and eps < best_eps:
                        best_eps = eps
                        best = cand_id
                return best

            # apply re-link for each target surface
            for target_sid, neigh_list in relink_surfaces.items():
                if target_sid not in surf_to_curves:
                    continue
                target_curve_ids = list(surf_to_curves[target_sid])
                preferred_surfs = set(neigh_list) if neigh_list else None

                for cid in target_curve_ids:
                    # already degenerate? skip, will be removed
                    if cid in removed_point:
                        continue
                    match = try_match(cid, preferred_surfs)
                    if match is not None:
                        canon_map[cid] = match  # remap this curve to the original candidate

        # --- (3) build global endpoint-based dedupe bins (after targeted re-link) ---
        dedup_groups = []
        canon_map = {cid: canon_map.get(cid, cid) for cid in ends}  # keep existing relinks
        if dedupe:
            # 1) coarse bin by rounded endpoints to avoid O(N^2)
            edge_bins = defaultdict(list)
            for cid, (p0, p1) in ends.items():
                ke = self._edge_key(p0, p1, sig_decimals)  # your existing rounded, orientation-agnostic key
                edge_bins[ke].append(cid)

            # 2) inside each bin, verify length + interior shape before merging
            N_SAMPLE = 31
            for key, ids in edge_bins.items():
                if len(ids) < 2:
                    continue
                # sort for stable canonical choice
                ids_sorted = sorted(ids)

                # build connected components of "truly identical curves"
                # (use strict tests to avoid merging different paths)
                groups = []
                used = set()
                for i, a in enumerate(ids_sorted):
                    if a in used:
                        continue
                    group = [a]
                    Pa = all_pts[a]
                    La = poly_len[a]
                    for b in ids_sorted[i+1:]:
                        if b in used:
                            continue
                        Pb = all_pts[b]
                        Lb = poly_len[b]

                        # length gate
                        if Lb == 0.0:
                            continue
                        if abs(La - Lb) / max(La, Lb) > len_rel_tol:
                            continue

                        # interior-shape gate
                        dev = _shape_distance(Pa, Pb, n=N_SAMPLE)
                        if dev <= endpoint_tol:   # reuse endpoint_tol or set a separate shape_tol
                            group.append(b)
                            used.add(b)

                    used.update(group)
                    if len(group) > 1:
                        groups.append(sorted(group))

                # 3) assign canonical per verified group only
                for g in groups:
                    can = g[0]  # policy: lowest ID
                    dedup_groups.append((key, g))
                    for dup in g:
                        prev_can = canon_map.get(dup, dup)
                        canon_map[dup] = min(prev_can, can)

        # --- (4) decide small-curve removals (after canonicalization for fairness) ---
        removed_small = set()
        if short_tol > 0.0 or rel_factor > 0.0:
            for blk in s4_blocks:
                chain_canon = []
                for cid in blk["ids"]:
                    c = canon_map.get(cid, cid)
                    if c in all_pts:
                        chain_canon.append(c)
                if not chain_canon:
                    continue
                uniq = list(dict.fromkeys(chain_canon))
                lens = [poly_len[c] for c in uniq]
                if not lens:
                    continue
                med = stats.median(lens) if len(lens) >= 3 else (sum(lens)/len(lens))
                for c in uniq:
                    L = poly_len[c]
                    if (short_tol > 0.0 and L <= short_tol) or (rel_factor > 0.0 and med > 0 and L < rel_factor*med):
                        removed_small.add(c)

        # --- total removal set ---
        removed_point_canon = {canon_map.get(cid, cid) for cid in removed_point}
        remove_all_canon = removed_point_canon | removed_small

        # --- build final mapping old_id -> new_sequential_id (only for referenced, kept canon) ---
        # collect referenced canonical IDs from final surfaces & segments to avoid emitting unused curves
        referenced_canon: Set[int] = set()
        def map_old_to_canon(old_id: int) -> Optional[int]:
            can = canon_map.get(old_id, old_id)
            return None if can in remove_all_canon else can

        # gather from Surfaces
        for blk in s4_blocks:
            for cid in blk["ids"]:
                m = map_old_to_canon(cid)
                if m is not None:
                    referenced_canon.add(m)
        # gather from Segments
        for a, b, _ in seg_entries:
            ma = map_old_to_canon(a)
            if ma is not None:
                referenced_canon.add(ma)
            mb = map_old_to_canon(b)
            if mb is not None:
                referenced_canon.add(mb)

        # renumber referenced canon only
        keep_canon = sorted(referenced_canon)
        renumber_map: Dict[int,int] = {cid: new_id for new_id, cid in enumerate(keep_canon, start=1)}

        def map_old_to_new(old_id: int) -> Optional[int]:
            can = canon_map.get(old_id, old_id)
            if can in remove_all_canon or can not in renumber_map:
                return None
            return renumber_map[can]

        # --- rewrite sections ---
        new_curves_lines = self._rewrite_curves_block_with_canonical(curves, canon_map, remove_all_canon, renumber_map)
        new_segments_lines = self._rewrite_segments_mapped(seg_entries, map_old_to_new)
        new_s4_lines = self._rewrite_surfaces4_mapped(s4_blocks, map_old_to_new)

        ncur_new = len(keep_canon)

        # --- assemble output ---
        out = []
        out.extend(self.lines[:hdr_line])
        out.append(f"{ncur_new:8d}{nsurf_old:8d}\n")
        out.extend(self.lines[hdr_line+1 : self.idx["Curves"]])

        out.extend(new_curves_lines)  # Curves

        out.extend(self.lines[self.idx["Surfaces1"] : self.idx["MeshGen"]])  # mid Surfaces unchanged

        out.extend(self.lines[self.idx["MeshGen"] : self.idx["MeshGen"]+1])  # MeshGen title
        mg_counts = self._parse_ints(self.lines[self.idx["MeshGen"]+1])
        flags = mg_counts[2:] if len(mg_counts) > 2 else []
        out.append(f"{ncur_new:8d}{nsurf_old:8d}" + "".join(f"{x:8d}" for x in flags) + "\n")

        out.extend(new_segments_lines)  # Segment Curves

        out.extend(self.lines[seg_end_idx : self.idx["Surfaces4"]])

        out.extend(new_s4_lines)  # final Surfaces

        if out_path:
            Path(out_path).write_text("".join(out), encoding="utf-8", errors="ignore")

        dedup_pairs = sum(len(ids)-1 for _, ids in dedup_groups) if dedupe else 0
        return {
            "curves_before": ncur_old,
            "curves_after": ncur_new,
            "surfaces": nsurf_old,
            "removed_point_curves": len(removed_point_canon),
            "removed_small_curves": len(removed_small),
            "dedup_dropped": dedup_pairs,
            "relinked_surfaces": sorted(list(relink_surfaces.keys())) if relink_surfaces else [],
        }

    # ---------------- internals ----------------
    
    @staticmethod
    def _parse_ints(line: str) -> List[int]:
        return [int(x) for x in line.strip().split()]

    def _find_sections(self, lines: List[str]) -> Dict[str, int]:
        idx = {}
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s == "Geometry" and "Geometry" not in idx: idx["Geometry"] = i
            elif s == "Curves" and "Curves" not in idx:   idx["Curves"] = i
            elif s == "Surfaces" and "Surfaces1" not in idx: idx["Surfaces1"] = i
            elif s == "Mesh Generation" and "MeshGen" not in idx: idx["MeshGen"] = i
            elif s == "Segment Curves" and "Segments" not in idx: idx["Segments"] = i
            elif s == "Surfaces" and "Surfaces4" not in idx and "Surfaces1" in idx: idx["Surfaces4"] = i
        need = ("Geometry","Curves","Surfaces1","MeshGen","Segments","Surfaces4")
        miss = [k for k in need if k not in idx]
        if miss: raise ValueError(f"Missing sections: {miss}")
        return idx

    # ---- numeric helpers ----
    @staticmethod
    def _xyz(line: str) -> List[float]:
        t = line.strip().split()
        return [float(t[0]), float(t[1]), float(t[2])]

    @staticmethod
    def _l2(a: List[float], b: List[float]) -> float:
        dx, dy, dz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
        return math.hypot(dx, math.hypot(dy, dz))

    def _polyline_len(self, P: List[List[float]]) -> float:
        return sum(self._l2(P[i], P[i+1]) for i in range(len(P)-1))

    @staticmethod
    def _edge_key(p0, p1, decimals=6):
        """Orientation-agnostic key with rounding."""
        a = (round(p0[0], decimals), round(p0[1], decimals), round(p0[2], decimals))
        b = (round(p1[0], decimals), round(p1[1], decimals), round(p1[2], decimals))
        return tuple(sorted((a, b)))

    # ---- curves parsing ----
    def _parse_curves_with_detection(
        self, lines: List[str], start: int, stop: int, tol: float, mode: str
    ) -> Tuple[List[Dict], List[int]]:
        curves, removed = [], []
        i = start + 1
        while i < stop:
            if not lines[i].strip(): i += 1; continue
            hdr = self._parse_ints(lines[i])
            if len(hdr) < 2: break
            cid, tag = hdr[0], hdr[1]
            npts = self._parse_ints(lines[i+1])[0]
            pts = lines[i+2:i+2+npts]
            # detect "point curve"
            try:
                p0 = self._xyz(pts[0]); p1 = self._xyz(pts[-1])
                if mode == "euclid":
                    dx = p1[0]-p0[0]; dy = p1[1]-p0[1]; dz = p1[2]-p0[2]
                    is_point = math.sqrt(dx*dx + dy*dy + dz*dz) <= tol
                else:
                    dx, dy, dz = abs(p1[0]-p0[0]), abs(p1[1]-p0[1]), abs(p1[2]-p0[2])
                    is_point = math.sqrt(dx*dx + dy*dy + dz*dz) <= tol
            except Exception:
                is_point = False
            curves.append({"orig_no": cid, "tag": tag, "npts": npts, "points": pts, "degen": is_point})
            if is_point: removed.append(cid)
            i += 2 + npts
        return curves, removed

    # ---- rewrite helpers ----
    def _rewrite_curves_block_with_canonical(
        self,
        curves: List[Dict],
        canon_map: Dict[int,int],
        remove_all_canon: set,
        renumber_map: Dict[int,int]
    ) -> List[str]:
        """Emit Curves section for kept canonical curves only, renumbered."""
        out = [" Curves\n"]
        emitted_canon = set()
        for c in curves:
            old = c["orig_no"]
            can = canon_map.get(old, old)
            if can in remove_all_canon:
                continue
            if can not in renumber_map:
                continue
            if can in emitted_canon:
                continue
            new_id = renumber_map[can]
            out.append(f"{new_id:10d}{c['tag']:8d}\n")
            out.append(f"{c['npts']:10d}\n")
            out.extend(c["points"])
            emitted_canon.add(can)
        return out

    @staticmethod
    def _parse_segments(lines: List[str], seg_idx: int, nxt_idx: int) -> Tuple[List[Tuple[int,int,int]], int]:
        entries, i = [], seg_idx + 1
        while i < nxt_idx:
            s = lines[i].strip()
            if not s: i += 1; continue
            tok = s.split()
            ok = lambda t: t.replace("-", "").isdigit()
            if len(tok) >= 3 and ok(tok[0]) and ok(tok[1]) and ok(tok[2]):
                entries.append((int(tok[0]), int(tok[1]), int(tok[2]))); i += 1
            else: break
        return entries, i

    @staticmethod
    def _rewrite_segments_mapped(entries, mapper) -> List[str]:
        out = ["  Segment Curves\n"]
        for a, b, c in entries:
            na = mapper(a)
            if na is None: continue
            nb = mapper(b)
            if nb is None: nb = na
            out.append(f"{na:10d}{nb:10d}{c:10d}\n")
        return out

    @staticmethod
    def _parse_surfaces4(lines: List[str], s4_idx: int) -> List[Dict]:
        blocks = []
        i = s4_idx + 1
        n = len(lines)
        while i < n:
            if not lines[i].strip(): i += 1; continue
            tok = lines[i].strip().split()
            if len(tok) < 4: break
            try:
                header = [int(x) for x in tok[:4]]
            except ValueError:
                break
            i += 1
            cnt = int(lines[i].strip().split()[0]); i += 1
            ids = []
            while len(ids) < cnt and i < n:
                for t in lines[i].strip().split():
                    if len(ids) < cnt: ids.append(int(t))
                i += 1
            blocks.append({"header_vals": header, "ids": ids})
        return blocks
    
    def _parse_surfaces4_2(self, lines, start_idx: int) -> List[Dict]:
        """Parse the final Surfaces block, using the header count to avoid stopping early."""
        hdr_line = self.idx["Geometry"] + 1
        _, nsurf_expected = self._parse_ints(self.lines[hdr_line])[:2]

        blocks = []
        i = start_idx + 1  # skip "Surfaces"
        for _ in range(nsurf_expected):
            header_vals = self._parse_ints(lines[i]); i += 1
            nids = int(lines[i].strip()); i += 1
            ids = []
            while len(ids) < nids:
                ids.extend(self._parse_ints(lines[i]))
                i += 1
            blocks.append({"header_vals": header_vals, "ids": ids})
        return blocks

    @staticmethod
    def _rewrite_surfaces4_mapped(blocks: List[Dict], mapper) -> List[str]:
        out = [" Surfaces\n"]
        for blk in blocks:
            a,b,c,d = blk["header_vals"][:4]
            out.append(f"{a:10d}{b:8d}{c:8d}{d:8d}\n")
            # map ids, drop None, drop duplicates while preserving order
            new_ids = []
            seen = set()
            for old in blk["ids"]:
                m = mapper(old)
                if m is None:
                    continue
                if m in seen:
                    continue
                seen.add(m)
                new_ids.append(m)
            out.append(f"{len(new_ids):10d}\n")
            for i in range(0, len(new_ids), 8):
                out.append("".join(f"{x:10d}" for x in new_ids[i:i+8]) + "\n")
        return out

    def combine_curves(self, curve_ids, n_points=25, new_id=999, tag=1):
        """
        Combine multiple curves (by their IDs) into a single resampled curve.
        
        Args:
            curve_ids (list[int]): List of curve IDs to stitch together in order.
            n_points (int): Number of points in the resampled curve.
            new_id (int): Curve ID to assign to the new combined curve.
            tag (int): Tag (2nd integer in curve header).
        
        Returns:
            str: Curve block in DAT format as a string.
        """

        # ---- collect points from all specified curves ----
        all_pts = []
        for i, cid in enumerate(curve_ids):
            # Find the curve in self.lines
            start_idx = None
            for j, ln in enumerate(self.lines):
                toks = ln.strip().split()
                if len(toks) >= 2 and toks[0].isdigit():
                    if int(toks[0]) == cid:
                        start_idx = j
                        break
            if start_idx is None:
                raise ValueError(f"Curve {cid} not found in DAT file.")

            npts = int(self.lines[start_idx+1].strip())
            pts_block = []
            for k in range(npts):
                x, y, z = map(float, self.lines[start_idx+2+k].split())
                pts_block.append([x, y, z])

            # append but drop first point for all except the first curve
            if i == 0:
                all_pts.extend(pts_block)
            else:
                all_pts.extend(pts_block[1:])

        all_pts = np.array(all_pts, float)

        # ---- resample to n_points equally spaced along arc length ----
        def arc_length_resample(points, n):
            d = np.sqrt(((points[1:] - points[:-1])**2).sum(axis=1))
            s = np.concatenate([[0.0], np.cumsum(d)])
            t = np.linspace(0, s[-1], n)
            idx = np.searchsorted(s, t, side="right") - 1
            idx = np.clip(idx, 0, len(points)-2)
            s0, s1 = s[idx], s[idx+1]
            w = (t - s0) / np.clip(s1 - s0, 1e-12, None)
            return points[idx]*(1-w)[:,None] + points[idx+1]*w[:,None]

        resampled = arc_length_resample(all_pts, n_points)

        # ---- format back into DAT-style block ----
        out = []
        out.append(f"{new_id:10d}{tag:8d}\n")
        out.append(f"{n_points:10d}\n")
        for p in resampled:
            out.append(f"{p[0]:12.7E} {p[1]:12.7E} {p[2]:12.7E}\n")

        return "".join(out)
    
    def _index_sections(self):
        """
        Build index dictionary pointing to line numbers of key sections.
        Distinguishes the first Surfaces block (Surfaces1) and the last one (Surfaces4).
        """
        idx = {}
        for i, line in enumerate(self.lines):
            if not line.strip():
                continue
            key = line.strip().split()[0]
            if key == "Geometry":
                idx["Geometry"] = i
            elif key == "Curves":
                idx["Curves"] = i
            elif key == "Surfaces":
                if "Surfaces1" not in idx:
                    idx["Surfaces1"] = i  # first Surfaces
                idx["Surfaces4"] = i      # always overwritten, ends at last Surfaces
            elif key == "Mesh":
                idx["MeshGen"] = i
            elif key == "Segment":
                idx["Segments"] = i
        self.idx = idx
    
    def clean_manual(
        self,
        out_path: Optional[str] = None,
        remove_surfaces: Optional[List[int]] = None,
        remove_curves: Optional[List[int]] = None,
    ) -> Dict:
        """
        Manual cleanup:
        - Remove unreferenced curves (not used in Surfaces4 or Segments).
        - Remove user-specified curves (remove_curves).
        - Remove user-specified surfaces (remove_surfaces).
        - Renumber curves and surfaces consistently.
        - Rewrite headers, Curves, Surfaces1, Segments, Surfaces4.
        """
        remove_surfaces = set(remove_surfaces or [])
        remove_curves = set(remove_curves or [])

        # --- header counts (old) ---
        hdr_line = self.idx["Geometry"] + 1
        ncur_old, nsurf_old = self._parse_ints(self.lines[hdr_line])[:2]

        # --- parse curves ---
        curves, _ = self._parse_curves_with_detection(
            self.lines, self.idx["Curves"], self.idx["Surfaces1"], tol=0.0, mode="euclid"
        )
        curves_by_id = {c["orig_no"]: c for c in curves}

        # --- parse Surfaces1 (geometry surfaces) ---
        s1_start = self.idx["Surfaces1"]
        s1_end = self.idx["MeshGen"]
        s1_lines = self.lines[s1_start+1 : s1_end]
        s1_blocks = []
        i = 0
        while i < len(s1_lines):
            if not s1_lines[i].strip():
                i += 1; continue
            header = self._parse_ints(s1_lines[i]); i += 1
            dims = self._parse_ints(s1_lines[i]); i += 1
            nx, ny = dims[:2]
            npts = nx * ny
            coords = s1_lines[i:i+npts]; i += npts
            s1_blocks.append({"header_vals": header, "dims": (nx, ny), "coords": coords})

        # --- parse segments ---
        seg_entries, seg_end_idx = self._parse_segments(self.lines, self.idx["Segments"], self.idx["Surfaces4"])

        # --- parse Surfaces4 (connectivity) ---
        s4_blocks = self._parse_surfaces4(self.lines, self.idx["Surfaces4"])

        # --- remove requested surfaces ---
        s1_blocks = [blk for blk in s1_blocks if blk["header_vals"][0] not in remove_surfaces]
        s4_blocks = [blk for blk in s4_blocks if blk["header_vals"][0] not in remove_surfaces]

        # --- collect referenced curves ---
        referenced = set()
        for blk in s4_blocks:
            referenced.update(blk["ids"])
        for a, b, _ in seg_entries:
            referenced.add(a); referenced.add(b)

        # --- final keep set of curves ---
        keep_curves = [cid for cid in sorted(curves_by_id)
                    if cid in referenced and cid not in remove_curves]
        curve_new_id = {old: i+1 for i, old in enumerate(keep_curves)}

        # --- renumber surfaces ---
        keep_surfaces = s4_blocks
        surf_new_id = {blk["header_vals"][0]: i+1 for i, blk in enumerate(keep_surfaces)}

        # --- rewrite Curves ---
        new_curves_lines = [" Curves\n"]
        for old in keep_curves:
            c = curves_by_id[old]
            new_id = curve_new_id[old]
            new_curves_lines.append(f"{new_id:10d}{c['tag']:8d}\n")
            new_curves_lines.append(f"{c['npts']:10d}\n")
            new_curves_lines.extend(c["points"])

        # --- rewrite Surfaces1 ---
        new_s1_lines = [" Surfaces\n"]
        for blk in s1_blocks:
            sid_old, b = blk["header_vals"][:2]
            sid_new = surf_new_id.get(sid_old, None)
            if sid_new is None:
                continue
            nx, ny = blk["dims"]
            new_s1_lines.append(f"{sid_new:10d}{b:8d}\n")
            new_s1_lines.append(f"{nx:10d}{ny:8d}\n")
            new_s1_lines.extend(blk["coords"])

        # --- rewrite Segments ---
        new_segments_lines = ["  Segment Curves\n"]
        for a, b, cat in seg_entries:
            if a not in curve_new_id or b not in curve_new_id:
                continue
            na, nb = curve_new_id[a], curve_new_id[b]
            new_segments_lines.append(f"{na:10d}{nb:10d}{cat:10d}\n")

        # --- rewrite Surfaces4 ---
        new_s4_lines = [" Surfaces\n"]
        for blk in s4_blocks:
            sid_old, b, c, d = blk["header_vals"][:4]
            sid_new = surf_new_id[sid_old]
            ids = [curve_new_id[cid] for cid in blk["ids"] if cid in curve_new_id]
            new_s4_lines.append(f"{sid_new:10d}{sid_new:8d}{c:8d}{d:8d}\n")
            new_s4_lines.append(f"{len(ids):10d}\n")
            for i in range(0, len(ids), 8):
                new_s4_lines.append("".join(f"{x:10d}" for x in ids[i:i+8]) + "\n")

        # --- counts ---
        ncur_new = len(keep_curves)
        nsurf_new = len(s1_blocks)

        # --- assemble new file ---
        out = []
        out.extend(self.lines[:hdr_line])
        out.append(f"{ncur_new:8d}{nsurf_new:8d}\n")
        out.extend(self.lines[hdr_line+1 : self.idx["Curves"]])
        out.extend(new_curves_lines)
        out.extend(new_s1_lines)
        out.extend(self.lines[self.idx["MeshGen"] : self.idx["MeshGen"]+1])  # "Mesh Generation"
        out.append(f"{ncur_new:8d}{nsurf_new:8d}\n")
        out.extend(new_segments_lines)
        out.extend(self.lines[seg_end_idx : self.idx["Surfaces4"]])
        out.extend(new_s4_lines)

        if out_path:
            Path(out_path).write_text("".join(out), encoding="utf-8", errors="ignore")

        return {
            "curves_before": ncur_old,
            "curves_after": ncur_new,
            "surfaces_before": nsurf_old,
            "surfaces_after": nsurf_new,
            "removed_surfaces": sorted(remove_surfaces),
            "removed_curves": sorted(remove_curves),
            "auto_dropped_curves": sorted(set(curves_by_id) - set(keep_curves)),
        }

    def convert_units(
        self,
        out_path: str = None,
        *,
        from_unit: str = None,
        to_unit: str = None,
        factor: float = None,
        decimals: int = None,
        zero_threshold: float = 1e-8,   # <-- anything with |value| < this becomes 0.0
    ) -> dict:
        """
        Scale ALL coordinates in the DAT file (Curves + Surfaces1).
        - Specify either (from_unit, to_unit) OR a numeric `factor`.
        - After scaling, any coord with |value| < `zero_threshold` is set to 0.0.
        Set zero_threshold=None to disable.
        - Sections after 'Mesh Generation' are copied verbatim.
        """

        # ---- 1) resolve scale factor ----
        unit_to_m = {
            "m": 1.0, "meter": 1.0, "metre": 1.0,
            "mm": 1e-3, "millimeter": 1e-3, "millimetre": 1e-3,
            "cm": 1e-2, "centimeter": 1e-2, "centimetre": 1e-2,
            "km": 1e3,
            "in": 0.0254, "inch": 0.0254,
            "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
        }
        if factor is None:
            if not (from_unit and to_unit):
                raise ValueError("Provide either (factor) or (from_unit and to_unit).")
            fu = from_unit.strip().lower()
            tu = to_unit.strip().lower()
            if fu not in unit_to_m or tu not in unit_to_m:
                raise ValueError(f"Unknown unit(s): {from_unit} -> {to_unit}")
            # value_new = value_old * factor
            factor = unit_to_m[fu] / unit_to_m[tu]

        # float formatter
        if decimals is None:
            fmt = lambda x: f"{x:12.7E}"
        else:
            fmt = lambda x: f"{x:.{decimals}E}"

        def _scale_and_zero(val: float) -> float:
            v = val * factor
            if (zero_threshold is not None) and (abs(v) < zero_threshold):
                return 0.0
            return v

        def _fmt_xyz(x, y, z):
            return f"{fmt(x)} {fmt(y)} {fmt(z)}\n"

        lines = self.lines

        # ---- 2) rewrite Curves block ----
        new_curves = [" Curves\n"]
        i = self.idx["Curves"] + 1
        end_curves = self.idx["Surfaces1"]
        num_curve_pts = 0
        num_zeroed = 0

        while i < end_curves:
            # header: id, tag
            id_tag = lines[i].strip().split()
            if len(id_tag) < 2:
                i += 1
                continue
            cid = int(id_tag[0]); tag = int(id_tag[1]); i += 1
            new_curves.append(f"{cid:10d}{tag:8d}\n")

            # npts
            npts = int(lines[i].strip().split()[0]); i += 1
            new_curves.append(f"{npts:10d}\n")

            # coords
            for _ in range(npts):
                x, y, z = map(float, lines[i].split()); i += 1
                x2, y2, z2 = _scale_and_zero(x), _scale_and_zero(y), _scale_and_zero(z)
                num_zeroed += (x2 == 0.0) + (y2 == 0.0) + (z2 == 0.0)
                new_curves.append(_fmt_xyz(x2, y2, z2))
            num_curve_pts += npts

        # ---- 3) rewrite Surfaces1 (geometry) ----
        s1_start = self.idx["Surfaces1"]
        s1_end   = self.idx["MeshGen"]
        i = s1_start + 1
        new_s1 = [" Surfaces\n"]
        num_surf_pts = 0

        while i < s1_end:
            # header: sid, tag
            id_tag = lines[i].strip().split()
            if len(id_tag) < 2:
                i += 1
                continue
            sid = int(id_tag[0]); stag = int(id_tag[1]); i += 1
            new_s1.append(f"{sid:10d}{stag:8d}\n")

            # dims: nx ny
            dims = list(map(int, lines[i].split()[:2])); i += 1
            nx, ny = dims[0], dims[1]
            new_s1.append(f"{nx:10d}{ny:8d}\n")

            # nx*ny coordinates
            npts = nx * ny
            for _ in range(npts):
                x, y, z = map(float, lines[i].split()); i += 1
                x2, y2, z2 = _scale_and_zero(x), _scale_and_zero(y), _scale_and_zero(z)
                num_zeroed += (x2 == 0.0) + (y2 == 0.0) + (z2 == 0.0)
                new_s1.append(_fmt_xyz(x2, y2, z2))
            num_surf_pts += npts

        # ---- 4) assemble new file ----
        out = []
        out.extend(lines[: self.idx["Curves"]])     # header + preamble untouched
        out.extend(new_curves)
        out.extend(new_s1)
        out.extend(lines[self.idx["MeshGen"] :])    # everything after MeshGen untouched

        if out_path:
            from pathlib import Path as _Path
            _Path(out_path).write_text("".join(out), encoding="utf-8", errors="ignore")

        return {
            "scale_factor": factor,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "zero_threshold": zero_threshold,
            "curve_points_scaled": num_curve_pts,
            "surface_points_scaled": num_surf_pts,
            "total_points_scaled": num_curve_pts + num_surf_pts,
            "coords_zeroed": int(num_zeroed),
            "output_path": out_path,
        }
