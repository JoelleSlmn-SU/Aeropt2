#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EnSight surface (.geo/.case)  -->  .fro

Dual use:
  - Library: engeo_to_fro(inpath, outpath, use_part_id=False, force_pyvista=False)
             -> returns an object with .write_file(outpath)
  - CLI:     python convertEnGeoToFro.py input.geo output.fro [--use-part-id] [--force-pyvista]

Two read modes:
  A) ASCII EnSight Gold .geo (no external deps)  --> direct parser
  B) Any PyVista-readable (.case, binary .geo)   --> pv.read (if pyvista installed)

Faces written as triangles; QUAD4 split into two TRIA.
Surface ID := per-part (sequential unless --use-part-id for ASCII .geo).
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import List, Tuple

__all__ = ["engeo_to_fro"]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def is_probably_ascii(path, sniff_bytes=2048):
    try:
        with open(path, "rb") as f:
            chunk = f.read(sniff_bytes)
        if b"\x00" in chunk:
            return False
        chunk.decode("utf-8", errors="ignore")
        return True
    except Exception:
        return False

def to_lower(s: str) -> str:
    return s.lower()

# ---------------------------------------------------------------------
# ASCII EnSight Gold (.geo) parser
# ---------------------------------------------------------------------
def read_ensight_ascii_geo(path, use_part_id: bool = False):
    """
    Returns:
        coords: list of (x,y,z) length N (1-based indexing when written)
        faces : list of (n1,n2,n3,surf_id) length F
        nsurf : int, number of parts seen
    """
    coords: List[Tuple[float, float, float]] = []
    faces:  List[Tuple[int, int, int, int]]   = []
    nsurf  = 0

    def next_nonempty_line(it):
        for line in it:
            s = line.strip()
            if s != "":
                return s
        return None

    # Buffer file for two logical passes
    with open(path, "r", encoding="utf-8", errors="ignore") as fr:
        content = fr.readlines()

    # ---- read coordinates
    it = iter(content)
    found_coords = False

    def parse_coord_line(s):
        toks = s.split()
        if len(toks) == 3:
            return float(toks[0]), float(toks[1]), float(toks[2])
        elif len(toks) == 4:
            return float(toks[1]), float(toks[2]), float(toks[3])
        else:
            raise ValueError(f"Bad coordinate line: {s!r}")

    while True:
        line = next_nonempty_line(it)
        if line is None:
            break
        lo = to_lower(line)
        if "coordinates" in lo:
            nline = next_nonempty_line(it)
            if nline is None:
                raise RuntimeError("Unexpected EOF after 'coordinates'")
            try:
                nnode = int(nline.split()[0])
            except Exception:
                raise RuntimeError("Failed reading node count after 'coordinates'")
            for _ in range(nnode):
                cl = next_nonempty_line(it)
                if cl is None:
                    raise RuntimeError("EOF while reading coordinates block")
                x, y, z = parse_coord_line(cl)
                coords.append((x, y, z))
            found_coords = True
            break

    if not found_coords:
        raise RuntimeError("Could not find a 'coordinates' section in ASCII .geo")

    # ---- read parts + surface elements
    it = iter(content)
    current_surf_id = 0

    while True:
        line = next_nonempty_line(it)
        if line is None:
            break
        lo = to_lower(line)

        if lo.startswith("part"):
            nsurf += 1
            raw = next_nonempty_line(it)
            if raw is not None:
                try:
                    part_id = int(raw.split()[0])
                    _ = next_nonempty_line(it)  # description
                except Exception:
                    part_id = nsurf
            else:
                part_id = nsurf
            current_surf_id = part_id if use_part_id else nsurf
            continue

        if lo.startswith("tria3"):
            nline = next_nonempty_line(it)
            if nline is None:
                raise RuntimeError("EOF after TRIA3 header")
            try:
                ntri = int(nline.split()[0])
            except Exception:
                raise RuntimeError("Could not parse TRIA3 count")
            for _ in range(ntri):
                el = next_nonempty_line(it)
                if el is None:
                    raise RuntimeError("EOF in TRIA3 data")
                t = el.split()
                if len(t) < 3:
                    raise RuntimeError(f"Bad TRIA3 line: {el!r}")
                n1, n2, n3 = int(t[0]), int(t[1]), int(t[2])
                faces.append((n1, n2, n3, current_surf_id))
            continue

        if lo.startswith("quad4"):
            nline = next_nonempty_line(it)
            if nline is None:
                raise RuntimeError("EOF after QUAD4 header")
            try:
                nquad = int(nline.split()[0])
            except Exception:
                raise RuntimeError("Could not parse QUAD4 count")
            for _ in range(nquad):
                el = next_nonempty_line(it)
                if el is None:
                    raise RuntimeError("EOF in QUAD4 data")
                t = el.split()
                if len(t) < 4:
                    raise RuntimeError(f"Bad QUAD4 line: {el!r}")
                j, k, q, r = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                faces.append((j, k, q, current_surf_id))
                faces.append((j, q, r, current_surf_id))
            continue

    return coords, faces, nsurf

# ---------------------------------------------------------------------
# PyVista loader (optional)
# ---------------------------------------------------------------------
def read_with_pyvista(path, use_part_id: bool = False):
    """
    Load .case/.geo via PyVista, triangulate, and build (coords, faces, nsurf).
    Each top-level block (or part) becomes a surface ID.
    """
    try:
        import pyvista as pv
        import numpy as np  # noqa: F401
    except Exception as e:
        raise RuntimeError("PyVista path requested but pyvista/numpy not available") from e

    dataset = pv.read(path)
    blocks = dataset if isinstance(dataset, pv.MultiBlock) else pv.MultiBlock([dataset])

    all_points: List[Tuple[float, float, float]] = []
    faces_out:  List[Tuple[int, int, int, int]]   = []
    nsurf = 0
    point_index = {}

    def get_pt_id(pt):
        key = (float(pt[0]), float(pt[1]), float(pt[2]))
        if key in point_index:
            return point_index[key]
        idx = len(all_points) + 1
        all_points.append(key)
        point_index[key] = idx
        return idx

    for block in blocks:
        if block is None:
            continue
        surf = block.extract_geometry().triangulate()
        if surf.n_cells == 0:
            continue

        nsurf += 1
        faces = surf.faces.reshape((-1, 4))
        pts = surf.points
        for f in faces:
            if f[0] != 3:
                continue
            n1 = get_pt_id(pts[f[1]])
            n2 = get_pt_id(pts[f[2]])
            n3 = get_pt_id(pts[f[3]])
            faces_out.append((n1, n2, n3, nsurf))

    return all_points, faces_out, nsurf

# ---------------------------------------------------------------------
# .fro writer
# ---------------------------------------------------------------------
def write_fro(path, coords, faces, nsurf):
    ipcount = len(coords)
    ifcount = len(faces)
    with open(path, "w", encoding="utf-8") as fw:
        fw.write(f"{ifcount} {ipcount} 1 0 0 {nsurf} 0 0\n")
        for i, (x, y, z) in enumerate(coords, start=1):
            fw.write(f"{i} {x:.10g} {y:.10g} {z:.10g}\n")
        for ie, (n1, n2, n3, sid) in enumerate(faces, start=1):
            fw.write(f"{ie} {n1} {n2} {n3} {sid}\n")

# ---------------------------------------------------------------------
# Library entrypoint expected by pipeline_remote.volume()
# ---------------------------------------------------------------------
class _FroResult:
    """Tiny return object to mirror vtm_to_fro contract (has .write_file())."""
    def __init__(self, coords, faces, nsurf):
        self._coords = coords
        self._faces  = faces
        self._nsurf  = nsurf
    def write_file(self, outpath: str):
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        write_fro(outpath, self._coords, self._faces, self._nsurf)
        return outpath

def engeo_to_fro(inpath: str, outpath: str, use_part_id: bool = False, force_pyvista: bool = False) -> _FroResult:
    """
    Convert EnSight (.case/.geo) to .fro and write to 'outpath'.
    Returns an object with .write_file(outpath) for compatibility with vtm_to_fro usage.
    """
    if not os.path.isfile(inpath):
        raise FileNotFoundError(inpath)

    coords = faces = None
    nsurf = 0

    # prefer explicit force path
    if force_pyvista:
        coords, faces, nsurf = read_with_pyvista(inpath, use_part_id=use_part_id)
    else:
        # Try ASCII .geo direct first if it looks like ASCII .geo
        if inpath.lower().endswith(".geo") and is_probably_ascii(inpath):
            coords, faces, nsurf = read_ensight_ascii_geo(inpath, use_part_id=use_part_id)
        else:
            # Fallback to PyVista for .case or binary .geo
            coords, faces, nsurf = read_with_pyvista(inpath, use_part_id=use_part_id)

    # Write immediately (pipeline checks existence right after)
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    write_fro(outpath, coords, faces, nsurf)

    return _FroResult(coords, faces, nsurf)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
'''def _main_cli():
    ap = argparse.ArgumentParser(description="Convert EnSight surface (.geo/.case) to .fro")
    ap.add_argument("input", help=".geo (ASCII/binary) or .case")
    ap.add_argument("output", help="output .fro")
    ap.add_argument("--use-part-id", action="store_true",
                    help="Use actual PART ids for surf_id (ASCII reader). Default is sequential per-part.")
    ap.add_argument("--force-pyvista", action="store_true",
                    help="Force using PyVista reader if available (useful for .case or binary .geo).")
    args = ap.parse_args()

    try:
        res = engeo_to_fro(args.input, args.output,
                           use_part_id=args.use_part_id,
                           force_pyvista=args.force_pyvista)
        # Already written; print a small summary
        print(f"[OK] Wrote {args.output}")
        print(f"  nodes: {len(res._coords)}")
        print(f"  faces: {len(res._faces)}")
        print(f"  nsurf: {res._nsurf}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    _main_cli()
'''