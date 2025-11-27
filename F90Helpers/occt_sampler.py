# FileRW/cad_fortran_sampler.py

import subprocess
import tempfile
from pathlib import Path

import numpy as np


def run_occt_sampler(cad_path: str, occt_exe: str) -> dict[int, dict]:
    """
    Call the Fortran occt_sampler on `cad_path` and read all surf_*.dat files.

    Args
    ----
    cad_path : str
        Path to STEP/IGES file.
    occt_exe : str
        Path to the compiled Fortran executable (CadSurfaceSampler).

    Returns
    -------
    surfaces : dict[int, dict]
        sid -> {
            "sid": sid,
            "numU": numU,
            "numV": numV,
            "grid": xyz_grid,   # (numU, numV, 3)
            "flat": xyz_flat,   # (numU*numV, 3)
        }
    """
    cad_path = Path(cad_path).resolve()
    occt_exe = Path(occt_exe).resolve()

    if not cad_path.exists():
        raise FileNotFoundError(f"CAD file not found: {cad_path}")
    if not occt_exe.exists():
        raise FileNotFoundError(f"occt_sampler executable not found: {occt_exe}")

    surfaces: dict[int, dict] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)

        cmd = [str(occt_exe), str(cad_path), str(outdir)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                "occt_sampler failed\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )

        for f in sorted(outdir.glob("surf_*.dat")):
            # Header: sid numU numV
            with f.open("r") as fh:
                header = fh.readline().strip()
            sid, numU, numV = map(int, header.split())

            data = np.loadtxt(f, skiprows=1)
            if data.ndim == 1:
                data = data[None, :]  # single-row edge case

            ij = data[:, 0:2].astype(int)  # i, j
            xyz = data[:, 2:5].astype(float)

            grid = np.zeros((numU, numV, 3), dtype=float)
            for (i, j), p in zip(ij, xyz):
                # Fortran is 1-based
                grid[i - 1, j - 1, :] = p

            flat = grid.reshape(-1, 3)

            surfaces[sid] = {
                "sid": sid,
                "numU": numU,
                "numV": numV,
                "grid": grid,
                "flat": flat,
            }

    return surfaces


def collect_control_points(surfaces: dict[int, dict], face_ids) -> np.ndarray:
    """
    Concatenate flat control-point lists for the given surface ids.

    face_ids are the *Fortran* surface ids (1-based).
    """
    pts = []
    for sid in face_ids:
        sid = int(sid)
        surf = surfaces.get(sid)
        if surf is None:
            continue
        pts.append(surf["flat"])
    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(pts)
