# Remote/runSimRemote.py

import os, sys, json
import numpy as np
import posixpath
import tempfile
from scipy.spatial import cKDTree

# project paths
sys.path.append(os.path.dirname("ConvertFileType"))
sys.path.append(os.path.dirname("MeshGeneration"))
sys.path.append(os.path.dirname("FileRW"))

from ConvertFileType.convertVtmtoFro import vtm_to_fro
from MeshGeneration.controlNodeDisp import (
    _surface_normals,
    _map_normals_to_control,
    _spectral_coeffs,
    estimate_normals,
    getDisplacements,
)
from MeshGeneration.MorphModel import MorphModel
from FileRW.FroFile import FroFile
from FileRW.BatchFile import Batchfile

import tarfile, pathlib, re


# ----------------------------
# Helpers
# ----------------------------
def _tar_vtm_dataset(vtm_path):
    vtm_path = pathlib.Path(vtm_path)
    stem = vtm_path.stem                  # "crm"
    sidecar_dir = vtm_path.parent / stem  # e.g., .../crm/
    tar_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz")
    with tarfile.open(tar_tmp.name, "w:gz") as tar:
        tar.add(vtm_path, arcname=vtm_path.name)
        if sidecar_dir.is_dir():
            tar.add(sidecar_dir, arcname=sidecar_dir.name)
    return tar_tmp.name


def _apply_bump_window(viewer, cn, cn_normals, d_ctrl, logger=None):
    """
    Same bump-windowing as local path: optional Gaussian window and one-sided clipping
    """
    if not getattr(viewer, "bump_enable", False):
        return d_ctrl

    c0 = getattr(viewer, "bump_center", None)
    r  = getattr(viewer, "bump_radius", None)
    if c0 is None or r is None:
        if logger: logger.log("[BUMP] Skipped: center/radius not set.")
        return d_ctrl

    try:
        c0 = np.asarray(c0, dtype=float).reshape(3)
        r  = float(r)
    except Exception:
        if logger: logger.log("[BUMP] Skipped: could not parse center/radius.")
        return d_ctrl

    if not np.isfinite(r) or r <= 0.0:
        if logger: logger.log(f"[BUMP] Skipped: invalid radius r={r}.")
        return d_ctrl

    dx = cn - c0[None, :]
    d2 = np.einsum("ij,ij->i", dx, dx)

    s2 = r * r
    z  = -0.5 * d2 / s2
    z  = np.maximum(z, -50.0)  # stability
    w  = np.exp(z)

    d_ctrl = d_ctrl * w[:, None]

    if getattr(viewer, "bump_one_sided", False):
        n = cn_normals
        n_norm = np.linalg.norm(n, axis=1, keepdims=True)
        n_safe = np.divide(n, np.clip(n_norm, 1e-12, None))
        a_n = np.einsum("ij,ij->i", d_ctrl, n_safe)
        neg = a_n < 0.0
        if np.any(neg):
            d_ctrl[neg] = d_ctrl[neg] - (a_n[neg, None] * n_safe[neg])
    return d_ctrl


class RemoteClient:
    def __init__(self, ssh_client, username="s.2268086", logger=None):
        self.ssh_client = ssh_client
        self.username = username
        self.logger = logger

        sftp = self.ssh_client.open_sftp()
        try:
            self.remote_home = sftp.normalize(".")  # e.g., "/home/s.2268086"
        finally:
            sftp.close()

        self.python_local = posixpath.join(self.remote_home, ".conda/envs/aeropt-hpc/bin/python")
        self.server = posixpath.join(self.remote_home, "aeropt", "Scripts")

        # Environment is already created by `conda env create -f environment.yml`
        self.install_remote_requirements()

    def _ensure_remote_dir(self, sftp, path):
        parts = [p for p in path.split("/") if p]
        cur = "/"
        for p in parts:
            cur = posixpath.join(cur, p)
            try:
                sftp.stat(cur)
            except IOError:
                sftp.mkdir(cur)
    
    def install_remote_requirements(self):
        if self.logger:
            self.logger.log("[HPC] install_remote_requirements: skipped (using pre-built aeropt-hpc conda env).")
        return

# ----------------------------
# core upload + config writer
# ----------------------------
def _ensure_remote_dirs_and_upload_mesh(viewer, remoteCli, n, logger):
    """
    Creates $HOME/aeropt/aeropt_out/<run>/surfaces/n_<n>, uploads VTK/VTM/FRO.

    Robust to MeshViewer variants that store the mesh path as either
    `input_filepath` (old) or `mesh_path` (new).
    """
    username = viewer.main_window.ssh_creds["username"]
    ssh = viewer.main_window.ssh_client

    # --- figure out where the baseline mesh is on the GUI side ---
    mesh_path = getattr(viewer, "input_filepath", None)
    if not mesh_path:
        mesh_path = getattr(viewer, "mesh_path", None)

    if not mesh_path:
        raise AttributeError(
            "MeshViewer has no 'input_filepath' or 'mesh_path'. "
            "Load a baseline surface mesh in the Mesh tab before running morph."
        )

    # --- ensure base remote output dir on the GUI side ---
    if not hasattr(viewer.main_window, "remote_output_dir"):
        from datetime import datetime
        run_id = datetime.now().strftime("auto_%Y%m%d_%H%M%S")
        viewer.main_window.remote_output_dir = f"$HOME/aeropt/aeropt_out/{run_id}"

    out_dir  = viewer.main_window.remote_output_dir          # may include $HOME
    rdir_sh  = posixpath.join(out_dir, "surfaces", f"n_{n}") # shell path (keeps $HOME)
    rdir_abs = rdir_sh.replace("$HOME", remoteCli.remote_home)

    ssh.exec_command(f"bash -lc 'mkdir -p {rdir_sh}'")

    base_name = os.path.basename(mesh_path)
    ext = os.path.splitext(base_name)[1].lower()

    if ext == ".vtm":
        tgz_local = _tar_vtm_dataset(mesh_path)
        remote_tgz = posixpath.join(rdir_abs, "mesh_bundle.tar.gz")
        sftp = ssh.open_sftp()
        sftp.put(tgz_local, remote_tgz)
        sftp.close()
        ssh.exec_command(f"bash -lc 'cd {rdir_sh} && tar -xzf mesh_bundle.tar.gz'")
        vtk_upload_name = base_name  # e.g. crm.vtm now present in rdir
        _in, _out, _err = ssh.exec_command(f"bash -lc 'ls -l {rdir_sh}'")
        logger.log(f"[HPC DEBUG] After extract, {rdir_sh}:\n{_out.read().decode()}")
    elif ext in (".vtu", ".vtk", ".fro"):
        sftp = ssh.open_sftp()
        sftp.put(mesh_path, posixpath.join(rdir_abs, base_name))
        sftp.close()
        vtk_upload_name = base_name
        _in, _out, _err = ssh.exec_command(f"bash -lc 'ls -l {rdir_sh}'")
        logger.log(f"[HPC DEBUG] After upload, {rdir_sh}:\n{_out.read().decode()}")
    else:
        raise ValueError(f"Unsupported mesh extension: {ext}")

    return out_dir, rdir_sh, rdir_abs, vtk_upload_name, ext


def _sample_surface_vertices(viewer, surface_ids, max_pts_per_surf=1000):
    """Sample points from GUI-side surface meshes by numeric IDs."""
    pts = []
    for sid in surface_ids:
        # viewer.mesh_obj supports name/ID lookup; accept either int sid or name → id
        try:
            name = viewer.mesh_obj.get_surface_name(int(sid))
        except Exception:
            name = str(sid)
        surf = viewer.mesh_obj.get_surface_mesh(name)
        if surf is None:
            continue
        P = np.asarray(surf.points, float)
        if len(P) == 0:
            continue
        if len(P) > max_pts_per_surf:
            idx = np.linspace(0, len(P)-1, max_pts_per_surf, dtype=int)
            P = P[idx]
        pts.append(P)
    return np.vstack(pts) if pts else np.zeros((0,3), float)


def _build_local_displacements_like_local(viewer, logger):
    """
    Replicate Local.runSurfMorph displacement chain:
      - T-surface vertex normals → map to CNs
      - modal coeffs sampling
      - remove rigid (mean) component
      - optional bump window
      - scale x10
      - add 0-disp anchors from U & C
    """
    # IDs: accept numeric or names
    t_ids = [int(s) if str(s).isdigit() else int(viewer.mesh_obj.get_surface_id(s)) for s in viewer.TSurfaces]
    u_ids = [int(s) if str(s).isdigit() else int(viewer.mesh_obj.get_surface_id(s)) for s in viewer.USurfaces]
    c_ids = [int(s) if str(s).isdigit() else int(viewer.mesh_obj.get_surface_id(s)) for s in viewer.CSurfaces]

    # T-surface vertices for normals
    t_verts = _sample_surface_vertices(viewer, t_ids, max_pts_per_surf=200000)  # large cap if needed

    cn = np.asarray(viewer.control_nodes, float)
    if len(t_verts) >= 4:
        surf_normals = _surface_normals(t_verts, knn=16)
        cn_normals   = _map_normals_to_control(cn, t_verts, surf_normals, k=8)
    else:
        logger.log("[WARN] Too few T vertices for surface normals; falling back to control-node PCA normals.")
        cn_normals = estimate_normals(cn, knn=12)

    # modal sampling
    k_modes  = getattr(viewer, "k_modes", 10)
    rng_seed = getattr(viewer, "seed", 0)
    p        = getattr(viewer, "spectral_p", 2.0)
    frac     = getattr(viewer, "coeff_frac", 0.15)

    coeffs = getattr(viewer, "modal_coeffs", None)
    if coeffs is None:
        coeffs = _spectral_coeffs(k_modes, cn, rng=rng_seed, p=p, frac=frac)

    d_ctrl = getDisplacements(
        viewer.output_dir,
        seed=rng_seed,
        control_nodes=cn,
        normals=cn_normals,
        coeffs=coeffs,
        k_modes=k_modes,
        normal_project=getattr(viewer, "normal_project", True)
    )

    # debug rigid translation + remove rigid
    vbar   = d_ctrl.mean(axis=0)
    spread = np.linalg.norm(d_ctrl - vbar, axis=1).max()
    logger.log(f"[DEBUG] mean disp = {vbar}, max dev from mean = {spread:.3e}")

    if getattr(viewer, "remove_rigid_component", True):
        d_ctrl = d_ctrl - vbar
        logger.log("[DEBUG] Removed rigid (mean) component from control-node displacements.")

    # bump window, then scale like local
    d_ctrl = _apply_bump_window(viewer, cn, cn_normals, d_ctrl, logger=logger)
    #d_ctrl = d_ctrl * 10

    # U & C anchors with zero displacement
    u_anchor = _sample_surface_vertices(viewer, u_ids, max_pts_per_surf=1000)
    c_anchor = _sample_surface_vertices(viewer, c_ids, max_pts_per_surf=1000)
    anchors  = np.vstack([u_anchor, c_anchor]) if (len(u_anchor)+len(c_anchor))>0 else np.zeros((0,3), float)

    if anchors.shape[0] > 0:
        cn_aug = np.vstack([cn, anchors])
        d_aug  = np.vstack([d_ctrl, np.zeros((anchors.shape[0], 3), float)])
    else:
        cn_aug, d_aug = cn, d_ctrl

    # stash to viewer for preview/debug (safe if local)
    try:
        viewer.cn_points        = cn
        viewer.cn_displacements = d_ctrl
        viewer.cn_targets       = cn + d_ctrl
        viewer.requestPlotCNs.emit()
    except Exception:
        pass

    if coeffs is not None:
        logger.log(f"[MODAL] k={k_modes}, |d|_max={float(np.linalg.norm(d_ctrl,axis=1).max()):.3e}, coeff_norm={np.linalg.norm(coeffs):.3e}")

    return cn_aug, d_aug, t_ids, u_ids, c_ids


def _write_and_upload_config(viewer, ssh, rdir_sh, rdir_abs, vtk_upload_name, ext,
                             n, out_dir, cn_aug, d_aug, t_ids, u_ids, c_ids, logger):
    """Write morph_config locally and upload to remote n_<n> directory."""
    source = getattr(viewer.main_window, "control_node_source", "mesh")
    morph_kind = "cad" if source == "cad" else "mesh"

    morph_config = {
        "mesh filetype": ext,
        "vtk_name": vtk_upload_name,
        "output_directory": out_dir,
        "n": int(n),
        "debug": bool(True),
        "morph_kind": morph_kind,
        "t_surfaces": t_ids,
        "u_surfaces": u_ids,
        "c_surfaces": c_ids,
        "control_nodes": cn_aug.tolist(),
        "displacement_vector": d_aug.tolist(),
    }
    local_json = os.path.join(viewer.output_dir, f"morph_config_n_{n}.json")
    with open(local_json, "w") as f:
        json.dump(morph_config, f)

    json_name         = f"morph_config_n_{n}.json"
    remote_json_abs   = posixpath.join(rdir_abs, json_name)
    remote_json_shell = posixpath.join(rdir_sh,  json_name)
    sftp = ssh.open_sftp()
    sftp.put(local_json, remote_json_abs)
    sftp.close()
    logger.log(f"[HPC] Uploaded morph config → {remote_json_shell}")
    return remote_json_shell


def _submit_batch(viewer, ssh, remoteCli, rdir_sh, vtk_upload_name, remote_json_shell, n, logger):
    """Submit sbatch to run remoteMorph.py using your Batchfile helper."""
    job_name = f"morph_n{n}"
    bf = Batchfile(job_name)

    # sbatch params — tune as needed
    bf.sbatch_params["job-name"]      = job_name
    bf.sbatch_params["time"]          = "02-00:00"
    bf.sbatch_params["mem"]           = "0"
    bf.sbatch_params["output"]        = "batch_%j.out"
    bf.sbatch_params["error"]         = "batch_%j.err"

    project_root = "$HOME/aeropt/Scripts"
    fro_target   = posixpath.join(rdir_sh, os.path.splitext(vtk_upload_name)[0] + ".fro")

    bf.lines.append('export PYTHONPATH="' + project_root + ':${PYTHONPATH:-}"')

    bf.lines.append('cd "' + project_root + '/Remote"')
    bf.lines.append('"' + remoteCli.python_local + '" remoteMorph.py "' + fro_target + '" "' + remote_json_shell + '"')

    # write locally and upload
    batch_name = f"batchfile_{job_name}"
    local_batch = os.path.join(viewer.main_window.output_directory, batch_name)
    os.makedirs(viewer.main_window.output_directory, exist_ok=True)
    with open(local_batch, "w", newline="\n") as f:
        f.write(str(bf))

    sftp = ssh.open_sftp()
    remote_batch = posixpath.join(rdir_sh.replace("$HOME", remoteCli.remote_home), batch_name)
    sftp.put(local_batch, remote_batch)
    sftp.chmod(remote_batch, 0o755)
    sftp.close()

    submit_cmd = f"cd {rdir_sh}; sbatch {batch_name}"
    viewer.logger.log(f"[HPC DEBUG] Submit: {submit_cmd}")
    stdin, stdout, stderr = ssh.exec_command(submit_cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if err:
        viewer.logger.log(f"[HPC DEBUG] sbatch stderr: {err}")
    viewer.logger.log(f"[HPC] sbatch: {out}")

    m = re.search(r"Submitted batch job (\d+)", out)
    jobid = m.group(1) if m else None
    if jobid:
        viewer.logger.log(f"[HPC] Morph job submitted as {jobid}")
    else:
        viewer.logger.log("[HPC] Could not parse job id from sbatch output.")
    return jobid


# ----------------------------
# Public entry point
# ----------------------------
def runSurfMorph(viewer, n=0, debug=True, run_as_batch=False):
    """
    Remote morph that mirrors Local.runSurfMorph displacement pipeline,
    with a single toggle to run via sbatch or immediate SSH execution.
    """
    username = viewer.main_window.ssh_creds["username"]
    ssh = viewer.main_window.ssh_client
    logger = viewer.logger

    remoteCli = RemoteClient(ssh, username, logger)

    # 1) ensure remote dir + upload mesh
    out_dir, rdir_sh, rdir_abs, vtk_upload_name, ext = _ensure_remote_dirs_and_upload_mesh(
        viewer, remoteCli, n, logger
    )

    # 2) build displacements exactly like local
    cn_aug, d_aug, t_ids, u_ids, c_ids = _build_local_displacements_like_local(viewer, logger)

    # 3) write & upload morph config
    remote_json_shell = _write_and_upload_config(
        viewer, ssh, rdir_sh, rdir_abs, vtk_upload_name, ext,
        n, out_dir, cn_aug, d_aug, t_ids, u_ids, c_ids, logger
    )

    # 4) run: sbatch or immediate exec
    project_root = "$HOME/aeropt/Scripts"
    remote_fro_target_shell = posixpath.join(rdir_sh, os.path.splitext(vtk_upload_name)[0] + ".fro")

    if run_as_batch:
        return _submit_batch(viewer, ssh, remoteCli, rdir_sh, vtk_upload_name, remote_json_shell, n, logger)

    morph_cmd = (
        "bash -lc '"
        "source ~/.bashrc; "
        "export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt; "
        "export MPLBACKEND=Agg; "
        "export PYVISTA_OFF_SCREEN=true; "
        "export PYVISTA_USE_PANEL=false; "
        "export LIBGL_ALWAYS_SOFTWARE=1; "
        f"export PYTHONPATH={project_root}:$PYTHONPATH; "
        f"cd {project_root}/Remote; "
        f"{remoteCli.python_local} remoteMorph.py "
        f"{remote_fro_target_shell} {remote_json_shell}"
        "'"
    )
    logger.log(f"[HPC] Running morph command on cluster...")
    logger.log(f"[HPC DEBUG] remoteMorph args:\n  fro:  {remote_fro_target_shell}\n  json: {remote_json_shell}")

    stdin, stdout, stderr = ssh.exec_command(morph_cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out: logger.log(f"[HPC] Morph stdout: {out}")
    if err: logger.log(f"[HPC] Morph stderr: {err}")

    return None  # immediate path returns None (like before)


def runCadMorph(geo_viewer, n=0, debug=True, run_as_batch=True):
    """
    CAD-driven morph on the cluster.

    The CAD FFD machinery has already populated control nodes and other
    parameters onto the MeshViewer via Geometry → 'Save control-node settings'.

    We simply delegate to runSurfMorph using the MeshViewer, but we
    tag morph_kind='cad' in the JSON (see _write_and_upload_config).
    """
    main = geo_viewer.main_window
    mesh_viewer = getattr(main, "mesh_viewer", None)

    if mesh_viewer is None:
        raise RuntimeError("[HPC][CAD] runCadMorph: main_window.mesh_viewer is None; cannot run CAD-driven morph.")

    logger = getattr(geo_viewer, "logger", None) or getattr(mesh_viewer, "logger", None)
    if logger:
        logger.log("[HPC][CAD] runCadMorph: using MeshViewer baseline mesh with CAD-derived control nodes.")

    # This will:
    #  1) upload the mesh,
    #  2) build displacements from mesh_viewer.control_nodes + T/U/C,
    #  3) write morph_config_n_<n>.json (with morph_kind='cad'),
    #  4) submit sbatch on remoteMorph.py and return jobid if run_as_batch=True.
    return runSurfMorph(mesh_viewer, n=n, debug=debug, run_as_batch=run_as_batch)


# ----------------------------
# Back-compat API & batch helper
# ----------------------------
def runSurfMesh(viewer, n=0, debug=True):
    """Legacy name kept for pipeline_remote; runs immediate (non-batch)."""
    return runSurfMorph(viewer, n=n, debug=debug, run_as_batch=False)


def runSurfMorph_batch(viewer, n=0, runafter=None, debug=True):
    """
    Kept for direct batch submissions if you still call it elsewhere.
    Internally, now it just calls runSurfMorph(..., run_as_batch=True).
    """
    return runSurfMorph(viewer, n=n, debug=debug, run_as_batch=True)
