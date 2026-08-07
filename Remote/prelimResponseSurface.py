#!/usr/bin/env python3
"""
prelimResponseSurface.py

Cluster-side preliminary Morris regional sensitivity screening.

Usage:
    python prelimResponseSurface.py <prelim_run_dir> <morph_basis_json> <bo_settings_json> <objective_json>

What it does:
    1. Loads the preliminary morph_basis.json created by AerOpt.
    2. Builds randomized Morris trajectories over the regional displacements.
    3. Runs morph -> volume -> prepro -> solver for all DOE cases using ClusterPipelineManager.
    4. Parses solver results and computes Morris elementary effects.
    5. Ranks regions by mu-star and records sigma as an interaction/nonlinearity signal.
    6. Keeps the top regions.
    7. Resamples the final control nodes inside the selected regions using farthest-point sampling.
    8. Updates morph_basis.json and bo_settings.json for the actual BO stage.

Notes:
    - The preliminary screening basis is forced to direct normal displacement so that
      each DOE variable corresponds cleanly to one region.
    - The final optimisation basis is restored to modal parameterisation unless the
      original morph_basis.json requested direct parameterisation.
"""

import os
import sys
import json
import time
import shutil
import traceback
import subprocess
import numpy as np

# ----------------------------------------------------------------------
# Path setup
# ----------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

for subdir in ["", "Optimisation", "FileRW", "Remote", "MeshGeneration", "ShapeParameterization"]:
    p = os.path.join(project_root, subdir) if subdir else project_root
    if p not in sys.path:
        sys.path.insert(0, p)

from remoteOpt import ClusterTestManager, _build_objective_callable
from pipeline_cluster import ClusterPipelineManager
from ShapeParameterization.controlNodeDisp import _surface_normals, _map_normals_to_control
from morris_design import generate_morris_trajectories, save_design
from Optimisation.response_surface_analysis import compute_elementary_effects


# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------
def log(msg, log_path):
    print(msg, flush=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")
    except Exception:
        pass


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def as_array_or_none(x, dtype=float):
    if x is None:
        return None
    arr = np.asarray(x, dtype=dtype)
    return arr


def farthest_point_sampling_indices(points, n_select, seed=0):
    pts = np.asarray(points, dtype=float)
    n_pts = len(pts)
    n_select = int(min(max(1, n_select), n_pts))

    if n_pts == 0:
        return np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n_pts))

    selected = [first]
    dist2 = np.full(n_pts, np.inf)

    for _ in range(1, n_select):
        last = pts[selected[-1]]
        d2 = np.sum((pts - last) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2)
        selected.append(int(np.argmax(dist2)))

    return np.asarray(selected, dtype=int)


def select_final_control_nodes(points, point_region_ids, selected_regions, final_n, seed=0):
    points = np.asarray(points, dtype=float)
    region_ids = np.asarray(point_region_ids, dtype=int)
    selected_regions = np.asarray(selected_regions, dtype=int)

    mask = np.isin(region_ids, selected_regions)
    candidate_idx = np.where(mask)[0]

    if len(candidate_idx) == 0:
        raise RuntimeError("No candidate T-surface points found inside selected preliminary regions.")

    candidate_pts = points[candidate_idx]
    local_idx = farthest_point_sampling_indices(candidate_pts, final_n, seed=seed)
    global_idx = candidate_idx[local_idx]

    control_nodes = points[global_idx]
    control_regions = region_ids[global_idx]

    return control_nodes, control_regions, global_idx


def bo_dimension_from_basis(basis):
    """Mirror the GUI's basic dimension logic enough for the updated BO JSON."""
    method = str(basis.get("parameterisation_method", "modal")).lower()

    if method == "direct":
        subtype = str(basis.get("direct_parameterisation_subtype", "normal") or "normal").lower()
        n_cn = len(basis.get("control_nodes", []) or [])
        return 3 * n_cn if subtype == "xyz" else n_cn

    if bool(basis.get("use_pca", False)):
        return int(basis.get("pca_k_final") or basis.get("pca_k_red") or basis.get("k_modes") or 1)

    n = 0
    if bool(basis.get("global_modes", False)):
        cfg = basis.get("global_mode_config", []) or []
        n += len(cfg) if cfg else 8

    use_local = bool(basis.get("use_local_modes", True))
    if bool(basis.get("global_only", False)):
        use_local = False

    if use_local:
        k = int(basis.get("k_modes", 1) or 1)
        if bool(basis.get("normal_project", True)):
            n += k
        else:
            n += 3 * k

    return int(max(1, n))


# ----------------------------------------------------------------------
# Restartable staged DOE orchestration
# ----------------------------------------------------------------------
def _jobs_still_active(job_ids):
    job_ids = [str(j) for j in job_ids if j]
    if not job_ids:
        return []
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i"],
            text=True,
        ).splitlines()
        active = {x.strip() for x in out if x.strip()}
        return [j for j in job_ids if j in active]
    except Exception:
        # Conservative: do not advance to the next stage if Slurm cannot be queried.
        return job_ids


def _wait_for_jobs(job_ids, poll_s, stage, log_path):
    job_ids = [str(j) for j in job_ids if j]
    if not job_ids:
        return
    log(f"[PRELIM][{stage.upper()}] Waiting for {len(job_ids)} submitted jobs.", log_path)
    while True:
        active = _jobs_still_active(job_ids)
        if not active:
            return
        log(
            f"[PRELIM][{stage.upper()}] {len(active)} jobs still pending/running; "
            f"checking again in {poll_s}s.",
            log_path,
        )
        time.sleep(int(max(10, poll_s)))


def _case_paths(root, base_name, gen_num, n_index, nc=1):
    return {
        "morph": os.path.join(root, "surfaces", f"n_{gen_num}", f"{base_name}_{n_index}.fro"),
        "volume": os.path.join(root, "volumes", f"n_{gen_num}", f"{base_name}_{n_index}.plt"),
        "prepro": os.path.join(root, "preprocessed", f"n_{gen_num}", str(n_index)),
        "solver": os.path.join(root, "solutions", f"n_{gen_num}", f"cond_{nc}", str(n_index), "SOLVER_DONE"),
    }


def _stage_complete(stage, root, base_name, gen_num, n_index, conds):
    paths = _case_paths(root, base_name, gen_num, n_index)
    if stage == "morph":
        p = paths[stage]
        return os.path.isfile(p) and os.path.getsize(p) > 1024
    if stage == "volume":
        p = paths[stage]
        return os.path.isfile(p) and os.path.getsize(p) > 1024
    if stage == "prepro":
        d = paths[stage]
        if not os.path.isdir(d):
            return False
        prefix = f"{base_name}_{n_index}.sol"
        return any(name.startswith(prefix) for name in os.listdir(d))
    if stage == "solver":
        return all(
            os.path.isfile(_case_paths(root, base_name, gen_num, n_index, nc)["solver"])
            for nc in range(1, len(conds) + 1)
        )
    raise ValueError(f"Unknown stage: {stage}")


def _make_pipe(root, base_name, input_dir, executables, prelim_basis_path, cad_units, parallel, gen_num, n_index, x):
    config = {
        "remote_output": root,
        "base_name": base_name,
        "input_dir": input_dir,
        "modal_coeffs": list(map(float, x)),
        "morph_basis_json": prelim_basis_path,
        "cad_units": cad_units,
        "parallel_processes": parallel,
        **executables,
    }
    return ClusterPipelineManager(config, gen=int(gen_num), n=int(n_index))


def run_restartable_doe_pipeline(
    X, conds, root, base_name, input_dir, executables, prelim_basis_path,
    cad_units, parallel, gen_num, poll_s, batch_size, log_path,
):
    """Run complete stage barriers: all morph -> all volume -> all prepro -> all solver."""
    stages = ("morph", "volume", "prepro", "solver")
    n_cases = len(X)

    for stage in stages:
        pending = [
            i for i in range(1, n_cases + 1)
            if not _stage_complete(stage, root, base_name, gen_num, i, conds)
        ]
        done_count = n_cases - len(pending)
        log(
            f"[PRELIM][RESUME] Stage {stage}: {done_count}/{n_cases} already complete; "
            f"{len(pending)} remaining.",
            log_path,
        )

        if not pending:
            continue

        for start in range(0, len(pending), batch_size):
            wave = pending[start:start + batch_size]
            job_ids = []
            log(f"[PRELIM][{stage.upper()}] Submitting cases {wave}.", log_path)

            for n_index in wave:
                x = X[n_index - 1]
                pipe = _make_pipe(
                    root, base_name, input_dir, executables, prelim_basis_path,
                    cad_units, parallel, gen_num, n_index, x,
                )
                try:
                    if stage == "morph":
                        jid = pipe.morph(n=n_index)
                        if jid:
                            job_ids.append(jid)
                    elif stage == "volume":
                        jid = pipe.volume()
                        if jid:
                            job_ids.append(jid)
                    elif stage == "prepro":
                        jid = pipe.prepro()
                        if jid:
                            job_ids.append(jid)
                    else:
                        for nc, cond in enumerate(conds, 1):
                            done = _case_paths(root, base_name, gen_num, n_index, nc)["solver"]
                            if os.path.isfile(done):
                                continue
                            jid = pipe.solver(cond, nc=nc)
                            if jid:
                                job_ids.append(jid)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to submit {stage} for DOE case {n_index}: {exc}"
                    ) from exc

            _wait_for_jobs(job_ids, poll_s, stage, log_path)

            failed = [
                i for i in wave
                if not _stage_complete(stage, root, base_name, gen_num, i, conds)
            ]
            if failed:
                raise RuntimeError(
                    f"Stage '{stage}' jobs left the queue but expected outputs are missing "
                    f"for DOE cases {failed}. Fix those cases and rerun; completed cases will be skipped."
                )

        log(f"[PRELIM][{stage.upper()}] All {n_cases} DOE cases complete.", log_path)


# ----------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) < 5:
        print("Usage: prelimResponseSurface.py <prelim_run_dir> <morph_basis_json> <bo_settings_json> <objective_json>", flush=True)
        sys.exit(2)

    prelim_run = os.path.abspath(sys.argv[1])
    morph_basis_path = os.path.abspath(sys.argv[2])
    bo_settings_path = os.path.abspath(sys.argv[3])
    objective_path = os.path.abspath(sys.argv[4])

    os.makedirs(prelim_run, exist_ok=True)
    log_path = os.path.join(prelim_run, "prelim_response_surface.log")

    try:
        log(f"[PRELIM] Starting preliminary response-surface study in {prelim_run}", log_path)
        log(f"[PRELIM] morph_basis_json = {morph_basis_path}", log_path)
        log(f"[PRELIM] bo_settings_json = {bo_settings_path}", log_path)
        log(f"[PRELIM] objective_json = {objective_path}", log_path)

        basis = load_json(morph_basis_path)
        settings = load_json(bo_settings_path)
        objective = load_json(objective_path)

        if not bool(basis.get("prelim_enabled", False)):
            log("[PRELIM] prelim_enabled is false. Nothing to do.", log_path)
            return

        points = as_array_or_none(basis.get("t_surface_points", None), float)
        if points is None:
            # In current GUI, full T points are not yet stored in morph_basis.
            # Fall back to using only point_region_ids length check later if output.vtk is available.
            output_vtk = os.path.join(os.path.dirname(os.path.dirname(morph_basis_path)), "..", "surfaces", "output.vtk")
            output_vtk = os.path.abspath(output_vtk)
            if os.path.exists(output_vtk):
                import pyvista as pv
                mesh = pv.read(output_vtk)
                points = np.asarray(mesh.points, dtype=float)
                log(f"[PRELIM] Loaded T-surface points from {output_vtk}", log_path)
            else:
                raise RuntimeError(
                    "No t_surface_points in morph_basis.json and could not find staged surfaces/output.vtk. "
                    "Please stage output.vtk and/or add t_surface_points to morph_basis.json."
                )

        point_region_ids = as_array_or_none(basis.get("point_region_ids", None), int)
        if point_region_ids is None or len(point_region_ids) != len(points):
            raise RuntimeError("point_region_ids missing or incompatible with T-surface points.")

        n_regions = int(basis.get("prelim_regions", int(point_region_ids.max()) + 1) or 1)
        amplitude = float(basis.get("prelim_doe_amplitude", 1.0) or 1.0)
        morris_r = int(basis.get("prelim_morris_trajectories", settings.get("prelim_morris_trajectories", 10)) or 10)
        morris_p = int(basis.get("prelim_morris_levels", settings.get("prelim_morris_levels", 4)) or 4)
        keep_fraction = float(basis.get("prelim_keep_fraction", 0.67) or 0.67)
        final_n = int(basis.get("prelim_final_control_nodes", len(basis.get("control_nodes", []))) or len(basis.get("control_nodes", [])))
        seed = int(basis.get("seed", 0) or 0)

        if amplitude <= 0.0:
            raise ValueError("prelim_doe_amplitude must be positive")
        region_centres = np.asarray(basis.get("control_nodes", []), dtype=float)
        if region_centres.shape != (n_regions, 3):
            raise RuntimeError(
                f"Morris requires one representative control node per region: expected "
                f"({n_regions}, 3), got {region_centres.shape}."
            )

        log(
            f"[PRELIM] method=Morris, R={n_regions}, trajectories={morris_r}, "
            f"levels={morris_p}, bounds=[{-amplitude}, {amplitude}], "
            f"keep_fraction={keep_fraction}, final_n={final_n}",
            log_path,
        )

        # ------------------------------------------------------------------
        # Make a temporary preliminary basis where each variable maps to one region CN.
        # ------------------------------------------------------------------
        prelim_basis = dict(basis)
        prelim_basis["_original_parameterisation_method"] = basis.get("parameterisation_method", "modal")
        prelim_basis["parameterisation_method"] = "direct"
        prelim_basis["direct_parameterisation_subtype"] = "normal"
        prelim_basis["k_modes"] = 0
        prelim_basis["use_local_modes"] = False
        prelim_basis["use_pca"] = False
        prelim_basis["global_modes"] = False
        prelim_basis["global_only"] = False
        # Direct coefficients are interpreted by pipeline_cluster as physical
        # normal displacements in the current mesh/CAD unit.
        prelim_basis["prelim_candidate_nodes"] = points.tolist()

        prelim_basis_path = os.path.join(prelim_run, "prelim_morph_basis.json")
        write_json(prelim_basis_path, prelim_basis)
        log(f"[PRELIM] Wrote preliminary direct-normal basis -> {prelim_basis_path}", log_path)

        design = generate_morris_trajectories(
            k=n_regions,
            r=morris_r,
            p=morris_p,
            lb=np.full(n_regions, -amplitude),
            ub=np.full(n_regions, amplitude),
            seed=seed,
        )
        design_path = os.path.join(prelim_run, "prelim_morris_design.json")
        if os.path.exists(design_path):
            existing = load_json(design_path)
            existing_X = np.asarray(existing.get("X", []), dtype=float)
            if existing_X.shape != design.X.shape or not np.allclose(existing_X, design.X):
                raise RuntimeError(
                    "The existing preliminary outputs belong to a different Morris design. "
                    "Use a new prelim_run_dir (or archive the old preliminary run) before rerunning."
                )
        else:
            save_design(design, design_path)
        X = design.X
        log(f"[PRELIM] Morris CFD cases = {len(X)} (= {morris_r} * ({n_regions} + 1))", log_path)

        # ------------------------------------------------------------------
        # Prepare pipeline manager, modelled on remoteOpt.py.
        # ------------------------------------------------------------------
        obj_func, obj_expr = _build_objective_callable(objective)
        conds = objective.get("conditions", []) or [{}]
        log(f"[PRELIM] Objective expression minimised: {obj_expr}", log_path)
        log(f"[PRELIM] Conditions: {conds}", log_path)

        input_dir = settings.get("input_dir", os.path.join(os.path.dirname(os.path.dirname(prelim_run)), "orig"))
        base_name = settings.get("base_name", "model")
        cad_units = settings.get("units", "mm")
        parallel = settings.get("parallel", 80)

        executables = {
            "parallel_domains": settings.get("parallel_domains", 1),
            "surface_mesher": settings.get("surface_mesher", "/home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Surf3D"),
            "volume_mesher": settings.get("volume_mesher", "/home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Mesh3D"),
            "prepro_exe": settings.get("prepro_exe", "/home/s.o.hassan/bin/Gen3d_jj"),
            "solver_exe": settings.get("solver_exe", "/home/s.o.hassan/bin/UnsMgnsg3d"),
            "combine_exe": settings.get("combine_exe", "/home/s.engevabj/codes/utilities/makeplot2"),
            "ensight_exe": settings.get("ensight_exe", "/home/s.engevabj/codes/utilities/engen_tet"),
            "splitplot_exe": settings.get("splitplot_exe", "/home/s.engevabj/codes/utilities/splitplot2"),
            "makeplot_exe": settings.get("makeplot_exe", "/home/s.engevabj/codes/utilities/makeplot2"),
            "intel_module": settings.get("intel_module", "module load compiler/intel/2020/0"),
            "gnu_module": settings.get("gnu_module", "module load compiler/gnu/12/1.0"),
            "mpi_intel_module": settings.get("mpi_intel_module", "module load mpi/intel/2020/0"),
        }

        tm = ClusterTestManager(
            remote_root=prelim_run,
            base_name=base_name,
            input_dir=input_dir,
            executables=executables,
            poll_s=settings.get("poll_interval", 120),
            morph_basis_json=prelim_basis_path,
            units=cad_units,
            parallel=parallel,
        )

        gen_num = 0
        stage_batch_size = int(settings.get("prelim_stage_batch_size", 5) or 5)
        stage_batch_size = max(1, stage_batch_size)
        poll_s = int(settings.get("poll_interval", 120) or 120)

        log(
            f"[PRELIM] Restartable staged execution enabled: batch_size={stage_batch_size}; "
            "order=morph -> volume -> prepro -> solver.",
            log_path,
        )
        run_restartable_doe_pipeline(
            X=X,
            conds=conds,
            root=prelim_run,
            base_name=base_name,
            input_dir=input_dir,
            executables=executables,
            prelim_basis_path=prelim_basis_path,
            cad_units=cad_units,
            parallel=parallel,
            gen_num=gen_num,
            poll_s=poll_s,
            batch_size=stage_batch_size,
            log_path=log_path,
        )

        log("[PRELIM] All solver outputs complete; parsing DOE results...", log_path)
        per_design = tm.evaluate_generation(X, gen_num, conds)

        Y = []
        for metrics_per_cond in per_design:
            y = 0.0
            for cond, m in zip(conds, metrics_per_cond):
                y += float(cond.get("Weight", 1.0)) * obj_func(m)
            Y.append(float(y))
        Y = np.asarray(Y, dtype=float)

        morris = compute_elementary_effects(X, Y, design.trajectories)
        scores = morris.mu_star
        order = np.argsort(scores)[::-1]
        n_keep = int(np.ceil(keep_fraction * n_regions))
        n_keep = max(1, min(n_regions, n_keep))
        selected_regions = np.sort(order[:n_keep])

        results = {
            "X": X.tolist(),
            "Y": Y.tolist(),
            "trajectories": design.trajectories,
            "k": design.k,
            "r": design.r,
            "p": design.p,
            "delta": design.delta,
            "lb": design.lb.tolist(),
            "ub": design.ub.tolist(),
            "method": "morris_elementary_effects",
            "mu": morris.mu.tolist(),
            "mu_star": morris.mu_star.tolist(),
            "sigma": morris.sigma.tolist(),
            "n_trajectories_used": morris.n_trajectories_used.tolist(),
            "region_scores": morris.mu_star.tolist(),
            "region_rank_desc": order.tolist(),
            "selected_regions": selected_regions.tolist(),
            "objective_expression_minimised": obj_expr,
        }
        results_path = os.path.join(prelim_run, "prelim_morris_results.json")
        write_json(results_path, results)
        log(f"[PRELIM] Morris mu*: {morris.mu_star.tolist()}", log_path)
        log(f"[PRELIM] Morris sigma: {morris.sigma.tolist()}", log_path)
        log(f"[PRELIM] Selected regions: {selected_regions.tolist()}", log_path)

        # ------------------------------------------------------------------
        # Build final control-node set inside selected regions.
        # ------------------------------------------------------------------
        control_nodes, cn_regions, cn_point_idx = select_final_control_nodes(
            points=points,
            point_region_ids=point_region_ids,
            selected_regions=selected_regions,
            final_n=final_n,
            seed=seed,
        )

        surf_normals = _surface_normals(points, knn=16)
        control_normals = _map_normals_to_control(control_nodes, points, surf_normals, k=12)

        updated_basis = dict(basis)
        updated_basis["control_nodes"] = control_nodes.tolist()
        updated_basis["control_normals"] = control_normals.tolist()
        updated_basis["control_node_region_ids"] = cn_regions.astype(int).tolist()
        updated_basis["control_node_point_indices"] = cn_point_idx.astype(int).tolist()
        updated_basis["prelim_selected_regions"] = selected_regions.astype(int).tolist()
        updated_basis["prelim_region_scores"] = scores.tolist()
        updated_basis["prelim_method"] = "morris_elementary_effects"
        updated_basis["prelim_mu"] = morris.mu.tolist()
        updated_basis["prelim_mu_star"] = morris.mu_star.tolist()
        updated_basis["prelim_sigma"] = morris.sigma.tolist()
        updated_basis["prelim_n_trajectories_used"] = morris.n_trajectories_used.tolist()
        updated_basis["prelim_results_json"] = results_path
        updated_basis["prelim_completed"] = True

        # Restore original parameterisation after screening.
        original_method = basis.get("parameterisation_method", "modal")
        updated_basis["parameterisation_method"] = original_method

        if str(original_method).lower() == "modal":
            max_k = max(1, len(control_nodes) - 1)
            updated_basis["k_modes"] = int(min(int(basis.get("k_modes", max_k) or max_k), max_k))
            updated_basis["use_local_modes"] = bool(basis.get("use_local_modes", True))

        write_json(morph_basis_path, updated_basis)
        log(f"[PRELIM] Updated final morph_basis.json -> {morph_basis_path}", log_path)

        # Also copy a record into prelim dir.
        write_json(os.path.join(prelim_run, "morph_basis_after_prelim.json"), updated_basis)

        # ------------------------------------------------------------------
        # Update BO settings dimensions/bounds for the final design space.
        # ------------------------------------------------------------------
        updated_settings = dict(settings)
        n_dim = bo_dimension_from_basis(updated_basis)
        updated_settings["n_dim"] = int(n_dim)

        old_lb = list(settings.get("lb", []))
        old_ub = list(settings.get("ub", []))
        default_lb = float(old_lb[0]) if old_lb else -1.0
        default_ub = float(old_ub[0]) if old_ub else 1.0

        updated_settings["lb"] = (old_lb[:n_dim] + [default_lb] * max(0, n_dim - len(old_lb)))[:n_dim]
        updated_settings["ub"] = (old_ub[:n_dim] + [default_ub] * max(0, n_dim - len(old_ub)))[:n_dim]
        updated_settings["prelim_completed"] = True
        updated_settings["prelim_method"] = "morris_elementary_effects"
        updated_settings["prelim_results_json"] = results_path

        write_json(bo_settings_path, updated_settings)
        log(f"[PRELIM] Updated bo_settings.json -> {bo_settings_path}", log_path)
        log("[PRELIM] Preliminary Morris screening complete.", log_path)

    except Exception as e:
        log(f"[PRELIM][ERROR] {e}", log_path)
        log(traceback.format_exc(), log_path)
        sys.exit(1)


if __name__ == "__main__":
    main()