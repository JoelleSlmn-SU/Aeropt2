# remoteOpt.py - FIXED FOR CLUSTER EXECUTION
# This script runs ON THE CLUSTER (not your local machine)
# It uses ClusterPipelineManager instead of HPCPipelineManager

import os, json, time, sys, re, subprocess
import numpy as np

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  

for subdir in ["", "Optimisation", "FileRW", "Remote", "MeshGeneration"]:
    path = os.path.join(project_root, subdir) if subdir else project_root
    if path not in sys.path:
        sys.path.insert(0, path)


from pipeline_cluster import ClusterPipelineManager


def _log(msg, log_path):
    """Log to both stdout and file"""
    print(msg, flush=True)
    try:
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _cond_tag(cond):
    """Create filesystem-safe tag for condition"""
    return f"AoA{cond.get('AoA',0)}_M{cond.get('Mach',1.0)}_Re{int(cond.get('Re',0))}_T{cond.get('TurbModel',0)}"


def _metrics_path(remote_root, n, gen, cond_index: int, base_name):
    """Path to metrics file for a given test and condition index (1-based)"""
    return os.path.join(
        remote_root,
        "solutions",
        f"n_{gen}",
        f"cond_{cond_index}",
        f"{n}",
        f"{base_name}_{n}.rsd",
    )
    
def _strip_outer_minmax(expr_raw: str):
    """Return (sense, inner_expr). sense is min or max."""
    expr_raw = (expr_raw or "").strip()
    m = re.match(r"^\s*(min|max)\s*\(\s*(.*)\s*\)\s*$", expr_raw, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    return "min", expr_raw


def _build_objective_callable(objective_dict: dict):
    """
    Build a safe scalar objective evaluator.

    New format:
      objective_dict["expression"] may reference any symbols produced by:
        - .rsd parser: CL, CD, CM, CL_over_CD
        - monitor parser: objective symbols such as PR_s5, DC60_s5, CD_s2_3_4

    Backwards compatible with old objective_type values: Drag, Lift, Lift-to-Drag.
    Returned objective is always in minimisation form for BO.
    """
    obj_type = (objective_dict.get("objective_type", "") or "").strip()
    expr_raw = (objective_dict.get("expression", "") or "").strip()

    if not expr_raw or expr_raw.lower() == "drag" or obj_type.lower() == "drag":
        expr_raw = "CD"
    elif expr_raw.lower() == "lift" or obj_type.lower() == "lift":
        expr_raw = "-CL"
    elif expr_raw.lower() in ("lift-to-drag", "lift to drag") or obj_type.lower() in ("lift-to-drag", "lift to drag"):
        expr_raw = "-(CL/CD)"

    sense, inner = _strip_outer_minmax(expr_raw)
    # expression symbols cannot contain '/', so also expose CL_over_CD
    inner = inner.replace("CL/CD", "CL_over_CD")

    allowed_funcs = {
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": lambda x: float(np.sqrt(x)),
        "log": lambda x: float(np.log(x)),
        "exp": lambda x: float(np.exp(x)),
    }
    safe_globals = {"__builtins__": {}}
    safe_globals.update(allowed_funcs)

    def obj_func(mdict: dict) -> float:
        try:
            local_vars = {}
            for k, v in (mdict or {}).items():
                if re.match(r"^[A-Za-z_]\w*$", str(k)):
                    try:
                        local_vars[str(k)] = float(v)
                    except Exception:
                        pass
            nan = float("nan")
            local_vars.setdefault("CL", nan)
            local_vars.setdefault("CD", nan)
            local_vars.setdefault("CM", nan)
            if "CL_over_CD" not in local_vars:
                local_vars["CL_over_CD"] = local_vars["CL"] / local_vars["CD"] if local_vars["CD"] else nan
            val = float(eval(inner, safe_globals, local_vars))
            if not np.isfinite(val):
                # A missing metric must NEVER turn into a fake number: with the
                # old 1e9 default a maximised term (e.g. -0.75*PR) became
                # -7.5e8, i.e. the apparent global optimum. NaN is imputed
                # explicitly (and logged) in eval_func instead.
                print(f"[OBJECTIVE][ERROR] '{inner}' evaluated to {val} (missing metric?) metrics={mdict}", flush=True)
                return nan
            return -val if sense == "max" else val
        except Exception as e:
            print(f"[OBJECTIVE][ERROR] Could not evaluate '{inner}' with metrics={mdict}: {e}", flush=True)
            return float("nan")

    pretty = f"-{inner}" if sense == "max" else inner
    return obj_func, pretty


def _reduce_values(values, reduction="last", default=1e9, window_frac=0.30, min_window=5, osc_rel_tol=0.02, unstable_policy="last"):
    vals = []
    for v in values:
        try:
            fv = float(v)
            if np.isfinite(fv):
                vals.append(fv)
        except Exception:
            pass
    if not vals:
        return float(default)
    reduction = str(reduction or "last").lower()
    if reduction == "time_average":
        n = len(vals)
        w = max(int(np.ceil(window_frac * n)), min_window)
        w = min(w, n)

        tail = np.asarray(vals[-w:], dtype=float)
        mean = float(np.mean(tail))

        amp = float(np.max(tail) - np.min(tail))
        scale = max(abs(mean), 1e-12)
        rel_amp = amp / scale

        if rel_amp <= osc_rel_tol:
            return mean

        if unstable_policy == "penalty":
            return float(default)

        return float(vals[-1])
    return float(vals[-1])


def _metric_aliases(metric: str):
    m = str(metric or "").strip().lower()
    aliases = {
        "pressure_recovery": ["pressure_recovery", "pr", "p0_recovery"],
        "distortion": ["distortion", "dc60", "DC60"],
        "drag": ["drag", "CD", "cd", "duct_drag"],
        "lift": ["lift", "CL", "cl"],
        "moment": ["moment", "CM", "cm"],
        "CD": ["CD", "cd", "drag", "duct_drag"],
        "CL": ["CL", "cl", "lift"],
        "CM": ["CM", "cm", "moment"],
    }
    return aliases.get(metric, aliases.get(m, [metric, m]))


# ---------------------------------------------------------------------------
# Monitor-CSV parsing (module level so the --check mode can reuse it exactly)
#
# paraview_cluster.py writes ONE wide CSV row per post-processed iteration,
# with column names that depend on the monitor TYPE and its "name" in
# monitors.json -- NOT on the objective symbol:
#     pressure_recovery -> "<name>_pressure_recovery"
#     distortion/dc60   -> "<name>"
#     drag              -> "<name>_over_q"
# The previous lookup only tried exact alias names ("pressure_recovery",
# "pr", ...), so a PR term never matched "<name>_pressure_recovery" and its
# symbol silently fell back to 1e9. The resolver below maps a term to its
# column using, in order:
#   1. an explicit "column" key on the term (manual override)
#   2. monitors.json in the solution dir: monitor with same type AND same
#      surface_ids as the term -> column via the naming rule above
#   3. exact symbol / "<symbol><suffix>"
#   4. legacy aliases
#   5. a UNIQUE column ending in the type's suffix (ambiguity -> no match)
# ---------------------------------------------------------------------------
_MONITOR_TYPE_OF_METRIC = {
    "pressure_recovery": "pressure_recovery", "pressure recovery": "pressure_recovery",
    "pr": "pressure_recovery", "p0_recovery": "pressure_recovery",
    "distortion": "distortion", "dc60": "distortion", "distortion_dc60": "distortion",
    "drag": "drag", "cd": "drag", "duct_drag": "drag",
}
# column suffix appended by paraview_cluster.main() for each monitor type
_COLUMN_SUFFIX = {"pressure_recovery": "_pressure_recovery", "distortion": "", "drag": "_over_q"}


def _canon_monitor_type(t):
    return _MONITOR_TYPE_OF_METRIC.get(str(t or "").strip().lower(), str(t or "").strip().lower())


def _load_monitor_cfg(sol_dir):
    p = os.path.join(sol_dir, "Monitors", "monitors.json")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _resolve_term_columns(term, columns, monitor_cfg):
    """Return (list_of_candidate_columns_in_priority_order, how)."""
    cols = [c for c in columns if c]
    colset = set(cols)
    symbol = str(term.get("symbol", "")).strip()
    mtype = _canon_monitor_type(term.get("metric", ""))
    suffix = _COLUMN_SUFFIX.get(mtype, "")

    explicit = str(term.get("column", "") or "").strip()
    if explicit:
        return ([explicit] if explicit in colset else []), f"explicit column '{explicit}'"

    want_sids = sorted(int(s) for s in (term.get("surface_ids") or []))
    for mon in (monitor_cfg or {}).get("monitors", []) or []:
        if not mon.get("enabled", True):
            continue
        if _canon_monitor_type(mon.get("type", "")) != mtype:
            continue
        if sorted(int(s) for s in (mon.get("surface_ids") or [])) != want_sids:
            continue
        name = str(mon.get("name", mon.get("type", ""))).strip() or str(mon.get("type", ""))
        col = f"{name}{suffix}"
        if col in colset:
            return [col], f"monitors.json match (name='{name}', surface_ids={want_sids})"

    for cand in (symbol, f"{symbol}{suffix}"):
        if cand and cand in colset:
            return [cand], "symbol match"

    for alias in _metric_aliases(term.get("metric", "")):
        if alias in colset:
            return [alias], f"legacy alias '{alias}'"

    if suffix:
        ends = [c for c in cols if c.endswith(suffix)]
        if len(ends) == 1:
            return ends, f"unique '*{suffix}' column"
        if len(ends) > 1:
            return [], f"AMBIGUOUS: {ends} all end in '{suffix}' -- set \"column\" on the term"
    return [], "no matching column"


def parse_rsd(path):
    """Last row of <base>_<n>.rsd -> CL/CD/CM (existing convention: tokens[2..4]).
    Missing/unreadable file -> NaN (NOT a fake 'good' number)."""
    nan = float("nan")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        last = None
        for raw in reversed(lines):
            toks = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
            if len(toks) >= 4:
                last = toks
                break
        if not last:
            print(f"[CLUSTER-TM][WARN] No numeric rows in {path}", flush=True)
            return {"CL": nan, "CD": nan, "CM": nan, "CL_over_CD": nan}
        CL = float(last[2]) if len(last) > 2 else nan
        CD = float(last[3]) if len(last) > 3 else nan
        CM = float(last[4]) if len(last) > 4 else nan
        return {"CL": CL, "CD": CD, "CM": CM, "CL_over_CD": CL / CD if CD else nan}
    except Exception as e:
        print(f"[CLUSTER-TM][WARN] Error parsing {path}: {e}", flush=True)
        return {"CL": nan, "CD": nan, "CM": nan, "CL_over_CD": nan}


def read_monitor_rows(sol_dir):
    import csv
    candidates = [
        os.path.join(sol_dir, "Monitors", "monitors.csv"),
        os.path.join(sol_dir, "Monitors", "pressure_recovery.csv"),
    ]
    rows = []
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                sample = f.read(4096)
                f.seek(0)
                if "," in sample and any(h in sample.lower() for h in ["iter", "pressure", "drag", "distortion", "dc60"]):
                    rdr = csv.DictReader(f)
                    rows.extend([{k: v for k, v in row.items()} for row in rdr])
                else:
                    for line in f:
                        toks = [x.strip() for x in line.split(",")]
                        if len(toks) >= 2:
                            rows.append({"pressure_recovery": toks[-1]})
        except Exception as e:
            print(f"[CLUSTER-TM][WARN] Failed reading monitor csv {path}: {e}", flush=True)
    return rows


def parse_monitors(sol_dir, base_metrics, objective_terms):
    metrics = dict(base_metrics)
    rows = read_monitor_rows(sol_dir)
    monitor_terms = [t for t in (objective_terms or []) if str(t.get("source", "")).lower() == "monitor"]
    if not rows:
        if monitor_terms:
            print(f"[CLUSTER-TM][WARN] No monitor rows in {sol_dir}/Monitors -- "
                  f"monitor symbols {[t.get('symbol') for t in monitor_terms]} set to NaN", flush=True)
        for t in monitor_terms:
            if t.get("symbol"):
                metrics[str(t["symbol"]).strip()] = float("nan")
        return metrics

    # Raw columns (sanitised names), last finite value -- unchanged behaviour.
    by_col = {}
    for row in rows:
        for k, v in row.items():
            if k is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            by_col.setdefault(str(k).strip(), []).append(fv)
    for k, vals in by_col.items():
        safe = re.sub(r"\W+", "_", k).strip("_")
        metrics[safe] = _reduce_values(vals, "last", default=float("nan"))

    columns = []
    for row in rows:
        for k in row.keys():
            if k is not None and k not in columns:
                columns.append(k)
    monitor_cfg = _load_monitor_cfg(sol_dir)

    for term in monitor_terms:
        symbol = str(term.get("symbol", "")).strip()
        if not symbol:
            continue
        reduction = term.get("reduction", "last")
        vals = []
        # Legacy long-format CSV (one row per monitor with a name column)
        for row in rows:
            row_name = str(row.get("name", row.get("monitor", row.get("objective_symbol", "")))).strip()
            if row_name and row_name == symbol:
                for alias in _metric_aliases(term.get("metric", "")) + ["value", symbol]:
                    if alias in row:
                        vals.append(row.get(alias))
        how = "long-format rows"
        if not vals:
            cols, how = _resolve_term_columns(term, columns, monitor_cfg)
            for row in rows:
                for c in cols:
                    if c in row:
                        vals.append(row.get(c))
        metrics[symbol] = _reduce_values(vals, reduction, default=float("nan"))
        if not np.isfinite(metrics[symbol]):
            print(f"[CLUSTER-TM][WARN] {sol_dir}: symbol '{symbol}' unresolved/non-finite "
                  f"({how}); available columns={columns}", flush=True)
        else:
            print(f"[CLUSTER-TM] {os.path.basename(os.path.normpath(sol_dir))}: {symbol} = "
                  f"{metrics[symbol]:.6g} via {how}", flush=True)
    return metrics


class ClusterTestManager:
    """
    Test manager that runs ON THE CLUSTER.
    Uses ClusterPipelineManager (no SSH/SFTP).
    """
    def __init__(self, remote_root, base_name, input_dir, executables, poll_s=120, morph_basis_json="", units="mm", parallel=80, monitor_config_json="", previous_solution=None):
        self.remote_root = os.path.abspath(remote_root)
        self.base_name = base_name
        self.input_dir = input_dir
        self.executables = executables
        self.poll_s = int(max(10, poll_s))
        self.jobs = {}
        self.morph_basis_json = morph_basis_json or ""
        self.units = units
        self.parallel = parallel
        self.monitor_config_json = monitor_config_json or ""
        self.previous_solution = previous_solution or {}
        
        # Create logs directory
        self.log_dir = os.path.join(self.remote_root, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _alloc_n_index(self, gen_num, local_idx):
        """Generate unique n-index for (generation, design) pair"""
        return int(local_idx)
    
    def _start_one(self, gen_num, n_index, x, conds):
        """Start pipeline for one design point"""
        print(f"[CLUSTER-TM] Starting gen={gen_num} n={n_index} with x={x}", flush=True)

        self.remote_output = self.remote_root

        config = {
            "remote_output": self.remote_output,
            "base_name": self.base_name,
            "input_dir": self.input_dir,
            "modal_coeffs": list(map(float, x)),
            "morph_basis_json": self.morph_basis_json,
            "cad_units": self.units,
            "parallel_processes": self.parallel,
            "monitor_config_json": self.monitor_config_json,
            "previous_solution": self.previous_solution,
            **self.executables,
        }

        # gen must be the BO generation number
        pipe = ClusterPipelineManager(config, gen=int(gen_num), n=int(n_index))

        try:
            morph_id = pipe.morph(n=n_index)
            vol_id   = pipe.volume(runafter=morph_id)
            pre_id   = pipe.prepro(runafter=vol_id)

            sol_ids = []
            for i, cond in enumerate(conds, 1):
                jid = pipe.solver(cond, nc=i)
                sol_ids.append(jid)

            self.jobs[n_index] = {
                "gen": int(gen_num),
                "morph": morph_id,
                "volume": vol_id,
                "prepro": pre_id,
                "solvers": sol_ids,
            }

            print(f"[CLUSTER-TM] Submitted gen={gen_num} n={n_index} -> jobs={self.jobs[n_index]}", flush=True)
            return sol_ids[-1] if sol_ids else None

        except Exception as e:
            print(f"[CLUSTER-TM] ERROR starting gen={gen_num} n={n_index}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None
    
    def init_generation(self, X_list, gen_num, conds):
        """Submit all designs for a generation"""
        print(f"[CLUSTER-TM] Initializing generation {gen_num} with {len(X_list)} designs", flush=True)

        for i, x in enumerate(X_list):
            n_index = self._alloc_n_index(gen_num, i + 1)
            self._start_one(gen_num, n_index, x, conds)
    
    def evaluate_generation(self, X_list, gen_num, conds):
        """Wait for solver completion markers, then parse .rsd plus monitor CSVs."""
        num_conds = len(conds)

        def _sol_dir(n_index: int, nc: int) -> str:
            return os.path.join(self.remote_root, "solutions", f"n_{gen_num}", f"cond_{nc}", f"{n_index}")

        def _done_path(n_index: int, nc: int) -> str:
            return os.path.join(_sol_dir(n_index, nc), "SOLVER_DONE")

        def _rsd_path(n_index: int, nc: int) -> str:
            return os.path.join(_sol_dir(n_index, nc), f"{self.base_name}_{n_index}.rsd")

        need_done = []
        for i, _x in enumerate(X_list, 1):
            n_index = self._alloc_n_index(gen_num, i)
            for nc in range(1, num_conds + 1):
                need_done.append((_done_path(n_index, nc), n_index, nc))

        print(f"[CLUSTER-TM] Waiting for {len(need_done)} SOLVER_DONE markers.", flush=True)
        unfinished = set(p for (p, _, _) in need_done)
        while unfinished:
            done_now = {p for p in list(unfinished) if os.path.exists(p)}
            unfinished -= done_now
            if unfinished:
                example = next(iter(unfinished))
                print(f"[CLUSTER-TM] Still waiting for {len(unfinished)} markers... e.g. {example}", flush=True)
                time.sleep(self.poll_s)

        print("[CLUSTER-TM] All SOLVER_DONE markers present. Parsing objective metrics.", flush=True)

        results = []
        for i, _x in enumerate(X_list, 1):
            n_index = self._alloc_n_index(gen_num, i)
            per_cond = []
            for nc in range(1, num_conds + 1):
                sol_dir = _sol_dir(n_index, nc)
                m = parse_rsd(_rsd_path(n_index, nc))
                m = parse_monitors(sol_dir, m, getattr(self, "objective_terms", []))
                per_cond.append(m)
                print(f"[CLUSTER-TM] gen={gen_num} n={n_index} cond={nc} metrics={m}", flush=True)
            results.append(per_cond)
        return results

_RSD_SYMBOLS = {"CL", "CD", "CM", "CL_over_CD"}
_EXPR_FUNCS = {"abs", "min", "max", "pow", "sqrt", "log", "exp"}


def _expression_symbols(expr):
    """Identifiers referenced by an objective/constraint expression."""
    import ast
    _, inner = _strip_outer_minmax(expr)
    inner = inner.replace("CL/CD", "CL_over_CD")
    try:
        tree = ast.parse(inner, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Expression {expr!r} is not valid Python syntax: {e}")
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - _EXPR_FUNCS


def _check_objective_symbols(objective, constraints):
    """Return a list of problems: symbols used in the expression/constraints
    that no term (or the .rsd parser) will ever produce. This is exactly the
    '-PR_s111' vs symbol 'PR' mismatch -- catch it at launch, not after
    days of CFD."""
    from Optimisation.BayesianOptimisation.optimiser import _parse_constraint
    provided = set(_RSD_SYMBOLS)
    for t in objective.get("terms", []) or []:
        if t.get("symbol"):
            provided.add(str(t["symbol"]).strip())
    problems = []
    expr = (objective.get("expression") or "").strip()
    if expr and expr.lower() not in ("drag", "lift", "lift-to-drag", "lift to drag"):
        missing = _expression_symbols(expr) - provided
        if missing:
            problems.append(f"expression '{expr}' uses {sorted(missing)}; terms provide {sorted(provided)}")
    for c in constraints or []:
        metric = _parse_constraint(c)["metric"]
        missing = _expression_symbols(metric) - provided
        if missing:
            problems.append(f"constraint '{c}' uses {sorted(missing)}; terms provide {sorted(provided)}")
    return problems


def _impute_nonfinite(new_vals, history, direction, label, log):
    """Replace NaN/inf in new_vals with a finite, deliberately BAD value:
    worst finite observation (history + this batch) plus 10% of the observed
    spread. direction=+1 -> larger is worse (minimised objective, '<='
    constraint); direction=-1 -> smaller is worse ('>=' constraint).
    Keeps the GP finite (NaN would poison C_mean/Y_mean for every point)
    while steering the search away from the failed design."""
    new_vals = np.asarray(new_vals, dtype=float).copy()
    bad = ~np.isfinite(new_vals)
    if not np.any(bad):
        return new_vals
    pool = np.concatenate([np.asarray(history, dtype=float).ravel(), new_vals[~bad]])
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        fill = 1e9 * direction
        log(f"[REMOTE-OPT][WARN] {label}: no finite values anywhere; imputing {fill:g}")
    else:
        worst = np.max(pool) if direction > 0 else np.min(pool)
        spread = float(np.ptp(pool)) if pool.size > 1 else abs(float(worst)) * 0.1
        fill = float(worst + direction * 0.1 * max(spread, 1e-12))
    log(f"[REMOTE-OPT][WARN] {label}: {int(bad.sum())} non-finite value(s) at batch index "
        f"{np.where(bad)[0].tolist()} -> imputed {fill:.6g} (treated as failed design)")
    new_vals[bad] = fill
    return new_vals


def check_mode(argv):
    """
    python remoteOpt.py --check <run_dir> <sol_dir> [<sol_dir> ...]

    Runs the EXACT metric/objective/constraint code path used during the
    optimisation on existing solution folders (single condition each), and
    prints the result. Use it on the baseline and a few designs from the
    Excel sheet before launching, to confirm CD / PR / DC60 match.
    """
    from Optimisation.BayesianOptimisation.objective_evaluator import ConstraintSet
    run_dir = os.path.abspath(argv[0])
    with open(os.path.join(run_dir, "objective.json")) as f:
        objective = json.load(f)
    with open(os.path.join(run_dir, "bo_settings.json")) as f:
        settings_json = json.load(f)
    constraints = settings_json.get("constraints") or objective.get("constraints", [])
    problems = _check_objective_symbols(objective, constraints)
    for p in problems:
        print(f"[CHECK][SYMBOL-ERROR] {p}")
    obj_func, pretty = _build_objective_callable(objective)
    cset = ConstraintSet(constraints)
    terms = objective.get("terms", []) or []
    base_name = settings_json.get("base_name", "model")
    print(f"[CHECK] objective (minimised): {pretty} | constraints: {constraints}")
    for sol_dir in argv[1:]:
        sol_dir = os.path.abspath(sol_dir)
        rsd = [f for f in os.listdir(sol_dir) if f.startswith(base_name) and f.endswith(".rsd")]
        rsd_path = os.path.join(sol_dir, sorted(rsd)[0]) if rsd else os.path.join(sol_dir, "missing.rsd")
        m = parse_monitors(sol_dir, parse_rsd(rsd_path), terms)
        shown = {k: m[k] for k in m if k in _RSD_SYMBOLS or any(k == t.get("symbol") for t in terms)}
        c = cset.evaluate([m])
        print(f"[CHECK] {sol_dir}\n        rsd={os.path.basename(rsd_path)} metrics={shown}\n"
              f"        objective={obj_func(m)} constraints={c}")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--check":
        if len(sys.argv) < 4:
            print("Usage: remoteOpt.py --check <run_dir> <sol_dir> [<sol_dir> ...]", flush=True)
            sys.exit(2)
        check_mode(sys.argv[2:])
        return

    if len(sys.argv) < 2:
        print("Usage: remoteOpt.py <run_directory>", flush=True)
        sys.exit(2)

    run_dir = os.path.abspath(sys.argv[1])
    log_path = os.path.join(run_dir, "remote_opt.log")
    os.makedirs(run_dir, exist_ok=True)
    
    _log(f"[REMOTE-OPT] Starting in {run_dir}", log_path)
    
    # Load configurations
    settings_path = os.path.join(run_dir, "bo_settings.json")
    objective_path = os.path.join(run_dir, "objective.json")
    
    if not os.path.exists(settings_path):
        _log(f"[ERROR] Settings file not found: {settings_path}", log_path)
        sys.exit(1)
    
    if not os.path.exists(objective_path):
        _log(f"[ERROR] Objective file not found: {objective_path}", log_path)
        sys.exit(1)
    
    with open(settings_path) as f:
        settings_json = json.load(f)
    
    with open(objective_path) as f:
        objective = json.load(f)
        
    morph_basis_json = settings_json.get("morph_basis_json", "")
    monitor_config_json = settings_json.get("monitor_config_json", "")
    
    _log(f"[REMOTE-OPT] Loaded settings: {settings_json}", log_path)
    _log(f"[REMOTE-OPT] Loaded objective: {objective}", log_path)
    
    # Import BO components
    from Optimisation.BayesianOptimisation.optimiser import BayesianOptimiser
    from Optimisation.BayesianOptimisation.kernels import (
        RBFKernel, SquaredExponentialKernel, ExponentialKernel, 
        Mat12Kern, Mat32Kern, Mat52Kern
    )
    from Optimisation.BayesianOptimisation.acquisition_functions import EI, POI, UCB
    
    # Map string names to classes
    kern_map = {
        "RBFKernel": RBFKernel,
        "Squared Exponential Kernel": SquaredExponentialKernel,
        "Exponential Kernel": ExponentialKernel,
        "Mat12Kern": Mat12Kern,
        "Mat32Kern": Mat32Kern,
        "Mat52Kern": Mat52Kern
    }
    
    acq_map = {
        "Expected Improvement": EI,
        "Probability of Improvement": POI,
        "Upper Confidence Bound": UCB
    }
    
    # Prepare settings
    settings = dict(settings_json)
    settings["kernel"] = kern_map[settings_json["kernel"]]
    settings["acquisition_function"] = acq_map[settings_json["acquisition_function"]]
    settings["sim_dir"] = run_dir
    
    # Get conditions and weights
    conds = objective.get("conditions", [])
    weights = [c.get("Weight", 1.0) for c in conds]
    
    # Build objective function from GUI config (Drag/Lift/Lift-to-Drag/Custom)
    obj_func, obj_expr = _build_objective_callable(objective)
    objective_terms = objective.get("terms", []) or []
    _log(f"[REMOTE-OPT] Objective expression (minimised): {obj_expr}", log_path)
    _log(f"[REMOTE-OPT] Objective terms: {objective_terms}", log_path)

    # ---- Constraints ----
    # Previously objective.json["constraints"] was never forwarded: the
    # optimiser only reads settings["constraints"], which this run's
    # bo_settings.json did not contain -> BO ran UNCONSTRAINED. Take them
    # from objective.json when bo_settings.json has none.
    obj_cons = list(objective.get("constraints", []) or [])
    set_cons = list(settings_json.get("constraints", []) or [])
    if set_cons and obj_cons and set_cons != obj_cons:
        _log(f"[REMOTE-OPT][WARN] bo_settings.json constraints {set_cons} differ from "
             f"objective.json constraints {obj_cons}; using bo_settings.json.", log_path)
    constraints = set_cons or obj_cons
    settings["constraints"] = constraints
    from Optimisation.BayesianOptimisation.objective_evaluator import ConstraintSet
    from Optimisation.BayesianOptimisation.optimiser import _parse_constraint
    cset = ConstraintSet(constraints)
    cons_parsed = [_parse_constraint(c) for c in constraints]
    _log(f"[REMOTE-OPT] Constraints: {constraints if constraints else 'NONE (unconstrained)'}", log_path)

    problems = _check_objective_symbols(objective, constraints)
    for p in problems:
        _log(f"[REMOTE-OPT][SYMBOL-ERROR] {p}", log_path)
    if problems and not settings_json.get("allow_unknown_symbols", False):
        _log("[REMOTE-OPT][ERROR] Aborting before any CFD is submitted. Fix objective.json "
             "(or set \"allow_unknown_symbols\": true in bo_settings.json if you reference raw "
             "monitor column names directly).", log_path)
        sys.exit(1)
    
    _log(f"[REMOTE-OPT] Conditions: {conds}", log_path)
    _log(f"[REMOTE-OPT] Weights: {weights}", log_path)
    
    # Determine remote root (parent of run_dir usually)
    # Adjust this based on your directory structure
    remote_root = settings_json.get("remote_root", os.path.dirname(run_dir))
    base_name = settings_json.get("base_name", "model")
    input_dir = settings_json.get("input_dir", os.path.join(remote_root, "orig"))
    parallel = settings_json.get("parallel", 80)
    cad_units = settings_json.get("units", "mm")
    previous_solution = settings_json.get("previous_solution", {}) or {}
    
    # Executable paths (customize for your cluster)
    executables = {
        "parallel_domains": settings_json.get("parallel_domains", 1),
        "surface_mesher": "/home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Surf3D",
        "volume_mesher": "/home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Mesh3D",
        "prepro_exe": "/home/s.engevabj/codes/PrePro_uns/Gen3d",
        "solver_exe": "/home/s.engevabj/codes/FLITE_uns/UnsMgnsg3d",
        "combine_exe": "/home/s.engevabj/codes/utilities/makeplot2",
        "ensight_exe": "/home/s.engevabj/codes/utilities/engen_tet",
        "splitplot_exe": "/home/s.engevabj/codes/utilities/splitplot2",
        "makeplot_exe": "/home/s.engevabj/codes/utilities/makeplot2",
        "intel_module": "module load compiler/intel/2020/0",
        "gnu_module": "module load compiler/gnu/12/1.0",
        "mpi_intel_module": "module load mpi/intel/2020/0",
        "interpu_script": "$HOME/aeropt/Scripts/Utilities/interpu.py",
    }
    
    # Create test manager (uses ClusterPipelineManager internally)
    tm = ClusterTestManager(
        remote_root=remote_root,
        base_name=base_name,
        input_dir=input_dir,
        executables=executables,
        poll_s=settings_json.get("poll_interval", 120),
        morph_basis_json=morph_basis_json,
        units = cad_units,
        parallel = parallel,
        monitor_config_json=monitor_config_json,
        previous_solution=previous_solution
    )
    tm.objective_terms = objective_terms
    
    # Define init and eval functions for BO
    def init_func(X_list, gen_num):
        _log(f"[REMOTE-OPT] Initializing generation {gen_num}: {len(X_list)} designs", log_path)
        tm.init_generation(X_list, gen_num, conds)
    
    def eval_func(X_list, gen_num):
        _log(f"[REMOTE-OPT] Evaluating generation {gen_num}", log_path)
        per_design = tm.evaluate_generation(X_list, gen_num, conds)

        # Reduce per-condition metrics -> scalar objective via expression
        Y = []
        for metrics_per_cond in per_design:
            y = 0.0
            for cond, m in zip(conds, metrics_per_cond):
                w = float(cond.get("Weight", 1.0))
                y += w * obj_func(m)          # NaN propagates -> imputed below
            Y.append(float(y))

        logf = lambda msg: _log(msg, log_path)
        Y = _impute_nonfinite(Y, bo.Y, +1, f"gen {gen_num} objective", logf)

        if not cons_parsed:
            _log(f"[REMOTE-OPT] Generation {gen_num} objectives: {Y.tolist()}", log_path)
            return Y

        # Constraint values: worst case across conditions (ConstraintSet),
        # returned as the (Y, C) tuple BayesianOptimiser expects.
        per_point = [cset.evaluate(mpc) for mpc in per_design]
        C = {}
        for cons in cons_parsed:
            name = cons["metric"]
            vals = np.array([pp.get(name, np.nan) for pp in per_point], dtype=float)
            direction = +1 if cons["sense"] == "<=" else -1
            hist = bo.C.get(name, np.array([]))
            if np.any(~np.isfinite(vals)):
                # make sure an imputed value is on the infeasible side
                pool = np.concatenate([np.asarray(hist, float).ravel(), vals[np.isfinite(vals)], [cons["limit"]]])
                vals = _impute_nonfinite(vals, pool, direction, f"gen {gen_num} constraint {name}", logf)
            C[name] = vals

        for i, y in enumerate(Y):
            desc = ", ".join(f"{k}={C[k][i]:.4g}" for k in C)
            feas = all((C[c['metric']][i] <= c['limit']) if c['sense'] == '<=' else (C[c['metric']][i] >= c['limit'])
                       for c in cons_parsed)
            _log(f"[REMOTE-OPT] gen {gen_num} point {i + 1}: Y={y:.6g} | {desc} | feasible={feas}", log_path)
        return Y, C
    
    # Run Bayesian Optimization
    _log("[REMOTE-OPT] Starting Bayesian Optimization...", log_path)
    bo = BayesianOptimiser(settings, eval_func=eval_func, init_func=init_func)
    X_best, Y_best = bo.optimise(cont=True)
    
    _log(f"[REMOTE-OPT] OPTIMIZATION COMPLETE!", log_path)
    _log(f"[REMOTE-OPT] Best X = {X_best}", log_path)
    _log(f"[REMOTE-OPT] Best Y = {Y_best}", log_path)
    
    # Save final results
    results_file = os.path.join(run_dir, "optimization_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "X_best": X_best.tolist() if hasattr(X_best, 'tolist') else X_best,
            "Y_best": float(Y_best),
            "settings": settings_json,
            "objective": objective
        }, f, indent=2)
    
    _log(f"[REMOTE-OPT] Results saved to {results_file}", log_path)


if __name__ == "__main__":
    main()