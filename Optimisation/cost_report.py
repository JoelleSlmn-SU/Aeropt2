"""
cost_report.py -- what an optimisation run cost, per design and in total.

For comparing optimisers fairly the cost must be counted the same way for every
run. This writes, into the run folder:

  cost_per_design.csv   one row per evaluated design (generation, n):
                        core-hours and elapsed time per stage, queue wait,
                        solver restarts, plus running totals
  cost_summary.json     totals: designs, core-hours (CFD and all), wall-clock
                        span, optimiser overhead (GP fits + acquisition search,
                        from optimiser_timing.csv), and the core-hours per design

Cluster (3D chain): read from Slurm accounting (sacct) by JOB NAME, so solver
restarts submitted by the convergence guard are counted too. Names:
    morph_n<gen>_<n>, vol_n<gen>_<n>, pre_n<gen>_<n>,
    sol_n<gen>_<n>_cond_<k>, solrestart_n<gen>_<n>_cond_<k>, guard_n<gen>_<n>_cond_<k>
core-hours = CPUTimeRAW / 3600 (= elapsed x allocated CPUs, what the
allocation is charged). Queue wait = Start - Submit.

Local (2D chain): wall time per design from designs.json; core-hours =
wall time x `cores` (FLITE 2D runs on one core).

    python Optimisation/cost_report.py cluster <run_dir> [--since 2026-11-01T00:00]
    python Optimisation/cost_report.py local2d <run_dir> [--cores 1]

remoteOpt.py and local2DOpt.py call these after every generation (errors are
logged, never raised), so the files are current while a run is going.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

JOB_RE = re.compile(r"^(?P<stage>morph|vol|pre|sol|solrestart|guard)_n(?P<gen>\d+)_(?P<n>\d+)(?:_cond_(?P<cond>\d+))?$")
STAGES = ("morph", "vol", "pre", "sol", "solrestart", "guard")
SACCT_FIELDS = "JobID,JobName,State,ElapsedRaw,AllocCPUS,CPUTimeRAW,Submit,Start,End"
STARTED_FILE = "run_started.txt"


def mark_run_start(run_dir: str) -> str:
    """Record when the run started (first call only); returns the ISO time."""
    p = os.path.join(run_dir, STARTED_FILE)
    if os.path.exists(p):
        with open(p) as f:
            return f.read().strip()
    t = datetime.now().replace(microsecond=0).isoformat()
    with open(p, "w") as f:
        f.write(t + "\n")
    return t


def _ts(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(s).strip())
    except ValueError:
        return None


def parse_sacct(text: str) -> List[dict]:
    """Rows of `sacct -X -P -n --format=SACCT_FIELDS` that belong to the pipeline."""
    rows = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 9:
            continue
        jid, name, state, el, cpus, cput, sub, start, end = parts[:9]
        m = JOB_RE.match(name.strip())
        if not m:
            continue
        sub_t, start_t, end_t = _ts(sub), _ts(start), _ts(end)
        rows.append({
            "job_id": jid, "name": name, "stage": m["stage"], "gen": int(m["gen"]), "n": int(m["n"]),
            "cond": int(m["cond"]) if m["cond"] else None, "state": state.split()[0] if state else "",
            "elapsed_s": float(el or 0), "cpus": int(cpus or 0), "core_h": float(cput or 0) / 3600.0,
            "queue_s": (start_t - sub_t).total_seconds() if (sub_t and start_t) else None,
            "submit": sub, "start": start, "end": end,
        })
    return rows


def _optimiser_overhead(run_dir: str) -> Dict[str, float]:
    p = os.path.join(run_dir, "optimiser_timing.csv")
    fit = acq = 0.0
    steps = 0
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                fit += float(r.get("gp_fit_s") or 0)
                acq += float(r.get("acquisition_s") or 0)
                steps += 1
    return {"optimiser_steps": steps, "gp_fit_s": round(fit, 3), "acquisition_s": round(acq, 3)}


def _restarts(remote_root: str, gen: int, n: int) -> Optional[int]:
    tot, seen = 0, False
    base = os.path.join(remote_root, "solutions", f"n_{gen}")
    if not os.path.isdir(base):
        return None
    for cdir in os.listdir(base):
        p = os.path.join(base, cdir, str(n), "convergence_state.json")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    tot += int(json.load(f).get("attempt", 0))
                seen = True
            except (OSError, ValueError):
                pass
    return tot if seen else None


def cluster_report(run_dir: str, remote_root: Optional[str] = None, since: Optional[str] = None,
                   sacct_text: Optional[str] = None) -> dict:
    run_dir = os.path.abspath(run_dir)
    remote_root = remote_root or os.path.dirname(run_dir)
    if sacct_text is None:
        since = since or mark_run_start(run_dir)
        cmd = ["sacct", "-X", "-P", "-n", f"--format={SACCT_FIELDS}", "-S", since]
        user = os.environ.get("USER")
        if user:
            cmd += ["-u", user]
        sacct_text = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    jobs = parse_sacct(sacct_text)
    with open(os.path.join(run_dir, "cost_jobs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()) if jobs else ["job_id"])
        w.writeheader()
        w.writerows(jobs)

    per = defaultdict(lambda: {**{f"{s}_core_h": 0.0 for s in STAGES}, **{f"{s}_elapsed_s": 0.0 for s in STAGES},
                               "queue_s": 0.0, "first_submit": None, "last_end": None, "states": set()})
    for j in jobs:
        d = per[(j["gen"], j["n"])]
        d[f"{j['stage']}_core_h"] += j["core_h"]
        d[f"{j['stage']}_elapsed_s"] += j["elapsed_s"]
        d["queue_s"] += j["queue_s"] or 0.0
        d["states"].add(j["state"])
        s, e = _ts(j["submit"]), _ts(j["end"])
        if s and (d["first_submit"] is None or s < d["first_submit"]):
            d["first_submit"] = s
        if e and (d["last_end"] is None or e > d["last_end"]):
            d["last_end"] = e

    rows, cum_cfd, cum_all = [], 0.0, 0.0
    for (gen, n) in sorted(per):
        d = per[(gen, n)]
        cfd = d["sol_core_h"] + d["solrestart_core_h"]
        allh = sum(d[f"{s}_core_h"] for s in STAGES)
        cum_cfd += cfd
        cum_all += allh
        span = ((d["last_end"] - d["first_submit"]).total_seconds()
                if d["first_submit"] and d["last_end"] else None)
        rows.append({
            "generation": gen, "n": n, **{k: round(v, 4) for k, v in d.items() if k.endswith("_core_h")},
            "cfd_core_h": round(cfd, 4), "total_core_h": round(allh, 4),
            "solver_elapsed_s": d["sol_elapsed_s"] + d["solrestart_elapsed_s"],
            "queue_s": round(d["queue_s"], 1), "submit_to_end_s": span,
            "restarts": _restarts(remote_root, gen, n), "states": ";".join(sorted(d["states"])),
            "cumulative_cfd_core_h": round(cum_cfd, 4), "cumulative_total_core_h": round(cum_all, 4),
        })
    _write_rows(os.path.join(run_dir, "cost_per_design.csv"), rows)
    firsts = [per[k]["first_submit"] for k in per if per[k]["first_submit"]]
    lasts = [per[k]["last_end"] for k in per if per[k]["last_end"]]
    summary = {
        "chain": "cluster", "designs": len(rows), "cfd_core_h": round(cum_cfd, 3), "total_core_h": round(cum_all, 3),
        "cfd_core_h_per_design": round(cum_cfd / len(rows), 3) if rows else None,
        "solver_restarts": sum(r["restarts"] or 0 for r in rows),
        "wall_clock_h": round((max(lasts) - min(firsts)).total_seconds() / 3600, 3) if firsts and lasts else None,
        "queue_h": round(sum(r["queue_s"] for r in rows) / 3600, 3),
        **_optimiser_overhead(run_dir),
    }
    _write_json(os.path.join(run_dir, "cost_summary.json"), summary)
    return summary


def local2d_report(run_dir: str, cores: int = 1) -> dict:
    run_dir = os.path.abspath(run_dir)
    with open(os.path.join(run_dir, "designs.json")) as f:
        designs = sorted(json.load(f)["designs"], key=lambda r: r["n"])
    rows, cum = [], 0.0
    for d in designs:
        if not d.get("done"):
            continue
        wall = float(d.get("wall_time_s") or 0.0)
        cum += wall
        its = sum(int(m.get("iterations") or 0) for m in (d.get("metrics") or []))
        rows.append({"generation": d.get("gen"), "n": d["n"], "status": d.get("status"),
                     "wall_s": wall, "core_h": round(wall * cores / 3600, 5), "solver_iterations": its,
                     "objective": d.get("objective"), "cumulative_wall_s": round(cum, 1),
                     "cumulative_core_h": round(cum * cores / 3600, 5)})
    _write_rows(os.path.join(run_dir, "cost_per_design.csv"), rows)
    summary = {
        "chain": "local2d", "designs": len(rows), "cores": cores,
        "cfd_core_h": round(cum * cores / 3600, 4), "total_core_h": round(cum * cores / 3600, 4),
        "cfd_core_h_per_design": round(cum * cores / 3600 / len(rows), 5) if rows else None,
        "wall_clock_h": round(cum / 3600, 4),
        "solver_iterations": sum(r["solver_iterations"] for r in rows),
        "designs_without_result": sum(1 for r in rows if r["objective"] is None),
        **_optimiser_overhead(run_dir),
    }
    _write_json(os.path.join(run_dir, "cost_summary.json"), summary)
    return summary


def _write_rows(path, rows):
    with open(path, "w", newline="") as f:
        if not rows:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chain", choices=["cluster", "local2d"])
    ap.add_argument("run_dir")
    ap.add_argument("--since", default=None)
    ap.add_argument("--remote-root", default=None)
    ap.add_argument("--cores", type=int, default=1)
    a = ap.parse_args(argv)
    s = (cluster_report(a.run_dir, a.remote_root, a.since) if a.chain == "cluster"
         else local2d_report(a.run_dir, a.cores))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    sys.exit(main())
