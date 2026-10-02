#!/usr/bin/env python3
"""
run_surface_sensitivity.py
==========================

Driver for surface_sensitivity.py on an Aeropt2 DOE folder.

It
  1. rebuilds <DOE>/designs.csv from the numbered case folders + the 'Results' sheet
     (so newly added optimiser designs are picked up automatically - just drop the new
     case folder in and add its row to the Results sheet),
  2. runs a fixed set of analyses (see RUNS below), each in its own output folder,
  3. writes a side-by-side skill table (summary_skill.csv) so you can see at a glance
     which objectives have an interpretable shape signal and whether the conclusion
     survives removing the old parameterisation or the optimiser designs.

A run is skipped if its output is newer than designs.csv and the surface_sensitivity.py
source (resume behaviour); use --force to rerun everything.

Usage (Windows / Anaconda prompt, from the folder that contains both scripts):

    python run_surface_sensitivity.py --doe "C:\\...\\Aeropt2\\examples\\DSI DOE"

    python run_surface_sensitivity.py --doe "DSI DOE" --runs all_T_ctrl newparam_T_ctrl
    python run_surface_sensitivity.py --doe "DSI DOE" --force --quick   # fast smoke test

On the cluster (one core is enough; ~1-2 min per run for 34 x 245k-node surfaces):

    sbatch -n 1 -t 00:30:00 --mem=8G --wrap "python run_surface_sensitivity.py --doe DSI_DOE"

Everything the runs need is in surface_sensitivity.py (same folder); no Aeropt2 imports.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "surface_sensitivity.py")

# ---------------------------------------------------------------------------
# Defaults for this study - edit here, not in the engine
# ---------------------------------------------------------------------------
RESULTS_XLSX = os.path.join(os.getcwd(), "examples", "DSI_DOE", "DSI_Optimisation_Results.xlsx")   # inside the DOE folder
RESULTS_SHEET = "Results"
BASELINE_FOLDER = 1
WEIGHTS = (0.25, 0.75)                           # J = w1*CD - w2*PR (as in the workbook)
OBJECTIVES = "PR:max,CD:min,J:min,DC:min"
SURFACES = "T"                                   # t_surfaces from the morph configs
GRID = (12, 8)
OLD_PARAM_BATCHES = ["corner_optimisation"]      # cases 15-28 (old parameterisation)

# name -> extra arguments for `surface_sensitivity.py analyse`
RUNS = {
    # main result: every design (LHS + optimiser), T surface, WHERE-effects at fixed size
    "all_T_ctrl":      ["--control-magnitude"],
    # same without size control - compare to see how much is just 'deformed more/less'
    "all_T_raw":       [],
    # new parameterisation only (LHS batch 2 + optimiser generations)
    "newparam_T_ctrl": ["--control-magnitude", "--exclude-batch", *OLD_PARAM_BATCHES],
    # robustness: LHS only (optimiser designs cluster near good regions and can
    # dominate correlations); if all_T_ctrl and lhs_T_ctrl disagree, trust neither
    "lhs_T_ctrl":      ["--control-magnitude", "--lhs-only"],
    # outlier robustness: rank-transformed response
    "all_T_ctrl_rank": ["--control-magnitude", "--rank-y"],
    # first 14 points: new-parameterisation LHS only (folders 2-14) + baseline
    "newparam_lhs_T_ctrl": ["--control-magnitude", "--lhs-only",
                            "--include-batch", "corner_optimisation_2"],
}
DEFAULT_RUNS = ["all_T_ctrl", "all_T_raw", "newparam_T_ctrl", "lhs_T_ctrl", "newparam_lhs_T_ctrl"]


def log(msg, fh=None):
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def up_to_date(outdir, inputs):
    rep = os.path.join(outdir, "report.md")
    if not os.path.exists(rep):
        return False
    t_out = os.path.getmtime(rep)
    return all(os.path.getmtime(p) < t_out for p in inputs if os.path.exists(p))


def run_cmd(cmd, logfile):
    with open(logfile, "w", encoding="utf-8") as fh:
        fh.write(" ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n\n")
        fh.flush()
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return p.returncode


def tail(path, n=15):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doe", required=True, help="DOE folder (contains 1/, 2/, ... and the results workbook)")
    ap.add_argument("--results", default=None, help=f"results workbook (default <doe>/{RESULTS_XLSX})")
    ap.add_argument("--sheet", default=RESULTS_SHEET)
    ap.add_argument("--out", default=None, help="output root (default <doe>/sensitivity)")
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS,
                    help=f"which runs (default {DEFAULT_RUNS}); available: {list(RUNS)}")
    ap.add_argument("--force", action="store_true", help="rerun even if outputs are up to date")
    ap.add_argument("--quick", action="store_true",
                    help="fewer permutations/bootstraps (smoke test only - not for reporting)")
    args = ap.parse_args()

    if not os.path.exists(ENGINE):
        sys.exit(f"surface_sensitivity.py not found next to this script ({ENGINE})")
    unknown = [r for r in args.runs if r not in RUNS]
    if unknown:
        sys.exit(f"unknown run(s) {unknown}; available: {list(RUNS)}")

    doe = os.path.abspath(args.doe)
    results = os.path.abspath(args.results or os.path.join(doe, RESULTS_XLSX))
    out_root = os.path.abspath(args.out or os.path.join(doe, "sensitivity"))
    os.makedirs(out_root, exist_ok=True)
    table = os.path.join(doe, "designs.csv")
    mlog = open(os.path.join(out_root, "run_log.txt"), "a", encoding="utf-8")
    log(f"=== run_surface_sensitivity  doe={doe}", mlog)

    # ---------------------------------------------------------------- 1. table
    table_new = table + ".new"
    cmd = [sys.executable, ENGINE, "from-doe", doe, "--results", results, "--sheet", args.sheet,
           "--baseline", str(BASELINE_FOLDER), "--weights", *map(str, WEIGHTS), "--out", table_new]
    lf = os.path.join(out_root, "from_doe.log")
    log("building designs.csv ...", mlog)
    if run_cmd(cmd, lf) != 0:
        log(f"from-doe FAILED - see {lf}\n{tail(lf)}", mlog)
        sys.exit(1)
    # replace designs.csv only if it changed, so up-to-date runs can be skipped
    new_txt = open(table_new, "rb").read()
    old_txt = open(table, "rb").read() if os.path.exists(table) else None
    if new_txt != old_txt:
        os.replace(table_new, table)
        log("designs.csv updated (new/changed designs or results)", mlog)
    else:
        os.remove(table_new)
        log("designs.csv unchanged", mlog)
    log(tail(lf, 4).strip(), mlog)
    warn = [l.strip() for l in open(lf, encoding="utf-8", errors="replace") if "WARNING" in l]
    for w in warn:
        log("  " + w, mlog)

    # ---------------------------------------------------------------- 2. runs
    common = ["--objectives", OBJECTIVES, "--weights", *map(str, WEIGHTS),
              "--surfaces", SURFACES, "--grid", *map(str, GRID)]
    if args.quick:
        common += ["--n-perm-nodal", "300", "--n-perm-patch", "500", "--n-boot", "100"]
    status = {}
    for name in args.runs:
        outdir = os.path.join(out_root, name)
        if not args.force and up_to_date(outdir, [table, ENGINE]):
            log(f"[{name}] up to date - skipped (use --force to rerun)", mlog)
            status[name] = "skipped"
            continue
        os.makedirs(outdir, exist_ok=True)
        cmd = [sys.executable, ENGINE, "analyse", table, "--outdir", outdir, *common, *RUNS[name]]
        log(f"[{name}] running ...", mlog)
        t0 = time.time()
        rc = run_cmd(cmd, os.path.join(outdir, "run.log"))
        if rc != 0:
            log(f"[{name}] FAILED (exit {rc}) after {time.time() - t0:.0f}s:\n"
                f"{tail(os.path.join(outdir, 'run.log'))}", mlog)
            status[name] = "failed"
            continue
        log(f"[{name}] done in {time.time() - t0:.0f}s", mlog)
        status[name] = "ok"

    # ---------------------------------------------------------------- 3. summary
    try:
        import pandas as pd
        rows = []
        for name in args.runs:
            f = os.path.join(out_root, name, "model_skill.csv")
            if os.path.exists(f):
                t = pd.read_csv(f)
                t.insert(0, "run", name)
                rows.append(t)
        if rows:
            summ = pd.concat(rows, ignore_index=True)
            cols = ["run", "objective", "n", "response", "rho_vs_deformation_size",
                    "q2_loo_size_only", "q2_loo_size_plus_shape", "q2_loo_nodal",
                    "frac_area_signif", "q2_loo_patch", "grid_agreement_jaccard"]
            summ = summ[[c for c in cols if c in summ.columns]]
            summ.to_csv(os.path.join(out_root, "summary_skill.csv"), index=False)
            log("summary (Q2 <= 0.2 -> maps not interpretable for that objective):\n"
                + summ.round(3).to_string(index=False), mlog)
    except Exception as e:  # summary is a convenience; never hide the run results behind it
        log(f"summary failed: {e}", mlog)

    log(f"status: {status}", mlog)
    log(f"outputs in {out_root}  (open <run>/sensitivity_surface.vtk in ParaView; "
        f"field definitions in VTK_FIELDS.txt)", mlog)
    mlog.close()
    sys.exit(0 if all(v != "failed" for v in status.values()) else 2)


if __name__ == "__main__":
    main()
