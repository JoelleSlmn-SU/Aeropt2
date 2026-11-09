#!/usr/bin/env python3
"""
convergenceGuard.py -- decision step of the cluster convergence guard.

Runs inside ONE small Slurm job after each solver run (afterany:<solver job>):

    python3 convergenceGuard.py <sol_dir> --inp <base_n>.inp --rsd <base_n>.rsd
                                --stdout solver_output --threshold -3
                                --max-retries 2

It calls convergenceCheck.py (same folder), updates convergence_state.json and
prints ONE word on stdout:

    restart   not converged and attempts left: the batch job prepares the
              restart files and re-submits the solver + this guard
    done      converged, or no attempts left: the batch job touches SOLVER_DONE

convergence_state.json (read by remoteOpt):
    {"attempt": <restarts so far>, "max_retries": N, "converged": bool,
     "reason": str, "final": bool, "history": [{"attempt", "converged", "reason"}...]}

`attempt` counts restarts: 0 after the first run. With --max-retries 2 the
solver runs at most 3 times. Pure Python, no Slurm calls, so it is unit-tested.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

STATE = "convergence_state.json"
LAST = "last_convergence.json"


def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def run_check(sol_dir, rsd, stdout_name, threshold, attempt, max_retries):
    """Call convergenceCheck.py; any failure counts as 'not converged'."""
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(sol_dir, "convergenceCheck.py")
    if not os.path.exists(script):
        script = os.path.join(here, "convergenceCheck.py")
    try:
        out = subprocess.run(
            [sys.executable, script, sol_dir, str(attempt), str(max_retries), str(threshold),
             "", rsd, stdout_name],
            capture_output=True, text=True, timeout=600,
        ).stdout.strip().splitlines()
        res = json.loads(out[-1]) if out else {"converged": False, "reason": "convergenceCheck printed nothing"}
    except Exception as e:  # noqa: BLE001 - the guard must always reach a decision
        res = {"converged": False, "reason": f"convergenceCheck failed: {e}"}
    _write_json(os.path.join(sol_dir, LAST), res)
    return res


def bump_restart_number(inp_path):
    """ivd%restartNumber -> +1 in the solver's own input file. Returns (old, new) or None."""
    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    pat = re.compile(r"(?im)^(?P<p>\s*ivd%restartNumber\s*=\s*)(?P<n>\d+)(?P<t>\s*,?)")
    m = pat.search(txt)
    if not m:
        return None
    old = int(m.group("n"))
    txt = txt[:m.start()] + m.group("p") + str(old + 1) + m.group("t") + txt[m.end():]
    with open(inp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    return old, old + 1


def _solver_rc(sol_dir):
    try:
        with open(os.path.join(sol_dir, "SOLVER_RC")) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def decide(sol_dir, inp, rsd, stdout_name, threshold, max_retries, check=run_check):
    state_path = os.path.join(sol_dir, STATE)
    st = _read_json(state_path, {})
    attempt = int(st.get("attempt", 0))
    res = check(sol_dir, rsd, stdout_name, threshold, attempt, max_retries)
    conv = bool(res.get("converged", False))
    reason = str(res.get("reason", ""))
    rc = _solver_rc(sol_dir)
    hist = list(st.get("history", [])) + [{"attempt": attempt, "converged": conv, "reason": reason,
                                            "solver_rc": rc}]

    action = "done"
    if not conv and rc not in (None, 0):
        # a crash is not slow convergence: restarting from its .res is pointless
        reason += f" | solver exited with code {rc}; not restarted"
    elif not conv and attempt < max_retries:
        bumped = None
        inp_path = os.path.join(sol_dir, inp)
        if os.path.exists(inp_path):
            bumped = bump_restart_number(inp_path)
        if bumped is None:
            reason += f" | cannot restart: ivd%restartNumber not found in {inp}"
        else:
            action = "restart"
            attempt += 1
            reason += f" | restart {attempt}/{max_retries} (restartNumber {bumped[0]}->{bumped[1]})"

    _write_json(state_path, {
        "attempt": attempt, "max_retries": int(max_retries), "converged": conv,
        "reason": reason, "final": action == "done", "history": hist,
    })
    return action


def mark_final(sol_dir, reason):
    """Used by the batch job when the restart could not be prepared/submitted:
    the run ends here (SOLVER_DONE is touched) with the last result."""
    state_path = os.path.join(sol_dir, STATE)
    st = _read_json(state_path, {})
    st["final"] = True
    st["reason"] = (str(st.get("reason", "")) + f" | restart aborted: {reason}").strip(" |")
    _write_json(state_path, st)
    return "done"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sol_dir")
    ap.add_argument("--inp", required=True)
    ap.add_argument("--rsd", required=True)
    ap.add_argument("--stdout", default="solver_output")
    ap.add_argument("--threshold", type=float, default=-3.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--mark-final", default=None,
                    help="restart preparation failed: record REASON, make the state final, print 'done'")
    a = ap.parse_args(argv)
    sol_dir = os.path.abspath(a.sol_dir)
    if a.mark_final is not None:
        print(mark_final(sol_dir, a.mark_final))
        return
    print(decide(sol_dir, a.inp, a.rsd, a.stdout, a.threshold, a.max_retries))


if __name__ == "__main__":
    main()
