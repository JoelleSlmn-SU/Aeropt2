#!/usr/bin/env python3
import os, re, sys, json
import numpy as np

"""
Usage:
  python3 convergenceCheck.py <sol_dir> <attempt> <max_retries> <res_threshold> <force_csv> <residual_csv> <stdout_file>

Notes for Aeropt:
- We assume <residual_csv> points to your .rsd file.
- Residual series is taken from column index 1 (2nd column) by default.
- Force oscillation check uses columns index 2 and 3 (3rd & 4th columns) per user requirement.
- If CSVs are missing, we fall back to parsing residual-like numbers from stdout.
- Output: a single JSON line: {"converged": bool, "reason": str}
"""

DEFAULT_RESIDUAL_COL = 1   # 2nd column
FORCE_COLS = (2, 3)        # 3rd & 4th columns (Cl, Cd)

def _safe_load_table(path):
    if not path or not os.path.exists(path):
        return None
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = re.split(r"[,\s;]+", s)
            # skip header-like rows that contain non-numeric tokens
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    nums.append(np.nan)
            if not all(np.isnan(x) for x in nums):
                rows.append(nums)
    if not rows:
        return None
    # ragged → pad with NaNs to same width
    width = max(len(r) for r in rows)
    arr = np.full((len(rows), width), np.nan, dtype=float)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r
    # drop rows all-NaN
    arr = arr[~np.isnan(arr).all(axis=1)]
    return arr if arr.size else None

def _parse_stdout_for_residuals(path):
    if not path or not os.path.exists(path):
        return None
    vals = []
    with open(path, "r", errors="ignore") as f:
        for ln in f:
            m = re.search(r"(?:RES|residual)[^\d\-+eE]*([\-+]?\d+\.?\d*(?:[eE][\-+]?\d+)?)", ln)
            if m:
                try:
                    vals.append(float(m.group(1)))
                except:
                    pass
    if not vals:
        return None
    it = np.arange(len(vals), dtype=float)
    return np.column_stack([it, np.array(vals, dtype=float)])

def _autocorr_peak(y, min_len=100):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < min_len:
        return 0.0
    y = (y - y.mean()) / (y.std() + 1e-12)
    n = 1
    L = len(y)
    while n < 2*L:
        n <<= 1
    fy = np.fft.rfft(y, n=n)
    ac = np.fft.irfft(fy * np.conj(fy), n=n)[:L]
    ac /= ac[0] + 1e-12
    if L <= 10:
        return 0.0
    return float(np.nanmax(ac[5:L//2]))  # ignore trivial small lags

def _forces_oscillatory(table, cols=FORCE_COLS, peak_thresh=0.5):
    ok = False
    peaks = []
    for c in cols:
        if c < table.shape[1]:
            series = table[:, c]
            p = _autocorr_peak(series, min_len=50)
            peaks.append(p)
            ok = ok or (p > peak_thresh)
    return ok, peaks

def _residual_converged(table, res_col=DEFAULT_RESIDUAL_COL, threshold=-3):
    if table is None or table.shape[1] <= res_col:
        return None, "no residual column"
    res = table[:, res_col]
    res = res[np.isfinite(res)]
    if res.size == 0:
        return None, "residual column empty"
    final = np.nanmedian(res[-5:]) if res.size >= 5 else res[-1]
    return bool(final <= threshold), f"final residual {final:.3e}"

def main():
    if len(sys.argv) != 8:
        print('{"converged": false, "reason": "bad argv"}')
        return
    sol_dir, attempt, max_retries, res_threshold, force_csv, residual_csv, stdout_file = sys.argv[1:]
    attempt = int(attempt)
    max_retries = int(max_retries)
    res_threshold = float(res_threshold)

    # Absolute paths
    force_csv = os.path.join(sol_dir, force_csv) if force_csv else ""
    residual_csv = os.path.join(sol_dir, residual_csv) if residual_csv else ""
    stdout_file = os.path.join(sol_dir, stdout_file) if stdout_file else ""

    # Load tables
    T = _safe_load_table(residual_csv)
    # Residual check
    conv = None
    reason = "unknown"
    if T is not None:
        conv, res_msg = _residual_converged(T, DEFAULT_RESIDUAL_COL, res_threshold)
        if conv is True:
            print(json.dumps({"converged": True, "reason": f"{res_msg} <= threshold {res_threshold:.1e}"}))
            return
        elif conv is False:
            reason = f"{res_msg} above threshold {res_threshold:.1e}"
        else:
            reason = res_msg

        # Force oscillation check on 3rd & 4th columns
        osc, peaks = _forces_oscillatory(T, FORCE_COLS, peak_thresh=0.5)
        if osc:
            print(json.dumps({"converged": False, "reason": f"oscillatory force history (autocorr peaks {peaks})"}))
            return

        # If residuals not OK and no oscillation, still not converged
        print(json.dumps({"converged": False, "reason": reason}))
        return

    # Fallback: try stdout residuals if no .rsd
    R = _parse_stdout_for_residuals(stdout_file)
    if R is not None and R.shape[1] >= 2:
        final = np.nanmedian(R[-5:, 1]) if R.shape[0] >= 5 else R[-1, 1]
        if final <= res_threshold:
            print(json.dumps({"converged": True, "reason": f"stdout residual {final:.3e} <= threshold {res_threshold:.1e}"}))
        else:
            print(json.dumps({"converged": False, "reason": f"stdout residual {final:.3e} above threshold {res_threshold:.1e}"}))
        return

    print(json.dumps({"converged": False, "reason": "no data available"}))

if __name__ == "__main__":
    main()
