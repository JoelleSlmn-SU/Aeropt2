# batch_compute_features.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from features import compute_features, append_jsonl


# -----------------------------------------------------------------------------
# Volume output pass/fail tokens
# -----------------------------------------------------------------------------
PASS_TOKENS: Tuple[str, ...] = ("Succesfully", "Successfully")
FAIL_TOKENS: Tuple[str, ...] = ("Error Stop",)


# -----------------------------------------------------------------------------
# Case-ID extraction
#   - morph meshes:     case_###.fro  ->  ###
#   - volume outputs:   volume_output_###(.*) -> ###
# -----------------------------------------------------------------------------
_CASE_RE = re.compile(r"(?:^|[^0-9])corner_(\d+)(?:[^0-9]|$)")
_VOL_RE = re.compile(r"(?:^|[^0-9])volume_output_(\d+)(?:[^0-9]|$)")


def infer_case_id_from_morph_path(p: Path) -> Optional[int]:
    """
    Extract integer id from names like case_120.fro (or any path containing case_120).
    Returns 120.
    """
    m = _CASE_RE.search(p.stem)  # stem: "case_120"
    if m:
        return int(m.group(1))

    m = _CASE_RE.search(str(p))
    if m:
        return int(m.group(1))

    return None


def _load_existing_morphed_paths(jsonl_path: Path) -> Set[str]:
    """
    Skip only cases that already have a *completed* row (non-empty features).
    This prevents "skip everything forever" if earlier rows were failures/empty.
    """
    seen: Set[str] = set()
    if not jsonl_path.exists():
        return seen

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            mf = rec.get("morphed_fro", "")
            if not mf:
                continue

            feats = rec.get("features") or {}
            if not isinstance(feats, dict) or len(feats) == 0:
                continue

            try:
                seen.add(str(Path(mf).resolve()))
            except Exception:
                seen.add(str(mf))

    return seen


def index_volume_outputs(vol_root: Path, pattern: str = "volume_output_*") -> Dict[int, Path]:
    """
    Scan volumes directory once and return mapping:
        case_id -> volume_output_file_path

    Looks for files named like:
        volume_output_120
        volume_output_120.out
        volume_output_120_rank0.txt
    anywhere under vol_root.

    If multiple matches exist for the same id, keeps the newest by mtime.
    """
    out: Dict[int, Path] = {}
    if not vol_root.exists():
        return out

    for fp in vol_root.rglob(pattern):
        if not fp.is_file():
            continue

        m = _VOL_RE.search(fp.name)
        if not m:
            continue

        k = int(m.group(1))

        if k not in out:
            out[k] = fp
        else:
            try:
                if fp.stat().st_mtime > out[k].stat().st_mtime:
                    out[k] = fp
            except Exception:
                pass

    return out


def parse_volume_output(fp: Optional[Path]) -> Tuple[Optional[int], str]:
    """
    Returns (label, reason):
      label: 1 pass, 0 fail, None unknown/not found
    """
    if fp is None or not fp.exists():
        return None, "missing_volume_output"

    txt = fp.read_text(encoding="utf-8", errors="ignore")

    # check fail first (more conservative)
    if any(tok in txt for tok in FAIL_TOKENS):
        return 0, "error_stop_found"
    if any(tok in txt for tok in PASS_TOKENS):
        return 1, "successfully_found"

    return None, "no_token_match"


def compute_features_for_directory(
    morph_dir: str,
    orig_fro: str,
    out_jsonl: str,
    morph_basis_json: Optional[str] = None,
    pattern: str = "*.fro",
    recursive: bool = True,
    limit: Optional[int] = None,
    run_dir: Optional[str] = None,
    skip_existing: bool = True,
    remote_output: Optional[str] = None,
    volume_output_pattern: str = "volume_output_*",
) -> Dict[str, Any]:
    """
    Walk morph_dir, find morphed meshes, compute features, attach volume-mesh label
    by matching:
        case_###  <->  volume_output_###

    Appends one JSON record per morphed mesh to out_jsonl.
    """
    morph_root = Path(morph_dir).expanduser().resolve()
    orig_fro_p = Path(orig_fro).expanduser().resolve()
    out_p = Path(out_jsonl).expanduser().resolve()

    if not morph_root.exists():
        raise FileNotFoundError(f"morph_dir not found: {morph_root}")
    if not orig_fro_p.exists():
        raise FileNotFoundError(f"orig_fro not found: {orig_fro_p}")

    remote_out_p = Path(remote_output).expanduser().resolve() if remote_output else None

    # Pre-index volume outputs once (fast + robust)
    vol_index: Dict[int, Path] = {}
    if remote_out_p:
        vol_root = remote_out_p
        vol_index = index_volume_outputs(vol_root, pattern=volume_output_pattern)

    # Skip only already-computed feature rows
    seen = _load_existing_morphed_paths(out_p) if skip_existing else set()

    # Collect morphed meshes
    files = sorted(morph_root.rglob(pattern) if recursive else morph_root.glob(pattern))
    files = [p for p in files if p.is_file()]
    if limit is not None:
        files = files[: int(limit)]

    n_total = len(files)
    n_done = 0
    n_skipped = 0
    n_failed = 0
    failed: List[Dict[str, str]] = []

    run_dir_final = run_dir or str(morph_root)

    for p in files:
        morphed_abs = str(p.resolve())

        if skip_existing and morphed_abs in seen:
            n_skipped += 1
            continue

        # case_id string for bookkeeping (relative path)
        rel = p.relative_to(morph_root)
        case_id = rel.with_suffix("").as_posix().replace("/", "__")

        # numeric id extracted from "case_###"
        case_num = infer_case_id_from_morph_path(p)

        # label lookup from "volume_output_###"
        vol_out = vol_index.get(case_num) if (remote_out_p and case_num is not None) else None
        label, reason = parse_volume_output(vol_out) if remote_out_p else (None, "remote_output_not_provided")

        try:
            rec = compute_features(
                orig_fro=str(orig_fro_p),
                morphed_fro=morphed_abs,
                run_dir=run_dir_final,
                case_id=case_id,
                morph_basis_json=morph_basis_json,
                extra={
                    "status": "morph_features_ok",
                    "case_num": case_num,
                    "label": label,  # 1/0/None
                    "reason": reason,
                    "paths": {
                        "volume_output": str(vol_out.resolve()) if vol_out else "",
                        "volumes_root": str((remote_out_p).resolve()) if remote_out_p else "",
                    },
                },
            )
            append_jsonl(str(out_p), rec)
            n_done += 1

        except Exception as e:
            n_failed += 1
            failed.append({"morphed_fro": morphed_abs, "error": repr(e)})

            err_rec = {
                "case_id": case_id,
                "run_dir": run_dir_final,
                "orig_fro": str(orig_fro_p),
                "morphed_fro": morphed_abs,
                "morph_basis_json": morph_basis_json or "",
                "status": "morph_features_failed",
                "error": repr(e),
                "case_num": case_num,
                "label": label,
                "reason": reason,
                "paths": {
                    "volume_output": str(vol_out.resolve()) if vol_out else "",
                    "volumes_root": str((remote_out_p).resolve()) if remote_out_p else "",
                },
                "features": {},
            }
            append_jsonl(str(out_p), err_rec)

    return {
        "morph_dir": str(morph_root),
        "orig_fro": str(orig_fro_p),
        "out_jsonl": str(out_p),
        "remote_output": str(remote_out_p) if remote_out_p else "",
        "pattern": pattern,
        "volume_output_pattern": volume_output_pattern,
        "recursive": recursive,
        "limit": limit,
        "total_found": n_total,
        "processed_ok": n_done,
        "skipped_existing": n_skipped,
        "processed_failed": n_failed,
        "failed_examples": failed[:5],
        "volume_index_size": len(vol_index),
    }


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--morph-dir", required=True, help="Directory containing morphed .fro files")
    ap.add_argument("--orig", required=True, help="Original/baseline .fro (same topology)")
    ap.add_argument("--out-jsonl", required=True, help="dataset.jsonl to append to")
    ap.add_argument("--basis", default="", help="morph_basis.json (optional)")
    ap.add_argument("--pattern", default="*.fro", help="Glob pattern for morphed meshes (default: *.fro)")
    ap.add_argument("--no-recursive", action="store_true", help="Do not search recursively")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N files")
    ap.add_argument("--run-dir", default="", help="Run directory metadata field (optional)")
    ap.add_argument("--no-skip-existing", action="store_true", help="Do not skip cases already in dataset.jsonl")

    ap.add_argument("--remote-output", default="", help="Root output dir containing volumes/ (for labeling)")
    ap.add_argument(
        "--volume-output-pattern",
        default="volume_output_*",
        help="Glob pattern for volume output files (default: volume_output_*)",
    )

    args = ap.parse_args()

    summary = compute_features_for_directory(
        morph_dir=args.morph_dir,
        orig_fro=args.orig,
        out_jsonl=args.out_jsonl,
        morph_basis_json=(args.basis or None),
        pattern=args.pattern,
        recursive=(not args.no_recursive),
        limit=args.limit,
        run_dir=(args.run_dir or None),
        skip_existing=(not args.no_skip_existing),
        remote_output=(args.remote_output or None),
        volume_output_pattern=args.volume_output_pattern,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
