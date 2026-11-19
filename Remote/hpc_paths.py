"""
hpc_paths.py - where AerOpt puts run data on the cluster. One definition, used everywhere.

On the new cluster scratch is per *project*, not per user:
    /scratch/<project code>          e.g. /scratch/scw2427
Everyone in the project shares that directory, so AerOpt keeps its own runs
under a per-user folder inside it:
    /scratch/<project>/<username>/aeropt/aeropt_out/<run>
Without the <username> level, two project members whose local output folders
have the same name (both "run1") would write into the same remote run.
"""
from __future__ import annotations

import posixpath
import re
import shlex
from typing import Tuple

SCRATCH_BASE = "/scratch"

# Project codes are short identifiers like scw2427. The rule is loose on purpose
# (the cluster check below is what really decides); it only has to keep the
# value safe inside a shell command and a path.
_PROJECT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{1,31}$")


def normalise_project_code(code: str) -> str:
    return (code or "").strip().lower()


def validate_project_code(code: str) -> Tuple[bool, str]:
    c = normalise_project_code(code)
    if not c:
        return False, "Enter your project code (for example scw2427)."
    if not _PROJECT_RE.match(c):
        return False, (f"'{code}' does not look like a project code. "
                       "Use letters, digits, '-' or '_' only, e.g. scw2427.")
    return True, ""


def project_scratch(project: str) -> str:
    return posixpath.join(SCRATCH_BASE, normalise_project_code(project))


def scratch_root(project: str, username: str) -> str:
    """Parent of all AerOpt run folders for this user in this project."""
    if not username or "/" in username:
        raise ValueError(f"bad username {username!r}")
    return posixpath.join(project_scratch(project), username, "aeropt", "aeropt_out")


def check_scratch(ssh_client, project: str, username: str, timeout: float = 30.0) -> Tuple[bool, str]:
    """
    On the cluster: does /scratch/<project> exist, and can we create and write our
    folder in it? Returns (ok, message). Creates scratch_root() as a side effect.
    """
    ok, msg = validate_project_code(project)
    if not ok:
        return False, msg
    proj_dir = project_scratch(project)
    root = scratch_root(project, username)
    q = shlex.quote
    cmd = (
        f"if [ ! -d {q(proj_dir)} ]; then echo NO_PROJECT_DIR; exit 3; fi; "
        f"mkdir -p {q(root)} 2>/dev/null || {{ echo NO_MKDIR; exit 4; }}; "
        f"t={q(root)}/.aeropt_write_test.$$; "
        f"( : > \"$t\" ) 2>/dev/null && rm -f \"$t\" || {{ echo NO_WRITE; exit 5; }}; "
        f"echo OK"
    )
    try:
        _in, out, err = ssh_client.exec_command(f"bash -c {q(cmd)}", timeout=timeout)
        status = out.channel.recv_exit_status()
        text = out.read().decode(errors="ignore").strip()
    except Exception as e:
        return False, f"Could not check {proj_dir} on the cluster: {e}"

    if status == 0 and text.endswith("OK"):
        return True, f"Using scratch folder {root}"
    if "NO_PROJECT_DIR" in text:
        return False, (f"{proj_dir} does not exist on the cluster. Check the project code "
                       "(it is shown on the MySCW / project portal page).")
    if "NO_MKDIR" in text or "NO_WRITE" in text:
        return False, (f"{proj_dir} exists but you cannot write to it. You may not be a "
                       f"member of project {normalise_project_code(project)} yet, or "
                       "membership can take a while to apply after you are added.")
    return False, f"Unexpected reply while checking {proj_dir} (exit {status}): {text!r}"
