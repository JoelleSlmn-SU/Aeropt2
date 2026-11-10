"""python -m pytest tests/test_hpc_paths.py -q
check_scratch's shell command is run for real with bash on a temporary 'scratch'."""
import io, os, sys, stat, subprocess
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from Remote import hpc_paths


class _Chan:
    def __init__(self, rc): self.rc = rc
    def recv_exit_status(self): return self.rc


class _Out(io.BytesIO):
    def __init__(self, data, rc):
        super().__init__(data); self.channel = _Chan(rc)


class LocalSSH:
    """Stands in for paramiko.SSHClient: runs the command with local bash."""
    def __init__(self): self.cmds = []
    def exec_command(self, cmd, timeout=None):
        self.cmds.append(cmd)
        p = subprocess.run(cmd, shell=True, capture_output=True, executable="/bin/bash")
        return None, _Out(p.stdout, p.returncode), io.BytesIO(p.stderr)


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(hpc_paths, "SCRATCH_BASE", str(tmp_path))
    return tmp_path


def test_paths():
    assert hpc_paths.scratch_root("SCW2427 ", "s.2268086") == \
        "/scratch/scw2427/s.2268086/aeropt/aeropt_out"


@pytest.mark.parametrize("code,ok", [("scw2427", True), ("SCW2427", True), ("", False),
                                     ("scw 2427", False), ("../etc", False),
                                     ("scw2427;rm -rf ~", False), ("2427", False)])
def test_validate(code, ok):
    assert hpc_paths.validate_project_code(code)[0] is ok


def test_ok_creates_user_folder(scratch):
    (scratch / "scw2427").mkdir()
    ok, msg = hpc_paths.check_scratch(LocalSSH(), "scw2427", "s.2268086")
    assert ok, msg
    assert (scratch / "scw2427" / "s.2268086" / "aeropt" / "aeropt_out").is_dir()
    assert not list((scratch / "scw2427" / "s.2268086" / "aeropt" / "aeropt_out").iterdir())


def test_wrong_code(scratch):
    ok, msg = hpc_paths.check_scratch(LocalSSH(), "scw9999", "u")
    assert not ok and "does not exist" in msg


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_not_a_member(scratch):
    d = scratch / "scw2427"; d.mkdir(); d.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        ok, msg = hpc_paths.check_scratch(LocalSSH(), "scw2427", "u")
        assert not ok and "cannot write" in msg
    finally:
        d.chmod(0o755)


def test_bad_code_never_reaches_shell(scratch):
    ssh = LocalSSH()
    ok, _ = hpc_paths.check_scratch(ssh, "x; touch pwned", "u")
    assert not ok and ssh.cmds == []


def test_dialog_requires_project(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, os.path.join(ROOT, "GUI"))
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from SSHLoginDialog import SshLoginDialog
    d = SshLoginDialog(saved={"host": "h", "username": "u"})
    d._try_accept()
    assert d.result() == 0 and "project code" in d.error_label.text().lower()
    d.project_input.setText(" SCW2427 ")
    d._try_accept()
    assert d.result() == 1
    c = d.get_credentials()
    assert c["project"] == "scw2427" and c["remember"] is True
