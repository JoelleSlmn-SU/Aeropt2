"""Run: python -m pytest tests/test_hpc_auth.py -q   (from the repo root)
A local paramiko server imitates an MFA login: round 1 password, round 2 a
Microsoft device-code message with a 'press Enter' prompt."""
import os, sys, socket, threading, time
import paramiko
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Remote import hpc_auth

HOST_KEY = paramiko.RSAKey.generate(2048)
DEVICE_MSG = ("To sign in, use a web browser to open https://microsoft.com/devicelogin "
              "and enter the code ABCD-EFGH to authenticate.")


class Server(paramiko.ServerInterface):
    def __init__(self, mode, pubkey=None):
        self.mode, self.pubkey, self.round = mode, pubkey, 0
        self.pk_ok = False

    def get_allowed_auths(self, u):
        if self.mode == "pk+kbd":
            return "keyboard-interactive" if self.pk_ok else "publickey"
        return "keyboard-interactive"

    def check_auth_publickey(self, u, key):
        if self.pubkey is not None and key == self.pubkey:
            self.pk_ok = True
            return paramiko.AUTH_PARTIALLY_SUCCESSFUL
        return paramiko.AUTH_FAILED

    def check_auth_interactive(self, u, submethods):
        if self.mode == "pk+kbd" and not self.pk_ok:
            return paramiko.AUTH_FAILED
        self.round = 1
        q = paramiko.InteractiveQuery("", "")
        q.add_prompt("Password: ", False)
        return q

    def check_auth_interactive_response(self, responses):
        if self.round == 1:
            if list(responses) != ["pw"]:
                return paramiko.AUTH_FAILED
            self.round = 2
            q = paramiko.InteractiveQuery("Microsoft sign-in", DEVICE_MSG)
            q.add_prompt("Press Enter when done", True)
            return q
        if self.round == 2:
            return paramiko.AUTH_SUCCESSFUL
        return paramiko.AUTH_FAILED

    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED

    def check_channel_exec_request(self, ch, cmd):
        def run():
            time.sleep(0.2); ch.sendall(b"ok:" + cmd)
            ch.send_exit_status(0)
            ch.close()
        threading.Thread(target=run, daemon=True).start()
        return True


def start_server(mode, pubkey=None, n_conn=1):
    ls = socket.socket()
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("127.0.0.1", 0))
    ls.listen(5)
    port = ls.getsockname()[1]
    transports = []

    def serve():
        for _ in range(n_conn):
            c, _ = ls.accept()
            t = paramiko.Transport(c)
            t.add_server_key(HOST_KEY)
            t.start_server(server=Server(mode, pubkey))
            transports.append(t)
    threading.Thread(target=serve, daemon=True).start()
    return port, transports


@pytest.fixture(autouse=True)
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(hpc_auth, "_private_keys", lambda: [])
    return tmp_path


def exec_(client, cmd):
    _, out, _ = client.exec_command(cmd)
    return out.read().decode()


def test_password_then_device_code():
    port, _ = start_server("kbd")
    seen = []

    def responder(title, instr, prompts):
        seen.append((title, instr, prompts))
        return [""] * len(prompts)

    c = hpc_auth.connect("127.0.0.1", "joelle", responder, password="pw", port=port)
    assert hpc_auth.is_alive(c)
    assert exec_(c, "squeue --me") == "ok:squeue --me"
    # password answered from the login box, only the device-code round reached the user
    assert len(seen) == 1 and "devicelogin" in seen[0][1]
    assert seen[0][2] == [("Press Enter when done", True)]
    c.close()


def test_password_prompt_goes_to_user_when_not_given():
    port, _ = start_server("kbd")
    rounds = []

    def responder(title, instr, prompts):
        rounds.append(prompts)
        return ["pw"] if prompts[0][0].startswith("Password") else [""]

    c = hpc_auth.connect("127.0.0.1", "joelle", responder, password=None, port=port)
    assert hpc_auth.is_alive(c) and len(rounds) == 2
    c.close()


def test_wrong_password_fails_cleanly():
    port, _ = start_server("kbd")
    with pytest.raises(hpc_auth.LoginFailed):
        hpc_auth.connect("127.0.0.1", "joelle", lambda *a: [""], password="nope", port=port)


def test_cancel():
    port, _ = start_server("kbd")

    def responder(*a):
        raise hpc_auth.LoginCancelled()
    with pytest.raises(hpc_auth.LoginCancelled):
        hpc_auth.connect("127.0.0.1", "joelle", responder, password=None, port=port)


def test_partial_publickey_then_mfa(monkeypatch):
    key = paramiko.Ed25519Key.generate() if hasattr(paramiko.Ed25519Key, "generate") else None
    if key is None:
        key = paramiko.ECDSAKey.generate()
    port, _ = start_server("pk+kbd", pubkey=key)
    monkeypatch.setattr(hpc_auth, "_private_keys", lambda: [key])
    c = hpc_auth.connect("127.0.0.1", "joelle", lambda t, i, p: [""] * len(p),
                         password="pw", port=port)
    assert hpc_auth.is_alive(c)
    c.close()


def test_slow_approval_beats_paramiko_default_30s():
    port, _ = start_server("kbd")

    def responder(title, instr, prompts):
        time.sleep(35)            # person takes 35 s on the Microsoft page
        return [""] * len(prompts)
    c = hpc_auth.connect("127.0.0.1", "joelle", responder, password="pw", port=port)
    assert hpc_auth.is_alive(c)
    c.close()


def test_reconnect_in_place_keeps_references_valid():
    port, srv = start_server("kbd", n_conn=2)
    r = lambda t, i, p: [""] * len(p)
    c = hpc_auth.connect("127.0.0.1", "joelle", r, password="pw", port=port)
    held = c                      # what TestManager / pipeline keep
    srv[0].close()                # cluster drops us
    time.sleep(0.5)
    assert not hpc_auth.is_alive(held)
    hpc_auth.connect("127.0.0.1", "joelle", r, password="pw", port=port, client=c)
    assert hpc_auth.is_alive(held)
    assert exec_(held, "hostname") == "ok:hostname"
    c.close()


def test_host_key_saved_then_change_detected(tmp_home):
    port, _ = start_server("kbd", n_conn=2)
    r = lambda t, i, p: [""] * len(p)
    hpc_auth.connect("127.0.0.1", "joelle", r, password="pw", port=port).close()
    kh = tmp_home / ".ssh" / "known_hosts"
    assert kh.exists()
    other = paramiko.RSAKey.generate(2048)
    kh.write_text(f"[127.0.0.1]:{port} ssh-rsa {other.get_base64()}\n")
    with pytest.raises(hpc_auth.LoginFailed, match="CHANGED"):
        hpc_auth.connect("127.0.0.1", "joelle", r, password="pw", port=port)


def test_qt_dialog_flow():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "GUI")))
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    import mfa_login
    app = QApplication.instance() or QApplication([])
    port, _ = start_server("kbd")
    shown = []

    def poke():
        for w in app.topLevelWidgets():
            if isinstance(w, mfa_login.PromptDialog) and w.isVisible() and w not in shown:
                shown.append(w)
                for e in w._edits:
                    e.setText("pw" if e.echoMode() else "")
                w.accept()
    timer = QTimer(); timer.timeout.connect(poke); timer.start(100)
    c = mfa_login.hpc_login(None, "127.0.0.1", "joelle", None, port=port)
    timer.stop()
    assert c is not None and hpc_auth.is_alive(c)
    assert len(shown) == 2       # password round, device-code round
    c.close()
