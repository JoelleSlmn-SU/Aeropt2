"""
hpc_auth.py - SSH login that works with multi-factor / Microsoft (Entra) sign-in.

No Qt in this file. The GUI supplies a `responder` callable; tests supply a fake.

Why not SSHClient.connect(password=...):
  * MFA logins arrive as SSH "keyboard-interactive" prompts (password, then a
    code / "approve on your phone" / "open https://microsoft.com/devicelogin and
    enter ABCD-EFGH, then press Enter"). connect() can only answer a single
    password prompt.
  * paramiko gives up after Transport.auth_timeout = 30 s. A phone approval or a
    browser device-code sign-in takes longer, so it is raised to AUTH_TIMEOUT_S.
  * paramiko calls the keyboard-interactive handler on its own background
    thread. The responder must therefore be thread-safe (the Qt one in
    GUI/mfa_login.py hands the prompt to the GUI thread and waits).

The authenticated transport is attached to an ordinary paramiko.SSHClient, so
exec_command / open_sftp and every existing caller keep working unchanged.
reconnect() swaps a new transport into the *same* SSHClient object, so objects
that kept a reference to it (TestManager.ssh, HPCPipelineManager, ...) stay valid.
"""
from __future__ import annotations

import os
import socket
import logging
from typing import Callable, List, Optional, Sequence, Tuple

import paramiko

log = logging.getLogger("aeropt.hpc_auth")

AUTH_TIMEOUT_S = 600      # time allowed for the person to finish sign-in
KEEPALIVE_S = 30          # stops idle-disconnects while the GUI sits open
CONNECT_TIMEOUT_S = 20

# responder(title, instructions, prompts) -> answers (same length as prompts)
#   prompts: list of (prompt_text, echo) ; echo=False means hide input (password)
#   raise LoginCancelled to abort
Responder = Callable[[str, str, Sequence[Tuple[str, bool]]], List[str]]


class LoginCancelled(Exception):
    pass


class LoginFailed(Exception):
    pass


def _private_keys() -> List[paramiko.PKey]:
    """Keys from ssh-agent, then ~/.ssh/id_*. Unreadable / passphrased keys are skipped."""
    keys: List[paramiko.PKey] = []
    try:
        keys.extend(paramiko.Agent().get_keys())
    except Exception:
        pass
    ssh_dir = os.path.join(os.path.expanduser("~"), ".ssh")
    for name, cls in (("id_ed25519", paramiko.Ed25519Key),
                      ("id_ecdsa", paramiko.ECDSAKey),
                      ("id_rsa", paramiko.RSAKey)):
        path = os.path.join(ssh_dir, name)
        if os.path.isfile(path):
            try:
                keys.append(cls.from_private_key_file(path))
            except Exception as e:  # encrypted key etc.
                log.debug("skip key %s: %s", path, e)
    return keys


def _allowed_types(t: paramiko.Transport, username: str) -> List[str]:
    try:
        t.auth_none(username)
        return []  # server let us in with no auth at all
    except paramiko.BadAuthenticationType as e:
        return list(e.allowed_types)


def _authenticate(t: paramiko.Transport, username: str, responder: Responder,
                  password: Optional[str], status: Callable[[str], None]) -> None:
    """
    Walk the server's method list, including partial success
    (e.g. publickey accepted, keyboard-interactive still required).
    """
    remaining = _allowed_types(t, username)
    status(f"Server offers: {', '.join(remaining) or 'none'}")
    tried_pk = False
    used_password = False

    while not t.is_authenticated():
        if "publickey" in remaining and not tried_pk:
            tried_pk = True
            for key in _private_keys():
                try:
                    remaining = t.auth_publickey(username, key)
                    status(f"Key {key.get_name()} accepted"
                           + (f"; server still wants: {', '.join(remaining)}" if remaining else ""))
                    break
                except paramiko.AuthenticationException:
                    continue
            continue

        if "keyboard-interactive" in remaining:
            status("Waiting for sign-in prompts from the cluster...")

            def handler(title, instructions, prompt_list):
                nonlocal used_password
                prompts = [(p, bool(echo)) for p, echo in prompt_list]
                # Answer a plain single "Password:" prompt from the login box once,
                # so the user is not asked twice. Everything else goes to the GUI.
                if (password is not None and not used_password and len(prompts) == 1
                        and not prompts[0][1] and "password" in prompts[0][0].lower()):
                    used_password = True
                    return [password]
                answers = responder(title or "", instructions or "", prompts)
                if len(answers) != len(prompts):
                    raise LoginFailed("responder returned wrong number of answers")
                return answers

            remaining = t.auth_interactive(username, handler)
            continue

        if "password" in remaining and password is not None and not used_password:
            used_password = True
            remaining = t.auth_password(username, password)
            continue

        raise LoginFailed(
            "Cluster requires an authentication method AerOpt cannot do: "
            f"{', '.join(remaining) or '(none listed)'}")


def open_transport(host: str, username: str, responder: Responder,
                   password: Optional[str] = None, port: int = 22,
                   host_key_policy: str = "auto_add",
                   status: Callable[[str], None] = lambda s: None,
                   on_transport: Callable[[paramiko.Transport], None] = lambda t: None
                   ) -> paramiko.Transport:
    sock = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S)
    sock.settimeout(None)
    t = paramiko.Transport(sock)
    on_transport(t)  # lets the GUI close it if the user presses Cancel
    t.auth_timeout = AUTH_TIMEOUT_S
    t.banner_timeout = 60
    try:
        t.start_client(timeout=CONNECT_TIMEOUT_S)
        _check_host_key(t, host, port, host_key_policy, status)
        _authenticate(t, username, responder, password, status)
    except paramiko.AuthenticationException as e:
        t.close()
        raise LoginFailed(f"Sign-in rejected by {host}: {e}") from e
    except BaseException:
        t.close()
        raise
    t.set_keepalive(KEEPALIVE_S)
    status("Signed in")
    return t


def _check_host_key(t, host, port, policy, status):
    """Same behaviour as before (AutoAddPolicy) but remembered in ~/.ssh/known_hosts,
    so a changed host key is noticed next time instead of silently accepted."""
    key = t.get_remote_server_key()
    kh_path = os.path.join(os.path.expanduser("~"), ".ssh", "known_hosts")
    hk = paramiko.HostKeys()
    if os.path.isfile(kh_path):
        try:
            hk.load(kh_path)
        except Exception:
            pass
    name = host if port == 22 else f"[{host}]:{port}"
    known = hk.lookup(name)
    if known and key.get_name() in known:
        if known[key.get_name()] != key:
            raise LoginFailed(
                f"Host key for {host} has CHANGED. If the cluster was rebuilt this is "
                f"expected: remove its line from {kh_path}. Otherwise do not connect.")
        return
    if policy == "reject":
        raise LoginFailed(f"Unknown host key for {host}")
    status(f"New host key for {host} ({key.get_name()}) saved")
    try:
        os.makedirs(os.path.dirname(kh_path), exist_ok=True)
        with open(kh_path, "a") as f:
            f.write(f"{name} {key.get_name()} {key.get_base64()}\n")
    except Exception as e:
        log.warning("could not save host key: %s", e)


def attach(client: paramiko.SSHClient, transport: paramiko.Transport) -> paramiko.SSHClient:
    """Put an authenticated transport inside an SSHClient (paramiko has no public API
    for this; exec_command/open_sftp only use client._transport)."""
    old = getattr(client, "_transport", None)
    client._transport = transport
    if old is not None and old is not transport:
        try:
            old.close()
        except Exception:
            pass
    return client


def connect(host, username, responder, password=None, port=22,
            status=lambda s: None, client: Optional[paramiko.SSHClient] = None,
            on_transport=lambda t: None) -> paramiko.SSHClient:
    """Sign in. Pass an existing `client` to reconnect it in place."""
    t = open_transport(host, username, responder, password, port,
                       status=status, on_transport=on_transport)
    return attach(client if client is not None else paramiko.SSHClient(), t)


def is_alive(client: Optional[paramiko.SSHClient]) -> bool:
    t = getattr(client, "_transport", None) if client is not None else None
    return bool(t is not None and t.is_active() and t.is_authenticated())
