"""
mfa_login.py - Qt front end for Remote/hpc_auth.py.

    from mfa_login import hpc_login
    client = hpc_login(parent, host, username, password_or_None)            # new
    hpc_login(parent, host, username, None, client=self.ssh_client)          # reconnect in place

Returns the signed-in paramiko.SSHClient, or None if cancelled / failed (the
error has already been shown to the user).

Threading: sign-in runs on a worker thread; paramiko asks its questions on its
own transport thread. Each question is passed to the GUI thread with a queued
signal and the asking thread waits on a threading.Event. Meanwhile the GUI
thread spins a local event loop (the status dialog), so the window stays alive
and this also works from MainWindow.__init__ before app.exec_() has started.
"""
from __future__ import annotations

import re
import html
import threading

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QUrl, QEventLoop
from PyQt5.QtGui import QDesktopServices, QGuiApplication
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QDialogButtonBox, QPlainTextEdit, QFormLayout)

from Remote import hpc_auth

_URL_RE = re.compile(r"https?://[^\s<>\"']+")
# Microsoft device codes look like ABCD-EFGH or 9 upper-case alphanumerics
_CODE_RE = re.compile(r"\b(?=[A-Z0-9-]*[A-Z])[A-Z0-9]{4,5}-[A-Z0-9]{4,5}\b|\b(?=[A-Z0-9]*[0-9])(?=[A-Z0-9]*[A-Z])[A-Z0-9]{9}\b")


class PromptDialog(QDialog):
    """Shows one round of keyboard-interactive prompts exactly as the cluster sent them."""

    def __init__(self, title, instructions, prompts, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title or "HPC sign-in")
        self.setMinimumWidth(480)
        lay = QVBoxLayout(self)

        text = "\n".join(s for s in (instructions,) if s).strip()
        # Some servers put the URL / code in the prompt text instead of the instructions
        all_text = text + "\n" + "\n".join(p for p, _ in prompts)
        urls = _URL_RE.findall(all_text)
        code = _CODE_RE.search(all_text)

        if text:
            esc = html.escape(text)
            esc = _URL_RE.sub(lambda m: f'<a href="{m.group(0)}">{m.group(0)}</a>', esc)
            lbl = QLabel(esc.replace("\n", "<br>"))
            lbl.setTextFormat(Qt.RichText)
            lbl.setOpenExternalLinks(True)
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
            lay.addWidget(lbl)

        if urls or code:
            row = QHBoxLayout()
            if urls:
                b = QPushButton("Open sign-in page")
                b.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(urls[0])))
                row.addWidget(b)
            if code:
                c = code.group(0)
                b2 = QPushButton(f"Copy code {c}")
                b2.clicked.connect(lambda: QGuiApplication.clipboard().setText(c))
                row.addWidget(b2)
            row.addStretch(1)
            lay.addLayout(row)

        form = QFormLayout()
        self._edits = []
        for p, echo in prompts:
            e = QLineEdit()
            if not echo:
                e.setEchoMode(QLineEdit.Password)
            form.addRow(QLabel(p.strip() or "Response:"), e)
            self._edits.append(e)
        lay.addLayout(form)
        if self._edits:
            self._edits[0].setFocus()

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Continue" if not self._edits or not any(
            not echo for _, echo in prompts) else "OK")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def answers(self):
        return [e.text() for e in self._edits]


class _Bridge(QObject):
    """Lives in the GUI thread. Worker/transport threads talk to it through signals."""
    ask = pyqtSignal(str, str, list, object)       # title, instructions, prompts, box
    status = pyqtSignal(str)
    done = pyqtSignal(object, object)              # client, error

    def __init__(self, parent_widget):
        super().__init__()
        self.parent_widget = parent_widget
        self.cancelled = threading.Event()
        self.ask.connect(self._on_ask, Qt.QueuedConnection)

    @pyqtSlot(str, str, list, object)
    def _on_ask(self, title, instructions, prompts, box):
        dlg = PromptDialog(title, instructions, prompts, self.parent_widget)
        if dlg.exec_() == QDialog.Accepted:
            box["answers"] = dlg.answers()
        else:
            box["cancel"] = True
        box["event"].set()

    # called on paramiko's transport thread
    def responder(self, title, instructions, prompts):
        if self.cancelled.is_set():
            raise hpc_auth.LoginCancelled()
        box = {"event": threading.Event()}
        self.ask.emit(title, instructions, list(prompts), box)
        while not box["event"].wait(0.2):
            if self.cancelled.is_set():
                raise hpc_auth.LoginCancelled()
        if box.get("cancel"):
            self.cancelled.set()
            raise hpc_auth.LoginCancelled()
        return box["answers"]


class _StatusDialog(QDialog):
    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Signing in to {host}")
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Follow any prompts. If the cluster sends a push to the "
                             "Microsoft Authenticator app, approve it on your phone."))
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumHeight(120)
        lay.addWidget(self.log)
        bb = QDialogButtonBox(QDialogButtonBox.Cancel)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


def hpc_login(parent, host, username, password=None, port=22, client=None, logger=None):
    bridge = _Bridge(parent)
    dlg = _StatusDialog(host, parent)
    result = {}
    transport_box = {}

    def say(msg):
        bridge.status.emit(msg)

    bridge.status.connect(dlg.log.appendPlainText, Qt.QueuedConnection)
    if logger is not None:
        bridge.status.connect(lambda m: logger.log(f"[HPC] {m}"), Qt.QueuedConnection)

    def work():
        try:
            c = hpc_auth.connect(host, username, bridge.responder, password, port,
                                 status=say, client=client,
                                 on_transport=lambda t: transport_box.setdefault("t", t))
            bridge.done.emit(c, None)
        except BaseException as e:
            bridge.done.emit(None, e)

    def finished(c, err):
        result["client"], result["err"] = c, err
        dlg.accept()

    bridge.done.connect(finished, Qt.QueuedConnection)

    def on_cancel():
        bridge.cancelled.set()
        t = transport_box.get("t")
        if t is not None:            # unblocks a wait for a phone approval
            try:
                t.close()
            except Exception:
                pass

    dlg.rejected.connect(on_cancel)
    th = threading.Thread(target=work, name="hpc-login", daemon=True)
    th.start()
    dlg.exec_()                      # local event loop: prompts + status are serviced here
    # user cancelled: let the worker unwind (closing the transport makes this quick)
    from PyQt5.QtWidgets import QApplication
    while th.is_alive():
        QApplication.processEvents(QEventLoop.AllEvents, 50)
        th.join(0.05)
    QApplication.processEvents()     # deliver the final queued `done`

    err = result.get("err")
    if err is None and result.get("client") is not None:
        return result["client"]
    if bridge.cancelled.is_set() or isinstance(err, hpc_auth.LoginCancelled):
        if logger is not None:
            logger.log("[HPC] Sign-in cancelled")
        return None
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.critical(parent, "HPC sign-in failed", f"Could not sign in to {host}:\n{err}")
    return None
