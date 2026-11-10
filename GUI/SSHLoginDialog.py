from PyQt5.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QFormLayout, QDialogButtonBox, QCheckBox
)

from Remote.hpc_paths import validate_project_code, normalise_project_code


class SshLoginDialog(QDialog):
    """Host / username / project code, and an optional password.

    Any further sign-in steps (Microsoft code, phone approval, ...) are asked
    by the cluster itself after OK; see GUI/mfa_login.py.
    """

    def __init__(self, parent=None, saved=None):
        super().__init__(parent)
        self.setWindowTitle("HPC Login")
        self.setMinimumWidth(420)
        saved = saved or {}

        # Create inputs
        self.host_input = QLineEdit(saved.get("host", ""))
        self.username_input = QLineEdit(saved.get("username", ""))
        self.project_input = QLineEdit(saved.get("project", ""))
        self.project_input.setPlaceholderText("e.g. scw2427")
        self.project_input.setToolTip("Your project code. Run data goes to /scratch/<project code>/<username>/aeropt")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        # Form layout
        form_layout = QFormLayout()
        form_layout.addRow("Host:", self.host_input)
        form_layout.addRow("Username:", self.username_input)
        form_layout.addRow("Project code:", self.project_input)
        form_layout.addRow("Password (optional):", self.password_input)

        self.remember_checkbox = QCheckBox("Remember host, username and project code")
        self.remember_checkbox.setChecked(bool(saved))
        form_layout.addRow("", self.remember_checkbox)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #c0392b;")
        self.error_label.setWordWrap(True)
        self.error_label.hide()

        # OK / Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self._try_accept)
        self.button_box.rejected.connect(self.reject)

        # Final layout
        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(self.error_label)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        # Put the cursor in the first empty field
        for w in (self.host_input, self.username_input, self.project_input, self.password_input):
            if not w.text():
                w.setFocus()
                break

    def show_error(self, msg):
        self.error_label.setText(msg)
        self.error_label.setVisible(bool(msg))

    def _try_accept(self):
        if not self.host_input.text().strip():
            return self.show_error("Enter the cluster host name.")
        if not self.username_input.text().strip():
            return self.show_error("Enter your cluster username.")
        ok, msg = validate_project_code(self.project_input.text())
        if not ok:
            self.project_input.setFocus()
            return self.show_error(msg)
        self.accept()

    def get_credentials(self):
        return {
            "host": self.host_input.text().strip(),
            "username": self.username_input.text().strip(),
            "project": normalise_project_code(self.project_input.text()),
            "password": self.password_input.text(),
            "remember": self.remember_checkbox.isChecked()
        }
