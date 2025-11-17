from PyQt5.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QFormLayout, QDialogButtonBox, QCheckBox
)

class SshLoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HPC Login")
        self.setFixedSize(400, 160)

        # Create inputs
        self.host_input = QLineEdit()
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        # Form layout
        form_layout = QFormLayout()
        form_layout.addRow("Host:", self.host_input)
        form_layout.addRow("Username:", self.username_input)
        form_layout.addRow("Password:", self.password_input)

        self.remember_checkbox = QCheckBox("Remember SSH credentials")
        form_layout.addRow("", self.remember_checkbox)
        
        # OK / Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Final layout
        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def get_credentials(self):
        return {
            "host": self.host_input.text().strip(),
            "username": self.username_input.text().strip(),
            "password": self.password_input.text(),
            "remember": self.remember_checkbox.isChecked()
        }
