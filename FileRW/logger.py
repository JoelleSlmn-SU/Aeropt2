# logger.py
import os
import datetime
import posixpath

class GuiLogger:
    def __init__(self, text_widget, output_dir_func, is_hpc_func, sftp_client_func=None):
        self.text_widget = text_widget
        self.output_dir_func = output_dir_func
        self.is_hpc_func = is_hpc_func
        self.sftp_client_func = sftp_client_func
        self.log_lines = []

    def _sftp_mkdirs(self, sftp, remote_dir):
        # recursively ensure remote_dir exists
        parts = remote_dir.strip("/").split("/")
        cur = ""
        for p in parts:
            cur = f"{cur}/{p}" if cur else f"/{p}"
            try:
                sftp.stat(cur)
            except IOError:
                sftp.mkdir(cur)

    def log(self, message):
        timestamped = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}"
        self.log_lines.append(timestamped)

        # GUI
        if self.text_widget:
            color = "black"
            if "ERROR" in message:
                color = "red"
            elif "WARN" in message:
                color = "orange"
            elif "INFO" in message or "✔" in message:
                color = "green"
            self.text_widget.append(f'<span style="color:{color};">{timestamped}</span>')

        # File (unchanged)
        output_dir = self.output_dir_func()
        if not output_dir:
            return

        log_path = posixpath.join(output_dir, "aeropt.log")

        if self.is_hpc_func() and self.sftp_client_func:
            try:
                sftp = self.sftp_client_func()
                self._sftp_mkdirs(sftp, output_dir)
                with sftp.file(log_path, "a") as f:
                    f.write((timestamped + "\n").encode("utf-8"))
                sftp.close()
                return
            except Exception as e:
                print(f"[Logger Error] Remote SFTP write failed: {e}")

        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(timestamped + "\n")
        except Exception as e:
            print(f"[Logger Error] Failed to write log file: {e}")


    def clear(self):
        self.log_lines.clear()
        if self.text_widget:
            self.text_widget.clear()
