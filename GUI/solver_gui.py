import pyvista as pv
from pyvistaqt import QtInteractor 
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os, io, fnmatch, json
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import posixpath


class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=3, dpi=120):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.spines["top"].set_visible(False)
        super().__init__(fig)

# --- SFTP recursive walk/download ---
def sftp_walk(sftp, remote_path):
    # yields (dirpath, dirnames, filenames)
    dirnames, filenames = [], []
    for attr in sftp.listdir_attr(remote_path):
        name = attr.filename
        rpath = posixpath.join(remote_path, name)
        if stat_isdir(attr.st_mode):
            dirnames.append(name)
        else:
            filenames.append(name)
    yield remote_path, dirnames, filenames
    for d in dirnames:
        for x in sftp_walk(sftp, posixpath.join(remote_path, d)):
            yield x

def stat_isdir(st_mode):
    import stat
    return stat.S_ISDIR(st_mode)

def download_tree(sftp, remote_root, local_root, patterns=None):
    os.makedirs(local_root, exist_ok=True)
    for dirpath, _, filenames in sftp_walk(sftp, remote_root):
        rel = os.path.relpath(dirpath, remote_root).replace("\\", "/")
        ldir = os.path.join(local_root, rel) if rel != "." else local_root
        os.makedirs(ldir, exist_ok=True)
        for fn in filenames:
            if patterns and not any(fnmatch.fnmatch(fn, p) for p in patterns):
                continue
            sftp.get(posixpath.join(dirpath, fn), os.path.join(ldir, fn))

def parse_rsd_bytes(data: bytes):
    # Robust .rsd: skip blank/header lines, whitespace-split
    xs, cols = [], [[], [], [], [], []]  # we’ll keep up to 5 y-cols (2..6 total cols)
    for raw in io.BytesIO(data).read().decode("utf-8", "ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "!", "%")):
            continue
        parts = line.replace(",", " ").split()
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            continue
        if len(nums) >= 2:
            xs.append(nums[0])
            # store up to 5 extra columns if present
            for i in range(1, min(6, len(nums))):
                cols[i-1].append(nums[i])
    X = np.array(xs)
    Ys = [np.array(c) if len(c) == len(xs) else None for c in cols]
    return X, Ys

class SolverViewer(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.setLayout(QVBoxLayout())

        # Top: controls
        top = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh jobs")
        self.load_btn    = QPushButton("Load & Plot .rsd")
        self.col_combo   = QComboBox()
        self.col_combo.addItems(["Rsd", "Lift Coefficient", "Drag Coefficient", "Lateral Force", "Skin Friction Coefficient"])
        top.addWidget(QLabel("Y-axis:"))
        top.addWidget(self.col_combo)
        top.addStretch(1)
        top.addWidget(self.refresh_btn)
        top.addWidget(self.load_btn)

        self.load_hist_btn = QPushButton("Load History")
        top.addWidget(self.load_hist_btn)  # put it near Refresh/Load buttons
        self.load_hist_btn.clicked.connect(self.load_history_into_list)
        
        self.layout().addLayout(top)
        
        # Job list
        self.jobs = QListWidget()
        self.jobs.setFixedHeight(100)
        self.layout().addWidget(self.jobs)

        # Plot
        self.canvas = MplCanvas()
        self.layout().addWidget(self.canvas)
        self.canvas.ax.set_xlabel("Iteration")
        
        self.refresh_btn.clicked.connect(self.refresh_jobs)
        self.load_btn.clicked.connect(self.load_and_plot)

    def _jobs_log_path(self):
        return os.path.join(os.getcwd(), "jobs_log.json")

    def _load_job_history(self):
        path = self._jobs_log_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except Exception:
            data = []
        # filter solver kind only
        return [d for d in data if isinstance(d, dict) and d.get("kind") == "solver"]

    def _solutions_root(self):
        # e.g. $HOME/aeropt/aeropt_out/<run>/solutions
        if not getattr(self.mw, "remote_output_dir", None):
            QMessageBox.information(self, "HPC output not set",
                                    "Please set an output directory and mirror to HPC first.")
            return None
        return posixpath.join(self.mw.remote_output_dir, "solutions")
    
    def load_history_into_list(self):
        hist = self._load_job_history()
        if not hist:
            QMessageBox.information(self, "History", "No solver jobs found in jobs_log.json.")
            return
        self.jobs.clear()
        for d in hist:
            # Display a friendly label, keep raw metadata on the item for later
            label = f"{d.get('jobid','?')} — n={d.get('n')} {d.get('cond','')} — {d.get('remote_dir','')}"
            it = QListWidgetItem(label)
            it.setData(32, d)  # Qt.UserRole
            self.jobs.addItem(it)

    def refresh_jobs(self):
        self.jobs.clear()
        try:
            sftp = self.mw.ssh_client.open_sftp()
        except Exception as e:
            QMessageBox.critical(self, "SSH error", str(e))
            return
        # Expect folder structure solutions/n_*/cond_*/ with .rsd somewhere inside
        try:
            sols_root = self._solutions_root()
            if not sols_root: return
            # list n_* levels
            for attr_n in sftp.listdir_attr(sols_root):
                nname = attr_n.filename
                npath = posixpath.join(sols_root, nname)
                if not stat_isdir(attr_n.st_mode) or not nname.startswith("n_"):
                    continue
                # list cond_* inside
                for attr_c in sftp.listdir_attr(npath):
                    cname = attr_c.filename
                    cpath = posixpath.join(npath, cname)
                    if not stat_isdir(attr_c.st_mode) or not cname.startswith("cond_"):
                        continue
                    # find .rsd files within this folder
                    try:
                        for attr_f in sftp.listdir_attr(cpath):
                            if attr_f.filename.lower().endswith(".rsd"):
                                self.jobs.addItem(posixpath.join(cpath, attr_f.filename))
                    except Exception:
                        pass
        finally:
            sftp.close()

    def load_and_plot(self):
        item = self.jobs.currentItem()
        if not item:
            QMessageBox.information(self, "No selection", "Please select a solver .rsd job.")
            return
        rpath = item.text()
        try:
            sftp = self.mw.ssh_client.open_sftp()
            with sftp.open(rpath, "rb") as f:
                data = f.read()
            sftp.close()
        except Exception as e:
            QMessageBox.critical(self, "SFTP error", str(e))
            return

        X, Ys = self.canvas.parse_rsd_bytes(data)
        if X.size == 0:
            QMessageBox.warning(self, "Parse error", "Could not parse any data from the .rsd.")
            return

        # Column mapping: "Column 2" -> Ys[0], ..., "Column 5" -> Ys[3]
        idx = self.col_combo.currentIndex()  # 0..3
        Y = Ys[idx]
        if Y is None or Y.size != X.size:
            QMessageBox.warning(self, "Data error", "Selected column not present / length mismatch.")
            return

        self.canvas.ax.clear()
        self.canvas.ax.plot(X, Y, lw=0.8)
        self.canvas.ax.set_xlabel("Iteration")
        self.canvas.ax.set_ylabel(self.col_combo.currentText())
        self.canvas.ax.grid(True, which="both", alpha=0.3)
        self.canvas.draw()
        
    def load_and_plot(self):
        item = self.jobs.currentItem()
        if not item:
            QMessageBox.information(self, "No selection", "Please select a solver job.")
            return
        meta = item.data(32)  # history dict if present, else None

        try:
            sftp = self.mw.ssh_client.open_sftp()
        except Exception as e:
            QMessageBox.critical(self, "SSH error", str(e))
            return

        try:
            if isinstance(meta, dict):
                # History path: use stored remote_dir
                rdir = meta.get("remote_dir")
                if not rdir:
                    raise RuntimeError("History record missing 'remote_dir'.")
                # search for rsd file in the directory
                rsd_candidates = []
                for attr in sftp.listdir_attr(rdir):
                    name = attr.filename
                    if name.lower().endswith(".rsd"):
                        rsd_candidates.append(posixpath.join(rdir, name))
                if not rsd_candidates:
                    raise FileNotFoundError(f"No .rsd found in {rdir}")
                rpath = sorted(rsd_candidates)[0]  # pick first deterministically
            else:
                # Current list item from refresh: we already stored the file path as text
                rpath = item.text()

            # download + parse
            with sftp.open(rpath, "rb") as f:
                data = f.read()
            sftp.close()

            X, Ys = parse_rsd_bytes(data)  # your existing robust parser
            idx = self.col_combo.currentIndex()  # 0..3 → Column 2..5
            Y = Ys[idx]
            if X.size == 0 or Y is None or Y.size != X.size:
                QMessageBox.warning(self, "Data error", "Selected column not present / size mismatch.")
                return

            self.canvas.ax.clear()
            self.canvas.ax.plot(X, Y, lw=1.5)
            self.canvas.ax.set_xlabel("Iteration")
            self.canvas.ax.set_ylabel(self.col_combo.currentText())
            self.canvas.ax.spines["bottom"].set_bounds(min(X), max(X))
            self.canvas.ax.grid(True, which="both", alpha=0.3)
            self.canvas.draw()

        except Exception as e:
            try:
                sftp.close()
            except Exception:
                pass
            QMessageBox.critical(self, "Load error", str(e))