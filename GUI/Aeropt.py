from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pyvistaqt import QtInteractor
from mesh_gui import MeshViewer
from solver_gui import SolverViewer
from post_gui import PostViewer
from geom_gui import GeomWindow
from geometry_gui import GeometryPanel
from SSHLoginDialog import SshLoginDialog
import paramiko
import os, sys

sys.path.append(os.path.dirname("FileRW"))
sys.path.append(os.path.dirname("Remote"))
sys.path.append(os.path.dirname("Local"))
from FileRW.logger import GuiLogger
from Remote.pipeline_remote import HPCPipelineManager
from Local.pipeline_local import PipelineManager
from FileRW.RungenInpFile import RungenInpFile

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.run_mode = self.ask_run_location()
        # Prompt run mode early
        if self.run_mode == "HPC":
            saved_creds = self.load_ssh_config()

            if saved_creds:
                use_saved = QMessageBox.question(
                    self,
                    "Use Saved Credentials?",
                    "Use previously saved SSH credentials?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if use_saved == QMessageBox.Yes:
                    self.ssh_creds = saved_creds
                    self.run_mode = "HPC"
                else:
                    creds = self.get_ssh_credentials_dialog()
                    if creds:
                        self.ssh_creds = {
                            "host": creds["host"],
                            "username": creds["username"],
                            "password": creds["password"]
                        }
                        self.run_mode = "HPC"
                        if creds.get("remember"):
                            self.save_ssh_config(self.ssh_creds)
                    else:
                        self.run_mode = "Local"
            else:
                creds = self.get_ssh_credentials_dialog()
                if creds:
                    self.ssh_creds = {
                        "host": creds["host"],
                        "username": creds["username"],
                        "password": creds["password"]
                    }
                    self.run_mode = "HPC"
                    if creds.get("remember"):
                        self.save_ssh_config(self.ssh_creds)
                else:
                    self.run_mode = "Local"
                    
        self.setup_window()
        self.create_buttons()
        self.create_stack_pages()
        self.create_display_section()
        
        self.control_nodes_saved = False
        self.input_directory_set = False
        self.output_directory_set = False
        self.prepro_settings_saved = False
        self.solver_settings_saved = False
        self.optimisation_settings_saved = False
        self.control_node_source = "mesh"
        
        self.use_pca_reduced = False
        self.restart_from_previous = False
        self.previous_solution_config = {
            "enabled": False,
            "location": "remote",
            "directory": "",
            "base": "",
            "Boundary_mode": "same_id",
            "num_comp": 6,
        }
        
        self.rbf_original = None
        self.rbf_current  = None

        # Initialise Logger
        self.logger = GuiLogger(
        text_widget=self.log_output,
        output_dir_func=lambda: (
            getattr(self, "remote_output_dir", None)
            if self.run_mode == "HPC"
            else getattr(self, "output_directory", os.path.join(os.getcwd(), "aeropt_logs"))
        ),
        is_hpc_func=lambda: self.run_mode == "HPC",
        sftp_client_func=lambda: self.ssh_client.open_sftp() if hasattr(self, "ssh_client") else None
    )

        if hasattr(self, "mesh_viewer"):
            self.mesh_viewer.set_logger(self.logger)
        if hasattr(self, "geo_viewer"):
            self.geo_viewer.set_logger(self.logger)

        # Test the SSH connection
        if self.run_mode == "HPC":
            ok = self.test_ssh_connection(self.ssh_creds)
            if not ok:
                QMessageBox.warning(self, "Connection Failed", "Falling back to local execution.")
                self.run_mode = "Local"
                
        self.monitor_config = {
            "interval": 50,
            "enabled": True,
            "monitors": [
                {
                    "type": "pressure_recovery",
                    "name": "pressure_recovery",
                    "enabled": True
                }
            ]
        }
        self.monitors_saved = False

    
    def get_project_basename(self):
        """Return self.base name of loaded geometry/mesh file without extension."""
        if hasattr(self, "input_filename") and self.input_filename:
            return os.path.splitext(self.input_filename)[0]
        return "project"

    def ask_run_location(self):
        reply = QMessageBox.question(
            self,
            "Run Location",
            "Do you want to run on the HPC cluster?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            return "HPC"
        else:
            return "Local"
        
    def get_cad_scale(self) -> float:
        """Return meters-per-CAD-unit scale from the dropdown."""
        units = getattr(self, "cad_units", "m")
        return {
            "mm": 1e-3,
            "cm": 1e-2,
            "m": 1.0,
            "in": 0.0254,
            "ft": 0.3048
        }.get(units, 1.0)
    
    def load_ssh_config(self):
        import os, json
        config_path = os.path.join(os.path.expanduser("~"), ".aeropt", "ssh_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[SSH Config] Failed to read: {e}")
        return None

    def save_ssh_config(self, creds):
        import os, json
        config_dir = os.path.join(os.path.expanduser("~"), ".aeropt")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "ssh_config.json")
        try:
            with open(config_path, "w") as f:
                json.dump(creds, f)
            if hasattr(self, "logger"):
                self.logger.log("[HPC] SSH credentials saved.")
        except Exception as e:
            print(f"[SSH Config] Failed to save: {e}")
    
    def get_ssh_credentials_dialog(self):
        dialog = SshLoginDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.get_credentials()
        return None
    
    def test_ssh_connection(self, creds):
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(creds["host"], username=creds["username"], password=creds["password"])
            self.logger.log("[HPC] SSH connection successful")
            return True
        except Exception as e:
            QMessageBox.critical(self, "SSH Failed", f"Could not connect to HPC:\n{e}")
            return False
    
    def apply_qss_theme(self, theme_name):
        if not theme_name or not hasattr(self, "themes_dir"):
            return

        qss_path = os.path.join(self.themes_dir, theme_name)
        try:
            with open(qss_path, "r") as f:
                QApplication.instance().setStyleSheet(f.read())
        except Exception as e:
            self.logger.log(f"[Theme Error] Failed to apply {theme_name}: {e}")
    
    def setup_window(self):
        """Set up the main window properties."""
        self.resize(1400, 900)  # Initial size, but resizable
        self.setMinimumSize(1000, 600)  # Optional, to avoid tiny sizes
        self.setWindowTitle("AerOpt")
        
        # 🔧 Create toolbar
        tb = self.addToolBar("File")
        tb.setIconSize(QSize(24, 24))

        new_icon = self.style().standardIcon(QStyle.SP_FileIcon)
        open_icon = self.style().standardIcon(QStyle.SP_DialogOpenButton)
        save_icon = self.style().standardIcon(QStyle.SP_DirIcon)

        new_action = QAction(new_icon, "", self)
        new_action.triggered.connect(self.new_file)
        tb.addAction(new_action)

        open_action = QAction(open_icon, "", self)
        open_action.triggered.connect(self.open_file)
        tb.addAction(open_action)

        save_action = QAction(save_icon, "", self)
        save_action.triggered.connect(self.save_file)
        tb.addAction(save_action)
        
        new_action.setToolTip("New Project")
        open_action.setToolTip("Open Directory")
        save_action.setToolTip("Set Output Directory")
        
        tb.addSeparator()
        spacer_small = QWidget()
        spacer_small.setFixedWidth(12)
        tb.addWidget(spacer_small)
        
        status_icon = self.style().standardIcon(QStyle.SP_FileDialogListView)
        cancel_icon = self.style().standardIcon(QStyle.SP_DialogCancelButton)

        status_action = QAction(status_icon, "", self)
        status_action.setToolTip("HPC Status (your jobs)")
        status_action.triggered.connect(lambda: self.on_hpc_status_clicked() if self.run_mode == "HPC" else None)
        tb.addAction(status_action)

        cancel_action = QAction(cancel_icon, "", self)
        cancel_action.setToolTip("Cancel HPC Job")
        cancel_action.triggered.connect(lambda: self.on_hpc_cancel_clicked() if self.run_mode == "HPC" else None)
        tb.addAction(cancel_action)
        
        if self.run_mode != "HPC":
            status_action.setEnabled(False)
            cancel_action.setEnabled(False)
        
        self.available_themes = []
        themes_dir = os.path.join(os.path.dirname(__file__), "themes")
        if os.path.exists(themes_dir):
            self.available_themes = [
                f for f in os.listdir(themes_dir) if f.endswith(".qss")
            ]
            self.themes_dir = themes_dir
        
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        
        # Theme dropdown
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(self.available_themes)
        self.theme_combo.setToolTip("Select Theme")
        self.theme_combo.setFixedWidth(180)
        self.theme_combo.currentTextChanged.connect(self.apply_qss_theme)

        tb.addWidget(self.theme_combo)
        
        # Outer vertical layout for entire window
        outer_layout = QVBoxLayout()

        # Horizontal layout for main UI (already exists)
        self.page_layout = QHBoxLayout()
        self.button_layout = QVBoxLayout()
        self.stack_layout = QStackedLayout()
        self.display_layout = QVBoxLayout()

        self.page_layout.addLayout(self.button_layout)
        self.page_layout.addLayout(self.stack_layout)
        self.page_layout.addLayout(self.display_layout)

        # Add main page layout to outer vertical layout
        outer_layout.addLayout(self.page_layout)

        # ADD the logger QTextEdit here
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #f9f9f9; color: black;")
        self.log_output.setFixedHeight(120)
        outer_layout.addWidget(self.log_output)

        # Attach everything to the window
        container = QWidget()
        container.setLayout(outer_layout)
        self.setCentralWidget(container)

    def _stage(self):
        import posixpath
        # --- Stage baseline inputs into orig/ (best-effort) ---
        base = getattr(self, "base", None) or self.get_project_basename()
        self.logger.log(f"[HPC] Staging baseline inputs using base='{base}'")

        remote_orig_dir = f"{self.base_hpc_dir}/orig"

        # Where to look for the files locally
        search_dirs = []
        candidates = [
            getattr(self, "input_directory", None),
            getattr(self, "output_directory", None),
            os.path.join(getattr(self, "output_directory", ""), "surfaces", "n_0"),
            os.path.dirname(getattr(self, "input_file_path", "") or ""),
        ]
        for d in candidates:
            if d and os.path.isdir(d) and d not in search_dirs:
                search_dirs.append(d)

        def find_local(fname: str):
            for d in search_dirs:
                p = os.path.join(d, fname)
                if os.path.exists(p):
                    return p
            return None

        file_list = [
            f"{base}.bac",
            f"{base}.bco",
            f"{base}.bpp",
            f"{base}.dat",
            f"{base}.fro",
            f"{base}.inp",
            "run.inp",
            "Mesh3D_v50.ctl",
            "Surf3D_v25.ctl",
            "rungen.inp",
        ]

        sftp = self.ssh_client.open_sftp()
        try:
            for fname in file_list:
                src = find_local(fname)
                remote_path = posixpath.join(remote_orig_dir, fname)
                if src:
                    try:
                        sftp.put(src, remote_path)
                        self.logger.log(f"[HPC] Uploaded: {fname}  (from {src})")
                    except Exception as e:
                        self.logger.log(f"[HPC][WARN] Failed upload {fname}: {e}")
                else:
                    self.logger.log(f"[HPC][WARN] Missing locally (searched {search_dirs}): {fname}")
        finally:
            try:
                sftp.close()
            except Exception:
                pass

    def _stage_before_hpc_run(self, action_name="run") -> bool:
        """
        Refresh remote orig/ from the latest local files immediately before
        submitting an HPC action.
        """
        if getattr(self, "run_mode", "") != "HPC":
            return True

        if not getattr(self, "ssh_client", None):
            self.logger.log(f"[{action_name}][HPC][ERROR] Not connected to HPC.")
            return False

        if not getattr(self, "remote_output_dir", None):
            self.logger.log(f"[{action_name}][HPC][ERROR] No remote output directory set.")
            return False

        try:
            self.logger.log(f"[{action_name}][HPC] Refreshing staged files in remote orig/ ...")

            # Stage the standard file set used by your pipeline
            self._stage()

            # Also do the more targeted orig refresh helper, since it searches
            # input/output dirs robustly for core FLITE inputs
            remote_orig = self._stage_orig_inputs_to_remote()
            if not remote_orig:
                self.logger.log(f"[{action_name}][HPC][WARN] orig/ refresh returned empty path.")

            return True
        except Exception as e:
            self.logger.log(f"[{action_name}][HPC][ERROR] Failed to refresh staged files: {e}")
            return False
        
    def stage_monitor_config_to_remote(self):
        if self.run_mode != "HPC" or not getattr(self, "ssh_client", None):
            return ""

        import os, posixpath, json, tempfile
        cfg = getattr(self, "monitor_config", None)
        if not cfg:
            return ""

        local_tmp = tempfile.mkdtemp()
        local_path = os.path.join(local_tmp, "monitors.json")
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        remote_path = posixpath.join(self.remote_output_dir, "orig", "monitors.json")
        sftp = self.ssh_client.open_sftp()
        try:
            sftp.put(local_path, remote_path)
        finally:
            sftp.close()

        self.logger.log(f"[MON] Uploaded monitor config to {remote_path}")
        return remote_path

    def new_file(self):
        """Start a fresh project: clear viewers, state, docks, logs."""
        # close any sub-dialogs
        if hasattr(self, "geom_window") and self.geom_window:
            try: self.geom_window.close()
            except Exception: pass
            self.geom_window = None

        # reset Mesh viewer
        if hasattr(self, "mesh_viewer") and self.mesh_viewer:
            try: self.mesh_viewer.reset_viewer()
            except Exception as e:
                self.logger.log(f"[WARN] MeshViewer reset failed: {e}")

        # reset Geometry panel
        if hasattr(self, "geo_panel") and self.geo_panel:
            try: self.geo_panel.reset()
            except Exception as e:
                self.logger.log(f"[WARN] GeometryPanel reset failed: {e}")

        # clear app state
        self.input_file_path = None
        self.input_directory = None
        self.input_filename = None
        self.output_directory = None
        self.remote_output_dir = None
        self.pipeline = None
        self.rbf_original = None
        self.rbf_current = None

        # clear log view
        try:
            if hasattr(self, "log_output") and self.log_output:
                self.log_output.clear()
        except Exception:
            pass

        # default page (pick what you want to land on)
        try:
            self.display_stack.setCurrentIndex(getattr(self, "IDX_MESH", 0))
        except Exception:
            pass

        self.logger.log("[INFO] New project started. State reset.")


    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "EnSight Case (*.case);;VTM Files (*.vtm);;VTK Files (*.vtk);;All Files (*);; CAD Files (*.step *.stp *.iges *.igs)"
        )
        if not filename:
            return

        # Bookkeeping
        self.input_file_path = filename
        self.input_directory = os.path.dirname(filename)
        self.input_filename  = os.path.basename(filename)

        self.logger.log(f"[INFO] Loading file: {self.input_file_path}")
        self._ensure_output_home()  # may auto-create local outputs/surfaces/n_0

        ext = os.path.splitext(filename)[1].lower()

        if ext in (".step", ".stp", ".iges", ".igs"):
            ## dialog to ask if CAD parameterisation
            self.display_stack.setCurrentIndex(self.IDX_GEO)
            self.geo_viewer.load_cad(filename)
            self.geo_viewer.set_input_filepath(self.input_file_path)
            self.logger.log("[INFO] Routed CAD to Geometry panel.")
        elif ext in (".case", ".vtk", ".vtm"):
            self.display_stack.setCurrentIndex(self.IDX_MESH)
            self.mesh_viewer.load_mesh_file(self.input_file_path)
            self.mesh_viewer.set_input_filepath(self.input_file_path)
            self.logger.log("[INFO] Mesh loaded into Mesh viewer.")
        else:
            self.logger.log(f"[WARN] Unsupported file type: {ext}.")
        
        self.input_directory_set = True
        self.create_pipeline()
        
    def _ensure_output_home(self):
        """Make sure output_directory + subfolders and rbf_original exist, and sync to MeshViewer."""
        if not getattr(self, "output_directory", None):
            default_dir = self.input_directory
            os.makedirs(default_dir, exist_ok=True)
            self.output_directory = default_dir

            # Create subfolders we rely on
            for sub in ["preprocessed", "solutions", "volumes", "surfaces", "postprocessed"]:
                os.makedirs(os.path.join(default_dir, sub), exist_ok=True)

            # Path used by runSurfMesh when converting VTM/VTK → .fro
            self.rbf_original = os.path.join(default_dir, "surfaces", "n_0")
            os.makedirs(self.rbf_original, exist_ok=True)

            # Keep MeshViewer in sync
            if hasattr(self, "mesh_viewer"):
                self.mesh_viewer.set_output_directory(default_dir)

            self.logger.log(f"[INFO] Auto-assigned output directory: {default_dir}")
        else:
            # Ensure rbf_original exists even if user picked an output dir already
            if not getattr(self, "rbf_original", None):
                self.rbf_original = os.path.join(self.output_directory, "surfaces", "n_0")
            os.makedirs(self.rbf_original, exist_ok=True)
                
    def save_file(self):
        import os
        import posixpath
        from PyQt5.QtWidgets import QFileDialog

        # 1) Pick output dir (local)
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return

        self.output_directory = output_dir
        os.makedirs(self.output_directory, exist_ok=True)

        # Make top-level subfolders
        for sub in ["preprocessed", "solutions", "surfaces", "volumes", "postprocessed"]:
            os.makedirs(os.path.join(self.output_directory, sub), exist_ok=True)

        # Ensure n_0 under surfaces exists
        self.rbf_original = os.path.join(self.output_directory, "surfaces", "n_0")
        os.makedirs(self.rbf_original, exist_ok=True)

        # Reflect in mesh viewer (local)
        if hasattr(self, "mesh_viewer") and self.mesh_viewer:
            self.mesh_viewer.output_dir = self.output_directory

        self.logger.log(f"[Local] Created output directory and subfolders under: {self.output_directory}")

        # 2) If HPC mode, mirror folders remotely (and stage baseline inputs to orig/)
        if getattr(self, "run_mode", "") == "HPC":
            try:
                # Name remote run folder after local output folder (sanitised)
                temp = os.path.basename(self.output_directory.rstrip('/\\')).replace(' ', '_')
                base_hpc_dir = f"/scratch/{self.ssh_creds['username']}/aeropt/aeropt_out/{temp}"
                self.base_hpc_dir = base_hpc_dir

                # Create remote dirs with bash (handles parents)
                subfolders = ["preprocessed", "solutions", "surfaces", "volumes", "postprocessed"]
                mkdir_parts = [f"mkdir -p '{base_hpc_dir}'"]
                mkdir_parts += [f"mkdir -p '{base_hpc_dir}/{sub}'" for sub in subfolders]
                mkdir_parts += [f"mkdir -p '{base_hpc_dir}/surfaces/n_0'"]
                mkdir_parts += [f"mkdir -p '{base_hpc_dir}/orig'"]

                mkdir_cmd = "bash -lc \"" + " && ".join(mkdir_parts) + "\""
                _in, _out, _err = self.ssh_client.exec_command(mkdir_cmd)
                err = _err.read().decode(errors="ignore").strip()
                if err:
                    self.logger.log(f"[HPC] mkdir stderr: {err}")

                self.remote_output_dir = base_hpc_dir
                self.logger.log(f"[HPC] Created output directory and subfolders under: {self.remote_output_dir}")

                # Tell the viewer where remote outputs live
                if hasattr(self, "mesh_viewer") and self.mesh_viewer:
                    self.mesh_viewer.set_remote_output_directory(self.remote_output_dir)

            except Exception as e:
                self.logger.log(f"[HPC] Remote setup failed: {e}")
            
        # 3) Mark as ready and (re)create the pipeline now that BOTH input & output are known
        self.output_directory_set = True
        self.check_run_morph_button_state()

        # Centralized creation; will no-op if prerequisites are missing
        if hasattr(self, "create_pipeline"):
            self.create_pipeline()

        # If you’re using the new train-classifier gating, refresh it here too
        if hasattr(self, "_update_train_classifier_button_state"):
            self._update_train_classifier_button_state()

    def list_jobs_me(self) -> str:
        """
        Return a compact table of all your pending/running jobs.
        Uses squeue --me, falls back to sacct if needed.
        """
        fmt = " %i   | %j   | %T   | %M   | %D   | %R"  # JobID|Name|State|Elapsed|Nodes|Reason/NodeList
        cmd = f"bash -lc \"squeue --me -o '{fmt}' --noheader | sort\""
        _in, _out, _err = self.ssh_client.exec_command(cmd)
        out = _out.read().decode().strip()
        err = _err.read().decode().strip()
        if not out:
            # maybe nothing in queue yet → show last few finished today
            cmd2 = "bash -lc \"sacct --user=$USER --state=ALL --format=JobID,JobName,State,Elapsed -X --parsable2 --noheader | tail -n 10\""
            _in2, _out2, _err2 = self.ssh_client.exec_command(cmd2)
            out = _out2.read().decode().strip()
            err = _err2.read().decode().strip()
        if err:
            self.logger.log(f"[HPC][status stderr] {err}")
        return out


    def cancel_job(self, jobid: str) -> bool:
        """
        Cancel a job by ID. Returns True on success, False on error.
        """
        jobid = jobid.strip()
        if not jobid.isdigit():
            self.logger.log(f"[HPC] Invalid job id: '{jobid}'")
            return False
        cmd = f"bash -lc 'scancel {jobid}'"
        _in, _out, _err = self.ssh_client.exec_command(cmd)
        err = _err.read().decode().strip()
        if err:
            self.logger.log("[HPC][cancel stderr] {err}")
            return False
        return True

    def _mesh_is_loaded(self) -> bool:
        mv = getattr(self, "mesh_viewer", None)
        return bool(mv is not None and getattr(mv, "mesh_obj", None) is not None)

    def on_hpc_status_clicked(self):
        try:
            txt = self.list_jobs_me()
            # Pretty-log each line
            if txt.strip() == "(no jobs)":
                self.logger.log("[HPC] No jobs in queue.")
            else:
                self.logger.log("[HPC] Jobs for you:")
                for line in txt.splitlines():
                    # normalize to a readable line
                    self.logger.log("  " + line)
        except Exception as e:
            self.logger.log(f"[HPC] Failed to get status: {e}")

    def on_hpc_cancel_clicked(self):
        from PyQt5.QtWidgets import QInputDialog
        jobid, ok = QInputDialog.getText(self, "Cancel Job", "Enter Slurm Job ID:")
        if not ok:
            return
        ok2 = False
        try:
            ok2 = self.cancel_job(jobid)
        except Exception as e:
            self.logger.log(f"[HPC] Cancel failed: {e}")
        if ok2:
            QMessageBox.information(self, "Cancel Job", f"Job {jobid} cancelled.")
        else:
            QMessageBox.warning(self, "Cancel Job", f"Could not cancel job '{jobid}'. Check the ID and try again.")
    
    def create_pipeline(self):
        """Create/refresh the pipeline only when both input & output are set."""
        # Preconditions
        if not getattr(self, "input_file_path", None):
            return

        if self.run_mode == "HPC":
            # Need remote_output_dir
            if not getattr(self, "remote_output_dir", None):
                return
            # Need a live SSH client
            if not hasattr(self, "ssh_client"):
                return

            # Create / refresh
            try:
                self.pipeline = HPCPipelineManager(main_window=self, n=0)
                if hasattr(self.pipeline, "_refresh_context"):
                    self.pipeline._refresh_context()
            except Exception as e:
                self.logger.log(f"[PIPE][HPC] Failed to create pipeline: {e}")
                return

        else:  # Local
            if not getattr(self, "output_directory", None):
                self.logger.log("[PIPE] Not creating pipeline yet: no local output dir.")
                return
            try:
                self.pipeline = PipelineManager(main_window=self, n=0)
                if hasattr(self.pipeline, "_refresh_context"):
                    self.pipeline._refresh_context()
            except Exception as e:
                self.logger.log(f"[PIPE][Local] Failed to create pipeline: {e}")
                return

        # Wire viewers
        try:
            if hasattr(self, "mesh_viewer") and self.mesh_viewer:
                self.mesh_viewer.set_pipeline(self.pipeline)
            if hasattr(self, "geo_viewer") and self.geo_viewer:
                self.geo_viewer.set_pipeline(self.pipeline)
        except Exception as e:
            self.logger.log(f"[PIPE] Failed to attach pipeline to viewers: {e}")

        # Optional: upload currently loaded file to HPC (if helper exists)
        if self.run_mode == "HPC" and hasattr(self.pipeline, "upload_geometry"):
            try:
                self.pipeline.upload_geometry(self.input_file_path)
                self.logger.log("[HPC] Uploaded input after pipeline creation.")
            except Exception as e:
                self.logger.log(f"[HPC] Geometry upload failed: {e}")

        self.logger.log("[PIPE] Pipeline initialised and linked.")
    
    def _update_train_classifier_button_state(self):
        ok = (
            getattr(self, "run_mode", "") == "HPC"
            and bool(getattr(self, "remote_output_dir", None))
            and bool(getattr(self, "ssh_client", None))
            and bool(getattr(self, "control_nodes_saved", False))
            and bool(getattr(self, "output_directory_set", False))
        )
        if hasattr(self, "train_mesh_classifier_btn"):
            self.train_mesh_classifier_btn.setEnabled(ok)
    
    def check_run_morph_button_state(self):
        if self.control_nodes_saved and self.output_directory_set:
            self.run_morph_btn.setEnabled(True)
            self.logger.log("[INFO] All requirements met. You can now run the simulation.")
        else:
            self.run_morph_btn.setEnabled(False)
            
    def check_run_sim_button_state(self):
        if self.solver_settings_saved and self.prepro_settings_saved:
            self.run_sim_btn.setEnabled(True)
            self.logger.log("[INFO] All requirements met. You can now run the solver.")
        else:
            self.run_sim_btn.setEnabled(False)

    def check_run_opt_button_state(self):
        if self.run_morph_btn.isEnabled() and self.run_sim_btn.isEnabled() and self.optimisation_settings_saved:
            self.run_opt_btn.setEnabled(True)
            self.logger.log("[INFO] All requirements met. You can now run optimisation.")
        else:
            self.run_opt_btn.setEnabled(False)
    
    def create_buttons(self):
        """Create navigation buttons."""
        buttons = [
            ("Geometry Definition", self.open_geometry_window),
            ("Solver Settings", self.activate_tab_3),
            ("Optimisation Settings", self.activate_tab_2),
            ("Monitors", self.open_monitor_editor),
            ("Train Mesh Classifier", self.on_train_mesh_classifier_clicked),
        ]
        
        self.button_layout.setAlignment(Qt.AlignTop)
        for text, handler in buttons:
            btn = QPushButton(text)
            btn.pressed.connect(handler)
            self.button_layout.addWidget(btn)
            self.button_layout.addSpacing(12)
            
            if text == "Train Mesh Classifier":
                self.button_layout.addSpacing(12)
                self.train_mesh_classifier_btn = btn
                
        self.train_mesh_classifier_btn.setEnabled(False)
            
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.button_layout.addItem(spacer)

        # ---- PCA reduced space toggle (optional) ----
        try:
            from PyQt5.QtWidgets import QCheckBox
            self.use_pca_reduced_cb = QCheckBox("Use PCA reduced space")
            self.use_pca_reduced_cb.setChecked(False)
            self.use_pca_reduced_cb.toggled.connect(self._on_toggle_use_pca_reduced)
            self.button_layout.addWidget(self.use_pca_reduced_cb)
        except Exception:
            self.use_pca_reduced_cb = None

        self.run_morph_btn = QPushButton("Run Morph")
        self.run_morph_btn.pressed.connect(self.run_morph)
        self.run_morph_btn.setEnabled(False)
        self.button_layout.addWidget(self.run_morph_btn)
        
        self.run_sim_btn = QPushButton("Run Simulation")
        self.run_sim_btn.pressed.connect(self.run_simulation)
        self.run_sim_btn.setEnabled(False)
        self.button_layout.addWidget(self.run_sim_btn)
        
        self.run_opt_btn = QPushButton("Run Optimisation")
        self.run_opt_btn.pressed.connect(self.run_optimisation)
        self.run_opt_btn.setEnabled(False)
        self.button_layout.addWidget(self.run_opt_btn)


    def _on_toggle_use_pca_reduced(self, checked: bool):
        """
        Global toggle in main UI. This should:
        1) store the preference in MainWindow
        2) propagate into MeshViewer so its CN panel defaults match
        """
        self.use_pca_reduced = bool(checked)

        mv = getattr(self, "mesh_viewer", None)
        if mv is None:
            return

        # If CN panel already created, sync its checkbox
        cb = getattr(mv, "use_pca_cb", None)
        if cb is not None:
            try:
                cb.blockSignals(True)
                cb.setChecked(self.use_pca_reduced)
            finally:
                cb.blockSignals(False)

        # Also store on MeshViewer so it persists even before the CN panel exists
        mv.use_pca = self.use_pca_reduced

        if hasattr(self, "logger") and self.logger:
            self.logger.log(f"[PCA] Use PCA reduced space = {self.use_pca_reduced}")


    def run_morph(self):
        """
        Run ONLY morph + volume on the cluster, via a cluster-side orchestrator (remoteMorph.py),
        matching the same pattern as run_optimisation().
        """
        if not getattr(self, "ssh_client", None):
            self.logger.log("[MORPH][ERROR] Not connected to HPC.")
            return

        if not self._stage_before_hpc_run("MORPH"):
            return

        # --- ask how many morphs ---
        from PyQt5.QtWidgets import QInputDialog
        dialog = MorphDialog(self, default_cases=5)
        if dialog.exec_() != QDialog.Accepted:
            return
        vals = dialog.values()
        n_cases = int(vals["n_cases"])
        run_mode = str(vals["run_mode"])

        import os, json, posixpath, tempfile
        from datetime import datetime

        remote_run = posixpath.join(self.remote_output_dir, "morph/")
        base = getattr(self, "base_name", None) or self.get_project_basename()

        # 1) Export & upload morph_basis.json
        remote_basis_path = self.export_morph_basis_for_opt(self.remote_output_dir)
        if not remote_basis_path:
            self.logger.log("[MORPH][WARN] Morph basis not available; displacements may be zero.")
        else:
            self.logger.log(f"[MORPH] Using morph basis: {remote_basis_path}")

        # 2) mkdir -p remote_run
        self.ssh_client.exec_command(f"bash -lc 'mkdir -p \"{remote_run}\"'")

        # 3) write morph_settings.json locally then upload
        local_tmp = tempfile.mkdtemp()
        morph_json = os.path.join(local_tmp, "morph_settings.json")

        settings = {
            "remote_output": self.remote_output_dir,  # <- base AerOpt out dir on cluster
            "run_dir": remote_run,                   # <- this run folder (where logs go)
            "base_name": base,
            "input_dir": posixpath.join(self.remote_output_dir, "orig"),
            "morph_basis_json": remote_basis_path or "",
            "cad_units": getattr(self, "cad_units", "mm"),
            "parallel_domains": int(getattr(self, "parallel_domains", 80)),
            "n_cases": int(n_cases),
            "run_mode": str(run_mode),

            # optional knobs (remoteMorph will default if missing)
            "coeff_sigma": 0.5,   # random coeff distribution
            "seed": None,         # or an int for repeatability
            "n_cases": n_cases,   # or however many
            "batch_size": 10,     # how many to run at once
            "poll_s": 45         # how long to wait to check if batch has been run
        }

        with open(morph_json, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)

        sftp = self.ssh_client.open_sftp()
        try:
            sftp.put(morph_json, posixpath.join(remote_run, "morph_settings.json"))
        finally:
            sftp.close()

        # 4) Create & sbatch orchestrator (same pattern as run_optimisation)
        batch = "\n".join([
            "#!/bin/bash -l",
            "#SBATCH --job-name=morph_orch",
            "#SBATCH --output=morph_orch.%J.out",
            "#SBATCH --error=morph_orch.%J.err",
            "#SBATCH --time=1-00:00",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "source ~/.bashrc",
            "set -euo pipefail",
            f"cd \"{remote_run}\"",
            f"/home/{self.ssh_creds['username']}/.conda/envs/aeropt-hpc/bin/python "
            f"/home/{self.ssh_creds['username']}/aeropt/Scripts/Remote/pipelineMorph.py \"{remote_run}\"",
        ])

        local_batch = os.path.join(local_tmp, "batchfile_morph_orchestrator")
        with open(local_batch, "w", newline="\n") as f:
            f.write(batch + "\n")

        sftp = self.ssh_client.open_sftp()
        try:
            sftp.put(local_batch, posixpath.join(remote_run, "batchfile_morph_orchestrator"))
        finally:
            sftp.close()

        _in, _out, _err = self.ssh_client.exec_command(
            f"bash -lc 'cd \"{remote_run}\"; sbatch batchfile_morph_orchestrator'"
        )
        out = _out.read().decode().strip()
        err = _err.read().decode().strip()
        if err:
            self.logger.log(f"[MORPH][HPC][WARN] sbatch stderr: {err}")
        self.logger.log(f"[MORPH][HPC] sbatch: {out}")


    def run_simulation(self):
        if not self.run_sim_btn.isEnabled():
            missing_items = []
            if not getattr(self, "output_directory_set", False):
                missing_items.append("output directory needs to be set")
            if not getattr(self, "solver_settings_saved", False):
                missing_items.append("solver settings need to be set")
            if missing_items:
                self.logger.log(f"[ERROR] Cannot run simulation: {' and '.join(missing_items)}.")
                return

        if hasattr(self, "output_directory") and self.output_directory:
            self.base = self.get_project_basename()
            default_inp = os.path.join(self.output_directory, f"{self.base}.inp")
            if os.path.exists(default_inp):
                self.solver_input_path = default_inp
        
        # Pull conditions from the parallel editor if present; otherwise fall back to the Objective editor
        conds = []
        if hasattr(self, "sim_config") and self.sim_config:
            conds = self.sim_config.get("conditions", [])
        if not conds:
            self.logger.log(
                "[SIM] No parallel flow conditions defined; "
                "running a single case using the solver .inp settings."
            )
            conds = [{}]
        if not conds:
            self.logger.log("[SIM][ERROR] Still no conditions. Aborting.")
            return
        
        if getattr(self, "run_mode", "") == "HPC":
            if not self._stage_before_hpc_run("SIM"):
                return

        self.stage_monitor_config_to_remote()

        # Launch on a worker thread
        from PyQt5.QtCore import QThread
        from GUI.workers import SimulationWorker

        self.sim_thread = QThread(self)
        # Slight tweak to your SimulationWorker: give it the cond list
        self.sim_worker = SimulationWorker(self, debug=True)
        self.sim_worker.conds = conds
        self.sim_worker.moveToThread(self.sim_thread)

        self.sim_worker.log.connect(self.logger.log)
        self.sim_worker.failed.connect(lambda msg: (self.logger.log(f"[SIM][ERROR] {msg}"), self.sim_thread.quit()))
        self.sim_worker.finished.connect(lambda: (self.logger.log("[SIM] Submitted all jobs."), self.sim_thread.quit()))

        self.sim_thread.started.connect(self.sim_worker.run)
        self.sim_thread.finished.connect(self.sim_thread.deleteLater)

        self.logger.log("[SIM] Preparing submission…")
        self.sim_thread.start()
        
    
    def _normalise_monitor_type_from_metric(self, metric: str) -> str:
        m = str(metric).strip().lower()

        return {
            "pressure recovery": "pressure_recovery",
            "pressure_recovery": "pressure_recovery",
            "pr": "pressure_recovery",
            "dc60": "distortion",
            "distortion": "distortion",
            "cd": "drag",
            "drag": "drag",
            "cl": "lift",
            "lift": "lift",
            "cm": "moment",
            "moment": "moment",
        }.get(m, m)


    def _sync_objective_terms_to_monitor_config(self):
        """
        Adds surface-based objective terms to self.monitor_config automatically.
        RSD terms are ignored because they come directly from the solver .rsd file.
        """
        obj = getattr(self, "objective_config", {}) or {}
        terms = obj.get("terms", []) or []

        if not hasattr(self, "monitor_config") or self.monitor_config is None:
            self.monitor_config = {
                "interval": 50,
                "enabled": True,
                "monitors": []
            }

        self.monitor_config.setdefault("interval", 50)
        self.monitor_config.setdefault("enabled", True)
        self.monitor_config.setdefault("monitors", [])

        def monitor_key(mon):
            return (
                str(mon.get("type", "")).lower(),
                tuple(int(x) for x in mon.get("surface_ids", []) or []),
                str(mon.get("direction", "")).lower(),
            )

        existing = {monitor_key(m) for m in self.monitor_config["monitors"]}

        for term in terms:
            source = str(term.get("source", "auto")).lower()
            metric = str(term.get("metric", "")).strip()
            surface_ids = [int(x) for x in term.get("surface_ids", []) or []]

            if source == "auto":
                source = "monitor" if surface_ids else "rsd"

            if source != "monitor":
                continue

            if not surface_ids:
                continue

            mon = {
                "type": self._normalise_monitor_type_from_metric(metric),
                "name": str(term.get("symbol", metric)),
                "enabled": True,
                "surface_ids": surface_ids,
                "objective_symbol": str(term.get("symbol", metric)),
                "reduction": str(term.get("reduction", "last")),
            }

            if mon["type"] in ("drag", "lift", "moment"):
                mon["direction"] = str(term.get("direction", "x"))
                mon["symmetry_factor"] = int(term.get("symmetry_factor", 1))

            k = monitor_key(mon)
            if k not in existing:
                self.monitor_config["monitors"].append(mon)
                existing.add(k)

        return self.monitor_config
        
    def export_morph_basis_for_opt(self, remote_run: str) -> str:
        """
        Build a morph_basis.json from the current MeshViewer settings and upload it
        into the optimisation run directory on the HPC.

        Returns the remote path to morph_basis.json (Unix-style).
        """
        import json, os, posixpath, tempfile

        # Sanity checks
        if not hasattr(self, "mesh_viewer") or self.mesh_viewer is None:
            self.logger.log("[OPT][ERROR] No MeshViewer available to export morph basis.")
            return ""

        mv = self.mesh_viewer
        # We expect save_controlnodes to have been called already
        if not hasattr(mv, "control_nodes") or mv.control_nodes is None:
            self.logger.log("[OPT][ERROR] Control nodes not defined; save them before running optimisation.")
            return ""
    

        # 1) Build local JSON from mesh viewer state
        basis_cfg = {
            "control_nodes": mv.control_nodes.tolist(),
            "control_normals": getattr(mv, "control_normals", None).tolist()
                if getattr(mv, "control_normals", None) is not None else None,

            "parameterisation_method": getattr(mv, "parameterisation_method", "modal"),
            "direct_parameterisation_subtype": getattr(mv, "direct_parameterisation_subtype", None),

            "selection_mode": getattr(mv, "control_node_selection_mode", None),
            "loaded_control_nodes_path": getattr(mv, "loaded_control_nodes_path", None),
            "loaded_control_normals_path": getattr(mv, "loaded_control_normals_path", None),

            "t_patch_scale": getattr(mv, "t_patch_scale", None),
            "amp_alpha": getattr(mv, "amp_alpha", 0.001),

            "TSurfaces": getattr(mv, "TSurfaces", []),
            "USurfaces": getattr(mv, "USurfaces", []),
            "CSurfaces": getattr(mv, "CSurfaces", []),
            
            # preliminary response-surface / regional screening metadata
            "prelim_enabled": bool(getattr(mv, "prelim_enabled", False)),
            "prelim_regions": int(getattr(mv, "prelim_regions", 1) or 1),
            "prelim_final_control_nodes": int(getattr(mv, "prelim_final_control_nodes", 0) or 0),
            "prelim_keep_fraction": float(getattr(mv, "prelim_keep_fraction", 0.67) or 0.67),
            "prelim_doe_amplitude": float(getattr(mv, "prelim_doe_amplitude", 1.0) or 1.0),

            "point_region_ids": (
                getattr(mv, "point_region_ids", None).tolist()
                if getattr(mv, "point_region_ids", None) is not None else None
            ),
            "control_node_region_ids": (
                getattr(mv, "control_node_region_ids", None).tolist()
                if getattr(mv, "control_node_region_ids", None) is not None else None
            ),
            "region_centres": (
                getattr(mv, "region_centres", None).tolist()
                if getattr(mv, "region_centres", None) is not None else None
            ),
            "control_node_point_indices": (
                getattr(mv, "control_node_point_indices", None).tolist()
                if getattr(mv, "control_node_point_indices", None) is not None else None
            ),

            "k_modes": getattr(mv, "k_modes", 0),
            "spectral_p": getattr(mv, "spectral_p", None),
            "coeff_frac": getattr(mv, "coeff_frac", None),
            "seed": getattr(mv, "seed", 0),

            "normal_project": getattr(mv, "normal_project", None),
            "vector_mode": getattr(mv, "vector_mode", None),
            "frame_knn": getattr(mv, "frame_knn", None),

            "use_local_modes": getattr(mv, "use_local_modes", False),
            "global_modes": getattr(mv, "global_modes_selected", False),
            "global_only": getattr(mv, "global_only", False),
            "global_mode_config": getattr(mv, "global_mode_config", []),
            "basis_axes": getattr(mv, "basis_axes", None),

            "use_pca": getattr(mv, "use_pca", False),
            "pca_cache_path": getattr(mv, "pca_cache_path", None),
            "pca_train_M": getattr(mv, "pca_train_M", None),
            "pca_energy": getattr(mv, "pca_energy", None),
            "pca_k_red": getattr(mv, "pca_k_red", None),
            "pca_k_final": getattr(mv, "pca_k_final", None),

            "bump_enable": getattr(mv, "bump_enable", False),
            "bump_center": getattr(mv, "bump_center", None),
            "bump_radius": getattr(mv, "bump_radius", None),
            "bump_one_sided": getattr(mv, "bump_one_sided", False),
            
            "use_protection": bool(
                getattr(mv, "use_protection", bool(getattr(mv, "protected_control_nodes", [])))
            ),
            "protected_control_nodes": [int(i) for i in getattr(mv, "protected_control_nodes", [])],
            "protection_radius": (
                float(getattr(mv, "protection_radius", 0.0))
                if getattr(mv, "protection_radius", None) is not None
                else None
            ),

            "rigid_translation": getattr(mv, "rigid_boundary_translation", True),
        }

        local_tmp = tempfile.mkdtemp()
        local_basis = os.path.join(local_tmp, "morph_basis.json")
        with open(local_basis, "w", encoding="utf-8") as f:
            json.dump(basis_cfg, f, indent=2)
        self.logger.log(f"[OPT] Wrote morph_basis.json → {local_basis}")

        # ---- 2) Upload to HPC under <remote_run>/morph/morph_basis.json ----
        if getattr(self, "run_mode", "") != "HPC":
            self.logger.log("[OPT] Not in HPC mode; skipping remote upload of morph basis.")
            return ""

        if not hasattr(self, "ssh_client"):
            self.logger.log("[OPT][ERROR] No SSH client for remote upload of morph basis.")
            return ""

        remote_morph_dir  = posixpath.join(remote_run, "morph")
        remote_basis_path = posixpath.join(remote_morph_dir, "morph_basis.json")

        # mkdir morph dir first
        cmd = f"bash -lc 'mkdir -p \"{remote_morph_dir}\"'"
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        err_text = stderr.read().decode().strip()
        if exit_code != 0 or err_text:
            self.logger.log(f"[OPT][HPC][ERROR] Failed to create morph dir '{remote_morph_dir}'. exit={exit_code}, stderr={err_text}")
            return ""

        # --- if use_pca: upload PCA cache to remote_morph_dir and rewrite JSON path ---
        if basis_cfg.get("use_pca", False):
            local_pca = basis_cfg.get("pca_cache_path", None)
            if (not local_pca) or (not os.path.exists(local_pca)):
                self.logger.log(f"[OPT][ERROR] use_pca=True but local PCA cache not found: {local_pca}")
                return ""

            remote_pca_path = posixpath.join(remote_morph_dir, "pca_basis.npz")
            sftp = self.ssh_client.open_sftp()
            try:
                self.logger.log(f"[OPT][HPC] Uploading PCA cache '{local_pca}' → '{remote_pca_path}'")
                sftp.put(local_pca, remote_pca_path)
            finally:
                sftp.close()

            basis_cfg["pca_cache_path"] = remote_pca_path  # <-- CRITICAL for cluster

            # rewrite local morph_basis.json with UPDATED pca_cache_path
            with open(local_basis, "w", encoding="utf-8") as f:
                json.dump(basis_cfg, f, indent=2)

        # finally upload morph_basis.json
        sftp = self.ssh_client.open_sftp()
        try:
            self.logger.log(f"[OPT][HPC] Uploading morph basis '{local_basis}' → '{remote_basis_path}'")
            sftp.put(local_basis, remote_basis_path)
        finally:
            sftp.close()

        return remote_basis_path

    def _stage_orig_inputs_to_remote(self) -> str:
        """
        Stage BAC/BPP/BCO + simple 'control' files from the local project
        into the cluster-side 'orig/' directory.

        Returns the remote 'orig' directory path (posix) or "" on failure.
        """
        if self.run_mode != "HPC" or not getattr(self, "ssh_client", None):
            self.logger.log("[OPT][HPC] Not staging orig inputs (not in HPC mode or no SSH).")
            return ""

        import os, posixpath, glob, tempfile

        base = getattr(self, "base_name", None) or self.get_project_basename()

        # Local search dirs (PC side)
        inp_dir = getattr(self, "input_directory", None)
        if not inp_dir and getattr(self, "input_file_path", None):
            inp_dir = os.path.dirname(self.input_file_path)

        out_dir = getattr(self, "output_directory", None) or os.getcwd()
        search_dirs = []
        for d in (inp_dir, out_dir):
            if d and d not in search_dirs:
                search_dirs.append(d)

        if not search_dirs:
            self.logger.log("[OPT][HPC][WARN] No local search dirs for orig staging.")
            return ""

        remote_orig = posixpath.join(self.remote_output_dir, "orig")
        # ensure remote orig/ exists
        cmd = f"bash -lc 'mkdir -p \"{remote_orig}\"'"
        _in, _out, _err = self.ssh_client.exec_command(cmd)
        exit_code = _out.channel.recv_exit_status()
        err_text = _err.read().decode().strip()
        if exit_code != 0 or err_text:
            self.logger.log(f"[OPT][HPC][WARN] Failed to create remote orig/: {err_text or exit_code}")
            return ""

        def find_first(relname: str):
            for d in search_dirs:
                cand = os.path.join(d, relname)
                if os.path.exists(cand):
                    return cand
            return None

        files_to_upload = {}

        # Core FLITE inputs
        for ext in ("bac", "bpp", "bco"):
            rel = f"{base}.{ext}"
            src = find_first(rel)
            if src:
                files_to_upload[rel] = src
            else:
                self.logger.log(f"[OPT][HPC][WARN] No local {rel} found in {search_dirs}")
        
        output_vtk = find_first(os.path.join("surfaces", "output.vtk"))
        if output_vtk:
            files_to_upload["output.vtk"] = {
                "src": output_vtk,
                "remote_dir": posixpath.join(
                    self.remote_output_dir,
                    "surfaces"
                ),
            }
        else:
            self.logger.log(
                "[OPT][HPC][WARN] Could not find surfaces/output.vtk locally."
            )

        # modal basis -> remote orig/
        modal_basis = find_first(
            os.path.join(
                "Control Nodes",
                "modal_basis_T_surface.npz"
            )
        )
        if modal_basis:
            files_to_upload["modal_basis_T_surface.npz"] = {
                "src": modal_basis,
                "remote_dir": remote_orig,
            }
        else:
            self.logger.log(
                "[OPT][HPC][WARN] Could not find "
                "Control Nodes/modal_basis_T_surface.npz locally."
            )
        
        # Generic "control" files (optional, very loose match)
        for pattern in (f"{base}*control*", "control*", f"Mesh3D_v50.ctl"):
            for d in search_dirs:
                for path in glob.glob(os.path.join(d, pattern)):
                    rel = os.path.basename(path)
                    if rel not in files_to_upload:
                        files_to_upload[rel] = path

        if not files_to_upload:
            self.logger.log("[OPT][HPC][WARN] No BAC/BPP/BCO/control files staged to orig/.")
            return remote_orig

        # Upload via SFTP
        try:
            sftp = self.ssh_client.open_sftp()
            try:
                for relname, src in files_to_upload.items():
                    dst = posixpath.join(remote_orig, relname)
                    sftp.put(src, dst)
                    self.logger.log(f"[OPT][HPC] Staged {src} → {dst}")
            finally:
                sftp.close()
        except Exception as e:
            self.logger.log(f"[OPT][HPC][WARN] Failed to upload orig inputs: {e}")
            return ""

        return remote_orig

    def _pick_surface_mesh_for_opt(self) -> str:
        """
        Best-effort selection of a *local* surface mesh to use as the baseline
        for optimisation, and (if in HPC mode) stage it into remote_output_dir/orig.

        Returns the remote 'orig' directory path on the cluster if upload succeeds,
        otherwise "" (and logs warnings).
        """
        import os, glob, posixpath

        tried = []

        def _add(p):
            if p and p not in tried:
                tried.append(p)

        self.base = self.get_project_basename()
        local_surf = None

        # 1) Prefer a canonical .fro in rbf_original (surfaces/n_0)
        surf_dir = getattr(self, "rbf_original", None)
        if surf_dir:
            cand = os.path.join(surf_dir, f"{self.base}.fro")
            if os.path.exists(cand):
                self.logger.log(f"[OPT] Using baseline FRO from rbf_original: {cand}")
                local_surf = cand
            else:
                _add(cand)
                # any other .fro in rbf_original
                for p in glob.glob(os.path.join(surf_dir, "*.fro")):
                    if os.path.exists(p):
                        self.logger.log(f"[OPT] Using baseline FRO found in rbf_original: {p}")
                        local_surf = p
                        break

        # 2) Loaded file itself, if it's a supported surface mesh
        if not local_surf:
            loaded = getattr(self, "input_file_path", None)
            if loaded and os.path.exists(loaded) and os.path.splitext(loaded)[1].lower() in (".fro", ".vtk", ".vtm"):
                self.logger.log(f"[OPT] Using loaded file as baseline surface: {loaded}")
                local_surf = loaded
            _add(loaded)

        # 3) Search the input_directory for any .fro/.vtm/.vtk
        if not local_surf:
            inp_dir = getattr(self, "input_directory", None) or ""
            for ext in (".fro", ".vtm", ".vtk"):
                for p in glob.glob(os.path.join(inp_dir, f"*{ext}")):
                    if os.path.exists(p):
                        self.logger.log(f"[OPT] Using baseline surface from input dir: {p}")
                        local_surf = p
                        break
                if local_surf:
                    break
                _add(os.path.join(inp_dir, f"*{ext}"))

        # 4) Nothing found → log and bail gracefully
        if not local_surf or not os.path.exists(local_surf):
            self.logger.log("[OPT][HPC][WARN] No baseline surface mesh found. Tried:")
            for t in tried:
                if t:
                    self.logger.log(f"    {t}")
            return ""

        # 5) If VTK/VTM, convert to FRO and use that instead
        ext = os.path.splitext(local_surf)[1].lower()
        if ext in (".vtm", ".vtk"):
            try:
                from ConvertFileType.convertVtmtoFro import vtm_to_fro
            except ImportError as e:
                self.logger.log(f"[OPT][ERROR] Cannot import vtm_to_fro to convert {local_surf}: {e}")
                return ""

            fro_dir = surf_dir
            if not fro_dir:
                if getattr(self, "output_directory", None):
                    fro_dir = os.path.join(self.output_directory, "surfaces", "n_0")
                else:
                    fro_dir = os.path.dirname(local_surf)

            os.makedirs(fro_dir, exist_ok=True)
            fro_out = os.path.join(fro_dir, f"{self.base}.fro")
            self.logger.log(f"[OPT] Converting {local_surf} → {fro_out}")
            try:
                vtm_to_fro(local_surf, fro_out)
            except Exception as e:
                self.logger.log(f"[OPT][ERROR] VTM/VYK → FRO conversion failed: {e}")
                return ""

            if not os.path.exists(fro_out):
                self.logger.log(f"[OPT][ERROR] Conversion did not produce '{fro_out}'.")
                return ""

            local_surf = fro_out

        # 6) Upload to remote 'orig/' if in HPC mode
        if (
            getattr(self, "run_mode", "") == "HPC"
            and getattr(self, "remote_output_dir", None)
            and hasattr(self, "ssh_client")
        ):
            remote_orig = posixpath.join(self.remote_output_dir, "orig")
            cmd = f"bash -lc 'mkdir -p \"{remote_orig}\"'"
            _in, _out, _err = self.ssh_client.exec_command(cmd)
            exit_code = _out.channel.recv_exit_status()
            err_text = _err.read().decode().strip()
            if exit_code != 0 or err_text:
                self.logger.log(f"[OPT][HPC][WARN] Failed to create remote 'orig/': {err_text or exit_code}")
                return ""

            try:
                sftp = self.ssh_client.open_sftp()
                try:
                    remote_mesh = posixpath.join(remote_orig, os.path.basename(local_surf))
                    sftp.put(local_surf, remote_mesh)
                    self.logger.log(f"[OPT][HPC] Staged baseline surface mesh → {remote_mesh}")
                finally:
                    sftp.close()
            except Exception as e:
                self.logger.log(f"[OPT][HPC][WARN] Failed to upload baseline surface mesh: {e}")
                return ""

            return remote_orig

        # Not in HPC mode → just return empty to indicate "no remote orig"
        return ""

    def run_prelim_response_surface(
        self,
        remote_run,
        remote_basis_path,
        bo_json_path,
        obj_json_path
    ):
        import os
        import posixpath
        import tempfile

        prelim_run = posixpath.join(remote_run, "prelim")
        user = self.ssh_creds["username"]

        py = f"/home/{user}/.conda/envs/aeropt-hpc/bin/python"
        script = (
            f"/home/{user}/aeropt/Scripts/Remote/"
            f"prelimResponseSurface.py"
        )

        # Create remote preliminary-study directory and WAIT for completion
        mkdir_cmd = f"bash -lc 'mkdir -p \"{prelim_run}\"'"
        _stdin, stdout, stderr = self.ssh_client.exec_command(mkdir_cmd)

        exit_code = stdout.channel.recv_exit_status()
        err_text = stderr.read().decode(errors="ignore").strip()

        if exit_code != 0:
            self.logger.log(
                f"[PRELIM][ERROR] Failed to create remote directory "
                f"'{prelim_run}': {err_text or f'exit code {exit_code}'}"
            )
            return None

        self.logger.log(
            f"[PRELIM] Remote directory ready: {prelim_run}"
        )

        batch = "\n".join([
            "#!/bin/bash -l",
            "#SBATCH --job-name=prelim_rs",
            "#SBATCH --output=prelim_rs.%J.out",
            "#SBATCH --error=prelim_rs.%J.err",
            "#SBATCH --time=3-00:00",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "source ~/.bashrc",
            "set -euo pipefail",
            f'cd "{prelim_run}"',
            (
                f'{py} "{script}" '
                f'"{prelim_run}" '
                f'"{remote_basis_path}" '
                f'"{bo_json_path}" '
                f'"{obj_json_path}"'
            ),
        ])

        local_tmp = tempfile.mkdtemp()
        local_batch = os.path.join(
            local_tmp,
            "batchfile_prelim_rs"
        )

        with open(local_batch, "w", newline="\n") as f:
            f.write(batch + "\n")

        remote_batch = posixpath.join(
            prelim_run,
            "batchfile_prelim_rs"
        )

        sftp = self.ssh_client.open_sftp()
        try:
            sftp.put(local_batch, remote_batch)
        finally:
            sftp.close()

        self.logger.log(
            f"[PRELIM] Uploaded batchfile: {remote_batch}"
        )

        _in, _out, _err = self.ssh_client.exec_command(
            f'bash -lc \'cd "{prelim_run}" && '
            f'sbatch batchfile_prelim_rs\''
        )

        out = _out.read().decode().strip()
        err = _err.read().decode().strip()

        if err:
            self.logger.log(
                f"[PRELIM][WARN] sbatch stderr: {err}"
            )

        self.logger.log(f"[PRELIM] Submitted: {out}")

        if "Submitted batch job" in out:
            return out.split()[-1]

        return None

    def run_optimisation(self):
        if not getattr(self, "optimisation_settings_saved", False):
            self.logger.log("[OPT] Please save optimisation settings first.")
            return

        if self.run_mode != "HPC":
            self.logger.log("[OPT] Optimisation is currently only wired for HPC mode.")
            return

        import os, posixpath, json, tempfile

        # ----------------------------------------------------------
        # 1) Resolve remote path for THIS optimisation run
        # ----------------------------------------------------------
        remote_run = posixpath.join(self.remote_output_dir, "opt")
        #remote_orig = self._pick_surface_mesh_for_opt()
        #if not remote_orig:
        #    self.logger.log(
        #        "[OPT][HPC][WARN] No baseline surface staged to 'orig/'. "
        #        "Optimiser will rely on whatever exists on the cluster."
        #    )
        #remote_orig = self._stage_orig_inputs_to_remote()
        #if not remote_orig:
        #    self.logger.log("[OPT][HPC][WARN] Could not stage BAC/BPP/BCO to orig/; "
        #                    "remoteOpt will rely on existing files on the cluster.")    

        # 1a) Export & upload morph_basis.json for this run
        if not self._stage_before_hpc_run("OPT"):
            return
        
        self._sync_objective_terms_to_monitor_config()
        monitor_remote_path = self.stage_monitor_config_to_remote()
        
        remote_basis_path = self.export_morph_basis_for_opt(remote_run)
        if not remote_basis_path:
            self.logger.log("[OPT][WARN] Morph basis not available; optimisation may run with zero morphing.")

        # 2) mkdir -p remote_run
        self.ssh_client.exec_command(
            f"bash -lc 'mkdir -p \"{remote_run}\"'"
        )

        # 3) Save JSONs locally then SFTP them up
        import json, os, posixpath, tempfile
        local_tmp = tempfile.mkdtemp()
        bo_json = os.path.join(local_tmp, "bo_settings.json")
        obj_json = os.path.join(local_tmp, "objective.json")
        
        remote_bo_json = posixpath.join(remote_run, "bo_settings.json")
        remote_obj_json = posixpath.join(remote_run, "objective.json")

        # turn your in-memory settings → json-safe (map class names)
        s = dict(self.bayes_settings)
        s["kernel"] = self.bo_kernel_combo.currentText()
        s["acquisition_function"] = self.bo_acq_combo.currentText()
        s["sim_dir"] = remote_run + "/"
        s["units"] = self.cad_units
        s["parallel"] = self.parallel
        # NEW: include path to morph_basis.json on the cluster
        s["morph_basis_json"] = remote_basis_path
        s["monitor_config_json"] = monitor_remote_path
        s["previous_solution"] = self.get_previous_solution_config_for_cluster()
        
        s["base_name"] = self.base
        s["input_dir"] = posixpath.join(self.remote_output_dir, "orig")

        with open(bo_json, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        with open(obj_json,"w", encoding="utf-8") as f:
            json.dump(self.objective_config, f, indent=2)

        sftp = self.ssh_client.open_sftp()
        try:
            sftp.put(bo_json, posixpath.join(remote_run, "bo_settings.json"))
            sftp.put(obj_json, posixpath.join(remote_run, "objective.json"))
        finally:
            sftp.close()

        # 4) Create & sbatch orchestrator
        batch = "\n".join([
            "#!/bin/bash -l",
            "#SBATCH --job-name=opt_orch",
            "#SBATCH --output=opt_orch.%J.out",
            "#SBATCH --error=opt_orch.%J.err",
            "#SBATCH --time=3-00:00",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "source ~/.bashrc",
            "set -euo pipefail",
            f"cd \"{remote_run}\"",
            f"/home/{self.ssh_creds['username']}/.conda/envs/aeropt-hpc/bin/python "
            f"/home/{self.ssh_creds['username']}/aeropt/Scripts/Remote/remoteOpt.py \"{remote_run}\"",
        ])

        local_batch = os.path.join(local_tmp, "batchfile_opt_orchestrator")
        with open(local_batch, "w", newline="\n") as f:
            f.write(batch + "\n")
        sftp = self.ssh_client.open_sftp()
        try:
            sftp.put(local_batch, posixpath.join(remote_run, "batchfile_opt_orchestrator"))
        finally:
            sftp.close()
            
        prelim_enabled = bool(getattr(self.mesh_viewer, "prelim_enabled", False))

        dependency = ""
        if prelim_enabled:
            prelim_jid = self.run_prelim_response_surface(
                remote_run=remote_run,
                remote_basis_path=remote_basis_path,
                bo_json_path=remote_bo_json,
                obj_json_path=remote_obj_json,
            )

            if prelim_jid:
                dependency = f"--dependency=afterok:{prelim_jid}"
            else:
                self.logger.log("[OPT][ERROR] Prelim submission failed; optimisation not submitted.")
                return

        _in,_out,_err = self.ssh_client.exec_command(
            f"bash -lc 'cd \"{remote_run}\"; "
            f"sbatch {dependency} batchfile_opt_orchestrator'"
        )
        out = _out.read().decode().strip()
        jid = out.split()[-1] if "Submitted batch job" in out else "?"
        self.logger.log(f"[OPT] Submitted headless optimisation job {jid}. You can now close the UI.")
        return

    def open_geometry_window(self):
        if hasattr(self, "mesh_viewer") and self.mesh_viewer.mesh_obj:
            self.geom_window = GeomWindow(self.mesh_viewer.mesh_obj, self)
            self.geom_window.show()
        else:
            QMessageBox.warning(self, "Error", "Load a mesh first.")
            
    def on_train_mesh_classifier_clicked(self):
        """UI entry-point: stage baseline to orig/, export morph basis, and submit a classifier training sweep on HPC."""
        if getattr(self, "run_mode", "") != "HPC":
            QMessageBox.warning(self, "HPC only", "Mesh-classifier training is currently configured for HPC mode.")
            return
        if not bool(getattr(self, "control_nodes_saved", False)):
            QMessageBox.warning(self, "Missing control nodes", "Save control nodes first (define CNs + bounds) before training.")
            return
        if not bool(getattr(self, "output_directory_set", False)) or not getattr(self, "remote_output_dir", None):
            QMessageBox.warning(self, "Missing output directory", "Set an output directory first.")
            return
        if not getattr(self, "ssh_client", None):
            QMessageBox.warning(self, "Not connected", "Connect to the HPC first.")
            return

        if not self._stage_before_hpc_run("TRAIN"):
            return

        dlg = TrainMeshClassifierDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        n_cases, batch_size, poll_s = dlg.values()

        import os, json, posixpath, tempfile
        from datetime import datetime

        # Ensure baseline is staged into remote_output_dir/orig (includes baseline .fro)
        '''remote_orig_dir = self._pick_surface_mesh_for_opt()
        if not remote_orig_dir:
            self.logger.log("[TRAIN][ERROR] Failed to stage baseline surface mesh into remote orig/.")
            return'''

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_run = posixpath.join(self.remote_output_dir, "train_mesh_classifier", stamp)
        self.ssh_client.exec_command(f"bash -lc 'mkdir -p \"{remote_run}\"'")

        # Export morph basis into remote_run/morph/morph_basis.json (and upload PCA cache if enabled)
        remote_basis_path = self.export_morph_basis_for_opt(remote_run)
        if not remote_basis_path:
            self.logger.log("[TRAIN][ERROR] Failed to export morph basis; cannot start training sweep.")
            return

        # Minimal sweep settings (keep morph basis logic inside export_morph_basis_for_opt)
        sweep_settings = {
            "remote_output": self.remote_output_dir,
            "run_dir": remote_run,
            "input_dir": posixpath.join(self.remote_output_dir, "orig"),
            "morph_basis_json": remote_basis_path,
            "cad_units": getattr(self, "cad_units", "mm"),
            "parallel_domains": int(getattr(self, "parallel_domains", 80)),
            "n_cases": int(n_cases),
            "batch_size": int(batch_size),
            "poll_s": int(poll_s),
            "dataset_path": posixpath.join(remote_run, "dataset.jsonl"),
        }

        local_tmp = tempfile.mkdtemp()
        local_settings = os.path.join(local_tmp, "sweep_settings.json")
        with open(local_settings, "w", encoding="utf-8") as f:
            json.dump(sweep_settings, f, indent=2)

        sftp = self.ssh_client.open_sftp()
        try:
            sftp.put(local_settings, posixpath.join(remote_run, "sweep_settings.json"))
        finally:
            try: sftp.close()
            except Exception: pass

        # Submit the sweep driver (submit_sweep.py) which will call run_case.py (which calls the pipeline)
        user = self.ssh_creds["username"]
        py = f"/home/{user}/.conda/envs/aeropt-hpc/bin/python"
        # NOTE: these scripts will live alongside other remote scripts (we'll create them next)
        submit_sweep = f"/home/{user}/aeropt/Scripts/MeshFailureClassifier/sweep/submit_sweep.py"

        batch_lines = [
            "#!/bin/bash -l",
            "#SBATCH --job-name=train_meshclf",
            "#SBATCH --output=train_meshclf.%J.out",
            "#SBATCH --error=train_meshclf.%J.err",
            "#SBATCH --time=3-00:00",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "set -euo pipefail",
            "source ~/.bashrc",
            f'cd "{remote_run}"',
            f'{py} "{submit_sweep}" "{remote_run}"',
        ]
        
        local_batch = os.path.join(local_tmp, "batchfile_train_meshclf")
        with open(local_batch, "w", newline="\n") as f:
            f.write("\n".join(batch_lines) + "\n")

        sftp = self.ssh_client.open_sftp()
        try:
            sftp.put(local_batch, posixpath.join(remote_run, "batchfile_train_meshclf"))
        finally:
            try: sftp.close()
            except Exception: pass

        _in, _out, _err = self.ssh_client.exec_command(f"bash -lc 'cd \"{remote_run}\"; sbatch batchfile_train_meshclf'")
        out = _out.read().decode().strip()
        err = _err.read().decode().strip()
        if err:
            self.logger.log(f"[TRAIN][HPC][WARN] sbatch stderr: {err}")
        self.logger.log(f"[TRAIN][HPC] Submitted: {out}")
        self.logger.log(f"[TRAIN][HPC] Run dir: {remote_run}")

    def open_monitor_editor(self):
        if not hasattr(self, "mesh_viewer") or self.mesh_viewer is None or self.mesh_viewer.mesh_obj is None:
            QMessageBox.warning(self, "No mesh loaded", "Please load a mesh first so surface IDs can be selected.")
            return

        dlg = MonitorEditor(
            parent=self,
            mesh_obj=self.mesh_viewer.mesh_obj,
            config=getattr(self, "monitor_config", None)
        )
        if dlg.exec_() == QDialog.Accepted:
            self.monitor_config = dlg.get_config()
            self.monitors_saved = True

            # save locally if output dir exists
            if getattr(self, "output_directory", None):
                import json, os
                out_path = os.path.join(self.output_directory, "monitors.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(self.monitor_config, f, indent=2)
                self.logger.log(f"[MON] Monitor settings saved to {out_path}")

            n_mon = len(self.monitor_config.get("monitors", []))
            self.logger.log(f"[MON] Saved {n_mon} monitor definition(s).")

    def create_stack_pages(self):
        """Create pages for the stacked layout."""
        self.stack_layout.addWidget(self.create_geometry_page())
        self.stack_layout.addWidget(self.create_optimisation_page())
        self.stack_layout.addWidget(self.create_solver_page())

    def create_geometry_page(self):
        """Create the Geometry page."""
        widget = QWidget()
        layout = QVBoxLayout()

        label = QLabel("Set number of bodies in geometry")
        spin_box = QSpinBox()
        spin_box.setRange(1, 2)

        layout.setAlignment(Qt.AlignTop)
        layout.addWidget(label)
        layout.addSpacing(5)
        layout.addWidget(spin_box)
        widget.setLayout(layout)

        return widget

    def create_optimisation_page(self):
        """Create the optimisation page with a secondary stacked layout."""
        widget = QWidget()
        layout = QVBoxLayout()

        dropdown = QComboBox()
        dropdown.addItem("Select an Optimisation Method")
        dropdown.setItemData(0, 0, Qt.UserRole - 1)
        dropdown.addItem("Bayesian Optimisation")
        dropdown.addItem("EA Optimisation")
        
        self.optimisation_stack = QStackedLayout()
        self.optimisation_stack.addWidget(self.create_bayesian_page())
        self.optimisation_stack.addWidget(self.create_ea_page())

        dropdown.currentIndexChanged.connect(
            lambda index: self.optimisation_stack.setCurrentIndex(index - 1)
            if index > 0 else None
        )

        layout.addLayout(self.optimisation_stack)
        widget.setLayout(layout)

        return widget

    def create_bayesian_page(self):
        """Create the Bayesian Optimisation page (bounds populated later)."""
        widget = QWidget()
        self.bo_layout = QVBoxLayout()
        self.bo_layout.setAlignment(Qt.AlignTop)

        title = QLabel("Bayesian Optimisation Parameters")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        self.bo_layout.addWidget(title)

        self.bo_form = QFormLayout()

        # Number of objectives
        self.bo_obj_spin = QSpinBox()
        self.bo_obj_spin.setRange(1, 10)
        self.bo_obj_spin.setValue(1)
        self.bo_form.addRow("Number of objectives:", self.bo_obj_spin)

        # Bounds summary + editor button (dialog-based)
        self.bound_fields = []
        self.bound_meta = []
        self.current_bounds = {"lb": [], "ub": []}

        self.bounds_summary_label = QLabel("Bounds not generated yet. Save control nodes first.")
        self.bo_form.addRow("Bounds:", self.bounds_summary_label)

        self.edit_bounds_btn = QPushButton("Edit Bounds")
        self.edit_bounds_btn.setEnabled(False)
        self.edit_bounds_btn.clicked.connect(self.open_bounds_dialog)
        self.bo_form.addRow("", self.edit_bounds_btn)

        # Count limit
        self.bo_count_spin = QSpinBox()
        self.bo_count_spin.setRange(1, 10000)
        self.bo_count_spin.setValue(50)
        self.bo_form.addRow("Count limit (generations):", self.bo_count_spin)

        # Initial samples
        self.bo_samples_spin = QSpinBox()
        self.bo_samples_spin.setRange(1, 1000)
        self.bo_samples_spin.setValue(5)
        self.bo_form.addRow("Initial samples:", self.bo_samples_spin)

        # Acquisition function
        self.bo_acq_combo = QComboBox()
        self.bo_acq_combo.addItems(["Expected Improvement", "Probability of Improvement", "Upper Confidence Bound"])
        self.bo_form.addRow("Acquisition function:", self.bo_acq_combo)

        # Kernel
        self.bo_kernel_combo = QComboBox()
        self.bo_kernel_combo.addItems([
            "RBFKernel", "Squared Exponential Kernel", "Exponential Kernel",
            "Mat12Kern", "Mat32Kern", "Mat52Kern"
        ])
        self.bo_form.addRow("Kernel:", self.bo_kernel_combo)

        self.bo_layout.addLayout(self.bo_form)

        objfunc_btn = QPushButton("Set Objective & Flow Conditions")
        objfunc_btn.setStyleSheet("padding: 6px; font-weight: bold;")
        objfunc_btn.clicked.connect(self.open_objective_editor)
        self.bo_layout.addWidget(objfunc_btn)

        save_btn = QPushButton("💾 Save Options")
        save_btn.setStyleSheet("padding: 6px; font-weight: bold;")
        save_btn.clicked.connect(self.save_bayesian_settings)
        self.bo_layout.addWidget(save_btn)

        widget.setLayout(self.bo_layout)
        
        self.objective_config = {
            "objective_type": "Drag",
            "objective_mode": "expression",
            "expression": "CD",
            "terms": [
                {
                    "metric": "CD",
                    "source": "rsd",
                    "surface_ids": [],
                    "reduction": "last",
                    "symbol": "CD",
                    "weight": 1.0,
                    "direction": "x",
                    "symmetry_factor": 2,
                }
            ],
            "conditions": [
                {
                    "AoA": 3.0,
                    "Mach": 1.3,
                    "Re": 6.9e6,
                    "TurbModel": 1,
                    "EngineFlow": 2,
                    "MassFlow": 0.189,
                    "Weight": 1.0,
                }
            ],
            "constraints": []
        }    
        return widget
    
    def open_objective_editor(self):
        dlg = ObjectiveEditor(self)

        if dlg.exec_() == QDialog.Accepted:
            self.objective_config = dlg.get_config()
            self._sync_objective_terms_to_monitor_config()

            self.logger.log(
                f"[OPT] Objective saved: expression = {self.objective_config.get('expression', '')}"
            )
            self.logger.log(
                f"[OPT] Objective terms = {len(self.objective_config.get('terms', []))}"
            )

    def _get_bo_dimension_info(self):
        """
        Returns a list of bound descriptors in optimisation-vector order.
        Each item is a dict: {"name": ..., "lb": ..., "ub": ...}
        """
        mv = getattr(self, "mesh_viewer", None)
        if mv is None:
            return []

        info = []

        parameterisation_method = str(
            getattr(mv, "parameterisation_method", "modal")
        ).strip().lower()

        # --------------------------------------------------
        # DIRECT CONTROL-NODE PARAMETERISATION
        # --------------------------------------------------
        if parameterisation_method == "direct":
            subtype = str(
                getattr(mv, "direct_parameterisation_subtype", "xyz") or "xyz"
            ).strip().lower()

            cn = getattr(mv, "control_nodes", None)
            n_cn = 0 if cn is None else len(cn)

            if subtype == "xyz":
                for i in range(n_cn):
                    for comp in ["x", "y", "z"]:
                        info.append({
                            "name": f"CN {i+1} {comp}",
                            "lb": -1.0,
                            "ub":  1.0
                        })
            elif subtype == "normal":
                for i in range(n_cn):
                    info.append({
                        "name": f"CN {i+1} normal",
                        "lb": -1.0,
                        "ub":  1.0
                    })
            else:
                self.logger.log(f"[Bayes][WARN] Unknown direct subtype '{subtype}'.")
                return []

            return info

        # --------------------------------------------------
        # PCA-REDUCED MODAL PARAMETERISATION
        # --------------------------------------------------
        use_pca = bool(getattr(mv, "use_pca", False))
        if use_pca:
            k_red = getattr(mv, "pca_k_red", None)
            k_final = getattr(mv, "pca_k_final", None)

            n_dim = k_final or k_red or getattr(mv, "k_modes", 0)
            for i in range(int(n_dim)):
                info.append({
                    "name": f"PCA Mode {i+1}",
                    "lb": -2.0,
                    "ub":  2.0
                })
            return info

        # --------------------------------------------------
        # STANDARD MODAL PARAMETERISATION
        # --------------------------------------------------
        k = int(getattr(mv, "k_modes", 0))
        normal_project = bool(getattr(mv, "normal_project", True))
        vector_mode = str(getattr(mv, "vector_mode", "local_frame") or "local_frame")
        global_modes = bool(getattr(mv, "global_modes_selected", False))
        global_cfg = getattr(mv, "global_mode_config", []) or []
        use_local_modes = bool(getattr(mv, "use_local_modes", True))
        global_only = bool(getattr(mv, "global_only", False))

        if global_only:
            use_local_modes = False

        # globals first, matching backend ordering
        if global_modes:
            if global_cfg:
                for i, g in enumerate(global_cfg, start=1):
                    gtype = g.get("type", "global")
                    gdir = g.get("direction", "")
                    label = f"Global {i}: {gtype}_{gdir}".rstrip("_")
                    info.append({
                        "name": label,
                        "lb": -0.5,
                        "ub":  0.5
                    })
            else:
                # fallback if config missing but global_modes is on
                for i in range(8):
                    info.append({
                        "name": f"Global {i+1}",
                        "lb": -0.5,
                        "ub":  0.5
                    })

        # locals only if enabled
        if use_local_modes:
            if normal_project:
                for i in range(k):
                    info.append({
                        "name": f"Local Mode {i+1}",
                        "lb": -1.0,
                        "ub":  1.0
                    })
            else:
                if vector_mode == "xyz":
                    for comp in ["x", "y", "z"]:
                        for i in range(k):
                            info.append({
                                "name": f"{comp} Mode {i+1}",
                                "lb": -1.0,
                                "ub":  1.0
                            })
                else:
                    # local_frame mode
                    for comp in ["t1", "t2", "n"]:
                        for i in range(k):
                            info.append({
                                "name": f"{comp} Mode {i+1}",
                                "lb": -1.0,
                                "ub":  1.0
                            })

        return info

    def populate_bayes_bounds(self):
        """
        Generate internal bounds storage once control nodes / basis are saved.
        This no longer clutters the main layout.
        """
        meta = self._get_bo_dimension_info()
        self.bound_meta = meta

        if not meta:
            self.current_bounds = {"lb": [], "ub": []}
            self.bounds_summary_label.setText("Bounds not generated yet. Save control nodes first.")
            self.edit_bounds_btn.setEnabled(False)
            self.logger.log("[Bayes][WARN] No optimisation dimensions found for current parameterisation.")
            return

        lb = [item["lb"] for item in meta]
        ub = [item["ub"] for item in meta]
        self.current_bounds = {"lb": lb, "ub": ub}

        self.bounds_summary_label.setText(
            f"{len(meta)} optimisation variables configured."
        )
        self.edit_bounds_btn.setEnabled(True)

        self.logger.log(f"[Bayes] Bounds regenerated for {len(meta)} optimisation variables.")

    def open_bounds_dialog(self):
        if not getattr(self, "bound_meta", None):
            QMessageBox.warning(self, "Bounds", "No bounds available yet. Save control nodes first.")
            return

        dlg = BoundsDialog(
            meta=self.bound_meta,
            lb=list(self.current_bounds["lb"]),
            ub=list(self.current_bounds["ub"]),
            parent=self
        )
        if dlg.exec_() == QDialog.Accepted:
            lb, ub = dlg.get_bounds()
            self.current_bounds = {"lb": lb, "ub": ub}
            self.bounds_summary_label.setText(f"{len(lb)} bounds configured.")
            self.logger.log(f"[Bayes] Updated bounds for {len(lb)} dimensions.")

    def save_bayesian_settings(self):
        """Collect Bayesian Optimisation settings and save to optimiser + JSON."""
        import json, os
        import numpy as np
        from Optimisation.BayesianOptimisation.optimiser import BayesianOptimiser
        from Optimisation.BayesianOptimisation.acquisition_functions import EI, POI, UCB
        from Optimisation.BayesianOptimisation.kernels import (
            RBFKernel, SquaredExponentialKernel, ExponentialKernel,
            Mat12Kern, Mat32Kern, Mat52Kern
        )

        # Map UI strings → actual functions/classes
        acq_map = {
            "Expected Improvement": EI,
            "Probability of Improvement": POI,
            "Upper Confidence Bound": UCB
        }
        kern_map = {
            "RBFKernel": RBFKernel,
            "Squared Exponential Kernel": SquaredExponentialKernel,
            "Exponential Kernel": ExponentialKernel,
            "Mat12Kern": Mat12Kern,
            "Mat32Kern": Mat32Kern,
            "Mat52Kern": Mat52Kern
        }

        # Parse bounds
        lb = list(getattr(self, "current_bounds", {}).get("lb", []))
        ub = list(getattr(self, "current_bounds", {}).get("ub", []))

        if not lb or not ub or len(lb) != len(ub):
            QMessageBox.warning(self, "Error", "Bounds are not defined. Please open 'Edit Bounds' first.")
            return

        for i, (lbi, ubi) in enumerate(zip(lb, ub), start=1):
            try:
                lbi = float(lbi)
                ubi = float(ubi)
            except Exception:
                QMessageBox.warning(self, "Error", f"Invalid bound for variable {i}.")
                return
            if lbi >= ubi:
                QMessageBox.warning(self, "Error", f"Lower bound must be smaller than upper bound for variable {i}.")
                return

        n_dim = len(lb)
        
        acq_func = acq_map[self.bo_acq_combo.currentText()]
        kern_cls = kern_map[self.bo_kernel_combo.currentText()]

        settings = {
            "n_dim": n_dim,
            "n_obj": self.bo_obj_spin.value(),
            "lb": lb,
            "ub": ub,
            "count_limit": self.bo_count_spin.value(),
            "n_samples": self.bo_samples_spin.value(),
            "acquisition_function": acq_func,
            "kernel": kern_cls,
            "sim_dir": getattr(self, "output_directory", os.getcwd())
        }
        self.bayes_settings = settings

        # Save JSON (store class/function names instead of raw objects)
        if hasattr(self, "output_directory") and self.output_directory:
            self.base = self.get_project_basename()
            out_path = os.path.join(self.output_directory, f"{self.base}_bo.json")

            json_settings = {
                **settings,
                "acquisition_function": self.bo_acq_combo.currentText(),
                "kernel": self.bo_kernel_combo.currentText()
            }
            with open(out_path, "w") as f:
                json.dump(json_settings, f, indent=2)
            self.logger.log(f"[Bayes] Settings saved to {out_path}")

        # Initialise optimiser
        #self.bayesian_optimiser = BayesianOptimiser(settings, eval_func=None)
        #self.logger.log("[Bayes] BayesianOptimiser initialised with current settings.")
        
        self.optimisation_settings_saved = True
        self.check_run_opt_button_state()

    def create_ea_page(self):
        """Create the EA Optimisation page."""
        widget = QWidget()
        layout = QVBoxLayout()

        layout.setAlignment(Qt.AlignTop)
        layout.addWidget(QLabel("EA Optimisation Parameters"))
        layout.addSpacing(20)
        layout.addWidget(QLabel("Population Size:"))
        layout.addSpacing(5)
        layout.addWidget(QSpinBox())

        widget.setLayout(layout)
        return widget
    
    def get_previous_solution_config_for_cluster(self) -> dict:
        cfg = dict(getattr(self, "previous_solution_config", {}) or {})
        cfg["enabled"] = bool(getattr(self, "restart_from_previous", False)) and bool(cfg.get("directory")) and bool(cfg.get("base"))
        cfg.setdefault("boundary_mode", "same_id")
        cfg.setdefault("num_comp", 7)
        return cfg
    
    def on_previous_solution_toggled(self, checked: bool):
        if not checked:
            self.restart_from_previous = False
            self.previous_solution_config["enabled"] = False
            if hasattr(self, "prev_solution_summary"):
                self.prev_solution_summary.setText("No previous solution selected")
            self.logger.log("[Solver] Previous-solution initialisation disabled.")
            return

        dlg = PreviousSolutionDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            self.prev_solution_cb.blockSignals(True)
            self.prev_solution_cb.setChecked(False)
            self.prev_solution_cb.blockSignals(False)
            self.restart_from_previous = False
            self.previous_solution_config["enabled"] = False
            return

        cfg = dlg.values()
        self.previous_solution_config = cfg
        self.restart_from_previous = True

        msg = f"{cfg['location']}: {cfg['directory']} | base={cfg['base']}"
        self.prev_solution_summary.setText(msg)
        self.logger.log(f"[Solver] Initialise from previous solution: {msg}")

    def create_solver_page(self):
        """Create the Solver page with a text editor for solver input files."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title = QLabel("Solver Input Editor")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(title)
        
        rung_grp = QGroupBox("Rungen (Preprocessor)")
        rung_form = QFormLayout(rung_grp)

        self.rg_project = QLabel(self.get_project_basename())

        self.rg_invvis = QComboBox()
        self.rg_invvis.addItems(["1 (Inviscid)", "2 (Viscous)"])
        self.rg_invvis.setCurrentIndex(1)  # default viscous

        self.rg_num_grids = QSpinBox(); self.rg_num_grids.setRange(1, 200); self.rg_num_grids.setValue(1)

        self.rg_hybrid = QCheckBox(); self.rg_hybrid.setChecked(True)

        self.rg_parallel = QSpinBox(); self.rg_parallel.setRange(1, 10000); self.rg_parallel.setValue(80)

        self.rg_roll = QCheckBox(); self.rg_roll.setChecked(False)

        self.rg_ground_angle = QSpinBox(); self.rg_ground_angle.setRange(0, 90); self.rg_ground_angle.setValue(45)
        self.rg_ground_angle.setEnabled(False)
        self.rg_roll.stateChanged.connect(lambda s: self.rg_ground_angle.setEnabled(self.rg_roll.isChecked()))

        self.rg_start_step = QSpinBox(); self.rg_start_step.setRange(0, 10_000); self.rg_start_step.setValue(1)
        self.rg_steps_per = QSpinBox(); self.rg_steps_per.setRange(1, 10_000); self.rg_steps_per.setValue(1)
        
        self.cad_units_combo = QComboBox()
        self.cad_units_combo.addItems(["mm", "cm", "m", "in", "ft"])
        self.cad_units_combo.setCurrentText("m")  # default
        self.cad_units = "m"
        self.cad_units_combo.currentTextChanged.connect(lambda u: setattr(self, "cad_units", u))
        rung_form.addRow("CAD units:", self.cad_units_combo)

        rung_form.addRow("Project name (auto):", self.rg_project)
        rung_form.addRow("Flow: Inviscid/Viscous:", self.rg_invvis)
        rung_form.addRow("Number of grids:", self.rg_num_grids)
        rung_form.addRow("Hybrid mesh:", self.rg_hybrid)
        rung_form.addRow("Parallel domains:", self.rg_parallel)
        rung_form.addRow("Rolling ground:", self.rg_roll)
        rung_form.addRow("Ground angle (deg):", self.rg_ground_angle)
        rung_form.addRow("Starting step in cycle:", self.rg_start_step)
        rung_form.addRow("Steps per cycle:", self.rg_steps_per)

        # load defaults from template if available
        try:
            # prefer templates/rungen.inp if you’ve put it there
            tmpl_guess = getattr(self, "solver_file", None)
            if not tmpl_guess:
                tmpl_guess = os.path.join(os.getcwd(), "templates", "rungen.inp")
            rg = RungenInpFile.from_template(tmpl_guess, self.get_project_basename())
            # reflect into form
            self.rg_invvis.setCurrentIndex(0 if rg.invvis == 1 else 1)
            self.rg_num_grids.setValue(rg.number_of_grids)
            self.rg_hybrid.setChecked(bool(rg.hybrid))
            self.rg_parallel.setValue(rg.parallel_domains)
            self.rg_roll.setChecked(bool(rg.roll_ground))
            self.rg_ground_angle.setValue(rg.ground_angle)
            self.rg_start_step.setValue(rg.starting_step_in_cycle)
            self.rg_steps_per.setValue(rg.steps_per_cycle)
            self.rg_ground_angle.setEnabled(self.rg_roll.isChecked())
        except Exception:
            pass

        # Buttons for Rungen
        rg_btns = QHBoxLayout()
        save_rungen_btn = QPushButton("💾 Save Rungen to Project")
        save_rungen_btn.clicked.connect(self.save_rungen_settings)
        rg_btns.addWidget(save_rungen_btn)
        rg_btns.addStretch(1)

        layout.addWidget(rung_grp)
        layout.addLayout(rg_btns)

        self.prev_solution_cb = QCheckBox("Initialise from previous solution")
        self.prev_solution_cb.setChecked(False)
        self.prev_solution_cb.toggled.connect(self.on_previous_solution_toggled)
        layout.addWidget(self.prev_solution_cb)

        self.prev_solution_summary = QLabel("No previous solution selected")
        self.prev_solution_summary.setStyleSheet("color: #555;")
        layout.addWidget(self.prev_solution_summary)

        # Text editor
        self.solver_editor = QPlainTextEdit()
        self.solver_editor.setPlaceholderText("# Solver settings go here…")
        layout.addWidget(self.solver_editor)

        # Load + Save buttons
        btn_row = QHBoxLayout()

        load_btn = QPushButton("Load Solver File…")
        load_btn.clicked.connect(self.load_solver_file)
        btn_row.addWidget(load_btn)

        save_btn = QPushButton("Save Solver Settings")
        save_btn.clicked.connect(self.save_solver_settings)
        btn_row.addWidget(save_btn)

        layout.addLayout(btn_row)
        
        cond_row = QHBoxLayout()
        self.parallel_btn = QPushButton("Parallel Flow Conditions…")
        self.parallel_btn.setToolTip("Define AoA/Mach/Re/etc. rows to run in parallel at solver step")
        self.parallel_btn.clicked.connect(self.open_sim_editor)
        cond_row.addWidget(self.parallel_btn)

        self.parallel_summary = QLabel("No parallel conditions set")
        self.parallel_summary.setStyleSheet("color: #555;")
        cond_row.addWidget(self.parallel_summary, 1)

        layout.addLayout(cond_row)

        return widget
    
    def save_rungen_settings(self):
        if not hasattr(self, "output_directory") or not self.output_directory:
            QMessageBox.warning(self, "Error", "Please set an output directory first.")
            return

        project = self.get_project_basename()
        # read UI
        invvis = 1 if self.rg_invvis.currentIndex() == 0 else 2
        num_grids = int(self.rg_num_grids.value())
        hybrid = bool(self.rg_hybrid.isChecked())
        self.parallel = int(self.rg_parallel.value())
        roll = bool(self.rg_roll.isChecked())
        ground_angle = int(self.rg_ground_angle.value())
        start_step = int(self.rg_start_step.value())
        steps_per = int(self.rg_steps_per.value())

        # create model
        rg = RungenInpFile(
            project=project,
            invvis=invvis,
            number_of_grids=num_grids,
            hybrid=hybrid,
            parallel_domains=self.parallel,
            roll_ground=roll,
            ground_angle=ground_angle,
            starting_step_in_cycle=start_step,
            steps_per_cycle=steps_per
        )

        # write to project root and preprocessed/n_<n>
        local_root = os.path.join(self.output_directory, "rungen.inp")
        pre_dir = os.path.join(self.output_directory, "preprocessed", f"n_{0}")  # n=0 for now; keep in sync with pipeline
        local_pre = os.path.join(pre_dir, "rungen.inp")

        try:
            rg.write(local_root)
            rg.write(local_pre)
            self.logger.log(f"[Solver] Wrote rungen.inp → {local_root}")
            self.logger.log(f"[Solver] Wrote rungen.inp → {local_pre}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not save rungen.inp:\n{e}")
            return

        # store for pipeline use
        self.rungen_local_path = local_pre
        # Mark solver settings saved so Run Simulation can be enabled
        self.prepro_settings_saved = True
        self.check_run_sim_button_state()

    def load_solver_file(self):
        start_dir = os.path.join(os.getcwd(), "templates")
        path, _ = QFileDialog.getOpenFileName(self, "Open Solver File", start_dir, "Text files (*.inp *.txt *.cfg *.dat);;All files (*)")
        if path:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    self.solver_editor.setPlainText(f.read())
                self.solver_file = path
                self.logger.log(f"[Solver] Loaded solver file: {path}")
            except Exception as e:
                QMessageBox.warning(self, "Open failed", f"Could not read:\n{path}\n\n{e}")

    def save_solver_settings(self):
        if not hasattr(self, "output_directory") or not self.output_directory:
            QMessageBox.warning(self, "Error", "Please set an output directory first.")
            return

        self.base = self.get_project_basename()
        save_path = os.path.join(self.output_directory, f"{self.base}.inp")
        folder = os.path.dirname(save_path)
        
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(self.solver_editor.toPlainText())
            self.logger.log(f"[Solver] Saved solver settings to {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not save solver settings:\n{e}")
            return

        self.solver_template_dir = folder
        self.solver_input_path = save_path
        self.logger.log(f"[Solver] Using solver template dir: {self.solver_template_dir}")

        # Mark solver settings saved → check run_sim_btn
        self.solver_settings_saved = True
        self.check_run_sim_button_state()
        
    def open_sim_editor(self):
        dlg = ParallelSimEditor(self)
        if dlg.exec_() == QDialog.Accepted:
            self.sim_config = dlg.get_config()
            n = len(self.sim_config.get("conditions", []))
            self.parallel_summary.setText(f"{n} condition(s) set for parallel solve")
            # Treat this as solver settings saved (so Run Simulation enables cleanly)
            self.solver_settings_saved = True
            self.check_run_sim_button_state()
            self.logger.log(f"[SIM] Parallel conditions saved ({n} rows).")

    def create_display_section(self):
        # Main display layout
        disp_layout = QVBoxLayout()

        # Secondary stacked layout
        self.display_stack = QStackedLayout()

        self.display_stack.addWidget(self.create_mesh_page())
        self.display_stack.addWidget(self.create_solv_page())
        self.display_stack.addWidget(self.create_post_page())
        
        self.geo_viewer = GeometryPanel(parent=self)
        geo_container = QWidget()
        geo_layout = QVBoxLayout(geo_container)
        geo_layout.addWidget(self.geo_viewer)
        
        self.display_stack.addWidget(geo_container)


        self.IDX_GEO  = 3
        self.IDX_MESH = 0
        self.IDX_SOLV = 1
        self.IDX_POST = 2

        # Show Mesh by default on startup (even though Geometry is index 0)
        self.display_stack.setCurrentIndex(self.IDX_MESH)

        # Button wiring
        btn_layout = QHBoxLayout()
        geo_btn = QPushButton("Geometry")
        geo_btn.pressed.connect(lambda: self.display_stack.setCurrentIndex(3))  # <- was 0

        mesh_btn = QPushButton("Mesh")
        mesh_btn.pressed.connect(lambda: self.display_stack.setCurrentIndex(0))

        solv_btn = QPushButton("Solver")
        solv_btn.pressed.connect(lambda: self.display_stack.setCurrentIndex(1))

        post_btn = QPushButton("Post-Processing")
        post_btn.pressed.connect(lambda: self.display_stack.setCurrentIndex(2))

        btn_layout.addWidget(geo_btn)
        btn_layout.addWidget(mesh_btn)
        btn_layout.addWidget(solv_btn)
        btn_layout.addWidget(post_btn)

        disp_layout.addLayout(btn_layout)
        disp_layout.addLayout(self.display_stack)

        self.display_layout.addLayout(disp_layout)

    def create_mesh_page(self):
        self.mesh_viewer = MeshViewer(parent=self)
        if getattr(self, "pipeline", None) is not None:
            self.mesh_viewer.set_pipeline(self.pipeline)
        self.mesh_viewer.control_ready.connect(self.enable_run_morph_btn)
        return self.mesh_viewer
    
    def enable_run_morph_btn(self):
        self.control_nodes_saved = True
        self.logger.log("[INFO] Control nodes and bounds defined.")
        self.check_run_morph_button_state()
    
        self._update_train_classifier_button_state()

        # Regenerate optimisation bounds once control nodes are saved
        if hasattr(self, "populate_bayes_bounds"):
            self.populate_bayes_bounds()
        
    def create_solv_page(self):
        return SolverViewer(self)

    def create_post_page(self):
        return PostViewer(self)

    def activate_tab_1(self):
        self.stack_layout.setCurrentIndex(0)

    def activate_tab_2(self):
        self.stack_layout.setCurrentIndex(1)

    def activate_tab_3(self):
        self.stack_layout.setCurrentIndex(2)
        
    def closeEvent(self, event):
        """Ensure sub-windows and plotters are cleaned up when closing the app."""
        try:
            if hasattr(self, "geom_window") and self.geom_window is not None:
                self.geom_window.close()
                self.geom_window = None
                
            try:
                mv = getattr(self, "mesh_viewer", None)
                if mv is not None and hasattr(mv, "modal_explorer_window"):
                    win = mv.modal_explorer_window
                    if win is not None:
                        win.close()
                    mv.modal_explorer_window = None
            except Exception as e:
                print(f"[DEBUG] Error closing modal explorer: {e}")

            if hasattr(self, "mesh_viewer") and self.mesh_viewer is not None:
                plotter = self.mesh_viewer.plotter
                if plotter is not None:
                    try:
                        plotter.disable_picking()
                    except Exception:
                        pass
                    try:
                        rw = plotter.ren_win
                        if rw is not None:
                            rw.Finalize()      # ✅ release OpenGL context
                    except Exception:
                        pass
                    try:
                        iren = plotter.interactor
                        if iren is not None:
                            iren.TerminateApp()  # ✅ stop interactor loop
                    except Exception:
                        pass
                    try:
                        plotter.close()
                    except Exception:
                        pass
                    self.mesh_viewer.plotter = None
        except Exception as e:
            print(f"[DEBUG] Error during MainWindow close: {e}")
        event.accept()


class PreviousSolutionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main = parent
        self.setWindowTitle("Initialise from Previous Solution")
        self.resize(760, 460)

        layout = QVBoxLayout(self)

        mode_row = QHBoxLayout()
        self.local_rb = QRadioButton("Local")
        self.remote_rb = QRadioButton("Remote")
        self.remote_rb.setChecked(getattr(parent, "run_mode", "") == "HPC")
        self.local_rb.setChecked(not self.remote_rb.isChecked())
        mode_row.addWidget(QLabel("Solution location:"))
        mode_row.addWidget(self.local_rb)
        mode_row.addWidget(self.remote_rb)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select or type previous solution directory...")
        self.browse_btn = QPushButton("Browse")
        self.up_btn = QPushButton("Up")
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.browse_btn)
        path_row.addWidget(self.up_btn)
        layout.addLayout(path_row)

        self.dir_list = QListWidget()
        layout.addWidget(self.dir_list, 1)

        base_row = QHBoxLayout()
        self.base_edit = QLineEdit()
        self.base_edit.setPlaceholderText("Previous base name, e.g. corner_1")
        self.detect_btn = QPushButton("Auto-detect base")
        base_row.addWidget(QLabel("Previous base:"))
        base_row.addWidget(self.base_edit, 1)
        base_row.addWidget(self.detect_btn)
        layout.addLayout(base_row)

        opts_row = QHBoxLayout()
        self.num_comp_spin = QSpinBox()
        self.num_comp_spin.setRange(1, 20)
        self.num_comp_spin.setValue(7)

        self.boundary_combo = QComboBox()
        self.boundary_combo.addItems(["same_id", "nearest", "none"])

        opts_row.addWidget(QLabel("Components:"))
        opts_row.addWidget(self.num_comp_spin)
        opts_row.addWidget(QLabel("Boundary mode:"))
        opts_row.addWidget(self.boundary_combo)
        opts_row.addStretch()
        layout.addLayout(opts_row)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #555;")
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.validate_btn = QPushButton("Validate")
        buttons.addButton(self.validate_btn, QDialogButtonBox.ActionRole)
        layout.addWidget(buttons)

        self.browse_btn.clicked.connect(self.browse)
        self.up_btn.clicked.connect(self.go_up)
        self.detect_btn.clicked.connect(self.auto_detect_base)
        self.validate_btn.clicked.connect(self.validate_selection)
        self.dir_list.itemDoubleClicked.connect(self.enter_selected_dir)
        self.local_rb.toggled.connect(self.refresh_mode)

        buttons.accepted.connect(self.accept_if_valid)
        buttons.rejected.connect(self.reject)

        cfg = getattr(parent, "previous_solution_config", {}) or {}
        self.path_edit.setText(cfg.get("directory", ""))
        self.base_edit.setText(cfg.get("base", ""))
        self.num_comp_spin.setValue(int(cfg.get("num_comp", 7)))
        self.boundary_combo.setCurrentText(cfg.get("boundary_mode", "same_id"))

        if self.path_edit.text().strip():
            self.refresh_listing()

    def is_remote(self):
        return self.remote_rb.isChecked()

    def refresh_mode(self):
        self.dir_list.clear()
        self.status_label.setText("")

    def browse(self):
        if self.is_remote():
            start = self.path_edit.text().strip()
            if not start:
                start = f"/scratch/{self.main.ssh_creds['username']}/aeropt/aeropt_out"
            self.path_edit.setText(start)
            self.refresh_listing()
        else:
            start = self.path_edit.text().strip() or getattr(self.main, "output_directory", os.getcwd())
            d = QFileDialog.getExistingDirectory(self, "Select Previous Solution Directory", start)
            if d:
                self.path_edit.setText(d)
                self.refresh_listing()
                self.auto_detect_base()

    def _remote_listdir(self, path):
        sftp = self.main.ssh_client.open_sftp()
        try:
            entries = sftp.listdir_attr(path)
        finally:
            sftp.close()

        dirs = []
        files = []
        import stat
        for e in entries:
            if stat.S_ISDIR(e.st_mode):
                dirs.append(e.filename)
            else:
                files.append(e.filename)
        return sorted(dirs), sorted(files)

    def refresh_listing(self):
        path = self.path_edit.text().strip()
        self.dir_list.clear()
        if not path:
            return

        try:
            if self.is_remote():
                dirs, files = self._remote_listdir(path)
            else:
                dirs = sorted([x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))])
                files = sorted([x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))])

            for d in dirs:
                self.dir_list.addItem(f"📁 {d}")
            for f in files:
                if f.endswith((".unk", ".res", ".plt", ".rsd", ".rst", ".inp", ".reg")):
                    self.dir_list.addItem(f"   {f}")

            self.status_label.setText(f"Listed: {path}")
        except Exception as e:
            self.status_label.setStyleSheet("color: #a33;")
            self.status_label.setText(f"Could not list directory: {e}")

    def enter_selected_dir(self, item):
        txt = item.text().strip()
        if not txt.startswith("📁"):
            return

        name = txt.replace("📁", "", 1).strip()
        cur = self.path_edit.text().strip()

        if self.is_remote():
            import posixpath
            new_path = posixpath.normpath(posixpath.join(cur, name))
        else:
            new_path = os.path.normpath(os.path.join(cur, name))

        self.path_edit.setText(new_path)
        self.refresh_listing()
        self.auto_detect_base()

    def go_up(self):
        cur = self.path_edit.text().strip()
        if not cur:
            return
        if self.is_remote():
            import posixpath
            self.path_edit.setText(posixpath.dirname(cur.rstrip("/")))
        else:
            self.path_edit.setText(os.path.dirname(cur.rstrip("/\\")))
        self.refresh_listing()

    def _list_files(self, path):
        if self.is_remote():
            _dirs, files = self._remote_listdir(path)
            return files
        return os.listdir(path)

    def auto_detect_base(self):
        path = self.path_edit.text().strip()
        if not path:
            return

        try:
            files = self._list_files(path)
            candidates = []
            for f in files:
                root, ext = os.path.splitext(f)
                if ext in [".unk", ".res"]:
                    candidates.append(root)

            # Prefer .unk base if available
            unk_bases = [os.path.splitext(f)[0] for f in files if f.endswith(".unk")]
            if unk_bases:
                self.base_edit.setText(sorted(unk_bases)[0])
            elif candidates:
                self.base_edit.setText(sorted(candidates)[0])

            if self.base_edit.text():
                self.status_label.setStyleSheet("color: #285;")
                self.status_label.setText(f"Detected base: {self.base_edit.text()}")
        except Exception as e:
            self.status_label.setStyleSheet("color: #a33;")
            self.status_label.setText(f"Auto-detect failed: {e}")

    def validate_selection(self):
        path = self.path_edit.text().strip()
        base = self.base_edit.text().strip()

        if not path or not base:
            self.status_label.setStyleSheet("color: #a33;")
            self.status_label.setText("Please select a directory and provide a base name.")
            return False

        try:
            files = self._list_files(path)
            has_unk = f"{base}.unk" in files
            has_res = f"{base}.res" in files
            has_plotreg = "plotreg.reg" in files

            if not has_unk and not has_res:
                self.status_label.setStyleSheet("color: #a33;")
                self.status_label.setText(f"Missing {base}.unk or {base}.res.")
                return False

            if has_res and not has_plotreg:
                self.status_label.setStyleSheet("color: #a33;")
                self.status_label.setText("Found .res but missing plotreg.reg, so makeplot2 cannot create .unk.")
                return False

            self.status_label.setStyleSheet("color: #285;")
            self.status_label.setText("Previous solution looks valid.")
            return True

        except Exception as e:
            self.status_label.setStyleSheet("color: #a33;")
            self.status_label.setText(f"Validation failed: {e}")
            return False

    def accept_if_valid(self):
        if self.validate_selection():
            self.accept()

    def values(self):
        return {
            "enabled": True,
            "location": "remote" if self.is_remote() else "local",
            "directory": self.path_edit.text().strip(),
            "base": self.base_edit.text().strip(),
            "boundary_mode": self.boundary_combo.currentText(),
            "num_comp": int(self.num_comp_spin.value()),
        }

# Objective Editor (GUI)
class ParallelSimEditor(QDialog):
    """
    Lets user define:
      - Flow-condition rows with: AoA, Mach, Re, TurbModel(1|2|3), EngineFlow(1|2), MassFlow, Weight
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define Parallel Flow Conditions")
        self.resize(900, 520)

        layout = QVBoxLayout(self)

        # --- Table of flow conditions ---
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Altitude (ft)", "AoA", "Mach", "Reynolds", "Turb Model (1|2|3)",
            "Engine Flow (1|2)", "Mass Flow"
        ])
        layout.addWidget(self.table)

        # Add/remove condition buttons
        row_btns = QHBoxLayout()
        add_btn = QPushButton("+ Add Condition")
        del_btn = QPushButton("– Remove Selected")
        add_btn.clicked.connect(self._add_row)
        del_btn.clicked.connect(self._remove_rows)
        row_btns.addWidget(add_btn)
        row_btns.addWidget(del_btn)
        row_btns.addStretch(1)
        layout.addLayout(row_btns)

        # Pre-fill one sensible default row
        self._add_row(defaults=["36000", "3", "1.3", "1e6", "1", "2", "1.0"])

        # OK/Cancel
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _add_row(self, defaults=None):
        row = self.table.rowCount()
        self.table.insertRow(row)
        defaults = defaults or ["36000", "3", "1.2", "1e6", "1", "2", "1.0"]
        for c, val in enumerate(defaults):
            self.table.setItem(row, c, QTableWidgetItem(val))

    def _remove_rows(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def get_config(self):

        # Conditions → list of dicts
        conds = []
        for r in range(self.table.rowCount()):
            def _txt(c):
                it = self.table.item(r, c)
                return it.text().strip() if it else ""
            try:
                conds.append({
                    "Altitude":   float(_txt(0)),
                    "AoA":        float(_txt(1)),
                    "Mach":       float(_txt(2))  if _txt(2)  else 1.2,
                    "Re":         float(_txt(3))  if _txt(3)  else 1e6,
                    "TurbModel":  int(_txt(4))    if _txt(4)  else 1,
                    "EngineFlow": int(_txt(5))    if _txt(5)  else 1,
                    "MassFlow":   float(_txt(6))  if _txt(6)  else 1.0
                })
            except Exception:
                # skip malformed row
                pass

        return {
            "conditions": conds
        }

class ObjectiveEditor(QDialog):
    """
    Objective editor with surface-aware objective terms.

    Design principle:
      - global CL/CD/CM without surfaces -> read from solver .rsd
      - surface-specific CL/CD/CM/pressure recovery/distortion/drag/lift -> require monitor output
      - custom expression is evaluated using generated symbols, e.g.:
          -1.0*PR_s5 + 0.25*DC60_s5 + 0.1*CD_s2_3_4
    """
    METRICS = ["CD", "CL", "CM", "CL/CD", "pressure_recovery", "distortion", "drag", "lift"]
    SOURCES = ["auto", "rsd", "monitor"]
    REDUCTIONS = ["last", "time_average"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define Objective & Flow Conditions")
        self.resize(1150, 760)
        self.parent_window = parent

        layout = QVBoxLayout(self)

        # --- top expression row ---
        obj_row = QHBoxLayout()
        obj_row.addWidget(QLabel("Objective preset:"))
        self.obj_type = QComboBox()
        self.obj_type.addItems(["Drag", "Lift", "Lift-to-Drag", "Custom Expression"])
        self.obj_type.currentTextChanged.connect(self._on_preset_changed)
        obj_row.addWidget(self.obj_type)

        obj_row.addWidget(QLabel("Expression:"))
        self.custom_expr = QLineEdit()
        self.custom_expr.setPlaceholderText("e.g. -1.0*PR_s5 + 0.25*DC60_s5 + 0.1*CD_s2_3_4")
        obj_row.addWidget(self.custom_expr, 1)
        layout.addLayout(obj_row)

        help_lbl = QLabel(
            "Use generated symbols from the table below. BO minimises this scalar, "
            "so maximise pressure recovery by using a negative coefficient."
        )
        help_lbl.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(help_lbl)

        # --- objective terms table ---
        layout.addWidget(QLabel("Objective terms / required monitors"))
        self.term_table = QTableWidget(0, 8)
        self.term_table.setHorizontalHeaderLabels([
            "Metric", "Source", "Surface IDs", "Reduction", "Symbol", "Weight", "Direction", "Symmetry"
        ])
        self.term_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.term_table)

        term_btns = QHBoxLayout()
        add_term = QPushButton("+ Add Term")
        del_term = QPushButton("– Remove Selected Term")
        rebuild_expr = QPushButton("Build Expression From Terms")
        add_term.clicked.connect(lambda: self._add_term_row())
        del_term.clicked.connect(self._remove_term_rows)
        rebuild_expr.clicked.connect(self._build_expression_from_terms)
        term_btns.addWidget(add_term)
        term_btns.addWidget(del_term)
        term_btns.addWidget(rebuild_expr)
        term_btns.addStretch(1)
        layout.addLayout(term_btns)

        # --- available surface IDs helper ---
        surfaces = self._surface_id_strings()
        if surfaces:
            surf_lbl = QLabel("Available surfaces: " + ", ".join(surfaces[:40]) + (" ..." if len(surfaces) > 40 else ""))
            surf_lbl.setStyleSheet("color: #777;")
            layout.addWidget(surf_lbl)

        # --- flow conditions ---
        layout.addWidget(QLabel("Flow conditions"))
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "Altitude (ft)", "AoA", "Mach", "Reynolds", "Turb Model (1|2|3)",
            "Engine Flow (1|2)", "Mass Flow", "Weight"
        ])
        layout.addWidget(self.table)

        row_btns = QHBoxLayout()
        add_btn = QPushButton("+ Add Condition")
        del_btn = QPushButton("– Remove Selected")
        add_btn.clicked.connect(self._add_row)
        del_btn.clicked.connect(self._remove_rows)
        row_btns.addWidget(add_btn)
        row_btns.addWidget(del_btn)
        row_btns.addStretch(1)
        layout.addLayout(row_btns)

        self._add_row(defaults=["36000", "3", "1.3", "6.9e6", "1", "2", "1.0", "1.0"])
        self._add_term_row(metric="CD", source="rsd", surfaces="", reduction="last", symbol="CD", weight="1.0")
        self._on_preset_changed(self.obj_type.currentText())

        layout.addWidget(QLabel("Constraints (optional, one per line; e.g. 'CL >= 0.3', 'CD <= 0.02')"))
        self.constraints_edit = QTextEdit()
        self.constraints_edit.setPlaceholderText("CL >= 0.3\nCD <= 0.02")
        self.constraints_edit.setFixedHeight(80)
        layout.addWidget(self.constraints_edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _combo(self, values, current):
        cb = QComboBox()
        cb.addItems(values)
        if current in values:
            cb.setCurrentText(current)
        return cb

    def _surface_id_strings(self):
        mv = getattr(self.parent_window, "mesh_viewer", None)
        mesh_obj = getattr(mv, "mesh_obj", None)
        if mesh_obj is None:
            return []
        out = []
        for name in mesh_obj.get_surface_names():
            try:
                sid = int(mesh_obj.get_surface_id(name))
                out.append(str(sid))
            except Exception:
                pass
        return out

    def _make_symbol(self, metric, surfaces):
        metric = str(metric or "metric")

        metric_map = {
            "Pressure Recovery": "PR",
            "pressure_recovery": "PR",
            "Distortion": "DC60",
            "distortion": "DC60",
            "Drag": "CD",
            "drag": "CD",
            "Lift": "CL",
            "lift": "CL",
            "Moment": "CM",
            "moment": "CM",
            "CD": "CD",
            "CL": "CL",
            "CM": "CM",
        }

        base = str(metric_map.get(metric, metric)).replace("/", "_over_").replace(" ", "_")

        surfaces = str(surfaces or "").strip()
        if surfaces:
            s = surfaces.replace(",", "_").replace(" ", "")
            return f"{base}_s{s}"

        return base

    def _add_term_row(self, metric="CD", source="rsd", surfaces="", reduction="last", symbol="", weight="1.0", direction="x", symmetry="1"):
        row = self.term_table.rowCount()
        self.term_table.insertRow(row)

        metric_cb = self._combo(self.METRICS, metric)
        source_cb = self._combo(self.SOURCES, source)
        reduction_cb = self._combo(self.REDUCTIONS, reduction)
        dir_cb = self._combo(["x", "y", "z"], direction)
        sym_cb = self._combo(["1", "2", "4"], str(symmetry))

        self.term_table.setCellWidget(row, 0, metric_cb)
        self.term_table.setCellWidget(row, 1, source_cb)
        self.term_table.setItem(row, 2, QTableWidgetItem(str(surfaces)))
        self.term_table.setCellWidget(row, 3, reduction_cb)
        self.term_table.setItem(row, 4, QTableWidgetItem(symbol or self._make_symbol(metric, surfaces)))
        self.term_table.setItem(row, 5, QTableWidgetItem(str(weight)))
        self.term_table.setCellWidget(row, 6, dir_cb)
        self.term_table.setCellWidget(row, 7, sym_cb)

        def refresh_symbol():
            surf_txt = self.term_table.item(row, 2).text() if self.term_table.item(row, 2) else ""
            self.term_table.setItem(row, 4, QTableWidgetItem(self._make_symbol(metric_cb.currentText(), surf_txt)))
            if source_cb.currentText() == "auto":
                source_cb.setCurrentText("monitor" if self._parse_surface_ids(surf_txt) or metric_cb.currentText() not in ("CD", "CL", "CM", "CL/CD") else "rsd")

        metric_cb.currentTextChanged.connect(lambda _t: refresh_symbol())

    def _remove_term_rows(self):
        rows = sorted({i.row() for i in self.term_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.term_table.removeRow(r)

    def _add_row(self, defaults=None):
        row = self.table.rowCount()
        self.table.insertRow(row)
        defaults = defaults or ["36000", "3", "1.3", "6.9e6", "1", "2", "1.0", "1.0"]
        for c, val in enumerate(defaults):
            self.table.setItem(row, c, QTableWidgetItem(val))

    def _remove_rows(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _parse_surface_ids(self, text):
        if not text:
            return []
        out = []
        for tok in str(text).replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(int(tok))
            except ValueError:
                pass
        return out

    def _term_dicts(self):
        terms = []
        for r in range(self.term_table.rowCount()):
            metric = self.term_table.cellWidget(r, 0).currentText()
            source = self.term_table.cellWidget(r, 1).currentText()
            surfaces_txt = self.term_table.item(r, 2).text().strip() if self.term_table.item(r, 2) else ""
            surface_ids = self._parse_surface_ids(surfaces_txt)
            if source == "auto":
                source = "monitor" if surface_ids or metric not in ("CD", "CL", "CM", "CL/CD") else "rsd"
            symbol = self.term_table.item(r, 4).text().strip() if self.term_table.item(r, 4) else self._make_symbol(metric, surfaces_txt)
            weight = float(self.term_table.item(r, 5).text()) if self.term_table.item(r, 5) and self.term_table.item(r, 5).text().strip() else 1.0
            terms.append({
                "metric": metric,
                "source": source,
                "surface_ids": surface_ids,
                "reduction": self.term_table.cellWidget(r, 3).currentText(),
                "symbol": symbol,
                "weight": weight,
                "direction": self.term_table.cellWidget(r, 6).currentText(),
                "symmetry_factor": int(self.term_table.cellWidget(r, 7).currentText()),
            })
        return terms

    def _build_expression_from_terms(self):
        parts = []
        for t in self._term_dicts():
            w = float(t.get("weight", 1.0))
            sym = t["symbol"]
            if w == 1.0:
                parts.append(sym)
            elif w == -1.0:
                parts.append(f"-{sym}")
            else:
                parts.append(f"{w:g}*{sym}")
        self.obj_type.setCurrentText("Custom Expression")
        self.custom_expr.setText(" + ".join(parts) if parts else "CD")

    def _on_preset_changed(self, text):
        presets = {
            "Drag": "CD",
            "Lift": "-CL",
            "Lift-to-Drag": "-(CL/CD)",
        }
        if text in presets:
            self.custom_expr.setText(presets[text])

    def _validate_and_accept(self):
        # Basic validation: monitor terms need surfaces, except custom monitor definitions may be global later.
        for t in self._term_dicts():
            if t["source"] == "monitor" and not t.get("surface_ids"):
                QMessageBox.warning(self, "Objective term", f"Monitor term '{t['symbol']}' needs at least one surface ID.")
                return
        if not self.custom_expr.text().strip():
            self._build_expression_from_terms()
        self.accept()

    def get_config(self):
        obj_type = self.obj_type.currentText()
        terms = self._term_dicts()
        expr = self.custom_expr.text().strip()
        if not expr:
            self._build_expression_from_terms()
            expr = self.custom_expr.text().strip()

        conds = []
        for r in range(self.table.rowCount()):
            def _txt(c):
                it = self.table.item(r, c)
                return it.text().strip() if it else ""
            try:
                conds.append({
                    "Altitude":   float(_txt(0)) if _txt(0) else 36000.0,
                    "AoA":        float(_txt(1)) if _txt(1) else 3.0,
                    "Mach":       float(_txt(2)) if _txt(2) else 1.2,
                    "Re":         float(_txt(3)) if _txt(3) else 6.9e6,
                    "TurbModel":  int(_txt(4)) if _txt(4) else 1,
                    "EngineFlow": int(_txt(5)) if _txt(5) else 2,
                    "MassFlow":   float(_txt(6)) if _txt(6) else 1.0,
                    "Weight":     float(_txt(7)) if _txt(7) else 1.0,
                })
            except Exception:
                pass

        cons = [ln.strip() for ln in self.constraints_edit.toPlainText().splitlines() if ln.strip()]

        return {
            "objective_type": obj_type,
            "objective_mode": "expression",
            "expression": expr,
            "terms": terms,
            "conditions": conds,
            "constraints": cons,
        }
        
class _SurfaceSelectionBox(QGroupBox):
    """Small reusable surface selector used by monitor dialogs."""
    def __init__(self, mesh_obj, title="Surface IDs", selected=None, parent=None):
        super().__init__(title, parent)
        self.mesh_obj = mesh_obj
        selected = set(int(x) for x in (selected or []) if str(x).strip() != "")

        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Select all")
        self.clear_btn = QPushButton("Clear")
        btn_row.addWidget(self.select_all_btn)
        btn_row.addWidget(self.clear_btn)
        layout.addLayout(btn_row)

        self.select_all_btn.clicked.connect(self.select_all)
        self.clear_btn.clicked.connect(self.clear_selection)
        self._populate(selected)

    def _populate(self, selected):
        names = []
        try:
            names = list(self.mesh_obj.get_surface_names())
        except Exception:
            names = []
        for nm in names:
            try:
                sid = int(self.mesh_obj.get_surface_id(nm))
            except Exception:
                sid = nm
            item = QListWidgetItem(f"{sid}    {nm}")
            item.setData(Qt.UserRole, sid)
            self.list_widget.addItem(item)
            try:
                if int(sid) in selected:
                    item.setSelected(True)
            except Exception:
                pass

    def select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setSelected(True)

    def clear_selection(self):
        self.list_widget.clearSelection()

    def selected_ids(self):
        out = []
        for it in self.list_widget.selectedItems():
            try:
                out.append(int(it.data(Qt.UserRole)))
            except Exception:
                pass
        return out
        
class MonitorItemDialog(QDialog):
    """Editor for one monitor definition."""
    MONITOR_TYPES = [
        ("pressure_recovery", "Pressure recovery"),
        ("drag", "Drag / force coefficient"),
        ("distortion", "DC60 distortion"),
        ("line_probe", "Line probe / shock location"),
        ("plane_integral", "Plane integral / mass flow"),
        ("separation", "Separation"),
    ]

    def __init__(self, mesh_obj, monitor=None, fixed_type=None, parent=None):
        super().__init__(parent)
        self.mesh_obj = mesh_obj
        self.monitor = dict(monitor or {})
        self.fixed_type = fixed_type
        self.setWindowTitle("Monitor definition")
        self.resize(760, 680)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.type_combo = QComboBox()
        for key, label in self.MONITOR_TYPES:
            self.type_combo.addItem(label, key)
        mtype = fixed_type or self.monitor.get("type", "pressure_recovery")
        idx = self.type_combo.findData(mtype)
        self.type_combo.setCurrentIndex(max(0, idx))
        self.type_combo.setEnabled(fixed_type is None)

        self.name_edit = QLineEdit(str(self.monitor.get("name", mtype)))
        self.enabled_cb = QCheckBox("Enabled")
        self.enabled_cb.setChecked(bool(self.monitor.get("enabled", True)))

        form.addRow("Type:", self.type_combo)
        form.addRow("Name:", self.name_edit)
        form.addRow("", self.enabled_cb)
        layout.addLayout(form)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)

        self.pages = {}
        for key, _label in self.MONITOR_TYPES:
            page = self._make_page(key)
            self.pages[key] = page
            self.stack.addWidget(page["widget"])

        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        self._on_type_changed()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _vec_edit(self, value, default):
        vals = value if value is not None else default
        try:
            txt = ", ".join(f"{float(v):g}" for v in vals)
        except Exception:
            txt = ", ".join(str(v) for v in default)
        e = QLineEdit(txt)
        e.setPlaceholderText("x, y, z")
        return e

    def _parse_vec(self, edit, name):
        raw = edit.text().replace(";", ",").split(",")
        try:
            vals = [float(x.strip()) for x in raw if x.strip() != ""]
        except Exception:
            raise ValueError(f"{name} must contain three numeric values.")
        if len(vals) != 3:
            raise ValueError(f"{name} must contain exactly three values.")
        return vals

    def _make_page(self, mtype):
        w = QWidget()
        layout = QVBoxLayout(w)
        form = QFormLayout()
        data = {"widget": w}
        m = self.monitor if self.monitor.get("type", mtype) == mtype else {}

        if mtype in ("pressure_recovery", "drag", "distortion", "separation"):
            surf = _SurfaceSelectionBox(self.mesh_obj, "Selected surfaces", m.get("surface_ids", []))
            data["surfaces"] = surf
            layout.addWidget(surf)

        if mtype == "pressure_recovery":
            data["center"] = self._vec_edit(m.get("center"), [11.3, 0.4, 0.21])
            data["radius"] = QDoubleSpinBox(); data["radius"].setRange(0.0, 1e6); data["radius"].setDecimals(6); data["radius"].setValue(float(m.get("radius", 0.37)))
            data["axis"] = QComboBox(); data["axis"].addItems(["x", "y", "z"]); data["axis"].setCurrentText(str(m.get("axis", "x")))
            form.addRow("Fallback AIP centre:", data["center"])
            form.addRow("Fallback AIP radius:", data["radius"])
            form.addRow("AIP normal axis:", data["axis"])
            layout.addLayout(form)
            layout.addWidget(QLabel("If surfaces are selected, they are used directly; otherwise the fallback geometric AIP selector is used."))

        elif mtype == "drag":
            data["direction"] = QComboBox(); data["direction"].addItems(["x", "y", "z"]); data["direction"].setCurrentText(str(m.get("direction", "x")))
            data["sym"] = QSpinBox(); data["sym"].setRange(1, 100); data["sym"].setValue(int(m.get("symmetry_factor", 1)))
            form.addRow("Force direction:", data["direction"])
            form.addRow("Symmetry factor:", data["sym"])
            layout.addLayout(form)

        elif mtype == "distortion":
            data["center"] = self._vec_edit(m.get("center"), [11.3, 0.4, 0.21])
            data["radius"] = QDoubleSpinBox(); data["radius"].setRange(1e-12, 1e6); data["radius"].setDecimals(6); data["radius"].setValue(float(m.get("radius", 0.37)))
            data["axis"] = QComboBox(); data["axis"].addItems(["x", "y", "z"]); data["axis"].setCurrentText(str(m.get("axis", "x")))
            data["n_angles"] = QSpinBox(); data["n_angles"].setRange(1, 360); data["n_angles"].setValue(int(m.get("n_angles", 5)))
            data["n_radial"] = QSpinBox(); data["n_radial"].setRange(1, 200); data["n_radial"].setValue(int(m.get("n_radial", 8)))
            data["sector"] = QDoubleSpinBox(); data["sector"].setRange(1.0, 360.0); data["sector"].setDecimals(2); data["sector"].setValue(float(m.get("sector_deg", 60.0)))
            form.addRow("AIP centre:", data["center"])
            form.addRow("AIP radius:", data["radius"])
            form.addRow("AIP normal axis:", data["axis"])
            form.addRow("Rake angles:", data["n_angles"])
            form.addRow("Radial samples per rake:", data["n_radial"])
            form.addRow("Sector angle [deg]:", data["sector"])
            layout.addLayout(form)

        elif mtype == "line_probe":
            data["point_a"] = self._vec_edit(m.get("point_a"), [0.0, 0.0, 0.0])
            data["point_b"] = self._vec_edit(m.get("point_b"), [1.0, 0.0, 0.0])
            data["n_samples"] = QSpinBox(); data["n_samples"].setRange(2, 10000); data["n_samples"].setValue(int(m.get("n_samples", 300)))
            data["method"] = QComboBox(); data["method"].addItems(["combined_pressure_mach_gradient", "max_dpds", "min_dMds", "raw_probe"])
            data["method"].setCurrentText(str(m.get("method", "combined_pressure_mach_gradient")))
            data["variables"] = QLineEdit(", ".join(m.get("variables", ["pressure", "mach", "density", "velocity_magnitude"])))
            data["variables"].setPlaceholderText("pressure, mach, density, velocity_magnitude")
            form.addRow("Point A:", data["point_a"])
            form.addRow("Point B:", data["point_b"])
            form.addRow("Number of samples:", data["n_samples"])
            form.addRow("Detection method:", data["method"])
            form.addRow("Variables:", data["variables"])
            layout.addLayout(form)
            layout.addWidget(QLabel("Use this for shock-location tracking or any centreline/duct line history."))

        elif mtype == "plane_integral":
            data["mode"] = QComboBox(); data["mode"].addItems(["slice", "surface"]); data["mode"].setCurrentText(str(m.get("mode", "slice")))
            data["origin"] = self._vec_edit(m.get("origin"), [0.0, 0.0, 0.0])
            data["normal"] = self._vec_edit(m.get("normal"), [1.0, 0.0, 0.0])
            data["quantity"] = QComboBox(); data["quantity"].addItems(["mass_flow", "area", "area_average_pressure", "area_average_total_pressure"])
            data["quantity"].setCurrentText(str(m.get("quantity", "mass_flow")))
            data["radius"] = QDoubleSpinBox(); data["radius"].setRange(0.0, 1e6); data["radius"].setDecimals(6); data["radius"].setValue(float(m.get("radius", 0.0)))
            surf = _SurfaceSelectionBox(self.mesh_obj, "Optional selected surface for surface-mode", m.get("surface_ids", []))
            data["surfaces"] = surf
            form.addRow("Plane mode:", data["mode"])
            form.addRow("Plane origin:", data["origin"])
            form.addRow("Plane normal:", data["normal"])
            form.addRow("Quantity:", data["quantity"])
            form.addRow("Optional clipping radius (0=off):", data["radius"])
            layout.addLayout(form)
            layout.addWidget(surf)
            layout.addWidget(QLabel("For mass-flow mismatch, create one plane_integral monitor for each section: capture, throat, AIP, etc."))

        elif mtype == "separation":
            data["streamwise"] = self._vec_edit(m.get("streamwise_direction"), [1.0, 0.0, 0.0])
            data["max_dist"] = QDoubleSpinBox(); data["max_dist"].setRange(1e-12, 1e6); data["max_dist"].setDecimals(6); data["max_dist"].setValue(float(m.get("max_wall_distance", 0.02)))
            data["normal_samples"] = QSpinBox(); data["normal_samples"].setRange(2, 500); data["normal_samples"].setValue(int(m.get("normal_samples", 20)))
            data["method"] = QComboBox(); data["method"].addItems(["reversed_flow_fraction", "separation_thickness"])
            data["method"].setCurrentText(str(m.get("method", "reversed_flow_fraction")))
            form.addRow("Streamwise direction:", data["streamwise"])
            form.addRow("Max wall-normal distance:", data["max_dist"])
            form.addRow("Normal samples:", data["normal_samples"])
            form.addRow("Method:", data["method"])
            layout.addLayout(form)
            layout.addWidget(QLabel("First robust metric is reversed-flow fraction; thickness can be added once the sampling is stable."))

        layout.addStretch(1)
        return data

    def _on_type_changed(self):
        mtype = self.type_combo.currentData()
        keys = [k for k, _ in self.MONITOR_TYPES]
        self.stack.setCurrentIndex(keys.index(mtype))
        if not self.name_edit.text().strip() or self.name_edit.text().strip() in [k for k, _ in self.MONITOR_TYPES]:
            self.name_edit.setText(mtype)

    def _current_page(self):
        return self.pages[self.type_combo.currentData()]

    def accept_if_valid(self):
        try:
            m = self.get_monitor()
            # basic sanity checks
            if m["type"] in ("drag", "separation") and not m.get("surface_ids"):
                raise ValueError(f"{m['type']} monitor needs at least one selected surface.")
            if m["type"] == "distortion" and not m.get("surface_ids"):
                # Allow geometric fallback, but radius/centre must be present.
                pass
            if m["type"] == "plane_integral" and m.get("mode") == "surface" and not m.get("surface_ids"):
                raise ValueError("Surface-mode plane integral needs at least one selected surface.")
        except Exception as e:
            QMessageBox.warning(self, "Invalid monitor", str(e))
            return
        self.accept()

    def get_monitor(self):
        mtype = self.type_combo.currentData()
        page = self._current_page()
        out = {
            "type": mtype,
            "name": self.name_edit.text().strip() or mtype,
            "enabled": bool(self.enabled_cb.isChecked()),
        }

        if "surfaces" in page:
            out["surface_ids"] = page["surfaces"].selected_ids()

        if mtype == "pressure_recovery":
            out.update({
                "center": self._parse_vec(page["center"], "AIP centre"),
                "radius": float(page["radius"].value()),
                "axis": page["axis"].currentText(),
            })
        elif mtype == "drag":
            out.update({
                "direction": page["direction"].currentText(),
                "symmetry_factor": int(page["sym"].value()),
            })
        elif mtype == "distortion":
            out.update({
                "center": self._parse_vec(page["center"], "AIP centre"),
                "radius": float(page["radius"].value()),
                "axis": page["axis"].currentText(),
                "n_angles": int(page["n_angles"].value()),
                "n_radial": int(page["n_radial"].value()),
                "sector_deg": float(page["sector"].value()),
                "denominator": "q_mean",
            })
        elif mtype == "line_probe":
            variables = [v.strip() for v in page["variables"].text().split(",") if v.strip()]
            out.update({
                "point_a": self._parse_vec(page["point_a"], "Point A"),
                "point_b": self._parse_vec(page["point_b"], "Point B"),
                "n_samples": int(page["n_samples"].value()),
                "method": page["method"].currentText(),
                "variables": variables,
            })
        elif mtype == "plane_integral":
            out.update({
                "mode": page["mode"].currentText(),
                "origin": self._parse_vec(page["origin"], "Plane origin"),
                "normal": self._parse_vec(page["normal"], "Plane normal"),
                "quantity": page["quantity"].currentText(),
                "radius": float(page["radius"].value()),
                "surface_ids": page["surfaces"].selected_ids(),
            })
        elif mtype == "separation":
            out.update({
                "streamwise_direction": self._parse_vec(page["streamwise"], "Streamwise direction"),
                "max_wall_distance": float(page["max_dist"].value()),
                "normal_samples": int(page["normal_samples"].value()),
                "method": page["method"].currentText(),
            })
        return out


class MonitorEditor(QDialog):
    """Top-level monitor editor for AerOpt."""
    def __init__(self, parent=None, mesh_obj=None, config=None):
        super().__init__(parent)
        self.mesh_obj = mesh_obj
        self.config = dict(config or {"interval": 50, "enabled": True, "monitors": []})
        self.monitors = [dict(m) for m in self.config.get("monitors", [])]

        self.setWindowTitle("Monitors")
        self.resize(900, 620)
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.enabled_cb = QCheckBox("Enable monitors")
        self.enabled_cb.setChecked(bool(self.config.get("enabled", True)))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 100000)
        self.interval_spin.setValue(int(self.config.get("interval", 50)))
        top.addWidget(self.enabled_cb)
        top.addWidget(QLabel("Monitor interval [iterations]:"))
        top.addWidget(self.interval_spin)
        top.addStretch(1)
        layout.addLayout(top)

        mid = QHBoxLayout()
        self.list_widget = QListWidget()
        mid.addWidget(self.list_widget, 1)

        btn_col = QVBoxLayout()
        self.add_btn = QPushButton("Add…")
        self.edit_btn = QPushButton("Edit")
        self.dup_btn = QPushButton("Duplicate")
        self.remove_btn = QPushButton("Remove")
        btn_col.addWidget(self.add_btn)
        btn_col.addWidget(self.edit_btn)
        btn_col.addWidget(self.dup_btn)
        btn_col.addWidget(self.remove_btn)
        btn_col.addSpacing(12)

        for key, label in MonitorItemDialog.MONITOR_TYPES:
            b = QPushButton(f"+ {label}")
            b.clicked.connect(lambda _=False, k=key: self.add_monitor(k))
            btn_col.addWidget(b)
        btn_col.addStretch(1)
        mid.addLayout(btn_col)
        layout.addLayout(mid, 1)

        hint = QLabel(
            "Tip: use line_probe for shock-location histories; use plane_integral for mass-flow planes; "
            "use separation for wall reversed-flow/separation metrics."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.add_btn.clicked.connect(lambda: self.add_monitor(None))
        self.edit_btn.clicked.connect(self.edit_current)
        self.dup_btn.clicked.connect(self.duplicate_current)
        self.remove_btn.clicked.connect(self.remove_current)
        self.list_widget.itemDoubleClicked.connect(lambda _item: self.edit_current())

        self.refresh_list()

    def _summary(self, mon):
        typ = mon.get("type", "?")
        name = mon.get("name", typ)
        enabled = "ON" if mon.get("enabled", True) else "OFF"
        extra = ""
        if mon.get("surface_ids"):
            extra = f" surfaces={mon.get('surface_ids')}"
        elif typ == "line_probe":
            extra = f" A={mon.get('point_a')} B={mon.get('point_b')}"
        elif typ == "plane_integral":
            extra = f" {mon.get('quantity','')} origin={mon.get('origin')} normal={mon.get('normal')}"
        return f"[{enabled}] {name}  ({typ}){extra}"

    def refresh_list(self):
        self.list_widget.clear()
        for mon in self.monitors:
            self.list_widget.addItem(self._summary(mon))

    def _current_index(self):
        row = self.list_widget.currentRow()
        return row if 0 <= row < len(self.monitors) else -1

    def add_monitor(self, fixed_type=None):
        dlg = MonitorItemDialog(self.mesh_obj, fixed_type=fixed_type, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.monitors.append(dlg.get_monitor())
            self.refresh_list()
            self.list_widget.setCurrentRow(len(self.monitors) - 1)

    def edit_current(self):
        idx = self._current_index()
        if idx < 0:
            return
        dlg = MonitorItemDialog(self.mesh_obj, monitor=self.monitors[idx], parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.monitors[idx] = dlg.get_monitor()
            self.refresh_list()
            self.list_widget.setCurrentRow(idx)

    def duplicate_current(self):
        idx = self._current_index()
        if idx < 0:
            return
        mon = dict(self.monitors[idx])
        mon["name"] = f"{mon.get('name', mon.get('type', 'monitor'))}_copy"
        self.monitors.insert(idx + 1, mon)
        self.refresh_list()
        self.list_widget.setCurrentRow(idx + 1)

    def remove_current(self):
        idx = self._current_index()
        if idx < 0:
            return
        del self.monitors[idx]
        self.refresh_list()
        self.list_widget.setCurrentRow(min(idx, len(self.monitors) - 1))

    def get_config(self):
        return {
            "interval": int(self.interval_spin.value()),
            "enabled": bool(self.enabled_cb.isChecked()),
            "monitors": [dict(m) for m in self.monitors],
        }
        
class TrainMeshClassifierDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train mesh classifier")
        self.resize(420, 180)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.n_cases = QSpinBox()
        self.n_cases.setRange(1, 1000)
        self.n_cases.setValue(200)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 200)
        self.batch_size.setValue(10)

        form.addRow("Number of training cases:", self.n_cases)
        form.addRow("Batch size (concurrent morph/volume):", self.batch_size)
        layout.addLayout(form)

        self.poll_s = QSpinBox()
        self.poll_s.setRange(10, 3600)
        self.poll_s.setValue(200)
        form.addRow("Poll interval (seconds):", self.poll_s)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def values(self):
        return int(self.n_cases.value()), int(self.batch_size.value()), int(self.poll_s.value())
    

class MorphDialog(QDialog):
    def __init__(self, parent=None, default_cases: int = 5):
        super().__init__(parent)
        self.setWindowTitle("Run Morph Cases")
        self.setModal(True)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.n_cases_spin = QSpinBox()
        self.n_cases_spin.setRange(1, 10000)
        self.n_cases_spin.setValue(int(default_cases))
        form.addRow("Number of cases:", self.n_cases_spin)
        layout.addLayout(form)

        mode_group = QGroupBox("Run mode")
        mode_layout = QVBoxLayout(mode_group)
        self.rb_displacements = QRadioButton("Generate displacement vectors only")
        self.rb_morph_only = QRadioButton("Generate morphs")
        self.rb_morph_and_volume = QRadioButton("Generate morphs and volume meshes")
        self.rb_morph_only.setChecked(True)
        mode_layout.addWidget(self.rb_displacements)
        mode_layout.addWidget(self.rb_morph_only)
        mode_layout.addWidget(self.rb_morph_and_volume)
        layout.addWidget(mode_group)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def values(self):
        if self.rb_displacements.isChecked():
            run_mode = "disp"
        elif self.rb_morph_and_volume.isChecked():
            run_mode = "vol"
        else:
            run_mode = "morph"
        return {"n_cases": int(self.n_cases_spin.value()), "run_mode": run_mode}
    

class BoundsDialog(QDialog):
    def __init__(self, meta, lb, ub, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Optimisation Bounds")
        self.resize(760, 520)

        self.meta = meta
        self.lb_widgets = []
        self.ub_widgets = []

        layout = QVBoxLayout(self)

        title = QLabel("Lower and upper bounds for optimisation variables")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        self.table = QTableWidget(len(meta), 3)
        self.table.setHorizontalHeaderLabels(["Variable", "Lower Bound", "Upper Bound"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)

        for i, item in enumerate(meta):
            name_item = QTableWidgetItem(item["name"])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, name_item)

            lb_edit = QLineEdit(str(lb[i]))
            ub_edit = QLineEdit(str(ub[i]))
            lb_edit.setAlignment(Qt.AlignRight)
            ub_edit.setAlignment(Qt.AlignRight)

            self.table.setCellWidget(i, 1, lb_edit)
            self.table.setCellWidget(i, 2, ub_edit)

            self.lb_widgets.append(lb_edit)
            self.ub_widgets.append(ub_edit)

        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()

        preset_btn = QPushButton("Reset to Presets")
        preset_btn.clicked.connect(self.reset_to_presets)
        btn_row.addWidget(preset_btn)

        btn_row.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        btn_row.addWidget(buttons)

        layout.addLayout(btn_row)

    def reset_to_presets(self):
        for i, item in enumerate(self.meta):
            self.lb_widgets[i].setText(str(item["lb"]))
            self.ub_widgets[i].setText(str(item["ub"]))

    def validate_and_accept(self):
        try:
            for i, item in enumerate(self.meta):
                lb = float(self.lb_widgets[i].text())
                ub = float(self.ub_widgets[i].text())
                if lb >= ub:
                    raise ValueError(f"{item['name']}: lower bound must be < upper bound")
        except Exception as e:
            QMessageBox.warning(self, "Invalid bounds", str(e))
            return
        self.accept()

    def get_bounds(self):
        lb = [float(w.text()) for w in self.lb_widgets]
        ub = [float(w.text()) for w in self.ub_widgets]
        return lb, ub

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])

    window = MainWindow()
    window.show()

    if window.available_themes:
        window.apply_qss_theme(window.available_themes[-1])

    sys.exit(app.exec())