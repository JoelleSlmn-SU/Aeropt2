# Solver view for local 2D runs, and the collapsing 2D basis panel

Date: 2026-10-06. Written into the working tree on Joelle's machine (not committed to git by Claude).

## Files

| File | Change |
|---|---|
| `GUI/solver_monitor_2d.py` | new: `Solver2DMonitor` (the 2D page of the Solver view), `PictureView` |
| `Local/flite2d_monitor.py` | new, no Qt: what is on disk (`design_numbers`, `run_state`, `latest_run`, `run_pictures`, `opt_designs`), `read_wall_outline`, `RsdFollower` |
| `Local/flite2d_post.py` | `plot_convergence` split into `draw_convergence(fig, ...)` + save; new `draw_wall_outline`, `draw_message` |
| `GUI/solver_gui.py` | `SolverViewer` has two pages: the cluster job browser (unchanged, 3D) and the 2D monitor; `sync_mode()` picks by `MainWindow._is_2d_case()` on show |
| `GUI/Aeropt.py` | keeps `self.solver_viewer`; `_solver_view_notify()` at start/end of `_run_simulation_2d` and `_run_optimisation_2d` |
| `GUI/mesh_gui.py` | after `save_basis_2d` the option widgets are hidden (`_collapse_basis_panel_2d`), bar with "Edit basis options" (`_expand_basis_panel_2d`) |
| `tests/test_flite2d_monitor.py`, `tests/test_solver_monitor_2d.py` | new (7 tests) |

## How the Solver view knows what is running

It does not ask the worker thread or the optimisation process; it reads the output folder once a
second, only while the view is on screen. Time stamps come from files that are written and closed
at fixed moments: `volumes/n_<n>/*.dat` (mesh exists), `preprocessed/n_<n>/runPrePro.inp`
(preprocessor about to start), `solutions/n_<n>/cond_<i>/runSolver.inp` (solver about to start).
`metrics.json` (removed at solver start, written after the pictures) marks the end. The `.rsd` and
`solver.log` are not used as time stamps (Windows does not update them reliably while open).
Consequences: works the same for Run Simulation, Run Optimisation and for browsing old designs;
"solver running" is shown until the pictures are done (a few seconds after the solver ended).

- "Follow the run" = stay on `latest_run()`. Picking a design/condition by hand turns it off; a new
  run turns it on.
- `RsdFollower`: incremental; only whole lines; resets when the file shrinks, when the bytes before
  the read position changed (file replaced and already longer), or when the iteration number goes
  back. Size is taken from the open file, not from `os.stat`.
- Pictures older than `runSolver.inp` are not shown as the result of the current run.
- Geometry: wall chains from the solver mesh (`read_wall_outline`, triangles skipped: 28 ms for the
  29,419-node mesh, 70 ms for 147,000 nodes), baseline (n = 0) dashed underneath, largest wall
  displacement in the subtitle. Coordinates are those of the `.dat`, i.e. after "Solver mesh scale".
- Design list labels take objective/status from `opt/designs.json` when it exists. After "Start new"
  the solution folders of the previous optimisation are still on disk until overwritten, so the list
  can show more designs than the new run has made.

## Evidence

- 56 tests pass (49 before) on Python 3.13 and on Python 3.11 with numpy 1.26, scipy 1.11.
- Real PrePro.exe / Solver.exe under Wine, driven through the real main window on a virtual display
  (her naca0012 mesh, 150 cycles): basis panel collapse / edit / back / reset; Run Simulation
  followed live (14 distinct row counts seen for 150 cycles); Stop Simulation; Run Optimisation in
  its own process (4 designs) followed design by design, preprocessor then solver; browsing by hand.
- `convergence.png` from the refactored function differs from the old one in 3 pixels by 1 level.
- On Windows: not run. The one Windows-specific assumption, that the `.rsd` can be read while
  Solver.exe writes it, is supported by her own `opt/local_opt.log` of 2026-10-06 (128 progress lines
  of the runner, 15 s apart, read from the open `.rsd`).

## Observations made on the way (not acted on)

- `requirements.txt` pins `pyvista>=0.42,<0.44`, but `MeshGeneration/meshFile.py` uses
  `MultiBlock._items()`, which pyvista 0.43.10 does not have (no `.vtm` loads with that pin). Works
  with pyvista 0.49.
- With a FIXED cycle count a diverged run does not end by itself: design n = 1 of the test
  optimisation wrote NaN rows for all 150 cycles. (In residual-target mode a NaN run ended within
  seconds, notes of 2026-10-04.) A fixed count of a few thousand cycles would spend the full time on
  every diverged design; a kill on the first NaN row is not implemented.
- Her optimisation of 2026-10-06 11:21 (unscaled mesh, target -4, limit 10 min, bounds
  [-11, 30], [-125, 125], [-10, 5], [-70, 70], [-2, 2], [-30, 30]): of the first 8 designs 2 gave a
  result (baseline and n = 5), 4 diverged within 1 s (wall displacement 106 to 130 on a chord of
  1009), 2 hit the time limit. Baseline lift tail range 38 on a lift of -16.4.
- The cycle-count guard for non-integer `numberOfMGIterations` (see
  `flite2d_control_file_pitfalls_notes.md`) is still not in the code.
