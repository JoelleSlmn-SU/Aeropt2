# FLITE 2D preprocessor and solver: integration notes

Date: 2026-10-02. Executables: `Aeropt2/Solver/PrePro.exe` (Aggl2d, multigrid agglomerator) and
`Aeropt2/Solver/Solver.exe` (MGNS2D), Intel Fortran builds from 2013, single precision. Windows
only, so 2D runs are local (decision: Joelle). The MATLAB scripts are not used: they only wrote the
mesh, called the two executables with piped answers, and read two text files.

## File formats (read from the original example folder)

Solver mesh `.dat` (not the FLITE geometry `.dat` that `EdgeMesh2D.read_dat_2d` reads):

```
1
title
ne  np  nb
<ne> <np> <nb>
connectivities
<elem id> <n1> <n2> <n3> 1
coordinates
<node id> <x> <y>
boundaries
<n1> <n2> <flag>
```

Boundary flags from the tutorial: 1 inviscid wall, 3 far field, 5 viscous wall. The solver also
has engine-face / inflow / outflow inputs (`enginesFrontMassFlow`, `engineBCRelaxation`,
`inflowField`, `inflowTemperature`); their flag numbers are not known yet.
The geometry-file codes assumed in `EdgeMesh2D.BC_NAMES` (1 far field, 4 wall) are a different
numbering and must not be mixed with these.

`PrePro.exe` questions, in the order the program asks them (read from a real run):
mesh file; hybrid grid (`.false.`); number of grids; directionality parameter;
number of directional agglomerations (asked ONLY if the directionality parameter is not 0);
visualization mode (0: none); output `.sol` file.
An earlier version of the runner sent "number of directional agglomerations" unconditionally and
no visualization answer; with directionality 0 the two happened to line up.

`Solver.exe` answers: control file (namelist `&inputVariables ivd%...`), `.sol` file, startup file
(blank), result file, residual file.

`.rsd` columns: iteration, residual, lift, drag, moment, pressure lift, pressure drag, friction
drag. Lift/drag/moment sit where `Remote/remoteOpt.parse_rsd` already reads CL, CD, CM.
`.res` columns: node, rho, u, v, E (total energy per unit mass), p; non-dimensional with
rho_inf = 1, |u_inf| = 1, p_inf = 1/(gamma M^2). Checked on `output.res`: Mach from the columns
is 0.5000. (The MATLAB pressure plot formula was missing a factor 1/2; column 6 is the pressure.)
- The node-id column is a 5-character field and prints `*****` from node 100,000 on: read rows by
  position, never by the id.
- Rows follow the node numbering of the solver mesh file (checked on naca0012: wall-normal velocity
  0.004 with that ordering against 0.13 with a shuffled one; far-field rows equal the free stream).

## Why the example run gave NaN

`save_mesh.m` wrote coordinates with `dlmwrite`'s default 5 significant digits. In the example
`mesh.dat` wall nodes 473 and 474 coincided at (-0.027081, 1.6395): one zero-length boundary edge
and one zero-area triangle (element 16360). `solverout.rsd` was a single NaN row. Not confirmed by
re-running that example.

## Defects in the executables (found 2026-10-02) and `Utilities/fix_flite2d_exes.py`

Found by running both programs under Wine with gdb. `python Utilities/fix_flite2d_exes.py Solver`
applies all three fixes, keeps `<name>.exe.bak`, checks file size and surrounding bytes first and
refuses any other build. `Local.flite2d_local.unfixed_executables()` reports what is not repaired;
the GUI shows it on the Solver Settings page and the runner warns in the log.

1. Stack reserve 1 MB (both). Work arrays sized by the node count live on the stack.
   Symptom: `forrtl: severe (170): Program Exception - stack overflow` at iteration 1.
   Fix: PE header stack reserve -> 1024 MB (`Utilities/set_exe_stack.py`).
2. `Solver.exe`, routine at RVA 0x6f20 in the time-stepping loop. It copies an N x 4 array into an
   N x 5 temporary and takes the column count from the destination, so it reads one extra column
   (4 N bytes) past the end of the source. Harmless on small meshes (the read lands in neighbouring
   heap memory); on Windows a block above about 1 MB is mapped on its own with nothing readable
   behind it, so meshes above roughly 65,000 nodes stop with
   `forrtl: severe (157): Program Exception - access violation` at iteration 1.
   Fix: one byte at file offset 0x6357 (`mov rcx,[rsi+0x6A8]` -> `mov rcx,[rsi+0x3A8]`).
   Check (naca0012, 137,882 nodes, Mach 0.8, alpha 2 deg, 150 cycles, original vs fixed under
   Wine): `.res` identical byte for byte; residual, lift, drag, moment identical; only `.rsd`
   column 6 (pressure lift) changes, from garbage to equal to the lift.
   Consequence: column 6 of any `.rsd` from the unrepaired solver is not to be trusted.
3. `PrePro.exe`, internal-outflow-register routine. With no engine-inlet sides and no inflow /
   outflow chains (every plain external-flow case) it returns early without setting the register's
   entry count at `[rbx+0x1EC]`. If the leftover memory is positive, PrePro writes from an array
   that was never allocated and stops after "Writing grid data # 1 to file..." with an access
   violation and a partial `.sol`. Whether it happens depends on the process environment, so the
   same mesh can work in one session and fail in the next.
   Fix: zero the count on that return path (jump to 18 bytes of new code in the unused tail of
   `.text`, section virtual size extended).
   Check: `.sol` byte-identical to the one the original writes when it happens to work
   (naca0012); completes on three meshes on which the original crashed.

Status of the checks: all under Wine on Linux, not on Windows. The byte-level comparisons do not
depend on the operating system; the claim that fix 2 removes the Windows crash rests on the
diagnosis and is confirmed only when the run goes through on Joelle's machine.

## Converter: `FileRW/FliteMesh2D.py`

- `write_flite2d_mesh(mesh, path, edge_flags, scale=1.0, write_map=False)`; `write_from_basis`
  takes the flags and scale from `morph_basis.json` (`solver_bc_flags`, `solver_mesh_scale`).
- Full-precision coordinates; triangles counter-clockwise; fluid on the left of every boundary
  edge; boundary nodes numbered first, wall loops first, in walking order (the original generator's
  numbering, so `ivd%tripNodes` keeps its meaning). `<stem>.map.npz` maps file numbering back to
  mesh numbering; the numbering is the same for every deformation of one baseline.
- Refuses inverted or zero-area triangles, coincident nodes and edges without a flag.
- GUI: boundary type per edge and "Solver mesh scale" in the 2D panel; saving the basis writes
  `<output>/volumes/n_0/<base>_0.dat`; the explorer can export a morphed mesh as `.dat`.
- `find_dat_for` ignores a solver mesh named `<stem>.dat`.

naca0012: 275,440 triangles, 137,882 nodes, 324 boundary edges (206 wall, 118 far field), 23 MB,
1.2 s to write. The source `.vtu` stores single-precision coordinates.

## Local runner: `Local/flite2d_local.py` (`Flite2DCase`)

Layout per design n:

```
volumes/n_<n>/<base>_<n>.dat             solver mesh
preprocessed/n_<n>/runPrePro.inp         answers piped to PrePro.exe
preprocessed/n_<n>/<base>_<n>.sol, prepro.log
solutions/n_<n>/cond_<i>/<base>.inp      control file for that flow condition
solutions/n_<n>/cond_<i>/runSolver.inp, <base>_<n>.res, <base>_<n>.rsd, solver.log, metrics.json
```

- Each executable runs with its working directory set to its own folder and is given short
  relative file names (confirmed to work on Windows with the long OneDrive path).
- Flow conditions edit the namelist: AoA -> `ivd%alpha`, Mach -> `ivd%MachNumber`,
  Re -> `ivd%ReynoldsNumber`, TurbModel -> `ivd%turbulenceModel`. Altitude, EngineFlow and MassFlow
  are not used in 2D.
- Checked before anything runs: the file is a 2D namelist; `numberOfGridsToUse` <= grids generated;
  Re > 0 requires a viscous wall (flag 5) in the mesh.
- PrePro counts as successful only if it finished, returned 0, wrote no `forrtl:` line and left a
  non-empty `.sol`; a partial `.sol` is deleted so a later run cannot pick it up.
- Old outputs are deleted before each run. Status per condition in `metrics.json`: `ok`,
  `diverged` (NaN in the last `.rsd` row), `timeout`, `stopped`, `failed` (no data), `crashed`
  (non-zero return code or a Fortran runtime error). Only `ok` carries usable lift/drag/moment.
- The solver time limit (GUI, default 60 min) is the stop for `numberOfMGIterations < 0`.
- GUI: Solver Settings shows a "2D Preprocessor (PrePro.exe)" group for a 2D mesh and loads
  `templates/flite2d_solver.inp` into the editor; "Run Simulation" runs the chain for n = 0 on a
  worker thread. Settings are also written to `<output>/flite2d_settings.json`.

Tests: 40 pass, with stand-in programs that ask the questions in the real order
(`tests/standins`). In addition the real, repaired executables were run through `Flite2DCase`
under Wine on a 12,200-node O-mesh (chord 1, Mach 0.5, inviscid): alpha 2 deg reached -3 orders in
303 cycles, lift 0.280, drag 0.0233.

## Stopping a run (added 2026-10-02)

GUI: "Stop Simulation" button, enabled only while a local 2D run is active. It sets a flag that the
worker thread polls twice a second; the programs cannot be asked to finish, so the process is killed.

- During PrePro: nothing usable is left; `RunStopped` is raised, no solver is started.
- During the solver: the `.rsd` holds every completed cycle (flushed per cycle). The `.res` is
  rewritten from scratch every `ivd%writeToFileInterval` cycles (about 0.3 s for 137,882 nodes) and
  once more when the solver finishes by itself. So after a stop or a time-out the flow field
  belongs to the last multiple of the interval, not to the last `.rsd` row.
  Verified with the real solver: stopped at cycle 252 with interval 100, the `.res` left on disk is
  identical (max difference 0.0) to the `.res` of a separate run of exactly 200 cycles.
- Before killing, the runner waits until the `.res` has stopped growing; afterwards its row count is
  compared with the node count of the mesh. `metrics.json`: `status` "stopped" / "timeout",
  `res_iteration`, `res_complete`, plus the lift/drag of the last cycle reached (flagged as not
  converged in the log and in the picture titles).
- Flow conditions that had not started are not run.

## Pictures: `Local/flite2d_post.py` (added 2026-10-02)

matplotlib `Figure` + Agg canvas only (no pyplot, no Qt), so it runs in the worker thread.
A picture that cannot be made is logged and never fails the run. `Flite2DCase(make_plots=False)`
turns them off.

Per design: `surfaces/n_<n>/<base>_<n>.png` - wall with its nodes on the near-body mesh; for n > 0
the baseline wall is dashed underneath and the largest wall displacement is stated. Written when the
GUI writes the solver mesh (Save 2D Basis) and before PrePro if missing or older than the mesh.

Per flow condition, in `solutions/n_<n>/cond_<i>/`:
`convergence.png` (residual, lift, drag, moment; four panels, force panels scaled to the last 80 %),
`surface_cp.png` (Cp against x per wall chain, critical Cp marked for subsonic free stream),
`<base>_<n>_surface.csv` (wall, node, x, y, arc length, Cp, Mach, p),
`mach.png` (sonic line marked), `cp.png`, `total_pressure.png` (p0/p0_inf, centred on 1).
View window: wall bounding box grown by half its larger dimension.

Formulas: Mach = |u| / sqrt(gamma p / rho); Cp = 2 (p - p_inf); p_inf = 1/(gamma M_inf^2);
p0/p0_inf isentropic. `.res` has 4 significant digits, so the pictures do too.

Observation from the naca0012 run (Mach 0.8, alpha 2 deg, cycle 1500, not converged): behind the
shock p0/p0_inf is about 0.96, consistent with a normal shock at Mach 1.4 (0.958). Along the whole
upper surface, starting at the leading edge and upstream of the shock, there is a wall layer with
p0/p0_inf down to 0.86. An inviscid flow cannot lose total pressure there, so that layer is
numerical (artificial dissipation on a wall spacing of 1 % chord). It feeds straight into drag and,
for an intake, into pressure recovery. To be quantified with a wall-spacing / dissipation-factor
study before either is used as an objective.

## Results so far (all inviscid, under Wine)

- Joelle's machine, first real run: PrePro accepted the naca0012 mesh (5 grids, `.sol` 43.6 MB,
  24 s); solver read 137,882 nodes, 324 boundary sides, 206 plot nodes; it then stopped with the stack overflow
  (defect 1) and, with the stack raised, with the access violation (defect 2).
  PrePro reports "Number of trailing edges found: 0" on the blunt-base mesh; effect unknown.
- naca0012 as saved (chord 1000, scale 1.0), Mach 0.8, alpha 2 deg, unrepaired solver under Wine:
  1535 cycles in 5 min, residual -2.54 and not monotone (-3 not reached), lift 492.06, drag 34.2
  (per chord 1000: about 0.49 and 0.034), maximum local Mach 1.41 at x about 681.
  Forces scale with the solver length, so a mesh scale of 0.001 gives coefficients directly.
- Synthetic O-mesh at alpha 0: lift 0.0387 where symmetry requires 0. Attributed to the crude
  one-diagonal triangulation of that test mesh but not verified. An alpha = 0 run on the real
  naca0012 mesh is the check: a lift not near zero would point at the mesh, the boundary
  treatment or the converter.

## Open points

- Confirm on Windows that the repaired executables complete the naca0012 run.
- Convergence criterion: -3 orders was not reached in 5 minutes on 137,882 nodes at Mach 0.8; an
  optimisation needs either a fixed cycle count or a force-convergence test, otherwise design
  comparisons mix convergence levels.
- Drag from an Euler solution on this mesh includes numerical dissipation; uniform 1%-chord wall
  spacing under-resolves the leading edge. Needs a mesh-sensitivity check before drag is used as
  an objective.
- Intake metrics (pressure recovery, distortion) are not in `.rsd`; they have to be computed from
  `.res` along a chosen edge, and the engine-face boundary flag is needed first.
- `ivd%tripNodes` in the template (203, 252) belong to the old example mesh.
- Next step: the 2D optimisation loop, i.e. `Flite2DCase` per design behind the same
  `init_generation` / `evaluate_generation` interface as `ClusterTestManager`.
