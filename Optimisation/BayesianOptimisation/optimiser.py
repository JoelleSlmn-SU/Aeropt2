import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

from Utilities.lhs import lhs
from FileRW.logger import GuiLogger
from Utilities.OutputSuppressor import suppress_stdout
from cma import fmin
from Optimisation.BayesianOptimisation.gp import GP
from Optimisation.BayesianOptimisation.kernels import Mat52Kern
from Optimisation.BayesianOptimisation.acquisition_functions import EI, constrained_EI
from FileRW.MultiArrayCsvFile import MultiArrayCsvFile

# ---------------------------------------------------------------------------
# Constraint parsing. Accepts either the structured form
# {"metric": "DC60_s111", "limit": 0.30, "sense": "<="} (what
# ConstraintSet.as_settings_list() in objective_evaluator.py produces) OR the
# raw free-text form the GUI's constraints box already writes, e.g.
# "DC60_s111 <= 0.30" -- so `settings["constraints"]` can just be
# `objective_config.get("constraints", [])` passed straight through with no
# extra plumbing required upstream.
# ---------------------------------------------------------------------------
_CONSTRAINT_PATTERN = re.compile(r"^(.*?)\s*(<=|>=)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")


def _parse_constraint(raw):
    if isinstance(raw, dict):
        return {"metric": raw["metric"], "limit": float(raw["limit"]), "sense": raw.get("sense", "<=")}
    match = _CONSTRAINT_PATTERN.match(str(raw).strip())
    if not match:
        raise ValueError(
            f"Could not parse constraint {raw!r}; expected '<metric> <= <number>' or '<metric> >= <number>'."
        )
    metric, sense, limit = match.group(1).strip(), match.group(2), float(match.group(3))
    return {"metric": metric, "limit": limit, "sense": sense}


class BayesianOptimiser:
    """Optimisation class"""
    def __init__(self, settings, eval_func, init_func=None):
        """
            Bayesian Optimisation Class for local and remote running.

            settings : dict
                all configurable settings.

            LOCAL:
                init_func - None\n
                eval_func  - Calculate evaluation function value
            REMOTE:
                init_func - starts tests running on the server\n
                eval_func - used to determine status of job on
                            server. should implement waits until server job is done.

            CONSTRAINED BO (new):
                settings["constraints"] - optional list of constraints, either
                    structured dicts {"metric","limit","sense"} or raw strings
                    like "DC60_s111 <= 0.30" (both accepted, see
                    _parse_constraint). Default [] reproduces the previous
                    unconstrained behaviour exactly.

                eval_func's contract changes when constraints are configured:
                it must return a tuple (Y, C) where Y is the objective array
                (as before) and C is a dict {metric_name: array}, aligned
                index-for-index with the evaluated points, containing the
                observed value of each constraint metric. If eval_func
                returns just Y (old contract), that's still accepted -- but
                then every constraint is treated as unobserved/infeasible
                for those points, which will stall the search. This is only
                correct if eval_func already returns the new (Y, C) tuple.
        """
        ## TODO - rename continuous to manual/automatic
        self.init_func = init_func
        self.eval_func = eval_func

        self.n_dim   = settings.get('n_dim', 1)
        self.n_obj   = settings.get('n_obj', 1)
        raw_lb = settings.get('lb', None)
        raw_ub = settings.get('ub', None)
        self.sim_dir = settings.get("sim_dir", "")
        self.kernel  = settings.get("kernel", Mat52Kern)
        self.count_limit = settings.get("count_limit", 5)
        self.n_samples   = settings.get("n_samples", 5)
        self.acquisition_function = settings.get("acquisition_function", EI)
        self.mll_maxfevals = settings.get("mll_maxfevals", 10000)
        self.af_maxfevals = settings.get("af_maxfevals", 10000)

        if raw_lb is None:
            self.lb = np.zeros(self.n_dim, dtype=float)
        else:
            self.lb = np.asarray(raw_lb, dtype=float)

        if raw_ub is None:
            self.ub = np.ones(self.n_dim, dtype=float)
        else:
            self.ub = np.asarray(raw_ub, dtype=float)

        self.mac = MultiArrayCsvFile(f"{self.sim_dir}/bo_data.mcsv")
        kern    = self.kernel(lengthscale=1.0, noise_variance=1.0)
        self.gp = GP(kernel=kern, noise_variance=0.0001)

        self.X = np.array([])
        self.Y = np.array([])
        self.X_uneval = []
        self.gen_num = 0

        # ---- constrained BO setup ----
        raw_constraints = settings.get("constraints", []) or []
        self.constraints = [_parse_constraint(c) for c in raw_constraints]
        self.constraint_gps = {}
        self.C = {}
        constraint_noise_variance = settings.get("constraint_noise_variance", 0.0001)
        for cons in self.constraints:
            name = cons["metric"]
            kern_c = self.kernel(lengthscale=1.0, noise_variance=1.0)
            self.constraint_gps[name] = GP(kernel=kern_c, noise_variance=constraint_noise_variance)
            self.C[name] = np.array([])

        if not os.path.exists(self.sim_dir):
            os.makedirs(self.sim_dir)

        log_path = os.path.join(self.sim_dir, "aeropt.log")
        self.logger = GuiLogger(
            text_widget=None,                      # no Qt widget in remote_opt
            output_dir_func=lambda: log_path,      # where to write the log file
            is_hpc_func=lambda: False,             # treat as local FS, no SFTP
            sftp_client_func=lambda: None          # no SSH client in this context
        )
        self.logger.log("Optimising using Bayesian Optimisation with settings: ")
        self.logger.log(f"Number of dimensions: {settings['n_dim']}")
        self.logger.log(f"Number of objectives: {settings['n_obj']}")
        self.logger.log(f"Training Samples    : {settings['n_samples']}")
        if self.constraints:
            desc = "; ".join(f"{c['metric']} {c['sense']} {c['limit']}" for c in self.constraints)
            self.logger.log(f"[OPT] Constraints active (feasibility-weighted EI): {desc}")
            if self.acquisition_function is not EI:
                self.logger.log(
                    "[OPT][WARN] A non-default acquisition_function is configured, but the "
                    "constrained path always uses feasibility-weighted EI internally -- only "
                    "constrained_EI is implemented, not constrained_POI/constrained_UCB."
                )

    # ------------------------------------------------------------------
    # Objective dataset
    # ------------------------------------------------------------------
    @property
    def Y_best(self):
        if len(self.Y) == 0:
            return None
        if self.constraints:
            return self.Y_best_feasible
        return np.min(self.Y)

    @property
    def X_best(self):
        if len(self.Y) == 0:
            return None
        if self.constraints:
            return self.X_best_feasible
        return self.X[np.argmin(self.Y)]

    @property
    def X_scaled(self):
        return (self.X-self.X_mean)/self.X_std

    @property
    def Y_scaled(self):
        if self.Y_std != 0:
            return (self.Y-self.Y_mean)/self.Y_std
        else:
            return self.Y.copy()

    @property
    def X_mean(self):
        return np.mean(self.X, axis=0)

    @property
    def Y_mean(self):
        return np.mean(self.Y)

    @property
    def X_std(self):
        return np.std(self.X, axis=0)

    @property
    def Y_std(self):
        return np.std(self.Y)

    # ------------------------------------------------------------------
    # Constraint dataset
    # ------------------------------------------------------------------
    def C_mean(self, name):
        return float(np.mean(self.C[name])) if len(self.C[name]) else 0.0

    def C_std(self, name):
        return float(np.std(self.C[name])) if len(self.C[name]) else 0.0

    def C_scaled(self, name):
        std = self.C_std(name)
        if std != 0:
            return (self.C[name] - self.C_mean(name)) / std
        return np.array(self.C[name], dtype=float)

    @property
    def feasible_mask(self):
        """Boolean mask over self.Y: True where every constraint is
        satisfied by the OBSERVED (not GP-predicted) value at that point.
        A missing/NaN constraint reading counts as infeasible -- it should
        never be possible for a broken monitor to silently look feasible."""
        if len(self.Y) == 0:
            return np.array([], dtype=bool)
        if not self.constraints:
            return np.ones(len(self.Y), dtype=bool)
        mask = np.ones(len(self.Y), dtype=bool)
        for cons in self.constraints:
            name = cons["metric"]
            vals = self.C.get(name, np.array([]))
            if len(vals) != len(self.Y):
                mask[:] = False
                continue
            vals = np.asarray(vals, dtype=float)
            if cons["sense"] == "<=":
                mask &= np.isfinite(vals) & (vals <= cons["limit"])
            else:
                mask &= np.isfinite(vals) & (vals >= cons["limit"])
        return mask

    @property
    def Y_best_feasible(self):
        if len(self.Y) == 0:
            return None
        mask = self.feasible_mask
        if not np.any(mask):
            return None
        return float(np.min(self.Y[mask]))

    @property
    def X_best_feasible(self):
        if len(self.Y) == 0:
            return None
        mask = self.feasible_mask
        if not np.any(mask):
            return None
        idx = np.where(mask)[0]
        return self.X[idx[np.argmin(self.Y[idx])]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_data(self):
        self.logger.log(f"[INFO] Current best: X = {self.X_best} | Y = {self.Y_best}")
        payload = {
            "X": np.array(self.X),
            "Y": np.array(self.Y).reshape([len(self.Y), 1]),
            "X_uneval": np.array(self.X_uneval),
            "gen_num": [self.gen_num],
        }
        # ASSUMPTION: MultiArrayCsvFile accepts arbitrary extra named arrays
        # beyond the original 4 keys. If it doesn't, constraint history will
        # NOT survive an HPC job restart -- a resumed run would come back up
        # constraint-blind (self.C empty) until new points are evaluated.
        # Verify this against your MultiArrayCsvFile implementation before
        # relying on restart/resume with constraints active.
        for cons in self.constraints:
            name = cons["metric"]
            vals = np.asarray(self.C.get(name, []), dtype=float)
            payload[f"C__{name}"] = vals.reshape([len(vals), 1]) if len(vals) else vals
        self.mac.write(payload)
        self.convergence(self.n_samples, True)
        self.visualise_generation()

    def load_data(self):
        data = self.mac.read()
        self.X = data["X"]
        self.Y = np.array([y[0] for y in data["Y"]])
        self.X_uneval = data["X_uneval"]
        self.gen_num = int(data["gen_num"][0])
        for cons in self.constraints:
            name = cons["metric"]
            key = f"C__{name}"
            if key in data:
                raw = data[key]
                self.C[name] = np.array(
                    [v[0] if hasattr(v, "__len__") else v for v in raw], dtype=float
                )
            # else: keep whatever self.C[name] already held (e.g. empty on
            # a first run, or a run started before this constraint existed)

            if len(self.C[name]) != len(self.Y):
                self.logger.log(
                    f"[OPT][WARN] Constraint '{name}' history has {len(self.C[name])} "
                    f"point(s) but the loaded objective history (Y) has {len(self.Y)}. "
                    f"This is expected when resuming a bo_data.mcsv written before this "
                    f"constraint was added (e.g. an old Drag-only run) -- it means the "
                    f"constraint GP has ZERO training data for those pre-existing "
                    f"{len(self.Y)} point(s), feasible_mask will read False for all of "
                    f"them until re-evaluated, and Y_best_feasible/X_best_feasible will "
                    f"stay None (pure feasibility-seeking acquisition) until at least one "
                    f"NEW point is evaluated with this constraint active. The 28-point "
                    f"history in bo_data.mcsv cannot be retroactively backfilled with "
                    f"'{name}' values unless you re-derive them for those specific "
                    f"designs and inject them by hand."
                )
        return None

    def get_training_data(self):
        print(self.mac.filename)
        if os.path.exists(self.mac.filename):
            self.logger.log("[INFO] Getting training data from file.")
            self.load_data()
            self.save_data()
        else:
            self.logger.log("[INFO] Getting training data by sampling using LHS.")
            self.X_uneval = lhs(self.n_dim, samples=self.n_samples, lb=self.lb, ub=self.ub)
            self.init_sample()

    def init_sample(self):
        if self.init_func is not None:
            self.init_func(self.X_uneval, self.gen_num)
        self.save_data()

    def eval_sample(self):
        self.load_data()
        if len(self.X_uneval) == 0:
            return

        result = self.eval_func(self.X_uneval, self.gen_num)
        if isinstance(result, tuple) and len(result) == 2:
            y_new, c_new = result
        else:
            # Old (Y-only) contract. Only correct if there are no active
            # constraints -- otherwise every constraint below will read as
            # "no data for this point" and be marked infeasible.
            y_new, c_new = result, {}
        y_new = np.asarray(y_new, dtype=float).flatten()
        c_new = c_new or {}

        if self.constraints and not c_new:
            self.logger.log(
                "[OPT][WARN] Constraints are configured but eval_func returned only Y "
                "(no constraint dict). Every new point will be treated as infeasible "
                "until eval_func returns (Y, {metric: values, ...})."
            )

        self.logger.log("Evaluated points:")
        for i, (x, y) in enumerate(zip(self.X_uneval, y_new)):
            line = f"X = {x} | Y = {y}"
            for cons in self.constraints:
                name = cons["metric"]
                vals = np.asarray(c_new.get(name, []), dtype=float).flatten()
                line += f" | {name} = {vals[i] if i < len(vals) else 'MISSING'}"
            self.logger.log(line)

        # update objective dataset
        if len(self.X) == 0:
            self.X = self.X_uneval
        else:
            self.X = np.concatenate([self.X, self.X_uneval])
        if len(self.Y) == 0:
            self.Y = y_new
        else:
            self.Y = np.concatenate([self.Y, y_new])

        # update constraint dataset(s)
        for cons in self.constraints:
            name = cons["metric"]
            vals = np.asarray(c_new.get(name, []), dtype=float).flatten()
            if len(vals) != len(y_new):
                self.logger.log(
                    f"[WARN] eval_func returned {len(vals)} values for constraint '{name}' "
                    f"but {len(y_new)} points were evaluated -- padding missing entries with "
                    f"NaN (NaN reads as infeasible, see feasible_mask)."
                )
                padded = np.full(len(y_new), np.nan)
                padded[:min(len(vals), len(y_new))] = vals[:len(y_new)]
                vals = padded
            if len(self.C.get(name, [])) == 0:
                self.C[name] = vals
            else:
                self.C[name] = np.concatenate([self.C[name], vals])

        self.X_uneval = []
        self.gen_num += 1
        self.save_data()

    def get_af(self):
        # construct af
        post_func = self.gp.posterior_predictor()
        constraint_post_funcs = {
            name: gp.posterior_predictor() for name, gp in self.constraint_gps.items()
        }

        def af(x):
            x_tr = (x-self.X_mean)/self.X_std
            m, cv = post_func(x_tr)
            # FIX: EI/POI/UCB take a STANDARD DEVIATION, but post_func returns
            # variance (np.diag of the covariance matrix). This was
            # previously being passed straight through as "sigma" -- see the
            # writeup accompanying this change. Fixed here for both the
            # constrained and unconstrained paths since get_af() is the one
            # function that has to be touched either way.
            sigma = np.sqrt(np.maximum(cv, 0.0))

            if not self.constraints:
                if self.Y_std != 0:
                    y_best_tr = (self.Y_best-self.Y_mean)/self.Y_std
                else:
                    y_best_tr = self.Y_best.copy()
                res = -self.acquisition_function(y_best_tr, m, sigma)
                return res

            # ---- constrained path: feasibility-weighted EI ----
            constraint_posteriors = []
            for cons in self.constraints:
                name = cons["metric"]
                cm, ccv = constraint_post_funcs[name](x_tr)
                c_mean, c_std = self.C_mean(name), self.C_std(name)
                # transform constraint posterior back to real (unscaled) units,
                # since `limit` is specified in real units
                mu_c = cm * c_std + c_mean if c_std != 0 else cm
                sigma_c = np.sqrt(np.maximum(ccv, 0.0)) * (c_std if c_std != 0 else 1.0)
                constraint_posteriors.append({
                    "mu": mu_c, "sigma": sigma_c, "limit": cons["limit"], "sense": cons.get("sense", "<=")
                })

            y_best_feasible_tr = None
            if self.Y_best_feasible is not None:
                y_best_feasible_tr = (
                    (self.Y_best_feasible - self.Y_mean) / self.Y_std if self.Y_std != 0 else self.Y_best_feasible
                )

            res = -constrained_EI(y_best_feasible_tr, m, sigma, constraint_posteriors, objective=-1)
            return res
        return af

    def optimise(self, cont=True):
        """
        Optimisation loop.

        Fixes:
        - CMA-ES is unstable/unsupported in 1D -> use bounded 1D search for n_dim==1
        - Ensure acquisition function returns a scalar float for CMA in n_dim>=2
        - (new) fits a separate GP per constraint each generation and drives
          acquisition with feasibility-weighted EI when constraints are set.
        """
        self.get_training_data()
        start = self.gen_num
        if start >= self.count_limit:
            self.logger.log("[INFO] All Finished")
            return self.X_best, self.Y_best

        def _af_to_scalar(val):
            """Force any af output to a Python float."""
            return float(np.asarray(val).reshape(-1)[0])

        def _minimise_af_1d(af, lb, ub, n_grid=81, n_refine=4):
            """
            Robust 1D minimisation on [lb, ub] without CMA/SciPy.
            Grid search + iterative local refinement. Returns float x_best.
            """
            lb = float(np.asarray(lb).reshape(-1)[0])
            ub = float(np.asarray(ub).reshape(-1)[0])
            if ub <= lb:
                return lb

            # coarse grid
            xs = np.linspace(lb, ub, int(n_grid))
            vals = np.empty_like(xs, dtype=float)
            for i, x in enumerate(xs):
                vals[i] = _af_to_scalar(af(np.array([x], dtype=float)))
            x_best = float(xs[int(np.argmin(vals))])

            # refine around best
            span = (ub - lb)
            for _ in range(int(n_refine)):
                w = span / 10.0
                a = max(lb, x_best - w)
                b = min(ub, x_best + w)
                xs = np.linspace(a, b, int(max(21, n_grid // 4)))
                vals = np.empty_like(xs, dtype=float)
                for i, x in enumerate(xs):
                    vals[i] = _af_to_scalar(af(np.array([x], dtype=float)))
                x_best = float(xs[int(np.argmin(vals))])
                span = (b - a)

            return x_best

        for gen_num in range(start, self.count_limit):
            self.gen_num = gen_num
            self.logger.log(f"[INFO] Evaluating generation {self.gen_num}")

            # evaluate any pending samples for this generation
            self.eval_sample()

            # fit objective GP
            self.gp.X = self.X_scaled
            self.gp.Y = self.Y_scaled
            self.gp.minimise_log_likelihood(maxiter=self.mll_maxfevals)

            # fit constraint GP(s)
            for cons in self.constraints:
                name = cons["metric"]
                cgp = self.constraint_gps[name]
                cgp.X = self.X_scaled
                cgp.Y = self.C_scaled(name)
                cgp.minimise_log_likelihood(maxiter=self.mll_maxfevals)

            # acquisition function
            af = self.get_af()

            # ---- acquire next point ----
            if self.n_dim == 1:
                # CMA-ES is unstable in 1D -> bounded 1D search
                x_best = _minimise_af_1d(af, self.lb, self.ub)
                x_opt = np.array([x_best], dtype=float)
                ei_val = _af_to_scalar(af(x_opt))
            else:
                # CMA-ES for n_dim >= 2
                cma_options = {
                    'bounds': [list(self.lb), list(self.ub)],
                    'maxfevals': self.af_maxfevals,
                    'verb_log': 0,
                    'CMA_stds': np.abs(self.ub - self.lb),
                }

                xinit = self.lb + (np.random.random(self.n_dim) * (self.ub - self.lb))
                xinit = np.atleast_1d(np.asarray(xinit, dtype=float))
                sigma0 = 0.25

                def af_scalar(x):
                    return _af_to_scalar(af(np.asarray(x, dtype=float)))

                with suppress_stdout():
                    res = fmin(af_scalar, xinit, sigma0, options=cma_options, bipop=True, restarts=9)

                x_opt = np.asarray(res[0], dtype=float).reshape(-1)
                ei_val = af_scalar(x_opt)

            print(self.gp.kernel)
            print(f"Optimal x from minimising af: {x_opt}")
            print(f"EI at this point: {ei_val}")
            if self.constraints and self.Y_best_feasible is None:
                self.logger.log(
                    "[OPT] No feasible point observed yet -- acquisition is currently pure "
                    "feasibility-seeking (ignores EI) until at least one point satisfies all "
                    "constraints."
                )

            # start evaluating the new test (this is what creates surfaces/n_<gen_num>/...)
            self.X_uneval = [x_opt]
            self.init_sample()

            self.logger.log(f"Generation {self.gen_num} complete...")
            if not cont:
                self.logger.log(f"[INFO] Generation {self.gen_num} started - continuous mode disabled. Exiting.")
                sys.exit()

        self.logger.log("[INFO] Final generation complete.")
        self.eval_sample()
        self.logger.log(f"[INFO] Best: X = {self.X_best} | Y = {self.Y_best}")

        return self.X_best, self.Y_best


    def visualise_generation(self):
        if self.init_func is not None:
            return
        if self.n_obj == 1 and self.n_dim > 1:
            pass
        elif self.n_obj == 1 and self.n_dim == 1:
            # PRE-EXISTING BUG (not introduced by the constrained-BO change):
            # this used to only guard on len(self.X)==0, but the first call
            # into here happens from inside eval_sample()->save_data() on
            # generation 0, BEFORE optimise() has fit self.gp for the first
            # time (self.gp.X is still None from __init__) -- self.X is
            # already non-empty by then, so the old guard didn't catch it
            # and post_func() below crashed with "TypeError: object of type
            # 'NoneType' has no len()" on every fresh 1D single-objective run.
            if len(self.X) == 0 or self.gp.X is None:
                return
            post_func = self.gp.posterior_predictor()
            af = self.get_af()
            plt.figure(self.gen_num, figsize=(8,10))

            # plot model
            plt.subplot(211)
            plt.title(f"Generation {self.gen_num}")
            plt.cla()
            tx = np.linspace(self.lb, self.ub, 1000).reshape(1000)
            # exact func
            ty = self.eval_func(tx, self.gen_num)
            plt.plot(tx, ty, ls='dashed', color="black", alpha=0.5, label="Analytical")

            # predicted function
            pred_y, pred_s = post_func(tx)
            pred_y = pred_y.flatten()
            pred_s = pred_s.flatten()

            acq = -af(tx)
            plt.scatter(self.X, self.Y, marker="x", color="blue", alpha=0.75, label="Evaluated Points (EP)")
            plt.scatter(self.X[-1], self.Y[-1], facecolor="none", edgecolor="black", s=80, label="Most recent EP")
            plt.plot(tx, pred_y, color="red", label="Mean")
            plt.fill_between(tx, pred_y-pred_s, pred_y+pred_s, color="red", alpha=0.3, label="Covariance")
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()

            # plot acquisition function
            plt.subplot(212)
            plt.cla()
            plt.plot(tx, acq)
            plt.xlabel('x')
            plt.ylabel('Acquisition (EI x feasibility)' if self.constraints else 'E[I(x)]')
            plt.savefig(f"{self.sim_dir}surrogate_g_{self.gen_num}.png")
            plt.savefig(f"{self.sim_dir}surrogate_g_{self.gen_num}.pdf")

    def convergence(self, training_data, normalize_y=True):
        if len(self.Y) == 0:
            return
        plt.figure()
        plt.xlabel("Iteration")
        plt.xlim([0, self.count_limit])
        plt.xticks(np.arange(-1, self.count_limit+1, 1.0))
        plt.grid(which='both')

        if normalize_y:
            y0 = self.Y[0]
            Y = [100*(y-y0)/y0 for y in self.Y]
            y0 = Y[0]
            plt.ylabel(r"\% Reduction in Y")
            plt.axhline(y=y0, color='red', linestyle='dashed', label="Original")
        else:
            Y = self.Y
            plt.ylabel("Y(x)")

        training_Y = Y[:training_data]
        iteration_y = Y[training_data:]

        xs = []
        ys = []

        for y in training_Y:
            xs.append(0)
            ys.append(y)

        for i, d in enumerate(iteration_y):
            xs.append(i+1)
            ys.append(d)


        plt.scatter(xs, ys, color='black', marker='x')
        plt.ylim([min(Y)-1, max(Y)+1])

        feasible = self.feasible_mask
        if self.constraints and len(feasible) == len(Y) and np.any(feasible[:training_data]):
            best_initial = min(y for y, ok in zip(training_Y, feasible[:training_data]) if ok)
        else:
            best_initial = min(training_Y)
        if self.constraints and len(feasible) == len(Y) and np.any(feasible):
            best_overall = min(y for y, ok in zip(Y, feasible) if ok)
        else:
            best_overall = min(Y)

        plt.axhline(y=best_initial, color='orange', linestyle='dotted',
                    label="Best Initial (feasible)" if self.constraints else "Best Initial")
        plt.axhline(y=best_overall, color='green', linestyle='solid',
                    label="Best Overall (feasible)" if self.constraints else "Best Overall")
        self.logger.log(f"Min y = {min(Y)}")
        plt.legend(prop={'size': 14})
        plt.savefig(f"{self.sim_dir}/bo_conv_hist_n_{training_data}_g_{self.gen_num}.png")
        plt.savefig(f"{self.sim_dir}/bo_conv_hist_n_{training_data}_g_{self.gen_num}.pdf")
        plt.close("all")