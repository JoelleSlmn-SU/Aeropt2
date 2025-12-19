import os
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
from Optimisation.BayesianOptimisation.acquisition_functions import EI
from FileRW.MultiArrayCsvFile import MultiArrayCsvFile

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

        self.mac = MultiArrayCsvFile(f"{self.sim_dir}bo_data.mcsv")
        kern    = self.kernel(lengthscale=1.0, noise_variance=1.0)
        self.gp = GP(kernel=kern, noise_variance=0.0001)

        self.X = np.array([])
        self.Y = np.array([])
        self.X_uneval = []
        self.gen_num = 0

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

    @property
    def Y_best(self):
        if len(self.Y) == 0:
            return None
        return np.min(self.Y)
    
    @property
    def X_best(self):
        if len(self.Y) == 0:
            return None
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

    def save_data(self):
        self.logger.log(f"[INFO] Current best: X = {self.X_best} | Y = {self.Y_best}")
        self.mac.write({"X":np.array(self.X), "Y":np.array(self.Y).reshape([len(self.Y),1]), "X_uneval":np.array(self.X_uneval), "gen_num":[self.gen_num]})
        self.convergence(self.n_samples, True)
        self.visualise_generation()

    def load_data(self):
        data = self.mac.read()
        self.X = data["X"]
        self.Y = np.array([y[0] for y in data["Y"]])
        self.X_uneval = data["X_uneval"]
        self.gen_num = int(data["gen_num"][0])
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
        
        y_new = self.eval_func(self.X_uneval, self.gen_num).flatten()
        self.logger.log("Evaluated points:")
        for x,y in zip(self.X_uneval, y_new):
            self.logger.log(f"X = {x} | Y = {y}")

        # update datasets
        if len(self.X) == 0:
            self.X = self.X_uneval
        else:
            self.X = np.concatenate([self.X, self.X_uneval])
        if len(self.Y) == 0:
            self.Y = y_new
        else:
            self.Y = np.concatenate([self.Y, y_new])
        self.X_uneval = []
        self.gen_num += 1
        self.save_data()
    
    def get_af(self):
        # construct af
        post_func = self.gp.posterior_predictor()

        def af(x):
            x_tr = (x-self.X_mean)/self.X_std
            m, cv = post_func(x_tr)
            if self.Y_std!=0:
                y_best_tr = (self.Y_best-self.Y_mean)/self.Y_std
            else:
                y_best_tr = self.Y_best.copy()
            res = -self.acquisition_function(y_best_tr, m, cv)
            return res
        return af

    def optimise(self, cont=True):
        self.get_training_data()
        start = self.gen_num
        if start > self.count_limit:
            print("All Finished")
            sys.exit()

        for gen_num in range(start,self.count_limit):
            self.gen_num = gen_num
            self.logger.log(f"[INFO] Evaluating generation {self.gen_num}")
            self.eval_sample()
            self.gp.X = self.X_scaled
            self.gp.Y = self.Y_scaled
            self.gp.minimise_log_likelihood(maxiter=self.mll_maxfevals)
            # evaluate new sample points
            af = self.get_af()
            
            # minimise af
            cma_options  = {'bounds':[list(self.lb), list(self.ub)],
                            'maxfevals':self.af_maxfevals,
                            'verb_log': 0,
                            'CMA_stds': np.abs(self.ub - self.lb)}
            xinit = self.lb + (np.random.random(self.n_dim) * (self.ub - self.lb))
            sigma0 = 0.25
            with suppress_stdout():
                res = fmin(af, xinit, sigma0, options=cma_options, bipop=True, restarts=9)
            x_opt = res[0]
            print(self.gp.kernel)
            print(f"Optimal x from minimising af: {x_opt}")
            print(f"EI at this point: {af(x_opt)}")
            
            # start evaluate new test.
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
        if self.n_obj == 1 and self.n_dim > 1:
            pass
        elif self.n_obj == 1 and self.n_dim == 1:
            if len(self.X) == 0:
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
            ty = self.eval_func(tx)
            plt.plot(tx, ty, ls='dashed', color="black", alpha=0.5, label="Analytical")
            
            # predicted function
            pred_y, pred_s = post_func(tx)
            pred_y = pred_y.flatten()
            pred_s = pred_s.flatten()
            
            ei = -af(tx)
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
            plt.plot(tx, ei)
            plt.xlabel('x')
            plt.ylabel('E[I(x)]')
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
        plt.axhline(y=min(training_Y), color='orange', linestyle='dotted', label="Best Initial")
        plt.axhline(y=min(Y),          color='green',  linestyle='solid',  label="Best Overall")
        self.logger.log(f"Min y = {min(Y)}")
        plt.legend(prop={'size': 14})
        plt.savefig(f"{self.sim_dir}/bo_conv_hist_n_{training_data}_g_{self.gen_num}.png")
        plt.savefig(f"{self.sim_dir}/bo_conv_hist_n_{training_data}_g_{self.gen_num}.pdf")
        plt.close("all")
