import random
import gpytorch
import botorch
import torch
import pandas as pd
import warnings 
warnings.simplefilter('ignore')
import os
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.optim import optimize_acqf
from sklearn.preprocessing import MinMaxScaler
from botorch.utils.sampling import draw_sobol_samples


from botorch.generation import MaxPosteriorSampling
import numpy as np
from pySOT.strategy import DYCORSStrategy
from pySOT.surrogate import gp
from pySOT.optimization_problems import OptimizationProblem
from pySOT.experimental_design import SymmetricLatinHypercube
from poap.controller import SerialController

DTYPE = torch.double
DEVICE = torch.device("cpu")

class DYCORSOptimProblem(OptimizationProblem):

    def __init__(self, X, y, dim=6, lower_bounds=None, upper_bounds=None):
        self.best_x = []
        self.dim = dim
        self.lb = lower_bounds
        self.ub = upper_bounds
        self.int_var = np.arange(0, dim) 
        self.cont_var = np.array([])
        self.info = str(dim) + "-dimensional black box"

        self.X = X
        self.y=y

    def train_surrogate(self, X, y):
        surrogate = gp.GPRegressor(6, self.lb, self.ub)
        surrogate.updated = False
        surrogate._X = X
        surrogate.fX =  y
        surrogate._fit()
        return surrogate
        
    def eval(self, x):
        self.best_x.append(x.astype(int))
        surrogate = self.train_surrogate(self.X, self.y)
        return surrogate.predict(x)[0]

class BayesianOptimizer(): 
    """ Implements 4 Bayesian optimization strategies: UCB, Expected Improvement, Thompson sampling and DYCORS"""
    def __init__(self, lower_bound:np.array, upper_bound: np.array, num_candidates=1,
                  num_restarts = 5, raw_samples = 100, is_scaler = False): 
        """ Initializes global atributes.

        Parameters
        ----------
        lower_bound : np.array
            Lower bound for X candidates search.
        upper_bound : np.array
            Upper bound for X candidates search.
        num_candidates : int, optional
            Num candiadates outputed by optimizers, by default 1
        num_restarts : int, optional
            Num restarts for EI and UCB, by default 5
        raw_samples : int, optional
            Num raw samples for EI and UCB, by default 100
        is_scaler : bool, optional
            Use MinMaxScaler, by default False
        """        
        self.num_candidates=num_candidates 
        self.num_restarts=num_restarts 
        self.raw_samples=raw_samples 
        self.lb = lower_bound
        self.ub = upper_bound
        if is_scaler:
            self.scaler = MinMaxScaler() 
        else:
            self.scaler = None
         
        self.bounds = torch.stack([torch.tensor(lower_bound,dtype=torch.float64), torch.tensor(upper_bound,dtype=torch.float64)]) 
     
    def train_botorch_surrogate(self, X: np.array, y: np.array): 
        """ Trains SingleTaskGP botorch surrogate model.

        Parameters
        ----------
        X : np.array
            Feature values.
        y : np.array
            Target values.

        Returns
        -------
        
            Trained surrogate model.
        """         
        if self.scaler is not None: 
            X = self.scaler.fit_transform(X) 
        X = torch.tensor(X) 
        y = torch.tensor(y)
        model_local = SingleTaskGP(X, y) 
        mll = ExactMarginalLogLikelihood(model_local.likelihood, model_local) 
        fit_gpytorch_mll(mll)  
        return model_local 

     
    def optimize_UCB(self, X: np.array, y: np.array,beta =0.1):
        """ """
        model_local = self.train_botorch_surrogate(X, y) 
        UCB = UpperConfidenceBound(model_local, beta=beta, maximize=True) 
        candidate, acq_value = optimize_acqf( 
        UCB,self.bounds, self.num_candidates, self.num_restarts, self.raw_samples
        ) 

        candidate = np.array(candidate)
         
        return candidate[0].astype(int)
         
         
    def optimize_EI(self, X: np.array, y: np.array,best_f =0.5): 
        """ """
        model_local = self.train_botorch_surrogate(X, y) 
        EI = ExpectedImprovement(model_local, best_f=best_f, maximize = True) 
        candidate, acq_value = optimize_acqf( 
        EI, self.bounds, self.num_candidates, self.num_restarts, self.raw_samples, 
        ) 
        candidate = np.array(candidate)
         
        return candidate[0].astype(int)
    
    def optimize_TS(self, X: np.array, y: np.array):
        """ """
        X_cand = draw_sobol_samples(self.bounds,1,1)[0]
        model_local = self.train_botorch_surrogate(X, y) 
        thompson_sampling = MaxPosteriorSampling(model=model_local, replacement=True)
        candidate = thompson_sampling(X_cand, num_samples=self.num_candidates)

        candidate = np.array(candidate)
         
        return candidate[0].astype(int)
    
    def optimize_DYCORS(self, X: np.array, y: np.array):
        """ """
        input_dimensionality = X.shape[1]
        optim = DYCORSOptimProblem(X, y,input_dimensionality, self.lb, self.ub)
        surrogate = gp.GPRegressor(input_dimensionality,  self.lb, self.ub)
        
        slhd = SymmetricLatinHypercube(dim=input_dimensionality,num_pts=2*input_dimensionality+1)

        # Create the DYCORSStrategy optimizer
        strategy = DYCORSStrategy(max_evals=self.num_candidates, opt_prob=optim, exp_design=slhd, 
                                surrogate=surrogate,
                                batch_size=1)

        controller = SerialController(objective=optim.eval)
        controller.strategy = strategy
        controller.run()
    

        return optim.best_x[0]
            
if __name__ == "__main__":
    # feature columns
    feature_cols = ['NUM_THREADS', 'TICKS_PER_SLOT', 'RECV_BATCH_MAX_CPU',
        'ITER_BATCH_SIZE', 'HASHES_PER_SECOND', 'TICKS_PER_SECOND']
    # y column
    target_col = ["AVERAGE_TPS_BENCH1"]
    # inital dataset for training
    df100_train = pd.read_csv("data/out_slhc_design_train_100.csv")

    #initial X, y for trainig
    X = df100_train[feature_cols].values
    y = df100_train[target_col].values

    # lower and upper bound for X candidates seacrh
    lb = np.array([3, 850, 850, 53, 1700000, 136])
    ub = np.array([5, 1150, 1150, 73, 2300000, 184])

    botorch_optim = BayesianOptimizer(lower_bound=lb, upper_bound=ub,is_scaler=True)
    print("UCB candidate {}:\nExpected Improvement cnadidate: {}\nTompson Sampling candidate: {}\nDYCORS candidate: {}".format(
        botorch_optim.optimize_UCB(X, y),
        botorch_optim.optimize_EI(X, y),
        botorch_optim.optimize_TS(X, y),
        botorch_optim.optimize_DYCORS(X, y),

        )
    )