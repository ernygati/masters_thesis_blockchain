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
from botorch.utils.transforms import unnormalize, normalize


from botorch.generation import MaxPosteriorSampling
import numpy as np
from pySOT.strategy import DYCORSStrategy
from pySOT.surrogate import gp
from pySOT.optimization_problems import OptimizationProblem
from pySOT.experimental_design import SymmetricLatinHypercube
from poap.controller import SerialController
from botorch.utils.transforms import unnormalize, normalize


DTYPE = torch.double
DEVICE = torch.device("cpu")

class DYCORSOptimProblem(OptimizationProblem):

    def __init__(self, X, y,objective:callable, dim=6,bounds=None, standard_bounds=None):
        self.best_x = []
        self.dim = dim
        self.lb = np.array(bounds[0])
        self.ub = np.array(bounds[1])
        self.int_var = []
        self.cont_var = np.arange(0, dim) 
        self.info = str(dim) + "-dimensional black box"

        self.X = np.array(normalize(torch.tensor(X), bounds))
        self.y=y
        self.objective = objective
        
    def obj(self,x: np.array):
        return self.objective(x)
        
    def eval(self, x):
        self.__check_input__(x)
        self.best_x.append(x)
        # print(x, -self.obj(x)[0])
        return -self.obj(x)[0]#surrogate.predict(x)[0]
    

class BayesianOptimizer(): 
    """ Implements 4 Bayesian optimization strategies: UCB, Expected Improvement, Thompson sampling and DYCORS"""
    def __init__(self, bounds:np.array, standard_bounds: np.array, num_candidates=1,
                  num_restarts = 5, raw_samples = 100, exp_name="exp"): 
        """ Initializes global atributes.

        Parameters
        ----------
        bounds : np.array
            Lower and Upper bounds of initial X parameters scale.
        standard_bounds : np.array
            [0,1]^d bounds after min-max normalization.
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
         
        self.bounds = torch.stack([torch.tensor(bounds[0],dtype=torch.float64), torch.tensor(bounds[1],dtype=torch.float64)])
        self.standard_bounds = torch.tensor(standard_bounds)
        self.save_optimal_points_path = f"data/opt_results/{exp_name}"
        os.makedirs(self.save_optimal_points_path, exist_ok=True)
        self.optimal_points_dict={}
     
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
        train_X = normalize(torch.tensor(X), self.bounds)
        y = torch.tensor(y)
        model_local = SingleTaskGP(train_X, y) 
        mll = ExactMarginalLogLikelihood(model_local.likelihood, model_local) 
        fit_gpytorch_mll(mll)  
        return model_local 

     
    def optimize_UCB(self, X: np.array, y: np.array,beta =0.1):
        """ """
        model_local = self.train_botorch_surrogate(X, y) 
        UCB = UpperConfidenceBound(model_local, beta=beta, maximize=True) 
        candidate, acq_value = optimize_acqf( 
        UCB,self.standard_bounds, self.num_candidates, self.num_restarts, self.raw_samples
        ) 
        self.optimal_points_dict["UCB"] = [candidate[0], acq_value]
        candidate = np.array(unnormalize(candidate ,self.bounds))
         
        return candidate[0].astype(int)
         
         
    def optimize_EI(self, X: np.array, y: np.array,best_f =0.5): 
        """ """
        model_local = self.train_botorch_surrogate(X, y) 
        EI = ExpectedImprovement(model_local, best_f=best_f, maximize = True) 
        candidate, acq_value = optimize_acqf( 
        EI, self.standard_bounds, self.num_candidates, self.num_restarts, self.raw_samples, 
        )
        self.optimal_points_dict["EI"] = [candidate[0], acq_value]
        candidate = np.array(unnormalize(candidate ,self.bounds))
         
        return candidate[0].astype(int)
    
    def optimize_TS(self, X: np.array, y: np.array, n_init=10):
        """ """
        X_cand = draw_sobol_samples(self.standard_bounds,1,n_init)[0]
        model_local = self.train_botorch_surrogate(X, y) 
        thompson_sampling = MaxPosteriorSampling(model=model_local, replacement=True)
        candidate = thompson_sampling(X_cand, num_samples=self.num_candidates)

        candidate = np.array(unnormalize(candidate ,self.bounds))
         
        return candidate[0].astype(int)
    
    def optimize_DYCORS(self, X: np.array, y: np.array, max_evals=50):
        """ """
        input_dimensionality = X.shape[1]
        model_local = self.train_botorch_surrogate(X,y)
        def DYCORS_objective(x, model_local, bounds):
            x = torch.tensor(x)
            return model_local.posterior(normalize(x, bounds).reshape(1,-1)).mean.cpu().detach().numpy()[0]
        X = np.array(X)
        y= -np.array(y)
        optim = DYCORSOptimProblem(X, y,lambda x: DYCORS_objective(x, model_local, self.bounds), input_dimensionality, self.bounds, self.standard_bounds)
        surrogate = gp.GPRegressor(input_dimensionality,   optim.lb, optim.ub)

        slhd = SymmetricLatinHypercube(dim=input_dimensionality,num_pts=2*input_dimensionality+1)

        # Create the DYCORSStrategy optimizer
        runs = 1
        for _ in range(runs):
            strategy = DYCORSStrategy(max_evals=max_evals, opt_prob=optim, exp_design=slhd,
                                    asynchronous=False,
                                    surrogate=surrogate,
                                    num_cand=100*optim.dim,
                                    batch_size=1)

            controller = SerialController(objective=optim.eval)
            controller.strategy = strategy
            controller.run()
        argmin = np.argmin([controller.fevals[i].value for i in range(len(controller.fevals))])
        self.optimal_points_dict["DYCORS"] = [normalize(torch.tensor(optim.best_x[argmin]),self.bounds), torch.tensor(-controller.best_point().value)]
        candidate =  np.array(optim.best_x)

    

        return candidate[-1].astype(int)
            
if __name__ == "__main__":
    # feature columns
    feature_cols = ['NUM_THREADS', 'TICKS_PER_SLOT', 'RECV_BATCH_MAX_CPU',
        'ITER_BATCH_SIZE', 'HASHES_PER_SECOND', 'TICKS_PER_SECOND']
    # y column
    target_col = ["AVERAGE_TPS_BENCH1"]
    # inital dataset for training
    df100_train = pd.read_csv("data/out_slhc_design_train_50.csv")

    #initial X, y for trainig
    X = df100_train[feature_cols].values
    y = df100_train[target_col].values

    # lower and upper bound for X candidates seacrh
    lb = torch.tensor([3, 53, 850, 850 , 1700000, 136])
    ub = torch.tensor([5, 73, 1150, 1150 , 2300000, 184])
    
    bounds = torch.stack([lb.type(torch.float64), ub.type(torch.float64)])
    standard_bounds = np.stack([[0.0]*6, [1.0]*6])

    botorch_optim = BayesianOptimizer(bounds, standard_bounds)
    
    #DYCORS strategy needs objective function for evaluation to predict next andidate points x
    
    cand_UCB = botorch_optim.optimize_UCB(X, y)
    cand_EI = botorch_optim.optimize_EI(X, y)
    cand_TS = botorch_optim.optimize_TS(X, y)
    cand_DYCORS = botorch_optim.optimize_DYCORS(X,y)
    print("UCB candidate {}:\nExpected Improvement cnadidate: {}\nTompson Sampling candidate: {}\nDYCORS candidate: {}".format(
        cand_UCB,
        cand_EI,
        cand_TS,
        cand_DYCORS
        )
    )
    print(botorch_optim.optimal_points_dict)
    # surrogate = botorch_optim.train_botorch_surrogate(X,y)
    # print(surrogate)
