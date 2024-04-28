import subprocess
import os
import numpy as np
import pandas as pd
from pathlib import Path
import toml
import random
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import DYCORSStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import Ackley, Levy, OptimizationProblem
from pySOT.utils import progress_plot
from poap.controller import ThreadController, SerialController, BasicWorkerThread

def parse_logs(file_name):
  with open(file_name) as f:
    contents = f.readlines()
  results = []
  tmp_res = [None]*2
  for line in contents:
    if "Average TPS:" in line:
      tmp = line.split(" ")
      tmp_res[0]=(float(tmp[-1].rstrip()))
    if "drop rate:" in line:
      tmp = line.split(" ")
      tmp_res[1]=(float(tmp[-1].rstrip()))
    if tmp_res[0]!=None and tmp_res[1]!=None:
      results.append(tmp_res)
      tmp_res = [None]*2
  return results

def get(config):
    #set blockchain params to change
    params = ['NUM_THREADS',
              'DEFAULT_TICKS_PER_SLOT',
              'ITER_BATCH_SIZE',
              'RECV_BATCH_MAX_CPU',
              'DEFAULT_HASHES_PER_SECOND',
              'DEFAULT_TICKS_PER_SECOND']
    out_in_keys = {}
    #gettin outter keys to get access to inner dicts
    for outter in config.keys():
        inner = list(config[outter].keys())
        #print(config[outter].items())
        for p in params:
            if p in inner:
                out_in_keys[p] = outter
    return out_in_keys

def get_x_bounds(config, var = 0.1):
    params = {}
    keys = get(config)
    bounds = [[],[]]
    for i,(inn, out) in enumerate(keys.items()):
        vals = config[out][inn]
        params[inn] = vals
        a = int(vals*(1-var))
        b = int(vals*(1+var))
        bounds[0].append(a)
        bounds[1].append(b)
    return np.array(list(params.values())), bounds

def weighted_Y(Y,weights = {'TPS':-0.8,'Droprate':+0.2}):
    return Y[0]*weights['TPS'] + Y[1]*weights['Droprate']

def optimizer(X, Y, method = 'surrogate', maxfev = 100):
    filepath = Path('optimization_results/Surrogare.txt')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open('optimization_results/Surrogare.txt', 'w+') as file:
        print('Input \n X: %s, \n Y: %s' % (X.T, Y))
        file.write('Input: \nX:\n%s, \n Y:\n%s' % (X.T, Y))
        Y =  weighted_Y(Y)
        if method == 'surrogate':
            kernel = GPy.kern.RBF(6, lengthscale=0.1)
            model = GPy.models.GPRegression(X.T, Y, kernel)
            # optimize surrogate function for better fit
            model.optimize(max_iters=maxfev, optimizer='lbfgs', messages=True)
            # finding minimal value of function and its argument
            Xmin, Ymin = np.array(model.X[np.argmin(model.Y)]).astype(int), float(min(model.Y))
            # save in file
            print ('\nOptimized result:\n')
            print('ymin:', Ymin, 'xmin:', Xmin)
            file.write('\n Optimized result:\n ymin: %s, xmin: %s' % (Ymin, Xmin))
        file.close()


'''
1) Config.toml preprocessing
'''
#path to directory in remote server containg factory scripts
path = '/Users/ernestgatiatullin/Desktop/scripts/'
#getting config dict from config.toml file
config = toml.load(path + 'config.toml')
#save initial configuration in save_config.toml

#path to save results
filepath = Path('results/out.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)

with open('save_config.toml', 'w') as f:
    toml.dump(config, f)
    f.close()

save_config = toml.load(path + 'save_config.toml')
#creating copy of config.toml for changes
config_new = toml.load(path + 'save_config.toml')
#user's number of iters input
#user's number of iters input
print("Input number of blockchain launches (maxiter):")
#maxiter = int(input())# scale = np.linspace(-0.1, 0.1, k)
out_in_keys = get(config)

"""
2) Changing config.toml file in loop:
"""
try:
    X_optim, bounds = get_x_bounds(config, var = 0.1)
    lower_bounds, upper_bounds = np.array(bounds[0]), np.array(bounds[1])
    input_dimensionality = X_optim.shape[0]

    with open('config.toml', 'w') as cfile:
        for i, (inn, out) in enumerate(out_in_keys.items()):
            config_new[out][inn] = X_optim[i]
        # config.tml update with new parameters
        toml.dump(config_new, cfile)
        cfile.close()
        '''
        3) Optimization via blackbox blockchain function calls
        '''

        Y = blackbox(path)
    #find optimal parameters:
    optimizer(X,Y)
    """
    4) Return config.toml params to initial configuration
    """

    with open('config.toml', 'w') as f:
        toml.dump(save_config, f)
        f.close()

except Exception as e:
    print('Error:', e)
    with open('config.toml', 'w') as f:
        toml.dump(config, f)
        f.close()

def blackbox(X):
    Y = random.randrange(50,58, 1),random.random()
    return Y

#optimizer(obj = func, method = 'surrogate', maxiter = 10, maxfev = 10, varience = 10)
#X,Y = blackbox(config,keys = out_in_keys, iter = maxiter, var = 0.1)
#print(X.shape)

# weights = [0.8,-0.2]
# Y = np.array([(y[0]*weights[0] + y[1]*weights[1])for y in Y]).reshape(-1,1)
# print(Y.shape)
def get_x_bounds(config, var = 0.1):
    params = {}
    keys = get(config)
    bounds = [[],[]]
    for i,(inn, out) in enumerate(keys.items()):
        vals = config[out][inn]
        params[inn] = vals
        a = int(vals*(1-var))
        b = int(vals*(1+var))
        bounds[0].append(a)
        bounds[1].append(b)
    return np.array(list(params.values())), bounds


X, bounds = get_x_bounds(config)
Y = blackbox(X)
print(Y)
lower_bounds, upper_bounds  = np.array(bounds[0]), np.array(bounds[1])
input_dimensionality = X.shape[0]
print(np.arange(0, input_dimensionality))

def rosen_const(x, w1 = 0.8, w2 = -0.2):
    """The Rosenbrock function"""
    return w1*(sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)) + w2*1

class BlackBox(OptimizationProblem):

    def __init__(self, dim=input_dimensionality):
        self.dim = dim
        self.lb = lower_bounds
        self.ub = upper_bounds
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional black box"

    def eval(self, X):
        return weighted_Y(blackbox(X))

max_evals = 11
rbf = RBFInterpolant(
    dim=input_dimensionality, lb=lower_bounds, ub=upper_bounds, kernel=CubicKernel(), tail=LinearTail(input_dimensionality))
slhd = SymmetricLatinHypercube(dim=input_dimensionality, num_pts=max_evals - 1)
black_box = BlackBox()
results_serial = np.zeros((max_evals, ))
controller = SerialController(objective=black_box.eval)
controller.strategy = DYCORSStrategy(
    max_evals=max_evals, opt_prob=black_box, asynchronous=False,
    exp_design=slhd, surrogate=rbf, num_cand=100*input_dimensionality,
    batch_size=1)

result = controller.run()
results_serial = np.array([o.value for o in controller.fevals if o.value is not None])

index = np.argmin(results_serial)
x = controller.fevals[index].params[0]
x = [int(elem) for elem in x]
res_pysot =  -results_serial[index], x
print(res_pysot)

