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
    tmp_res = [None] * 2
    for line in contents:
        if "Average TPS:" in line:
            tmp = line.split(" ")
            tmp_res[0] = (float(tmp[-1].rstrip()))
        if "drop rate:" in line:
            tmp = line.split(" ")
            tmp_res[1] = (float(tmp[-1].rstrip()))
        if tmp_res[0] != None and tmp_res[1] != None:
            results.append(tmp_res)
            tmp_res = [None] * 2
    return results


def get(config):
    # set blockchain params to change
    params = ['NUM_THREADS',
              'DEFAULT_TICKS_PER_SLOT',
              'ITER_BATCH_SIZE',
              'RECV_BATCH_MAX_CPU',
              'DEFAULT_HASHES_PER_SECOND',
              'DEFAULT_TICKS_PER_SECOND']
    out_in_keys = {}
    # gettin outter keys to get access to inner dicts
    for outter in config.keys():
        inner = list(config[outter].keys())
        # print(config[outter].items())
        for p in params:
            if p in inner:
                out_in_keys[p] = outter
    return out_in_keys


def get_x_bounds(config, var=0.1):
    """
    #gets blockchain parameters from config dict file
    # with +-10% upper and lower bounds arrays
    """
    params = {}
    keys = get(config)
    bounds = [[], []]
    for inn, out in keys.items():
        vals = int(config[out][inn])
        params[inn] = vals
        a = int(vals * (1 - var))
        b = int(vals * (1 + var))
        bounds[0].append(a)
        bounds[1].append(b)
    return np.array(list(params.values())), bounds


def weighted_Y(Y, weights={'TPS': -0.8, 'Droprate': +0.2}):
    return Y[0] * weights['TPS'] + Y[1] * weights['Droprate']


# function that take params X and output of blockchain blackbox Y,
# outputing such X in +-10% bounded range which
def optimization(func, max_evals, path, method='surrogate', ):

    with open('save_config.toml', 'w') as f:
        toml.dump(config, f)
        f.close()
    global save_config
    save_config = toml.load(path + 'save_config.toml')

    if method == 'surrogate':
        filepath = Path('optimization_results/Surrogare.txt')
        filepath.parent.mkdir(parents=True, exist_ok=True)

        _, bounds = get_x_bounds(config)
        lower_bounds, upper_bounds = np.array(bounds[0]), np.array(bounds[1])
        input_dimensionality = len(lower_bounds)
        class BlackBox(OptimizationProblem):

            def __init__(self, dim=input_dimensionality):
                self.dim = dim
                self.lb = lower_bounds
                self.ub = upper_bounds
                self.int_var = np.array([])
                self.cont_var = np.arange(0, dim)
                self.info = str(dim) + "-dimensional black box"

            def eval(self, X):
                return func(X,path = path)

        rbf = RBFInterpolant(
            dim=input_dimensionality, lb=lower_bounds, ub=upper_bounds, kernel=CubicKernel(),
            tail=LinearTail(input_dimensionality))
        slhd = SymmetricLatinHypercube(dim=input_dimensionality, num_pts=2*input_dimensionality+1)
        black_box = BlackBox()
        results_serial = np.zeros((max_evals,))
        controller = SerialController(objective=black_box.eval)
        controller.strategy = DYCORSStrategy(
            max_evals=max_evals, opt_prob=black_box, asynchronous=False,
            exp_design=slhd, surrogate=rbf, num_cand=100 * input_dimensionality,
            batch_size=1)

        result = controller.run()
        results_serial = np.array([o.value for o in controller.fevals if o.value is not None])
        index = np.argmin(results_serial)
        x = controller.fevals[index].params[0]
        x = [int(elem) for elem in x]
        print([controller.fevals[i].params[0] for i in range(max_evals)])

        # save in file

        with open('optimization_results/Surrogare.txt', 'w+') as file:
            print('\nOptimized result:\n')
            print('ymin:', -results_serial[index], 'xmin:', x)
            file.write('\n Optimized result:\n ymin: %s, xmin: %s' % (-results_serial[index], x))
            file.close()

    """
    Return config.toml params to initial configuration
    """
    with open('config.toml', 'w') as f:
        toml.dump(save_config, f)
        f.close()


def chain_stop(chain_id):
    chain_stop = subprocess.run('python3 stop_chain.py -u ' + chain_id, shell=True)
    if not chain_stop.returncode:
        print('Chain %s successfully stopped' % chain_id)
    else:
        print(chain_stop.stderr)


def blackbox(X,path):
    """ #load the last parameters from config.toml;
        Stop chains in chains folder, start blockchain, get chain UID,
        get public ip of a node,launch transaction with docker,
        record results in file, stop blockchain, parse logs \n
        output Y = (TPS, DROPRATE)
    """
    #get the latest parameters
    config = toml.load(path + 'config.toml')
    X_optim, _ = get_x_bounds(config, var=0.1)
    #config copy to be changed
    config_new = toml.load(path + 'config.toml')
    with open('config.toml', 'w') as cfile:
        # config.toml update with new parameters
        for i, (inn, out) in enumerate(get(config).items()):
            config_new[out][inn] = int(X[i])
        print('new params: %s'% [int(x) for x in X])
        # save updated version
        toml.dump(config_new, cfile)
        cfile.close()

    # print('Checking for started chains ...')
    # os.chdir(path + 'chains')
    # init_chain_id = subprocess.run('ls', capture_output=True, text=True, shell=True).stdout.split('\n')[:-1]
    # os.chdir(path)
    # if init_chain_id != []:
    #     for chain in init_chain_id:
    #         chain_stop(chain)
    # else:
    #     print('Chain folder is empty')
    # print('Starting blockchain...')
    # subprocess.run('python3 start_chain.py -v 3 -c config.toml', shell=True)
    # os.chdir(path + 'chains')
    # chain_id = subprocess.run('ls', capture_output=True, text=True, shell=True).stdout
    # os.chdir(path)
    # get_chain_ip = subprocess.run('python3 get_public_ip.py -u ' + chain_id, capture_output=True, shell=True,
    #                               text=True)
    # public_ip = get_chain_ip.stdout.split(' ')[-1][:-1]
    # print('Starting transactions...')
    #
    # # file_to_parse = open("current.txt", "w")
    #
    # with open('output.txt', 'w') as out_file:
    #     subprocess.run(
    #         "sudo docker run -it --rm --net=host -e NDEBUG=1 timofeykulakov/solana_simulations:1.0 bash -c \"./multinode-demo/bench-tps.sh --entrypoint " + public_ip + ":8001 --faucet " + public_ip + ":9900 --duration 5 --tx_count 50 \"",
    #         shell=True, text=True, stdout=out_file)
    #     subprocess.run('printf \'\n NEW TRANSACTION \n \'', shell=True, text=True, stdout=out_file)
    #     subprocess.run('printf \'\n  \'', shell=True, text=True, stdout=out_file)
    #     # TODO сделать запись файла
    # print('End transactions')
    # chain_stop(chain_id)

    # get results from logs
    current_results = parse_logs('output.txt')
    #print(current_results)
    #Y = current_results[0]
    Y = (random.randrange(50,58, 1),random.random())
    return weighted_Y(Y)


'''
1) Config.toml preprocessing
'''
# path to directory in remote server containg factory scripts
config_path = '/Users/ernestgatiatullin/Desktop/scripts/'

# getting config dict from config.toml file
config = toml.load(config_path + 'config.toml')
# save initial configuration in save_config.toml


# user's number of iters input
print("Input number of blockchain launches (max_eval):")
max_evals = int(input())


try:
    optimization(func=blackbox, max_evals=max_evals, path=config_path)

except Exception as e:
    print ('Error:', e)
    with open('config.toml', 'w') as f:
        toml.dump(config, f)
        f.close()


