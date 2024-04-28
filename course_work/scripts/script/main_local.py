import os
import argparse
import subprocess
import toml
import numpy as np
import get_config_params
import chain_docker

''' 
1) Changing initial config.toml params for -+ 10%
'''
#path = '/home/ubuntu/ernest/factory'
path = '/Users/ernestgatiatullin/Desktop/scripts/'
os.path.basename(path)
#get path to config.toml file
from_path = path + '/' + 'solana_factory/config.toml'
#from_path = path + '/' + 'config.toml'
# read toml file, get dict of dicts
config = toml.load(from_path)

#save initial configuration and forbid changes
try:
    with open('../save_config.toml', 'x') as init_file:
        toml.dump(config, init_file)
        f.close()
except: FileExistsError

#get path to unchangable config file with initial params
save_path = path + '/' + 'script/save_config.toml'

out_in_keys  = get_config_params.get(config)

config_n = toml.load(save_path)
# set param grid
#scale = np.linspace(-0.1, 0.1, 2)
scale = np.linspace(-0.1,0.1,2)
for s in scale:
    with open('../config.toml', 'w') as f:
        new = []
        for inn,out in out_in_keys.items():
           #save initial values
           vals = config[out][inn]
           new_vals = int(vals + s*vals)
           #print(inn,vals, s*vals)
           config_n[out][inn] = new_vals
           new.append(new_vals)
        print(new)
        #write new config file
        toml.dump(config_n, f)
        #chain_docker.start(path)
        f.close()



#return to initial file
save_config = toml.load(save_path)
with open('../config.toml', 'w') as init_file:
    toml.dump(save_config, init_file)
    init_file.close()

''''''

# parser = argparse.ArgumentParser()
# parser.add_argument("-ip","--ip", type = str,
#                     help="public ip of genesis node")
# parser.add_argument("-n", "--n", type = int,
#                     help="number of iterations")
# args = parser.parse_args()
# n = args.n
# ip = args.ip
# with open('output.txt', 'w') as f:
#     subprocess.call('docker images', shell = True, stdout=f, stderr=f)
# for i in range(n):
#     #os.system execute given command in a subshell
#     subprocess.run(["docker run"])
#     os.system('printf \'\n NEW TRANSACTION \' ')
subprocess.run('ls', shell = True)
subprocess.run('printf \'\n NEW TRANSACTION \n \'', shell = True)
subprocess.run('printf \'\n  \'', shell = True)
subprocess.run('ls', shell = True)
