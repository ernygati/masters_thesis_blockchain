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
path = '/home/ubuntu/ernest/factory/'
filename = 'config.toml'
# path to config file in factory folder
config_path = path + filename
init_path = os.getcwd() + '/config.toml'
# read toml file, get dict of dicts
config = toml.load(init_path)
#save initial configuration and forbid changes
with open('save_config.toml', 'w') as init_file:
    toml.dump(config, init_file)
    init_file.close()

#get path to unchangable config file with initial params
save_path = path + 'script/save_config.toml'
out_in_keys = get_config_params.get(config)
config_n = toml.load(save_path)
# set param grid
print("Input number of blockchain launches:")
k = int(input())
scale = np.linspace(-0.1, 0.1, k)
try:
    for s in scale:
        with open(config_path, 'w') as f:
            for inn,out in out_in_keys.items():
               #save initial values
               vals = config[out][inn]
               new_vals = vals + s*vals
               #print(inn,vals, s*vals)
               config_n[out][inn] = new_vals
            #write new config file
            toml.dump(config_n, f)
            f.close()
        chain_docker.start(path)
    #return to initial file
    save_config = toml.load(save_path)
    with open(config_path, 'w') as init_file:
        toml.dump(save_config, init_file)
        init_file.close()
except:
    print('Error')
    save_config = toml.load(save_path)
    with open(config_path, 'w') as init_file:
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
