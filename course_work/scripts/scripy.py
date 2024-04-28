import subprocess
import os
import numpy as np
import pandas as pd
from pathlib import Path  
import toml


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

#outputs keys for tuning parameters from config dictionary
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

#gets UID of chain and calls stop_chain.py
def chain_stop(chain_id):
    chain_stop = subprocess.run('python3 stop_chain.py -u ' + chain_id, shell=True)
    if not chain_stop.returncode:
        print('Chain %s successfully stopped' % chain_id)
    else:
        print(chain_stop.stderr)

'''
1) Config.toml preprocessing`
'''


#path to directory in remote server containg factory scripts
path = '/home/ubuntu/factory/'
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
print("Input number of blockchain launches (maxiter):")
k = int(input())
#set changes in +-10% range
scale = np.linspace(-0.1, 0.1, k)
out_in_keys = get(config)
"""
2) Changing config.toml file in loop:
"""
try:
    column_names = ["NUM_THREADS", "DEFAULT_TICKS_PER_SLOT", "ITER_BATCH_SIZE" , "RECV_BATCH_MAX_CPU", "DEFAULT_HASHES_PER_SECOND", "DEFAULT_TICKS_PER_SECOND" , "AVERAGE_TPS", "AVERAGE_DROPRATE"]
    df = pd.DataFrame(columns = column_names)
    row_count = 0

    for s in scale:
        with open('config.toml', 'w') as cfile:
            new = []
            for inn, out in out_in_keys.items():
                vals = config[out][inn]
                new_vals = int(vals + s * vals)
                config_new[out][inn] = new_vals
                new.append((inn,new_vals))

            # ПОДСТАВИТЬ ЗНАЧЕНИЯ
            df = df.append({"NUM_THREADS":int(new[0][1]),
                            "DEFAULT_TICKS_PER_SLOT":int(new[1][1]),
                            "ITER_BATCH_SIZE":int(new[2][1]),
                            "RECV_BATCH_MAX_CPU":int(new[3][1]), 
                            "DEFAULT_HASHES_PER_SECOND":int(new[4][1]),
                            "DEFAULT_TICKS_PER_SECOND":int(new[5][1]), 
                            "AVERAGE_TPS":np.NaN,
                            "AVERAGE_DROPRATE":np.NaN},
                            ignore_index=True)



            #config.tml update with new parameters
            toml.dump(config_new, cfile)
            cfile.close()

        """
        3)Stop chains in chains folder, start blockchain, get chain UID, 
        get public ip of a node,launch transaction with docker, 
        record results in file, stop blockchain
        """
        print('Checking for started chains ...')
        os.chdir(path + 'chains')
        init_chain_id = subprocess.run('ls', capture_output=True, text=True, shell=True).stdout.split('\n')[:-1]
        os.chdir(path)
        if init_chain_id != []:
            for chain in init_chain_id:
                chain_stop(chain)
        else:
            print('Chain folder is empty')
        print('Starting blockchain...')
        subprocess.run('python3 start_chain.py -v 3 -c config.toml', shell = True)
        os.chdir(path + 'chains')
        chain_id = subprocess.run('ls', capture_output=True, text=True, shell=True).stdout
        os.chdir(path)
        get_chain_ip = subprocess.run('python3 get_public_ip.py -u ' + chain_id, capture_output=True, shell=True, text = True)
        public_ip = get_chain_ip.stdout.split(' ')[-1][:-1]
        print('Starting transactions...')


        #file_to_parse = open("current.txt", "w")

        with open('output.txt', 'w') as out_file:
            subprocess.run("sudo docker run -it --rm --net=host -e NDEBUG=1 timofeykulakov/solana_simulations:1.0 bash -c \"./multinode-demo/bench-tps.sh --entrypoint " + public_ip + ":8001 --faucet " + public_ip + ":9900 --duration 5 --tx_count 50 \"",
                           shell = True, text = True,
                           #stdout=out_file
                           )
            subprocess.run('printf \'\n NEW TRANSACTION \n \'', shell=True, text=True, stdout=out_file )
            subprocess.run('printf \'\n  \'', shell=True, text=True, stdout=out_file)
            #TODO сделать запись файла
        print('End transactions')
        chain_stop(chain_id)


        # get results from logs
        current_results = parse_logs('output.txt')
        # add them to dataframe
        df.at[len(df)-1, 'AVERAGE_TPS']=current_results[0][0]
        df.at[len(df)-1, 'AVERAGE_DROPRATE']=current_results[0][1]
        # save current state to scv
        df.to_csv(filepath, index=False)

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
