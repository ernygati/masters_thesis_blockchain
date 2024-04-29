import subprocess
import os
import docker_transactions

def chain_stop(chain_id):
    chain_stop = subprocess.run('python3 stop_chain.py -u ' + chain_id, capture_output= True, shell=True)
    if not chain_stop.returncode:
        print('Chain %s successfully stopped' % chain_id)
    else:
        print(chain_stop.stderr)

def start(path):
    # checking if there are started chains in chain dir
    print('Checking for started chains ...')
    os.chdir(path + 'chains')
    init_chain_id = subprocess.run('ls', capture_output=True, text=True, shell=True).stdout.split('\n')[:-1]
    os.chdir(path)
    if init_chain_id != []:
        for chain in init_chain_id:
            chain_stop(chain)
    else:
        print('Chain folder is empty')

    # starting blockchain
    os.chdir(path)
    subprocess.call('python3 start_chain.py -h', shell = True)
    print('Enter arguments of start_chain.py (exp: -v 3 -e "solana-nr4" -c config.toml):')
    args = input()
    print('Chain creation ...')
    p1 = subprocess.run("python3 start_chain.py " + args.strip(),
                        capture_output=True, shell = True)

    #os.chdir(path + '/'+'solana_factory/chains')
    os.chdir(path + 'chains')

    chain_id = subprocess.run('ls', capture_output=True, text = True, shell = True).stdout

    print("UID:", chain_id)

    #os.chdir(path + '/' + 'solana_factory/')
    os.chdir(path)

    get_chain_ip = subprocess.run('python3 get_public_ip.py -u ' + chain_id, capture_output=True, shell=True, text = True)
    public_ip = get_chain_ip.stdout.split(' ')[-1][:-1]
    print(get_chain_ip.stdout)
    print('\n public_ip:' , public_id)
    if not get_chain_ip.returncode:
        print('Chain public ip:', public_ip)
    else:
        print(get_chain_ip.stderr)

    # #calling docker run for transactions bombing
    # print('\nStarting docker run process ...')
    # #docker_transactions.docker_run(public_id)
    # print('\nFinish docker run process')
    #
    # print('Stopping chain ...')
    # chain_stop(chain_id)


    #os.chdir(path)

