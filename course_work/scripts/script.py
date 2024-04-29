import os
import argparse
import subprocess  


parser = argparse.ArgumentParser()
parser.add_argument("-ip","--ip", type = str,
                    help="public ip of genesis node")
parser.add_argument("-n", "--n", type = int,
                    help="number of iterations")
args = parser.parse_args()
n = args.n
ip = args.ip

for i in range(n):
    #os.system execute given command in a subshell
    os.system("sudo docker run -it --rm --net=host -e NDEBUG=1 timofeykulakov/solana_simulations:1.0 bash -c \"./multinode-demo/bench-tps.sh --entrypoint " + args.ip + ":8001 --faucet " + args.ip + ":9900 --duration 5 --tx_count 50 \"")
    os.system('printf \'\n NEW TRANSACTION \' ')
