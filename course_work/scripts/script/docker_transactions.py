import subprocess

def docker_run(public_ip):
    print('launch sudo docker run -it --rm --net=host -e NDEBUG=1 timofeykulakov/solana_simulations:1.0 bash -c \"./multinode-demo/bench-tps.sh --entrypoint " + public_ip + ":8001 --faucet " + public_ip + ":9900 --duration 5 --tx_count 50 \""')
    subprocess.run("sudo docker run -it --rm --net=host -e NDEBUG=1 timofeykulakov/solana_simulations:1.0 bash -c \"./multinode-demo/bench-tps.sh --entrypoint " + public_ip + ":8001 --faucet " + public_ip + ":9900 --duration 5 --tx_count 50 \"",
                  shell = True)