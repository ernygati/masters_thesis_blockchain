import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-m","--move", type = str,
                    help="the direction of file movement between remote/local servers",
                    choices = ['rl', 'lr'])
parser.add_argument("-opt","--option", type = str,
                    help="file or directory?",
                    choices = ['f', 'd'])
parser.add_argument("-ssh","--ssh", type = str,
                    help="ssh key of remote server")
parser.add_argument("-rd","--remote_dir", type = str,
                    help="address of remote directory")
parser.add_argument("-ld","--local_dir", type = str,
                    help="addres of local directory")

args = parser.parse_args()
ld = args.local_dir
rd = args.remote_dir

if args.move == 'rl':
    if args.option == 'f':
        os.system("scp " + args.ssh + ":" + rd + " " + ld)
    else:
        os.system("scp -r " + args.ssh + ":" + rd + " " + ld)

if args.move == 'lr':
    if args.option == 'f':
        os.system("scp " + ld + " " + args.ssh + ":" + rd)
    else:
        os.system("scp -r " + ld + " " + args.ssh + ":" + rd)
