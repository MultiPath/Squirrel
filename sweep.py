import ipdb
import itertools
from datetime import datetime
from time import localtime, strftime
from pytz import timezone

import argparse
import random
import sys
import time
import os
import re
import subprocess
import pprint
import json
import pytz

zone = timezone('US/Eastern')

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    vals = [v if isinstance(v, list) else [v] for v in vals]
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def argfy(param):
    arguments = []
    for k, v in param.items():
        if v is True:
            arguments.append("--{}".format(k))
        elif v is False:
            pass
        else:
            arguments.append("--{} {}".format(k,v))

    return " ".join(arguments)

parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str)
parser.add_argument('--shuffle', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--queue', type=str, choices=['dev', 'priority', 'uninterrupted'])
parser.add_argument('--memory', type=int, default=256)
parser.add_argument('--hours', type=int, default=12)
parser.add_argument('--gpus', type=int, default=8)
parser.add_argument('--cpus', type=int, default=48)


args = parser.parse_args()
fmt = '%Y-%m-%d-%H-%M%Z'

date = zone.localize(datetime.now()).strftime(fmt)

params = json.load(open( args.json))
experiment = params['prefix']
jobname = "{}_{}".format(experiment, date)

params['prefix'] = jobname
params['workspace_prefix'] += experiment

params = list(product_dict(**params))
for k, p in enumerate(params):
    p['prefix'] += '.A{}'.format(k)
params = [argfy(p) for p in params]

run = "ez_run.py"
slurm_dir = "/private/home/jgu/slurm/msdecoder/"
slurm_dir = slurm_dir + jobname
slurm_file = "/private/home/jgu/work/slurm.sh"

if not args.test and not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)

slurm_options = "sbatch -J {} --partition={} --mem {}GB --gres=gpu:{} -c {} -C volta --time={}:00:00 --output={}/%j.out".format(
                jobname, args.queue, args.memory, args.gpus, args.cpus, args.hours, slurm_dir)

cmds = []
job_idx = 0
for param in params:
    job_idx += 1
    cmd = "python -m torch.distributed.launch --nproc_per_node={} --master_port=23456 {} {}".format(args.gpus, run, param)
    cmd = '{} {} \"{}\"'.format(slurm_options, slurm_file, cmd)
    cmds.append(cmd)

# if args.shuffle:
#     random.shuffle(cmds)

if not args.test:
    for cmd in cmds:
        print(cmd)
        subprocess.call(cmd, shell=True)
        time.sleep(0.01)
    print ("Submitted {} jobs".format(job_idx))
else:
    for cmd in cmds:
        print(cmd)
    print ("Printing {} jobs".format(job_idx))

