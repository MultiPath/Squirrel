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

# run = "scripts/ZeroNMT_EuroDeEsFr/baseline_europarl.sh"
run = "scripts/ZeroNMT_EuroDeEsFr/baseline_europarl_na.sh"
slurm_options = "sbatch -J {} --partition={} --mem {}GB --gres=gpu:{} -c {} -C volta --time={}:00:00 --output={}/".format(
                'baselines', 'learnfair', 128, 8, 48, 52, '/checkpoint/jgu/space/slurm')
slurm_file = "/private/home/jgu/work/slurm.sh"

cmds = []
pairs = [('en', a) for a in ['de', 'es', 'fr']] + [(a, 'en') for a in ['de', 'es', 'fr']]
for src, trg in pairs:
    cmd = "bash {} {} {}".format(run, src, trg)
    slurm_options_ = slurm_options + (src + '-' + trg + '.baseline_%j.out')
    cmd = '{} {} \"{}\"'.format(slurm_options_, slurm_file, cmd)
    cmds.append(cmd)

random.shuffle(cmds)

# if not args.test:
for cmd in cmds:
    print(cmd)
    subprocess.call(cmd, shell=True)
    time.sleep(0.01)
print ("Submitted {} jobs".format(len(cmds)))
# subprocess.call("bash {}", shell=True)