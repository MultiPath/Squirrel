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
run = "scripts/ZeroNMT_EuroDeEsFr/baseline_europarl_test_on_train_set.sh"
slurm_options = "sbatch -J {} --partition={} --mem {}GB --gres=gpu:{} -c {} -C volta --time=2:12:12 --output={}/".format(
                'baselines', 'priority', 128, 2, 48, '/checkpoint/jgu/space/slurm')
slurm_file = "/private/home/jgu/work/slurm.sh"

lans = ['es', 'fr', 'de']
cmds = []
additional = ['es-de-fr-eval 12.29_02.16.18', 'es-de-fr-eval-nonoverlap 12.29_03.37.35']
for a in additional:
    for src in lans:
        for trg in lans:
            for b in [1, 4]:
                if src == trg:
                    continue
                cmd = "bash {} {} {} {} {}".format(run, src, trg, b, a)
                slurm_options_ = slurm_options + (src + '-' + trg + '-' + str(b) + '-' + a.split()[0] + '.trans_on_train_%j.out')
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