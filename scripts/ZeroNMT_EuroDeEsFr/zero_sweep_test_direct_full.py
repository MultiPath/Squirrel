#!/usr/bin/env python 
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
run = "scripts/ZeroNMT_EuroDeEsFr/zero_europarl_test_direct_full.sh"
slurm_options = "sbatch -J {} --partition={} --mem {}GB --gres=gpu:{} -c {} -C volta --time=02:00:00 --output={}/".format(
                'zs-translation', 'priority', 128, 4, 48, '/checkpoint/jgu/space/ZeroNMT_EuroDeEsFr/slurm_logs')
slurm_file = "/private/home/jgu/work/slurm.sh"

lans = ['es', 'fr', 'de', 'en']
cmds = []

# model = "12.27_05.16.53.MT.es-de-fr-eval_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_14400_from_12.26_22.57.59.iter=20000"
# model = "12.27_05.15.00.MT.es-de-fr-eval_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_14400_from_12.26_22.57.59.iter=0"
# model = "01.03_20.50.55.MT.es-de-fr-eval_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_57600_from_12.26_22.57.59.iter=0"
model = "01.04_19.42.23.MT.es-de-fr-eval_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_57600_from_12.26_22.57.59.iter=20000"
dataset = "es-de-fr-eval"

port_id = 20018

for src in lans:
    for trg in lans:
        #if src != trg:
        #    continue
        # if (src == 'fr') and ((trg == 'es') or (trg == 'de')):
        port_id += 1
        cmd = "bash {} {} {} {} {} {} {} {} {}".format(run, src, trg, 1, dataset, model, 4, 19920206, port_id)
        slurm_options_ = slurm_options + (src + '-' + trg + '-' + "1" + '-' + dataset + '-' + 'full_test.56K.trans_%j.out')
        cmd = '{} {} \"{}\"'.format(slurm_options_, slurm_file, cmd)
        cmds.append(cmd)


# random.shuffle(cmds)
# if not args.test:
for cmd in cmds:
    print(cmd)
    subprocess.call(cmd, shell=True)
    time.sleep(0.01)
print ("Submitted {} jobs".format(len(cmds)))
# subprocess.call("bash {}", shell=True)
