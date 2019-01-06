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
run = "scripts/ZeroNMT_EuroDeEsFr/zero_europarl_test_direct.sh"
slurm_options = "sbatch -J {} --partition={} --mem {}GB --gres=gpu:{} -c {} -C volta --time=00:15:00 --output={}/".format(
                'baselines', 'priority', 128, 2, 48, '/checkpoint/jgu/space/slurm')
slurm_file = "/private/home/jgu/work/slurm.sh"

lans = ['es', 'fr', 'de']
cmds = []

models = [
    "12.27_05.15.00.MT.es-de-fr-eval_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_14400_from_12.26_22.57.59.iter=0",
    "12.27_05.16.09.MT.es-de-fr-eval_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_14400_from_12.26_22.57.59.iter=5000",
    "12.27_05.16.29.MT.es-de-fr-eval_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_14400_from_12.26_22.57.59.iter=10000",
    "12.27_05.16.53.MT.es-de-fr-eval_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_14400_from_12.26_22.57.59.iter=20000",
    "12.27_02.12.46.MT.es-de-fr-eval-nonoverlap_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_14400_from_12.26_23.15.06.",
    "12.27_02.14.35.MT.es-de-fr-eval-nonoverlap_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_14400_from_12.26_23.15.06.",
    "12.27_02.15.41.MT.es-de-fr-eval-nonoverlap_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_14400_from_12.26_23.15.06.",
    "12.27_02.18.34.MT.es-de-fr-eval-nonoverlap_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_14400_from_12.26_23.15.06.iter=20000"]

datasets = ["es-de-fr-eval", "es-de-fr-eval", "es-de-fr-eval", "es-de-fr-eval", 
            "es-de-fr-eval-nonoverlap", "es-de-fr-eval-nonoverlap", "es-de-fr-eval-nonoverlap", "es-de-fr-eval-nonoverlap"]

lms = ["0", "5k", "10k", "20k", "0", "5k", "10k", "20k"]

for a in range(len(models)):
    for src in lans:
        for trg in lans:
            if src == trg:
                continue

            dataset, model, lm = datasets[a], models[a], lms[a]
            cmd = "bash {} {} {} {} {} {}".format(run, src, trg, 1, dataset, model)
            slurm_options_ = slurm_options + (src + '-' + trg + '-' + "1" + '-' + dataset + '-' + 'lm{}'.format(lm) + '.zero_pivot_%j.out')
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