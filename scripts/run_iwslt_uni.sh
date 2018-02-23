#!/usr/bin/env bash
python run_realtime.py   --prefix [time] --gpu 2
                --eval-every 500 \
                --dataset iwslt \
                --tensorboard \
                --level subword \
                --use_mask \
                --data_prefix "/export/home/jiatao/work/data/" \
                --use_wo \
                --share_embeddings \
                --debug
                #--params base-t2t \
                #--fast --use_alignment --diag --positional_attention \
