#!/usr/bin/env bash
python run_realtime.py   \
                --prefix [time] \
                --gpu 1 \
                --eval-every 500 \
                --dataset iwslt \
                --tensorboard \
                --data_prefix "/data0/data/transformer_data/" \
                --use_wo \
                --share_embeddings \
                --debug
                #--params base-t2t \
                #--fast --use_alignment --diag --positional_attention \
