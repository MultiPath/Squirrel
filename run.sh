#!/usr/bin/env bash
CUDA=${1}
MODE=${2:-train}

export CUDA_VISIBLE_DEVICES=${CUDA}

python ez_run.py \
                --prefix [time] \
                --gpu 0 \
                --mode ${MODE} \
                --data_prefix "/data0/data/transformer_data/" \
                --dataset "wmt16" \
                --src "ro" --trg "en" \
                --train_set "train.bpe" \
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --workspace_prefix "/data0/jiatao/work/Squirrel/" \
                --params "t2t-base" \
                --eval-every 500  \
                --batch_size 2048 \
                --inter_size 2 \
                --use_wo \
                --share_embeddings \
                --tensorboard \
                --debug \
                # --char \
                # --causal_enc \
                # --encoder_lm \
                
