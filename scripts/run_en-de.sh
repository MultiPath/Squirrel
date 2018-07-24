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
                --src "en" --trg "de" \
                --train_set "train.tok.clean.bpe.32000.shuf" \
                --dev_set   "newstest2013.tok.bpe.32000"   \
                --test_set  "newstest2014.tok.bpe.32000"  \
                --load_lazy \
                --workspace_prefix "/data0/workspace/squirrel/" \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size 1250 \
                --inter_size 4 \
                --label_smooth 0.1 \
                --share_embeddings \
                --tensorboard \
                --beam 5 \
                

