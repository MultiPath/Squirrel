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
                --train_set "train.bpe.shuf" \
                --dev_set   "test.bpe"   \
                --test_set  "test.bpe"  \
                --load_lazy \
                --workspace_prefix "/data0/workspace/squirrel/" \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size 10000 \
                --inter_size 4 \
                --label_smooth 0.0 \
                --share_embeddings \
                --tensorboard \
                --beam 5 \
                --alpha 0.6 \
                --load_from "07.24_10.48.wmt16_t2t-base_ro_en_w_0.1_5000"

