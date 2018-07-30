#!/usr/bin/env bash
CUDA=${1}
MODE=${2:-train}

export CUDA_VISIBLE_DEVICES=${CUDA}
python ez_run.py \
                --prefix [time] \
                --gpu 0 \
                --mode test \
                --data_prefix "/data0/data/transformer_data/" \
                --dataset "wmt16" \
                --src "en" --trg "de" \
                --train_set "train.tok.clean.bpe.32000.shuf" \
                --dev_set   "newstest2014.tok.bpe.32000"   \
                --test_set  "newstest2014.tok.bpe.32000"  \
                --load_lazy \
                --workspace_prefix "/data0/workspace/squirrel/" \
                --load_from "07.24_14.20.wmt16_t2t-base_en_de_w_0.1_15000" \
                --params "t2t-base" \
                --batch_size 12500 \
                --share_embeddings \
                --tensorboard \
                --beam 1 --alpha 0.6 \
                --debug

                


