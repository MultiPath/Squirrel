#!/usr/bin/env bash
CUDA=${1}

export CUDA_VISIBLE_DEVICES=${CUDA}
python ez_run.py \
                --prefix [time] \
                --gpu 0 \
                --mode test \
                --data_prefix "/data0/data/transformer_data/" \
                --dataset "wmt16" \
                --src "ro" --trg "en" \
                --train_set "train.bpe.shuf" \
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --load_lazy \
                --workspace_prefix "/data0/workspace/squirrel/" \
                --load_from "07.24_15.26.wmt16_t2t-base_ro_en_causal_lm_w_0.1_5000" \
                --params "t2t-base" \
                --batch_size 12500 \
                --share_embeddings \
                --tensorboard \
                --causal_enc \
                --encoder_lm \
                --beam 1 --alpha 0.6 \
                --debug
                

