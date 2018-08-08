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
                --test_set  "dev.bpe"  \
                --load_lazy \
                --workspace_prefix "/data0/workspace/squirrel_io/" \
                --load_from "07.25_09.57.wmt16_t2t-base_ro_en_causal_w_0.1_5000" \
                --params "t2t-base" \
                --batch_size 1250 \
                --share_embeddings \
                --beam 5 --alpha 0.6 \
                --causal_enc
                # --debug
                

