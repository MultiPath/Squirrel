#!/usr/bin/env bash
CUDA=${1}
MODE=${2:-train}

export CUDA_VISIBLE_DEVICES=${CUDA}
python ez_run.py \
                --prefix [time] \
                --gpu 0 \
                --mode ${MODE} \
                --data_prefix "/private/home/jgu/data/" \
                --dataset "wmt16" \
                --src "ro" --trg "en" \
                --train_set "train.bpe" \
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --load_lazy \
                --workspace_prefix "/private/home/jgu/space/squirrel/" \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size 3000 \
                --inter_size 3 \
                --label_smooth 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --pry_io \
                --debug
                #--normalize_emb \
                #--debug

                #--debug

                # --pry_io \
                # --pry_depth 2 \
                #--debug
                #--debug
                #--cross_attn_fashion "reverse" \
                #--debug
                #--causal_enc \
                # --encoder_lm
                #--debug
                

