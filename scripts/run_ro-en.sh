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
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --load_lazy \
                --workspace_prefix "/data0/workspace/squirrel_io/" \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size 1250 \
                --inter_size 4 \
                --label_smooth 0.1 \
                --share_embeddings \
                --tensorboard \
                
                #--cross_attn_fashion "reverse" \
                #--debug
                #--causal_enc \
                # --encoder_lm
                #--debug
                

