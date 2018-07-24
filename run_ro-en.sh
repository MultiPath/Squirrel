#!/usr/bin/env bash
CUDA=${1}
MODE=${2:-train}

export CUDA_VISIBLE_DEVICES=${CUDA}
export CUDA_LAUNCH_BLOCKING=1
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
                --workspace_prefix "/data0/jiatao/work/Squirrel/" \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size 1200 \
                --inter_size 3 \
                --label_smooth 0.1 \
                --share_embeddings \
                --tensorboard \
                --beam 5 \
                
                # --debug

                # --debug \
                # --char \
                # --causal_enc \
                # --encoder_lm \
                