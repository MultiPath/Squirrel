 python -m torch.distributed.launch --nproc_per_node=2 --master_port=23456 \
                ez_run.py \
                --prefix [time] \
                --mode train \
                --data_prefix "/private/home/jgu/data/" \
                --dataset "wmt16" \
                --src "ro" --trg "en" \
                --train_set "train.bpe" \
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --load_lazy \
                --workspace_prefix "/private/home/jgu/space/blockwise/" \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size 500 \
                --inter_size 8 \
                --label_smooth 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --multi_width 5 \
                #--debug
