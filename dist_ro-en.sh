 python -m torch.distributed.launch --nproc_per_node=1  \
                ez_run.py \
                --prefix [time] \
                --mode train \
                --data_prefix "./data/" \
                --dataset "wmt16" \
                --src "ro" --trg "en" \
                --train_set "train" \
                --dev_set   "dev"   \
                --test_set  "test"  \
                --char \
                --load_lazy \
                --workspace_prefix "/data1/workspace/ro-en-character-dist/" \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size 1200 \
                --inter_size 4 \
                --label_smooth 0.1 \
                --share_embeddings \
                --tensorboard \
