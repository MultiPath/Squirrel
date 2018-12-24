python -m torch.distributed.launch --nproc_per_node=${1} --master_port=23456 \
                ez_run.py \
                --prefix [time] \
                --mode train \
                --data_prefix "/private/home/jgu/data/" \
                --dataset "wmt16" \
                --src "en" --trg "ro" \
                --train_set "train" \
                --dev_set   "dev"   \
                --test_set  "test"  \
                --load_lazy \
                --base "char" \
                --workspace_prefix "/private/home/jgu/space/${2}/" \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size 3072 \
                --inter_size 8 \
                --label_smooth 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "forward" \
                
