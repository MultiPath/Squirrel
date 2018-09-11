 NUM_GPU=8
 NUM_CPU=48

 srun --gres=gpu:${NUM_GPU} -c ${NUM_CPU} -C volta --partition=dev --time=24:00:00 --pty \
 python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=23456 \
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
                --batch_size 2048 \
                --inter_size 2 \
                --label_smooth 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --warmup 4000 \
               

                # --pry_io \
                # --pry_depth 2
            

