NUM_GPU=8
NUM_CPU=48

srun --gres=gpu:${NUM_GPU} -c ${NUM_CPU} -C volta --partition=dev --time=24:00:00 --mem 256GB --pty \
 python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=23456 \
                ez_run.py \
                --prefix [time] \
                --mode train \
                --data_prefix "/private/home/jgu/data/" \
                --dataset "wmt16" \
                --src "ro" --trg "en" \
                --train_set "train" \
                --dev_set   "dev"   \
                --test_set  "test"  \
                --load_lazy \
                --char \
                --workspace_prefix "/private/home/jgu/space/char_debug/" \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size 3072 \
                --inter_size 8 \
                --label_smooth 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --warmup 4000 \
                --load_from 09.14_17.49.51..wmt16_t2t-base_ro_en_c_0.1_196608_M1 \
                --resume

                #--debug
                # --pry_io \
                # --pry_depth 2
            

