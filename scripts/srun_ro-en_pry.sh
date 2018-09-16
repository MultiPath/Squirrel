 NUM_GPU=8
 NUM_CPU=48

 srun --gres=gpu:${NUM_GPU} -c ${NUM_CPU} -C volta --partition=dev --time=18:00:00 --mem 256GB --pty \
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
                --workspace_prefix "/private/home/jgu/space/blockwise_debug/" \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size  800 \
                --inter_size 5 \
                --label_smooth 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --warmup 4000 \
                --multi_width 4 \
                --dyn \
                --constant_penalty 0.8

                #--debug \

       
            

