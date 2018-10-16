mode=${3:-train}
load_from=${4:-none}  # --load_from name --resume
python -m torch.distributed.launch --nproc_per_node=${1} --master_port=23456 \
                ez_run.py \
                --prefix [time] \
                --mode train \
                --data_prefix "/private/home/jgu/data/" \
                --dataset "kftt" \
                --src "ja" --trg "en" \
                --train_set "train" \
                --dev_set   "dev"   \
                --test_set  "test"  \
                --load_lazy \
                --base "byte" \
                --workspace_prefix "/private/home/jgu/space/${2}/" \
                --eval_every 500  \
                --batch_size 3277 \
                --inter_size 5 \
                --label_smooth 0.1 \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --load_from ${load_from}


                #--load_from "09.24_02.09.20..kftt_t2t-base_ja_en_byte_0.1_131080_M1" --resume
                
                #--debug

