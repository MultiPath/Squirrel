gpus=${1:-2}
jbname=${2:-MSDecoder2}
mode=${3:-train}
load_from=${4:-none}  # --load_from name --resume
python -m torch.distributed.launch --nproc_per_node=${1} --master_port=23456 \
                ez_run.py \
                --prefix [time] \
                --mode ${mode} \
                --data_prefix "/private/home/jgu/data/" \
                --dataset "wmt16" \
                --src "ro" --trg "en" \
                --train_set "dev.bpe" \
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/private/home/jgu/space/${2}/" \
                --eval_every 500  \
                --batch_size 1024 \
                --inter_size 1 \
                --label_smooth 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --load_from ${load_from} \
                --length_ratio 2 \
                --debug --no_valid
                # --debug
                #--load_from "09.24_02.09.20..kftt_t2t-base_ja_en_byte_0.1_131080_M1" --resume
                
                #--debug

