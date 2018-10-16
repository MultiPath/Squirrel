eval_set=${1:-dev.bpe}
load_from=${2:-none}  # --load_from name --resume
python -m torch.distributed.launch --nproc_per_node=2 --master_port=23456 \
                ez_run.py \
                --prefix [time] \
                --mode test \
                --data_prefix "/private/home/jgu/data/" \
                --dataset "wmt16" \
                --src "en" --trg "en" \
                --train_set "train.bpe" \
                --dev_set   ${eval_set}   \
                --test_set  "test.bpe"  \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/private/home/jgu/space/UnsupMT/" \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size 4096 \
                --inter_size 1 \
                --label_smooth 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --model 'AutoTransformer2' \
                --load_from ${load_from} \
                
                #--variational \
                #--debug
                    #--debug
                # 10.05_01.33.09..wmt16_t2t-base_ro_en_bpe_0.1_32768_M1

