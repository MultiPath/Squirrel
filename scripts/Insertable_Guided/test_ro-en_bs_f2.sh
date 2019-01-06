gpus=${1:-2}
jbname=${2:-Insertable_Guided}
mode=${3:-train}
load_from=${4:-none}  # --load_from name --resume

python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port 23456\
                ez_run.py \
                --prefix fairseq \
                --mode ${mode} \
                --data_prefix "/private/home/jgu/data/" \
                --dataset "wmt16" \
                --src "ro" --trg "en" \
                --train_set "train.bpe.l2r" \
                --dev_set   "dev.bpe"  \
                --vocab_file "ro-en/vocab.ins.pt" \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/checkpoint/jgu/space/${jbname}/" \
                --eval_every 500  \
                --print_every 10 \
                --batch_size 2000 \
                --sub_inter_size 4 \
                --inter_size 1 \
                --label_smooth 0.1 \
                --lr 0.0005 \
                --weight_decay 0.0001 \
                --drop_ratio 0.1 --attn_drop_ratio 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --load_from ${load_from} \
                --length_ratio 2 \
                --beam_size 10 \
                --path_temp 1 \
                --relative_pos \
                --model TransformerIns \
                --insertable --insert_mode word_first \
                --order optimal --beta 8 \
                --search_with_dropout \
                --debug
                # --debug --no_valid \
                # --use_gumbel \
                # --uniform_embedding_init \
                # 
                # --debug --no_valid
                # --debug --no_valid \
                
                #                 --epsilon 0.3 \
