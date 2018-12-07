gpus=${1:-2}
jbname=${2:-Insertable}
mode=${3:-train}
load_from=${4:-none}  # --load_from name --resume

python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port 23456\
                ez_run.py \
                --prefix opt_e03 \
                --mode ${mode} \
                --data_prefix "/private/home/jgu/data/" \
                --dataset "wmt16" \
                --src "ro" --trg "en" \
                --train_set "train.bpe.l2r" \
                --dev_set   "dev.bpe"  \
                --vocab_file "ro-en/vocab.ins.pt" \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/private/home/jgu/space/${jbname}/" \
                --eval_every 500  \
                --print_every 10 \
                --batch_size 2000 \
                --sub_inter_size 3 \
                --inter_size 2 \
                --label_smooth 0.1 \
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
                --order optimal --gsteps 3000 --beta 4 --gamma 1.0 \
                --use_gumbel --no_weight --search_with_dropout \
                
                # --debug --no_valid
                # --debug --no_valid
                # --debug --no_valid \
                
                #                 --epsilon 0.3 \
