gpus=${1:-2}
jbname=${2:-Insertable}
mode=${3:-train}
load_from=${4:-none}  # --load_from name --resume
python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port=23456 \
                ez_run.py \
                --prefix opt \
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
                --att_plot_every 2000 \
                --print_every 20 \
                --batch_size 2000 \
                --inter_size 2 \
                --label_smooth 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --load_from ${load_from} \
                --length_ratio 2 \
                --beam_size 10 \
                --relative_pos \
                --model TransformerIns \
                --insertable --word_first \
                --order optimal --epsilon 0.0 --esteps 1 --gsteps 3000 --beta 4 --gamma 1.0 \
                --resume 
                

