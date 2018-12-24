gpus=${1:-2}
jbname=${2:-Insertable_DE}
mode=${3:-train}
load_from=${4:-none}  # --load_from name --resume
python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port=23456 \
                ez_run.py \
                --prefix l2r \
                --mode ${mode} \
                --data_prefix "/private/home/jgu/data/" \
                --dataset "wmt16" \
                --src "en" --trg "de" \
                --train_set "train.tok.clean.bpe.32000.l2r.shuf" \
                --dev_set  "newstest2013.tok.bpe.32000"  \
                --dev_set  "newstest2014.tok.bpe.32000"  \
                --vocab_file "en-de/vocab.ins.pt" \
                --load_lazy \
                --share_embeddings \
                --base "bpe" \
                --workspace_prefix "/private/home/jgu/space/${jbname}/" \
                --eval_every 500  \
                --batch_size 4000 \
                --sub_inter_size 1 \
                --inter_size 1 \
                --label_smooth 0.1 \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --load_from ${load_from} \
                --length_ratio 2 \
                --beam_size 4 --alpha 0.6 \
                --relative_pos \
                --model TransformerIns \
                --insertable --insert_mode word_first \
                --order fixed \
                # --debug --no_valid
                #--debug --no_valid
                
