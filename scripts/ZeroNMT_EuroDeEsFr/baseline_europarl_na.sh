jbname=ZeroNMT_EuroDeEsFr
src=${1:-de}
trg=${2:-en}  # --load_from name --resume
seed=${3:-19920206}
port=${4:-22220}

python -m torch.distributed.launch --nproc_per_node=8 --master_port=${port} \
                ez_run.py \
                --prefix 'MT' \
                --mode train \
                --data_prefix "/private/home/jgu/data/Europarl/" \
                --dataset "es-de-fr-eval-nonoverlap" \
                --src ${src} --trg ${trg} \
                --test_src ${src}  --test_trg ${trg} \
                --train_set "train.bpe.shuf" \
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --vocab_file "es-de-fr.s.w.pt" \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/checkpoint/jgu/space/${jbname}/" \
                --params "t2t-base" \
                --lm_steps 0 \
                --eval_every 500  \
                --batch_size 1800 \
                --valid_batch_size 4800 \
                --save_every 10000 \
                --inter_size 1 \
                --label_smooth 0.1 \
                --drop_ratio 0.1 \
                --lr 0.0005 \
                --weight_decay 0.0001 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "last_layer" \
                --model 'Transformer' \
                --lang_as_init_token \
                --seed ${seed} \
        
        