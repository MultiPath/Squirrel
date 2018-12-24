gpus=${1:-2}
jbname=${2:-ZeroNMT}
mode=${3:-train}
load_from=${4:-none}  # --load_from name --resume

python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port=12348 \
                ez_run.py \
                --prefix [time] \
                --mode ${mode} \
                --data_prefix "/private/home/jgu/data/Europarl/" \
                --dataset "es-fr-eval" \
                --src "es,en,fr,en" --trg "en,es,en,fr" \
                --test_src "fr,es"  --test_trg "es,fr" \
                --train_set "train.bpe.shuf" \
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --vocab_file "vocab.es-fr-eval.s.w.pt" \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/private/home/jgu/space/${jbname}/" \
                --params "t2t-base" \
                --eval_every 500  \
                --batch_size 1200 \
                --inter_size 4 \
                --label_smooth 0.1 \
                --drop_ratio 0.1 \
                --lr 0.0005 \
                --weight_decay 0.0001 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "last_layer" \
                --model 'Transformer' \
                --lang_as_init_token \
                --load_from ${load_from} --resume
                
                # --debug --no_valid


