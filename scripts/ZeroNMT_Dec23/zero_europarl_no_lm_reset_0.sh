gpus=${1:-2}
jbname=${2:-ZeroNMT2}
mode=${3:-train}
load_from=${4:-none}  # --load_from name --resume
seed=${5:-19920206}
port=${6:-11113}

python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port=${port} \
                ez_run.py \
                --prefix [time] \
                --mode ${mode} \
                --data_prefix "/private/home/jgu/data/Europarl/" \
                --dataset "es-fr-eval-nonoverlap" \
                --src "es,en,fr,en" --trg "en,es,en,fr" \
                --test_src "fr,es,es,en"  --test_trg "es,fr,en,fr" \
                --train_set "train.bpe.shuf" \
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --vocab_file "vocab.es-fr-eval.s.w.pt" \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/private/home/jgu/space/${jbname}/" \
                --params "t2t-base" \
                --lm_steps 0 \
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
                --load_from 12.23_01.52.51.LM.es-fr-eval-nonoverlap_t2t-base_es,en,fr,en_en,es,en,fr_Transformer_wf_lm300000_bpe_0.1_9600__iter=0 \
                --seed ${seed} \
        