gpus=${1:-2}
jbname=${2:-ZeroNMT_EuroDeEsFr}
mode=${3:-train}
load_from=${4:-none}  # --load_from name --resume
seed=${5:-19920206}
port=${6:-22222}

export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port=${port} \
                ez_run.py \
                --prefix 'MT' \
                --mode ${mode} \
                --data_prefix "/private/home/jgu/data/Europarl/" \
                --dataset "es-de-fr-eval-nonoverlap" \
                --src "de,en,fr,en,es,en" --trg "en,de,en,fr,en,es" \
                --test_src "fr,de,es,fr,de,es,de,en"  --test_trg "de,fr,fr,es,es,de,en,fr" \
                --track_best "0,1,2,3,4,5" \
                --train_set "train.bpe.shuf" \
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --vocab_file "es-de-fr.s.w.pt" \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/checkpoint/jgu/space/${jbname}/" \
                --params "t2t-base" \
                --lm_steps 0 \
                --eval_every 1000  \
                --batch_size 1200 \
                --valid_batch_size 4800 \
                --save_every 5000 \
                --inter_size 6 \
                --label_smooth 0.1 \
                --drop_ratio 0.1 \
                --lr 0.0005 \
                --weight_decay 0.0001 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "last_layer" \
                --model 'Transformer' \
                --lang_as_init_token \
                --load_from 12.26_23.15.06.LM.es-de-fr-eval-nonoverlap_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_lm300000_bpe_0.1_14400__iter=10000 \
                --seed ${seed} \
        