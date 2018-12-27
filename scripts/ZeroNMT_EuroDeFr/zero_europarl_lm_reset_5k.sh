gpus=${1:-2}
jbname=${2:-ZeroNMT_EuroDeFr}
mode=${3:-train}
load_from=${4:-none}  # --load_from name --resume
seed=${5:-19920206}
port=${6:-22221}

export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port=${port} \
                ez_run.py \
                --prefix 'MT' \
                --mode ${mode} \
                --data_prefix "/private/home/jgu/data/Europarl/" \
                --dataset "de-fr-eval" \
                --src "de,en,fr,en" --trg "en,de,en,fr" \
                --test_src "fr,de,de,en"  --test_trg "de,fr,en,fr" \
                --train_set "train.bpe.shuf" \
                --dev_set   "dev.bpe"   \
                --test_set  "test.bpe"  \
                --vocab_file "vocab.de-fr-eval.s.w.pt" \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/checkpoint/jgu/space/${jbname}/" \
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
                --load_from 12.25_02.52.21.LM.de-fr-eval_t2t-base_de,en,fr,en_en,de,en,fr_Transformer_wf_lm300000_bpe_0.1_9600__iter=5000 \
                --seed ${seed} \
        
        