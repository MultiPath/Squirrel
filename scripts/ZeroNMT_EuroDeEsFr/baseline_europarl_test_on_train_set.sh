jbname=ZeroNMT_EuroDeEsFr
src=${1:-de}
trg=${2:-fr}  # --load_from name --resume
beam_size=${3:-1}
dataset=${4:-"es-de-fr-eval"}
model=${5:-"12.29_02.16.18"}
gpus=${6:-8}
seed=${7:-19920206}
port=${8:-22220}

load_from="${model}.MT.${dataset}_t2t-base_en_${src}_Transformer_wf_bpe_0.1_14400_"

python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port=${port} \
                ez_run.py \
                --prefix 'MT' \
                --mode test \
                --output_decoding_files \
                --data_prefix "/private/home/jgu/data/Europarl/" \
                --dataset ${dataset} \
                --src "en"  --trg ${trg} \
                --force_translate_to ${src} \
                --dev_set   "train.bpe.shuf"   \
                --vocab_file "es-de-fr.s.w.pt" \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/checkpoint/jgu/space/${jbname}/" \
                --params "t2t-base" \
                --valid_batch_size 30000 \
                --label_smooth 0.1 \
                --drop_ratio 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "last_layer" \
                --model 'Transformer' \
                --beam_size ${beam_size} \
                --lang_as_init_token \
                --load_from ${load_from} \
                --seed ${seed} \
                --decoding_path "/checkpoint/jgu/space/${jbname}/pivot_translation/baseline_en${src}_with_en${trg}_pairs"


