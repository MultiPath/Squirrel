jbname=ZeroNMT_EuroDeEsFr
src=${1:-de}
trg=${2:-fr}  # --load_from name --resume
beam_size=${3:-1}
dataset=${4:-"es-de-fr-eval"}
model=${5:-"12.29_02.16.18"}
gpus=${6:-2}
seed=${7:-19920206}
port=${8:-22220}

load_from_1="${model}.MT.${dataset}_t2t-base_${src}_en_Transformer_wf_bpe_0.1_14400_"
load_from_2="${model}.MT.${dataset}_t2t-base_en_${trg}_Transformer_wf_bpe_0.1_14400_"

python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port=${port} \
                ez_run.py \
                --prefix 'MT' \
                --mode test \
                --output_decoding_files \
                --data_prefix "/private/home/jgu/data/Europarl/" \
                --dataset ${dataset} \
                --src ${src}  --trg ${trg} \
                --force_translate_to "en" \
                --dev_set   "dev.bpe"   \
                --vocab_file "es-de-fr.s.w.pt" \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/checkpoint/jgu/space/${jbname}/" \
                --params "t2t-base" \
                --valid_batch_size 4800 \
                --label_smooth 0.1 \
                --drop_ratio 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "last_layer" \
                --model 'Transformer' \
                --beam_size ${beam_size} \
                --lang_as_init_token \
                --load_from ${load_from_1} \
                --seed ${seed} \

decoding_data_prefix="/checkpoint/jgu/space/${jbname}/decodes/${load_from_1}"
decoding_dataset="dev.bpe.b=${beam_size}_a=1"

python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port=${port} \
                ez_run.py \
                --prefix 'MT' \
                --mode test \
                --output_decoding_files \
                --data_prefix ${decoding_data_prefix} \
                --dataset ${decoding_dataset} \
                --src ${src}  --trg ${trg} \
                --force_translate_from "en" \
                --dev_set "dev.bpe" \
                --suffix_src ".dec" --suffix_trg ".trg" \
                --vocab_file "/private/home/jgu/data/Europarl/${dataset}/es-de-fr.s.w.pt" \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/checkpoint/jgu/space/${jbname}/" \
                --params "t2t-base" \
                --valid_batch_size 4800 \
                --label_smooth 0.1 \
                --drop_ratio 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "last_layer" \
                --model 'Transformer' \
                --lang_as_init_token \
                --load_from ${load_from_2} \
                --seed ${seed} \