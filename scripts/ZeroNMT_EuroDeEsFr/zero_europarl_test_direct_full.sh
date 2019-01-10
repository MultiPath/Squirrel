jbname=ZeroNMT_EuroDeEsFr
src=${1:-de}
trg=${2:-fr}  # --load_from name --resume
beam_size=${3:-1}
dataset=${4:-"es-de-fr-eval"}
load_from=${5:-"12.27_05.16.53.MT.es-de-fr-eval_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_14400_from_12.26_22.57.59.iter=20000"}
gpus=${6:-2}
seed=${7:-19920206}
port=${8:-22220}

#"12.27_05.15.00.MT.es-de-fr-eval_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_14400_from_12.26_22.57.59.iter=0"
#"12.27_02.12.46.MT.es-de-fr-eval-nonoverlap_t2t-base_de,en,fr,en,es,en_en,de,en,fr,en,es_Transformer_wf_bpe_0.1_14400_from_12.26_23.15.06."

for i in {0..200000..5000}
do 
    model=${load_from}_iter=${i}
    decoding_path="/checkpoint/jgu/space/${jbname}/zero_translation/${model}/"
    mkdir -p ${decoding_path}

    python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port=${port} \
                ez_run.py \
                --prefix 'MT' \
                --mode test \
                --output_decoding_files \
                --data_prefix "/private/home/jgu/data/Europarl/" \
                --dataset ${dataset} \
                --src ${src}  --trg ${trg} \
                --dev_set   "dev.bpe"   \
                --vocab_file "es-de-fr.s.w.pt" \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/checkpoint/jgu/space/${jbname}/" \
                --params "t2t-base" \
                --valid_batch_size 6000 \
                --label_smooth 0.1 \
                --drop_ratio 0.1 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "last_layer" \
                --model 'Transformer' \
                --beam_size ${beam_size} \
                --lang_as_init_token \
                --load_from ${model} \
                --seed ${seed} \
                --decoding_path ${decoding_path}
done






# mkdir -p ${decoding_path}


                
