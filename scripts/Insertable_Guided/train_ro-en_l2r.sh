gpus=${1:-2}
jbname=${2:-Insertable_Guided}
mode=${3:-train}
load_from=${4:-none}  # --load_from name --resume
beamsize=${5:-1}
python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port=23456 \
                ez_run.py \
                --prefix l2r \
                --mode ${mode} \
                --data_prefix "/private/home/jgu/data/" \
                --dataset "wmt16" \
                --src "ro" --trg "en" \
                --train_set "train.bpe.l2r" \
                --dev_set   "dev.bpe"   \
                --vocab_file "ro-en/vocab.ins.pt" \
                --load_lazy \
                --base "bpe" \
                --workspace_prefix "/checkpoint/jgu/space/${jbname}/" \
                --eval_every 500  \
                --save_every 2000 \
                --batch_size 2000 \
                --inter_size 1 \
                --sub_inter_size 1 \
                --label_smooth 0.1 \
                --lr 0.0005 \
                --weight_decay 0.0001 \
                --share_embeddings \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --load_from ${load_from} \
                --length_ratio 2 \
                --beam_size ${beamsize} \
                --relative_pos \
                --insertable --order fixed \
                --model TransformerIns \
                --debug
                # --debug --no_valid
            
                #--resume --load_from 10.22_06.52.36..wmt16_customize_ro_en_rp_bpe_0.1_32000_M1 \
                #--debug --no_valid
                #--debug --no_valid
                #--relative_pos \
                #--debug
                # --debug
                #--load_from "09.24_02.09.20..kftt_t2t-base_ja_en_byte_0.1_131080_M1" --resume
                
                #--debug

