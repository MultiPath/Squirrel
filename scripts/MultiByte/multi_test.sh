gpus=${1:-2}
jbname=${2:-MultiByte}
mode=${3:-train}
load_from=${4:-none}  # --load_from name --resume
python -m torch.distributed.launch --nproc_per_node=${gpus} --master_port=23456 \
                ez_run.py \
                --prefix [time] \
                --mode train \
                --data_prefix "/private/home/jgu/data/" \
                --dataset "MultiUN" \
                --src "ar,es,fr,ru,zh" --trg "en,en,en,en,en" \
                --multi \
                --sample_prob 0.17574147 0.20493178 0.23570797 0.20753076 0.17608802 \
                --train_set "train" \
                --dev_set   "dev"   \
                --test_set  "test"  \
                --load_lazy \
                --base "byte" \
                --workspace_prefix "/private/home/jgu/space/${jbname}/" \
                --eval_every 1000  \
                --batch_size 3000 \
                --inter_size 5 \
                --label_smooth 0.1 \
                --tensorboard \
                --cross_attn_fashion "forward" \
                --load_from ${load_from} --resume \
                --maxlen 1000 \

                

