set -x
script=${1}
queue=${2:-dev}
mode=${3:-train}
nodes=${4:-2}
gpus=${5:-2}
jname=${6:-TransformerB}
load_from=${7:-}
hour=${8:-24}

world_size=$((${gpus} * ${nodes}))

srun --job-name=${jname} \
    --gres=gpu:${gpus} --nodes=${nodes} --cpus-per-task 48 -C volta \
    --partition=${queue} --comment "Deadline for ICML 2019" \
    --time=${hour}:00:00 --mem 128GB \
    bash ${script} ${gpus} ${jname} ${mode} ${load_from} 
