script=${1}
queue=${2:-dev}
mode=${3:-train}
jname=${4:-TransformerB}
load_from=${5:-}
hour=${6:-72}
gpus=${7:-8}

srun --job-name=${jname} \
    --gres=gpu:${gpus} -c 48 -C volta16gb -v \
    --partition=${queue} --comment "Deadline for ICML2019" \
    --time=${hour}:00:00 --mem 128GB --pty \
    bash ${script} ${gpus} ${jname} ${mode} ${load_from}
    
