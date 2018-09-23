script=${1}
queue=${2:-dev}
jname="TransformerB"

srun --job-name=${jname} \
    --gres=gpu:8 -c 48 -C volta \
    --partition=${queue} \
    --time=24:00:00 --mem 256GB --pty \
    bash ${script} 8 ${jname}