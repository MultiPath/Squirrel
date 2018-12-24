queue=${1:-dev}
hour=${2:-24}

srun --job-name=${jname} \
    --gres=gpu:8 -c 48 -C volta16gb -v \
    --partition=${queue} --comment "Deadline for ICML 2019" \
    --time=${hour}:00:00 --mem 128GB --pty \
    bash
