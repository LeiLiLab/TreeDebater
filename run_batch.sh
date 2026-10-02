#!/usr/bin/env bash
set -euo pipefail

cd /home/danqingwang/workspace/clone/TreeDebater/src
source /home/danqingwang/anaconda3/etc/profile.d/conda.sh
conda activate debate

MAX_PARALLEL=2
CONFIG_NAME=overlap_debate_2_deepseek-v4-flash.yml
LOG_NAME=overlap_debate_2_deepseek-chat.log

for case_id in $(seq 3 15); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n 2>/dev/null || sleep 1
  done

  gpu=$((4 + (case_id - 1) % 2))
  log_dir="../logs/emnlp/case${case_id}"
  mkdir -p "$log_dir"

  echo "Starting case${case_id} on GPU ${gpu}"
  env CUDA_VISIBLE_DEVICES="${gpu}" python -m streaming.overlap \
    --config "configs/emnlp/case${case_id}/${CONFIG_NAME}" \
    < /dev/null > "${log_dir}/${LOG_NAME}" 2>&1 &
done

wait
echo "All cases 1-15 finished."
