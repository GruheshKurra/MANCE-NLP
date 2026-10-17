#!/usr/bin/env bash
# Every experiment in the results table. Runs JOBS trainings at once on one GPU.
#   JOBS=4 scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiments/logs

{
  for seed in 1 2 3; do
    echo "configs/ag_news.yaml --seed $seed"
    echo "configs/dbpedia.yaml --seed $seed"
    echo "configs/germeval14.yaml --seed $seed"
  done
  echo "configs/ag_news.yaml --seed 1 --tag word_baseline --set model.embedding=word"
  echo "configs/ag_news.yaml --seed 1 --tag avg_chars --set model.char_pool=mean"
  echo "configs/ag_news.yaml --seed 1 --tag no_char_order --set model.char_positions=false"
  echo "configs/dbpedia.yaml --seed 1 --tag word_baseline --set model.embedding=word"
  echo "configs/germeval14.yaml --seed 1 --tag word_baseline --set model.embedding=word"
} | xargs -P "${JOBS:-4}" -L 1 sh -c \
  'log="experiments/logs/$(echo "$*" | tr " /=" "___").log"; python run.py train "$@" > "$log" 2>&1 || echo "failed: $*"' _

python run.py table
