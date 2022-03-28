#!/usr/bin/env bash

langs=java,python,en_XX
SOURCE=source
TARGET=target
PRETRAIN="$1"
DATADIR="$2"
SAVEDIR="$3"
OUTPUT_FILE="$4"
USER_DIR="$5"
BATCH_SIZE=16
UPDATE_FREQ=1

# Derived from https://github.com/wasiahmad/PLBART/blob/main/codeXglue/code_to_code/refin_run.sh

fairseq-train $DATADIR \
  --user-dir $USER_DIR \
  --langs $langs --task translation_in_same_language \
  --arch mbart_base --layernorm-embedding \
  --source-lang $SOURCE --target-lang $TARGET \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ --max-epoch 30 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 5e-05 --min-lr -1 \
  --warmup-updates 500 --max-update 100000 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
  --seed 1234 --log-format json --log-interval 100 \
  --restore-file $PRETRAIN --reset-dataloader \
  --reset-optimizer --reset-meters --reset-lr-scheduler \
  --eval-bleu --eval-bleu-detok space --eval-tokenized-bleu \
  --eval-bleu-remove-bpe sentencepiece --eval-bleu-args '{"beam": 5}' \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --no-epoch-checkpoints --patience 10 \
  --ddp-backend no_c10d --save-dir $SAVEDIR 2>&1 | tee ${OUTPUT_FILE}