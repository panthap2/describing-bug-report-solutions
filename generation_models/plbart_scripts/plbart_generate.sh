#!/usr/bin/env bash

langs=java,python,en_XX
SOURCE=source
TARGET=target
DATADIR="$1"
SAVEDIR="$2"
RESULTDIR="$3"
USER_DIR="$4"

fairseq-generate ${DATADIR} \
    --user-dir $USER_DIR \
    --path ${SAVEDIR}/checkpoint_best.pt \
    --task translation_in_same_language \
    --gen-subset test \
    -t $TARGET -s $SOURCE \
    --scoring sacrebleu --remove-bpe 'sentencepiece' \
    --max-len-b 200 --beam 5 \
    --batch-size 4 --langs $langs \
    --results-path ${RESULTDIR}

grep ^T ${RESULTDIR}/generate-test.txt | cut -f2- > ${RESULTDIR}/target.txt
grep ^H ${RESULTDIR}/generate-test.txt | cut -f3- > ${RESULTDIR}/hypotheses.txt
grep ^T ${RESULTDIR}/generate-test.txt | cut -f1 > ${RESULTDIR}/order.txt