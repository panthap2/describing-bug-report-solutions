#!/usr/bin/env bash

PREFIX="$1"
SPM="$2"
MODEL="$3"
PRETRAIN="$4"

${SPM} --model=${MODEL} < ${PREFIX}.source > ${PREFIX}.spm.source &
${SPM} --model=${MODEL} < ${PREFIX}.target > ${PREFIX}.spm.target &