
#!/usr/bin/env bash

SPM="$1"
MODEL="$2"
TRAIN_PREFIX="$3"
VALID_PREFIX="$4"
TEST_PREFIX="$5"
DEST_DIR="$6"
DICT="$7"

${SPM} --model=${MODEL} < ${TRAIN_PREFIX}.source > ${TRAIN_PREFIX}.spm.source &
${SPM} --model=${MODEL} < ${TRAIN_PREFIX}.target > ${TRAIN_PREFIX}.spm.target &

${SPM} --model=${MODEL} < ${VALID_PREFIX}.source > ${VALID_PREFIX}.spm.source &
${SPM} --model=${MODEL} < ${VALID_PREFIX}.target > ${VALID_PREFIX}.spm.target &

${SPM} --model=${MODEL} < ${TEST_PREFIX}.source > ${TEST_PREFIX}.spm.source &
${SPM} --model=${MODEL} < ${TEST_PREFIX}.target > ${TEST_PREFIX}.spm.target &


fairseq-preprocess \
  --source-lang source \
  --target-lang target \
  --trainpref ${TRAIN_PREFIX}.spm  \
  --validpref ${VALID_PREFIX}.spm \
  --testpref ${TEST_PREFIX}.spm \
  --destdir ${DEST_DIR} \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 70