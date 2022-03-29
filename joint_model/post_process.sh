#!/usr/bin/env bash

RESULTDIR="$1"
PARTITION="$2"

grep ^T ${RESULTDIR}/generate-${PARTITION}.txt | cut -f2- > ${RESULTDIR}/target.txt
grep ^H ${RESULTDIR}/generate-${PARTITION}.txt | cut -f3- > ${RESULTDIR}/hypotheses.txt
grep ^T ${RESULTDIR}/generate-${PARTITION}.txt | cut -f1 > ${RESULTDIR}/order.txt