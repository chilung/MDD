#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

PROJ_ROOT="/home/ccma/MDD"
PROJ_NAME="MDD"
LOG_FILE="${PROJ_ROOT}/log/${PROJ_NAME}-`date +'%Y-%m-%d-%H-%M-%S'`.log"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python ${PROJ_ROOT}/trainer/train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset Office-31 \
    --src_address /home/ccma/MDD/data/dslr.txt \
    --tgt_address /home/ccma/MDD/data/amazon.txt \
    >> ${LOG_FILE}  2>&1
