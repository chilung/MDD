## usage: bash scripts/train.sh -g 1 -s dslr -t amazon
#!/bin/bash

while getopts g:s:t: flag
do
    case "${flag}" in
        g) gpu=${OPTARG};;
        s) src=${OPTARG};;
        t) tgt=${OPTARG};;
    esac
done

export CUDA_VISIBLE_DEVICES=$gpu

PROJ_ROOT=$HOME'/MDD'
PROJ_NAME="MDD"
LOG_FILE="${PROJ_ROOT}/log/${PROJ_NAME}-`date +'%Y-%m-%d-%H-%M-%S'`.log"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python ${PROJ_ROOT}/trainer/train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset Office-31 \
    --src_address $HOME/MDD/data/$src.txt \
    --tgt_address $HOME/MDD/data/$tgt.txt \
#     >> ${LOG_FILE}  2>&1
