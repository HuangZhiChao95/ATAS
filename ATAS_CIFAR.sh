#!/usr/bin/env bash

BATCH_SIZE=128
WD=5e-4
LR=0.1

DATASET=$1

EPOCHS=30
DECAY_STEPS="24 28"
EPSILON=$2

C=0.01
MAX_STEP_SIZE=14
MIN_STEP_SIZE=4

ARCH=PreActResNet18
MODEL_DIR=${DATASET}_${ARCH}_eps${EPSILON}_ATAS
python3 -u ATAS.py \
  --dataset ${DATASET}\
  --batch-size ${BATCH_SIZE}\
  --epochs ${EPOCHS}\
  --wd ${WD} \
  --lr ${LR}\
  --arch ${ARCH}\
  --decay-steps ${DECAY_STEPS}\
  --epsilon ${EPSILON}\
  --max-step-size ${MAX_STEP_SIZE} \
  --min-step-size ${MIN_STEP_SIZE} \
  --model-dir results/${MODEL_DIR}\
  --c ${C}\
  > log/${MODEL_DIR}.log 2>&1

LOG_NAME=log/attack_${MODEL_DIR}.log
MODEL_DIR=results/${MODEL_DIR}
python3 -u attack.py --dataset ${DATASET} --model-dir ${MODEL_DIR} --arch ${ARCH} --epsilon ${EPSILON} > ${LOG_NAME} 2>&1

ARCH=WideResNet
MODEL_DIR=${DATASET}_${ARCH}_eps${EPSILON}_ATAS
python3 -u ATAS.py \
  --dataset ${DATASET}\
  --batch-size ${BATCH_SIZE}\
  --epochs ${EPOCHS}\
  --wd ${WD} \
  --lr ${LR}\
  --arch ${ARCH}\
  --decay-steps ${DECAY_STEPS}\
  --epsilon ${EPSILON}\
  --max-step-size ${MAX_STEP_SIZE} \
  --min-step-size ${MIN_STEP_SIZE} \
  --model-dir results/${MODEL_DIR}\
  --c ${C}\
  > log/${MODEL_DIR}.log 2>&1

LOG_NAME=log/attack_${MODEL_DIR}.log
MODEL_DIR=results/${MODEL_DIR}
python3 -u attack.py --dataset ${DATASET} --model-dir ${MODEL_DIR} --arch ${ARCH} --epsilon ${EPSILON} > ${LOG_NAME} 2>&1