#!/bin/bash

DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101' 'bach' 'brain' 'eyedr'
SEED=$2
SHOTS=$3
BATCH_SIZE=$4
LR=$5
KD_WEIGHT=$6
REDUCTION=$7
RESIDUAL_RATIO=$8

CUDA_VISIBLE_DEVICES=0 python train.py \
    --root DATA \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/mtkd.yaml \
    --output-dir output/${DATASET}/shots_${SHOTS}/MTKD/seed_${SEED} \
    --kd-weight ${KD_WEIGHT} \
    --num-shots ${SHOTS} \
    --reduction ${REDUCTION} \
    --residual-ratio ${RESIDUAL_RATIO} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --second-phase \
