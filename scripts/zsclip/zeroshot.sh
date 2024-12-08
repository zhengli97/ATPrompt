#!/bin/bash

# custom config
DATA=/path/to/datasets

DATASET=oxford_flowers
TRAINER=ZeroshotCLIP
CFG=vit_b16  # rn50, rn101, vit_b32 or vit_b16
DIR=output/${TRAINER}/${CFG}/${DATASET}/test

CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml \
    --output-dir ${DIR} \
    --eval-only \
    DATASET.SUBSAMPLE_CLASSES all 
