#!/bin/bash

# custom config
DATA=/root/prompt_dataset
CFG=vit_b16_ep20_bs4_lr35
SHOTS=16

TRAINER=DePT
NCTX=4
DATASET=$1 # caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101 imagenet
WEIGHT=0.7

for SEED in 1 2 3 4 5
do
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/CTX_${NCTX}_epo_${EPO}/seed${SEED}

        CUDA_VISIBLE_DEVICES=0 python3 train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES base \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.LINEAR_PROBE.WEIGHT ${WEIGHT} 
done
