#!/bin/bash

# custom config
DATA=/root/prompt_dataset
TRAINER=MaPLe
CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16

DATASET=imagenet

for NCTX in 2 4
do
for EPO in 3
do
for SEED in 1 2 3 4 5
do
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_few_shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/CTX_${NCTX}_epo_${EPO}/seed${SEED}

        CUDA_VISIBLE_DEVICES=0 python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/MaPLe/${CFG}.yaml \
                --output-dir ${DIR} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES all \
                OPTIM.MAX_EPOCH ${EPO}
done
done
done
