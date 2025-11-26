#!/bin/bash

# custom config
DATA=/root/prompt_dataset
TRAINER=MaPLe_ATP
CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16

DATASET=$1 # caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101
NCTX=4

for EPO in 3
do
for SEED in 1 2 3 4 5
do
        DIR=output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots_few_shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/CTX_${NCTX}_epo_${EPO}/seed${SEED}

        CUDA_VISIBLE_DEVICES=0 python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/MaPLe/${CFG}.yaml \
                --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
                --model-dir ${DIR} \
                --load-epoch ${EPO} \
                --eval-only \
                DATASET.SUBSAMPLE_CLASSES all \
                TRAINER.ATPROMPT.USE_ATPROMPT True \
                TRAINER.ATPROMPT.N_ATT1 ${NCTX} \
                TRAINER.ATPROMPT.N_ATT2 ${NCTX} \
                TRAINER.ATPROMPT.N_ATT3 ${NCTX} \
                OPTIM.MAX_EPOCH ${EPO}
done
done
