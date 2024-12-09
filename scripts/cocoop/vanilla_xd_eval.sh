#!/bin/bash

# custom config
DATA=/root/prompt_dataset
TRAINER=CoCoOp

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16
NCTX=2
EPO=5

# imagenetv2 imagenet_a imagenet_r imagenet_sketch
DATASET=$1 # caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101

for SEED in 1 2 3 4 5
do
        DIR=output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots_few_shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/CTX_${NCTX}_epo_${EPO}/seed${SEED}

        CUDA_VISIBLE_DEVICES=0 python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/CoCoOp/${CFG}.yaml \
                --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
                --model-dir ${DIR} \
                --load-epoch ${EPO} \
                --eval-only \
                DATASET.SUBSAMPLE_CLASSES all \
                TRAINER.COCOOP.N_CTX ${NCTX} \
                DATASET.NUM_SHOTS ${SHOTS} \
                OPTIM.MAX_EPOCH ${EPO}
done
