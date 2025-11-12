#!/usr/bin/env bash

EPOCH_NUM=600
for epoch in `seq 1 $EPOCH_NUM`
do
    CUDA_VISIBLE_DEVICES=0 python eval_metric_epoch.py --cfg config/hot-resnet50dilated-c1.yaml --epoch ${epoch}
done