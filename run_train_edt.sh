#!/bin/bash

# train edt with different datset size
train_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
for train_ratio in "${train_ratios[@]}"
do
    /bin/python3.6 train.py --cf ./ConfigMemb/train_edt_discrete.txt --train_ratio $train_ratio || break
done