#!/bin/bash

module load anaconda/2020.11
source activate fe

# train
# bash ./scripts/train.sh

# test
#bash ./scripts/test.sh

# inference
bash ./scripts/inference.sh
