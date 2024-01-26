#!/bin/bash

# Loop over chromosomes 1 to 16
for chr in {1..16}; do
    echo "Training on chromosome: $chr"
    
    # Run the train.py script with the current chromosome
    python hicocorigami/train.py --window=16 --itype=Combined --data-root=processed/gaussian --save_path=Combined/gaussian/16/checkpoint_$chr
    python hicocorigami/test.py --window=16 --val_chr=$chr --itype=Combined --data_root=processed --gaussian
done


