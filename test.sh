#!/bin/bash

# Loop over chromosomes 1 to 16
for chr in {1..16}; do
    echo "Training on chromosome: $chr"
    
    # Run the train.py script with the current chromosome
    python hicocorigami/reduce_size.py --dir=Tandem-/gaussian/16/checkpoint_$chr/result/csv/
    python hicocorigami/reduce_size.py --dir=Combined/gaussian/16/checkpoint_$chr/result/csv/
done


