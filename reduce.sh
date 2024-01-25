#!/bin/bash

# Define the arrays for each parameter
types=("Combined" "Inward" "Outward" "Tandem+" "Tandem-")
processes=("gaussian" "notransform")
windows=(16 64 32)
valchrs=$(seq 1 16) # Creates a sequence from 1 to 16

# Loop over each combination
for type in "${types[@]}"; do
    for process in "${processes[@]}"; do
        for window in "${windows[@]}"; do
            for valchr in $valchrs; do
                dir="$type/$process/$window/checkpoint_$valchr/result"
                echo "Processing directory: $dir"
                python your_python_script.py --dir "$dir"
            done
        done
    done
done
