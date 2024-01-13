#!/bin/bash

# Array of itypes
itypes=("Tandem+" "Tandem-" "Outward" "Inward" "Combined")

# Array of window sizes
window_sizes=(16 32 64)

# Loop over each itype
for itype in "${itypes[@]}"
do
    # Loop over each window size
    for window in "${window_sizes[@]}"
    do
        # Loop over val_chr values
        for val_chr in {1..16}
        do
            # Execute the Python script with given parameters
            python Documents/hicocorigami/plot.py --val_chr=${val_chr} --itype=${itype} --data_root=Desktop/results --window=${window}

            # Remove the specified directory
        done
    done
done
