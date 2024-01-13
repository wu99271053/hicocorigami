#!/bin/bash

# Array of categories
categories=("Tandem+" "Tandem-" "Outward" "Inward" "Combined")

# Array of sizes
sizes=(16 32 64)

# Nested loops to iterate over categories and sizes
for category in "${categories[@]}"
do
    for size in "${sizes[@]}"
    do
        # Inner loop for checkpoints
        for j in {1..16}
        do
            # Create the local directory structure
            mkdir -p Desktop/results/${category}/${size}/${j}/

            # scp command to copy files from remote to local directory
            scp -r user@10.228.53.5:/home/user/yixuan/${category}/gaussian/${size}/checkpoint_${j}/result/ Desktop/results/${category}/${size}/${j}/
        done
    done
done

echo "Files copied successfully."

