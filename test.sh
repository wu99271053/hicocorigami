
#!/bin/bash

# Array of itypes
itypes=("Combined" "Tandem+" "Tandem-" "Inward" "Outward")

# Array of timesteps
timesteps=(0 4 8 15 30 60)

# Loop over each itype
for itype in "${itypes[@]}"
do
    # Loop over each val_chr value
    for val_chr in {1..16}
    do
        # Inner loop for timesteps for inference
        for timestep in "${timesteps[@]}"
        do
            # Execute the Python inference script with given parameters
            python hicocorigami/train.py --window=16 --itype=${itype} --data-rootdir=processed --val_chr=${val_chr} --save_path=
        done

        # After all timesteps are processed, execute the plotting script
        python inferencehicocorigami/inference_plot.py --val_chr=${val_chr} --itype=${itype} --data_root=inference_processed/ --window=64 --gaussian
    done
done
