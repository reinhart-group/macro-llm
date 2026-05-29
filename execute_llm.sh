#!/bin/bash

# Define the number of times to run the script
N=1

# morphologies=("liquid" "string" "membrane" "vesicle" "wormlike micelle" "spherical micelle")
morphologies=("spherical micelle")

for ((i=1; i<=N; i++)); do
    for MORPH in "${morphologies[@]}"; do
        python run_claude.py "$MORPH" "prompts/prompt-oracle-v4.4.yml" --gen_random --pad_random --nproc 5 --sonnet35
    done
done