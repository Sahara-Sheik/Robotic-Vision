#!/bin/bash

# Define the notebook file and output directory
NOTEBOOK="/AnonymousProject/src/visual_proprioception/Train_VisualProprioception.ipynb"
OUTPUT_DIR="output_runs"
mkdir -p $OUTPUT_DIR
export PYTHONPATH="/AnonymousProject/src/:$PYTHONPATH"


# List of different values for the 'run' variable

RUN_VALUES=("vit_large" "vit_base" "vp_aruco_128" "vp_convvae_128" "vp_ptun_vgg19_128" "vp_ptun_resnet50_128" "vp_convvae_256" "vp_ptun_vgg19_256" "vp_ptun_resnet50_256" "vp_aruco_128")

# Loop through each run value and execute the notebook
for RUN in "${RUN_VALUES[@]}"
do
    OUTPUT_FILE="$OUTPUT_DIR/output_${RUN}.ipynb"

    echo "Running notebook with run=${RUN}"

    papermill $NOTEBOOK $OUTPUT_FILE -p run "$RUN" &
done

# Wait for all background jobs to finish
wait

echo "All notebooks executed!"
