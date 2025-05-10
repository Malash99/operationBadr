#!/bin/bash

# Configuration
SEQUENCE_LENGTH=2
TRAIN_CAMERAS="0 1 2 3"
TEST_CAMERA=4
MODEL_NAME="flownet_regressor"

echo "=== Starting Pipeline ==="

# Data Extraction
echo "Extracting images from ROS bag..."
python3 src/data_processing/rosbag_parser.py \
    --input data/ariel_2023-12-21-14-26-32_2.bag \
    --output data/sequences_${SEQUENCE_LENGTH} \
    --sequence-length ${SEQUENCE_LENGTH}

# Training
echo "Training model..."
python3 src/training/train.py \
    --data-dir data/sequences_${SEQUENCE_LENGTH} \
    --cameras ${TRAIN_CAMERAS} \
    --sequence-length ${SEQUENCE_LENGTH} \
    --model-output output/models/${MODEL_NAME}_sl${SEQUENCE_LENGTH}.pt

# Evaluation
echo "Evaluating on test camera..."
python3 src/evaluation/evaluate.py \
    --model-path output/models/${MODEL_NAME}_sl${SEQUENCE_LENGTH}.pt \
    --data-dir data/sequences_${SEQUENCE_LENGTH}/cam${TEST_CAMERA} \
    --sequence-length ${SEQUENCE_LENGTH}

echo "=== Pipeline Complete ==="