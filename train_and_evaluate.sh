#!/bin/bash


PYTHON=python3  

rm -rf ./model/
rm -rf ./Output/

# Ejecutar los scripts
echo "Starting training..."
$PYTHON train_model.py -d ./Challenge_Data/ -m ./model/
echo "Running on test set..."
$PYTHON run_model.py -d ./Test_Data/ -m ./model/ -o ./Output/
echo "Starting evaluation of the model..."
$PYTHON evaluate_model.py -d ./Test_Data/ -o ./Output/
