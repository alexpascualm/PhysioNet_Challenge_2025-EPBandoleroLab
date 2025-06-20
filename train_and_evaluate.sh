#!/bin/bash


PYTHON=python3  

rm -rf ./model/
rm -rf ./Output/

# Ejecutar los scripts
$PYTHON train_model.py -d ./Challenge_Data/ -m ./model/
$PYTHON run_model.py -d ./Test_Data/ -m ./model/ -o ./Output/
$PYTHON evaluate_model.py -d ./Test_Data/ -o ./Output/
