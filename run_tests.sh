#!/bin/bash
mkdir results
mkdir results/hochuli
mkdir results/hochuli_deep
mkdir results/hochuli_double
mkdir results/mobilenet

device="cuda"

python3 hochuli_test.py < $device
python3 hochuli_deep_test.py < $device
python3 hochuli_double_test.py < $device
python3 mobilenet_test.py < $device
