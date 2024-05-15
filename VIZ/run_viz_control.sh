#!/bin/bash

# Loop 16 times and run aaa.py in each iteration
for i in {1..16}; do
    python viz_control.py --start_idx "$i"
done
