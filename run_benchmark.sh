#!/bin/bash

for exp in 2 3 4; do
    rollouts=$((2 ** exp))
    echo "Running benchmark with rollouts=$rollouts"
    uv run benchmark.py --rollouts $rollouts --model "7-B" --output_path "logs/benchmark/7B/"
done
