#!/bin/bash

python3 answer_gen.py --model_name "meta-llama/Meta-Llama-3-8B" --output_dir "outputs/llama-8b" --max_length 128 --dataset_path "../../dataset/annotations/dataset.csv" --prompt_name basic