#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
    --model-path ./models/llava-pretrain-vicuna-7b-v1.3 \
    --model-base ./models/vicuna-7b-v1.3 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/stage1-llava_baseline.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment stage1-llava_baseline

cd eval_tool

python calculation.py --results_dir answers/stage1-llava_baseline
