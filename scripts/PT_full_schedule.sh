#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
MODEL_VERSION="vicuna-7b-v1.3"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################
PT_len_llm=10
PT_len_vision_encoder=10
learning_rate=7e-4
max_len=1024
tune_mm_mlp_adapter=true
tune_lm_header=true

deepspeed M2PT/train/train_memPT.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path ./models/vicuna-7b-v1.3 \
    --version $PROMPT_VERSION \
    --data_path ./playground/Vision-Flan/annotation_191-task_1k.json \
    --image_folder ./playground/Vision-Flan/images_191task_1k \
    --vision_tower ./models/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./models/llava-pretrain-vicuna-7b-v1.3/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/PT_LLM$PT_len_llm-VisEncoder$PT_len_vision_encoder-$lr-maxlen$max_len-TuneProj$tune_mm_mlp_adapter-TuneLMHead$tune_lm_header \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $max_len \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --PT_len_llm $PT_len_llm \
    --PT_len_vision_encoder $PT_len_vision_encoder \
    --tune_mm_mlp_adapter $tune_mm_mlp_adapter \
    --tune_lm_header $tune_lm_header \
    --load_best_model_at_end True \
    --metric_for_best_model "train_loss" \
    --greater_is_better false \
    --report_to tensorboard
