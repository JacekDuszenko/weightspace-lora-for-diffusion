#!/bin/bash

accelerate launch --num_processes=1 multi_lora_train_dreambooth.py \
  --pretrained_model_name_or_path=stable-diffusion-v1-5/stable-diffusion-v1-5  \
  --instance_data_dir=test-opt-lora/ \
  --output_dir=out \
  --instance_prompt="a photo of sks $1" \
  --resolution=256 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --max_train_steps=200 \
  --checkpointing_steps=1000 \
  --pre_compute_text_embeddings \
  --tokenizer_max_length=77 \
  --num_validation_images=0 \
  --rank=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --class_label=$1 \
  --folder_to_crawl=data/weightspace-images/$1 # e.g. airplane

