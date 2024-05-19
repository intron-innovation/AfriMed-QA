#!/bin/bash

pretrained_model_path="johnsnowlabs/JSL-MedLlama-3-8B-v2.0"
data_path="data/afri_med_qa_10k_v1_1_phase_1.csv"

question_types=("mcq" "saq" "consumer_queries")
prompt_type=base

num_few_shot_values=(0)

for q_type in "${question_types[@]}"; do
    prompt_file_path="prompts/${prompt_type}_${q_type}.txt"
    echo $prompt_file_path
    for num_few_shot in "${num_few_shot_values[@]}"; do
        python bin/main_predictions.py \
            --data_path $data_path \
            --prompt_file_path $prompt_file_path \
            --pretrained_model_path $pretrained_model_path \
            --q_type $q_type \
            --num_few_shot $num_few_shot
    done
done
