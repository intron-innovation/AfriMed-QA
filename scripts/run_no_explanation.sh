#!/bin/bash

pretrained_model_path="microsoft/Phi-3-medium-128k-instruct"
#pretrained_model_path="meta-llama/Meta-Llama-3-8B"

#data_path="data/MedQA-USMLE-4-options-test.csv" #medqa
#data_path="data/afri_med_qa_10k_v1_1_phase_1.csv" #afrimed-qa
data_path="data/afri_med_qa_24k_v2.3_phase_2_24348_expert.csv"
question_types=("mcq")
prompt_type=base
explanation=False

num_few_shot_values=(0)

for q_type in "${question_types[@]}"; do
    prompt_file_path="prompts/base_mcq_no_exp.txt"
    echo $prompt_file_path
    for num_few_shot in "${num_few_shot_values[@]}"; do
        python bin/main_predictions.py \
            --data_path $data_path \
            --prompt_file_path $prompt_file_path \
            --explanation $explanation \
            --pretrained_model_path $pretrained_model_path \
            --q_type $q_type \
            --num_few_shot $num_few_shot
    done
done
