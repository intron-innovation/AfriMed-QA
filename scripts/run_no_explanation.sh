#!/bin/bash

declare -A data_paths
data_paths["medqa"]="data/MedQA-USMLE-4-options-test.csv"
data_paths["afrimed-qa-v1"]="data/afri_med_qa_10k_v1_1_phase_1.csv"
data_paths["afrimed-qa-v2"]="data/afri_med_qa_24k_v2.3_phase_2_24348_expert.csv"

declare -A model_paths
model_paths["microsoft-phi"]="microsoft/Phi-3-medium-128k-instruct"
model_paths["meta-llama"]="meta-llama/Meta-Llama-3-8B"
model_paths["meta-llama-405b"]="meta/llama3-405b-instruct-maas"

source="afrimed-qa-v2"
pretrained_model_choice="meta-llama-405b"

# Retrieve paths 
data_path=${data_paths[$source]}
pretrained_model_path=${model_paths[$pretrained_model_choice]}
question_types=("mcq")
prompt_type="base"
explanation=False
num_few_shot_values=(0)

for q_type in "${question_types[@]}"; do
    prompt_file_path="prompts/${prompt_type}_mcq_no_exp.txt"
    echo $prompt_file_path
    for num_few_shot in "${num_few_shot_values[@]}"; do
        python bin/main_predictions.py \
            --data_path "$data_path" \
            --prompt_file_path "$prompt_file_path" \
            --explanation "$explanation" \
            --pretrained_model_path "$pretrained_model_path" \
            --source "$source" \
            --q_type "$q_type" \
            --num_few_shot "$num_few_shot"
    done
done
