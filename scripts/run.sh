#!/bin/bash
export PYTHONPATH=.
declare -A data_paths
data_paths["medqa"]="data/MedQA-USMLE-4-options-test.csv"
data_paths["afrimed-qa-v1"]="data/afri_med_qa_10k_v1_1_phase_1.csv"
data_paths["afrimed-qa-v2"]="data/afri_med_qa_24k_v2.3_phase_2_24348_expert.csv"
data_paths["afrimed-qa-v2.5"]="data/afri_med_qa_15k_v2.5_phase_2_15275.csv"


declare -A model_paths
model_paths["microsoft-phi-med128"]="microsoft/Phi-3-medium-128k-instruct"
# model_paths["microsoft-phi-mini128"]="microsoft/Phi-3-mini-128k-instruct"
# model_paths["meta-llama"]="meta-llama/Meta-Llama-3-8B"
# model_paths["meta-llama-405b"]="meta/llama3-405b-instruct-maas"
# model_paths["claude-opus"]="claude-3-opus-20240229"

source="afrimed-qa-v2.5"
pretrained_model_choice="claude-opus"

# Retrieve paths 
data_path=${data_paths[$source]}
pretrained_model_path=${model_paths[$pretrained_model_choice]}
question_types=( "saq")
prompt_type="base"
explanation=True
num_few_shot_values=(0)

for model_key in "${!model_paths[@]}"; do
    pretrained_model_path=${model_paths[$model_key]}
    data_path=${data_paths[$source]}
    
    echo "Running model: $model_key"
    echo "Using data: $data_path"
    
    # Loop through question types and few-shot values
    for q_type in "${question_types[@]}"; do
        prompt_file_path="prompts/${prompt_type}_${q_type}.txt"
        echo "Prompt file: $prompt_file_path"
        
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
done
