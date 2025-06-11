#!/bin/bash

declare -A data_paths
# data_paths["medqa"]="data/MedQA-USMLE-4-options-test.csv"
# data_paths["afrimed-qa-v1"]="data/afri_med_qa_10k_v1_1_phase_1.csv"
#data_paths["afrimed-qa-v2"]="data/afri_med_qa_24k_v2.3_phase_2_24348_expert.csv"
data_paths["afrimed-qa-v2.5"]="/home/chinemelu/AfriMed-QA/data/afri_med_qa_15k_v2.5_phase_2_15275.csv"
# data_paths["afrimed-qa-v2.5"]="/mnt/external_aka/data/AfriMed-QA/data/sample_10.csv"
#data_paths["test_data"]="data/sample_10.csv"


declare -A model_paths
# model_paths["microsoft-phi-med128"]="microsoft/Phi-3-medium-128k-instruct"
# model_paths["microsoft-phi-mini128"]="microsoft/Phi-3-mini-128k-instruct"
# model_paths["meta-llama"]="meta-llama/Meta-Llama-3-8B"
# model_paths["meta-llama-405b"]="meta/llama3-405b-instruct-maas"
#model_paths["claude-3-7-sonnett"]="claude-3-7-sonnet-20250219"
# model_paths["gpt4o"]="gpt-4o"
# model_paths["microsoft-phi-med128"]="microsoft/Phi-3-medium-128k-instruct"
# model_paths["microsoft-phi-mini128"]="microsoft/Phi-3-mini-128k-instruct"
# model_paths["meta-llama"]="meta-llama/Meta-Llama-3-8B"
#model_paths["Meta-Llama-3-70B-Instruct"]="meta-llama/Meta-Llama-3-70B-Instruct"
#model_paths["gemma-2-2b"]="gemma-2-2b"
# model_paths["gemma-3-27b-it"]="gemma-3-27b-it"
model_paths["google/medgemma-4b-it"]="google/medgemma-4b-it"
#model_paths["Meditron-70B"]="epfl-llm/meditron-70b"
#model_paths["gpt-o3"]="o3"
#model_paths["gemini-2.5-pro-preview-05-06"]="gemini-2.5-pro-preview-05-06"
#model_paths["gemini-2.5-flash-preview-04-17"]="gemini-2.5-flash-preview-04-17"
# model_paths["deepseek-R1"]="deepseek-R1"

#source="test_data"
source="afrimed-qa-v2.5"

#pretrained_model_choice="gemma-3-27b-it"
#source="afrimed-qa-v1"
pretrained_model_choice="google/medgemma-4b-it"
#pretrained_model_choice="gpt-o3"
#pretrained_model_choice="Meta-Llama-3-70B-Instruct"

# Retrieve paths 
data_path=${data_paths[$source]}
question_types=("mcq")
prompt_type="base"
explanation=False
num_few_shot_values=(0)

for model_key in "${!model_paths[@]}"; do
    pretrained_model_path=${model_paths[$model_key]}
    data_path=${data_paths[$source]}
    
    echo "Running model: $model_key"
    echo "Using data: $data_path"
    
    # Loop through question types and few-shot values
    for q_type in "${question_types[@]}"; do
        #prompt_file_path="prompts/${prompt_type}_mcq_no_exp.txt"
        prompt_file_path="prompts/${prompt_type}_mcq.txt"
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
