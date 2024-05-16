#!/bin/sh

pretrained_model_path=microsoft/Phi-3-mini-128k-instruct
data_path="data/afri_med_qa_10k_v1_1_phase_1.csv"
prompt_file_path="prompts/base_mcq.txt"
python bin/main_predictions.py \
    --data_path $data_path \
    --prompt_file_path $prompt_file_path \
    --pretrained_model_path ${pretrained_model_path} \
    --q_type mcq

python bin/main_predictions.py \
    --data_path $data_path \
    --prompt_file_path  $prompt_file_path\
    --pretrained_model_path ${pretrained_model_path} \
    --q_type saq

python bin/main_predictions.py \
    --data_path $data_path \
    --prompt_file_path $prompt_file_path\
    --pretrained_model_path ${pretrained_model_path} \
    --q_type consumer_queries