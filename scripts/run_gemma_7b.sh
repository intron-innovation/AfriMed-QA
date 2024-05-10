
pretrained_model_path=${HF_MODELS}unsloth/gemma-7b
data_path="data/afri_med_qa_10k_v1_1_phase_1.csv"
# python src/evals/eval_clm.py \
#     --data_path $data_path \
#     --pretrained_model_path ${pretrained_model_path} \
#     --q_type mcq

python src/evals/eval_clm.py \
    --data_path $data_path \
    --pretrained_model_path ${pretrained_model_path} \
    --q_type saq

python src/evals/eval_clm.py \
    --data_path $data_path \
    --pretrained_model_path ${pretrained_model_path} \
    --q_type comnsumer_queries