import os
import logging
from typing import List, Tuple
from tqdm import tqdm
import torch
import pandas as pd
from src.utils.utils import logging_cuda_memory_usage, read_txt_file


logger = logging.getLogger(__name__)


def mcq_inference(args, row, model, **kwargs):

    sys_msg = "The following are multiple choice questions (MCQs)."

    sys_msg += """ You should directly answer the question by choosing the correct option and then provide a rationale for your answer. 
                """

    if args.prompt_file_path != None:
        user_msg = read_txt_file(args.prompt_file_path)
        sys_msg += "\n\n" + user_msg
    row = row[
        ~row.isna()
    ]  # this line drops the empty options if there are just 2 valid options like true/false
    question = row["question"]
    options = dict(row.drop(["question", "sample_id", "answer", "rationale"]))
    formatted_options = ""
    for key, value in options.items():
        formatted_options += f"{key}.  {value}\n"

    option_ids = list(row.drop(["sample_id", "answer", "rationale", "question"]).keys())
    input_text = sys_msg + "\n\n"
    few_shot_samples = []
    if args.num_few_shot > 0:
        for s in few_shot_samples[: args.num_few_shot]:
            input_text += s + "\n\n"
    input_text += "Question: " + question + "\n\n"
    input_text += formatted_options

    pred = model.predict(input_text)
    return pred


def saq_inference(args, row, model, **kwargs):
    sys_msg = "The following are short answer questions(SAQs)."

    sys_msg += " You should directly answer the question by providing a short answer and then provide a rationale for your answer."

    if args.prompt_file_path != None:
        user_msg = read_txt_file(args.prompt_file_path)
        sys_msg += "\n\n" + user_msg
    question = row["question"]

    input_text = sys_msg + "\n\n"
    few_shot_samples = []
    if args.num_few_shot > 0:
        for s in few_shot_samples[: args.num_few_shot]:
            input_text += s + "\n\n"
    input_text += "Question: " + question 

    
    pred = model.predict(input_text)
    return pred


def consumer_queries_inference(args, row, model, **kwargs):
    sys_msg = "The following are open-ended question."

    sys_msg += " You should directly answer the question freely"

    if args.prompt_file_path != None:
        user_msg = read_txt_file(args.prompt_file_path)
        sys_msg += "\n\n" + user_msg

    prompt = row["prompt"]
    question = row["question"]
    input_text = sys_msg + "\n\n"
    few_shot_samples = []
    if args.num_few_shot > 0:
        for s in few_shot_samples[: args.num_few_shot]:
            input_text += s + "\n\n"

    input_text += "Question: " + prompt + "\n\n"
    input_text += question

    pred = model.predict(input_text)
    return pred


def infer(args, row, model, **kwargs):
    if args.q_type == "mcq":
        pred = mcq_inference(args, row, model, **kwargs)
        return pred
    elif args.q_type == "saq":
        pred = saq_inference(args, row, model, **kwargs)
        return pred
    elif args.q_type == "consumer_queries":
        pred = consumer_queries_inference(args, row, model, **kwargs)
        return pred


def run_inference(args, model, data) -> Tuple[List[str], List[str]]:
    outputs = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Running Inference"):
        output = infer(args, row, model)
        outputs.append(output)
    logging_cuda_memory_usage()
    torch.cuda.empty_cache()

    return outputs
