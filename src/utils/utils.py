
import os
import numpy as np
import random
import logging
import torch
import argparse
from bert_score import score as bert_score
from rouge import Rouge
import torch.nn.functional as F 

logger = logging.getLogger(__name__)

def parse_arguments():
    logger.info(f'cuda is available {torch.cuda.is_available()}')
    logger.info(f'cuda device count {torch.cuda.device_count()}')
    logger.info(f'cuda device name {torch.cuda.get_device_name()}')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--q_type", type=str, required=True,
                        help='eval tasks or question group type')
    
    args = parser.parse_args()

    args.model_name = args.pretrained_model_path.split('/')[-1]
    args.num_few_shot = 0

    return args

def mcq_eval(args, row, model, toker, **kwargs):
    sys_msg = 'The following are multiple choice questions (MCQs).'

    sys_msg += ' You should directly answer the question by choosing the correct option.'

    row = row[~row.isna()] # this line drops the empty options if there are just 2 valid options like true/false
    question = row['question']
    options = dict(row.drop(['question', 'sample_id', 'answer', 'rationale']))
    formatted_options = ""
    for key, value in options.items():
        formatted_options += f"{key}.  {value}\n"

    option_ids = list(row.drop(['sample_id', 'answer', 'rationale', 'question']).keys())
    input_text = sys_msg + '\n\n'
    few_shot_samples = []
    if args.num_few_shot > 0:
        for s in few_shot_samples[:args.num_few_shot]:
            input_text += s + '\n\n'
    input_text += "Question: " + question  + '\n\n'
    input_text += formatted_options
    

    input_ids = toker(input_text, return_tensors="pt").input_ids.to(model.device)
    input_ids = input_ids[..., -1536:]
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
        ).logits[:, -1].view(-1)

    option_indices = [toker(f': {e}').input_ids[-1] for e in option_ids] + \
            [toker(f':{e}').input_ids[-1] for e in option_ids]
    probs = F.softmax(
        logits[..., option_indices], dim=-1
    ).detach().cpu().to(torch.float32).numpy()
    probs = probs.reshape(2, len(option_ids)).sum(axis=0)
    pred = option_ids[np.argmax(probs)]
    return pred

    
def saq_eval(args, row, model, tokenizer, **kwargs):
    sys_msg = 'The following are short answer questions(SAQs).'

    sys_msg += ' You should directly answer the question by providing a short and consie response'

    prompt = row['prompt']
    question = row['question']
    
    input_text = sys_msg + '\n\n'
    few_shot_samples = []
    if args.num_few_shot > 0:
        for s in few_shot_samples[:args.num_few_shot]:
            input_text += s + '\n\n'
    input_text += "Question: "  + prompt + '\n\n'
    input_text +=  question 
    

    pred  = model(
        input_text,
        max_length=200,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )['generated_text']
    return pred

def consumer_queries_eval(args, row, model, tokenizer, **kwargs):
    sys_msg = 'The following are open-ended question.'

    sys_msg += ' You should directly answer the question freely'

    question = row['question']
    
    input_text = sys_msg + '\n\n'
    few_shot_samples = []
    if args.num_few_shot > 0:
        for s in few_shot_samples[:args.num_few_shot]:
            input_text += s + '\n\n'
    input_text += "Question: "  + question 
    input_text +=  question 

    pred  = model(
        input_text,
        max_length=200,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )['generated_text']
    return pred


def compute_score(args, data):
    if args.q_type == "mcq":
        data['correct'] = data['answer'] == data['preds']
        score = data['correct'].mean()
        return data, score
    else:
        p, r, f1 = bert_score(data['pred'], data['answer'], lang="en", verbose=True)
        data['BERTScore_Precision'] = p.numpy()
        data['BERTScore_Recall'] = r.numpy()
        data['BERTScore_F1'] = f1.numpy()

        rouge = Rouge()
        rouge_scores = data.apply(lambda row: rouge.get_scores(row['pred'], row['answer']), axis=1)
        data['ROUGE-1'] = [score[0]['rouge-1']['f'] for score in rouge_scores]
        data['ROUGE-2'] = [score[0]['rouge-2']['f'] for score in rouge_scores]
        data['ROUGE-L'] = [score[0]['rouge-l']['f'] for score in rouge_scores]
        
        score = data['ROUGE-1'].mean()
        return data, score

def get_bootstrap_accuracy_std(data, num_samples=1000):
    rng = random.Random(123)
    vals = [e['data']["correct"] for e in data if e['type'] == 'result']
    return np.std([np.mean(rng.sample(vals, len(vals) // 2)) for _ in range(num_samples)])


def _norm(x):
    return ' '.join(x.strip().split())


def chunklist(lst, n):
    avg = len(lst) // n
    remainder = len(lst) % n
    chunks = []
    start = 0
    for i in range(n):
        end = start + avg + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


def patch_open():
    import builtins
    import io

    prev_open = open

    def new_open(*args, **kwargs):
        buffer_size = kwargs.pop("buffering", io.DEFAULT_BUFFER_SIZE)
        kwargs["buffering"] = min(io.DEFAULT_BUFFER_SIZE, buffer_size)
        return prev_open(*args, **kwargs)

    builtins.open = new_open


def _purple(str: str) -> str:
    return f"\033[1;35m{str}\033[0m"
def _orange(str: str) -> str:
    return f"\033[1;31m{str}\033[0m"
def _blue(str: str) -> str:
    return f"\033[1;34m{str}\033[0m"