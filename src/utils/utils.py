import os
import numpy as np
import logging
import torch
import argparse
from bert_score import score as bert_score
from rouge import Rouge

import gc
import pynvml
pynvml.nvmlInit()

logger = logging.getLogger(__name__)

def parse_arguments():
    logger.info(f"cuda is available {torch.cuda.is_available()}")
    logger.info(f"cuda device count {torch.cuda.device_count()}")
    logger.info(f"cuda device name {torch.cuda.get_device_name()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--prompt_file_path", type=str, default=None)

    parser.add_argument(
        "--q_type", type=str, required=True, help="eval tasks or question group type"
    )

    args = parser.parse_args()

    args.model_name = args.pretrained_model_path.split("/")[-1]
    args.num_few_shot = 0

    return args


def compute_score(args, data):
    if args.q_type == "mcq":
        data["correct"] = data["answer"] == data["preds"]
        score = data["correct"].mean()
        return data, score
    elif args.q_type == "saq":
        p, r, f1 = bert_score(data["preds"], data["answer"], lang="en", verbose=True)
        data["BERTScore_Precision"] = p.numpy()
        data["BERTScore_Recall"] = r.numpy()
        data["BERTScore_F1"] = f1.numpy()

        rouge = Rouge()
        rouge_scores = data.apply(
            lambda row: rouge.get_scores(row["pred"], row["rationale"]), axis=1
        )
        data["ROUGE-1"] = [score[0]["rouge-1"]["f"] for score in rouge_scores]
        data["ROUGE-2"] = [score[0]["rouge-2"]["f"] for score in rouge_scores]
        data["ROUGE-L"] = [score[0]["rouge-l"]["f"] for score in rouge_scores]

        score = data["ROUGE-1"].mean()
        return data, score
    else:
        score = ""
        return data, score


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

def logging_cuda_memory_usage():
    logger.info("******** Memory usage ********")
    n_gpus = pynvml.nvmlDeviceGetCount()
    for i in range(n_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info(
            "GPU {}: {:.2f} GB / {:.2f} GB".format(
                i, meminfo.used / 1024 ** 3, meminfo.total / 1024 ** 3
            )
        )

def write_results(data, args, score):
    file_name = os.path.basename(args.data_path)
    results_fname = f"results/{file_name.split('.csv')[0]}_{args.model_name.replace('/', '_')}_{args.q_type}_score-{score}_{len(data)}.csv"
    data.to_csv(results_fname, index=False)
    logger.info(f"Results saved to: {results_fname}")
    return results_fname

def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content
