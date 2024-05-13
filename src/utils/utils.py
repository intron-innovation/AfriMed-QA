import os
import numpy as np
import logging
import torch
import argparse
import builtins
import io
import re


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

def patch_open():
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
    results_fname = f"results/{file_name.split('.csv')[0]}_{args.model_name.replace('/', '_')}_{args.q_type}_score-{score:.4f}_{len(data)}.csv"
    data.to_csv(results_fname, index=False)
    logger.info(f"Results saved to: {results_fname}")
    return results_fname

def post_process_output(model_output: str) -> str:
    cleaned_output = []
    for output in model_output:
        matched_pieces = re.findall(r"(?i)OPTION [ABCDE] IS CORRECT", output)
        if len(matched_pieces) == 0:  # no matched piece
            predicted_option = ""
        else:
            predicted_option = matched_pieces[0].split()[1]
        cleaned_output.append(predicted_option)
    return cleaned_output

def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content
