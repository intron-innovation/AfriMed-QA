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

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--prompt_file_path", type=str, default=None)
    parser.add_argument("--explanation", type=str, default=True)
    parser.add_argument(
        "--q_type", type=str, required=True, help="eval tasks or question group type"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="The source of the data to be evaluated",
    )
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_few_shot", type=int, default=0)

    args = parser.parse_args()
    args.model_name = args.pretrained_model_path.split("/")[-1]

    args.explanation = False if args.explanation == "False" else True

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
    pynvml.nvmlInit()
    logger.info(f"cuda is available {torch.cuda.is_available()}")
    logger.info(f"cuda device count {torch.cuda.device_count()}")
    logger.info(f"cuda device name {torch.cuda.get_device_name()}")

    logger.info("******** Memory usage ********")
    n_gpus = pynvml.nvmlDeviceGetCount()
    for i in range(n_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info(
            "GPU {}: {:.2f} GB / {:.2f} GB".format(
                i, meminfo.used / 1024**3, meminfo.total / 1024**3
            )
        )


def write_results(data, args, score):
    prompt_type = args.prompt_file_path.split("/")[-1].split("_")[0]
    q_type = args.q_type.split("_")[0]
    explanation = explanation = "_no_exp" if not args.explanation else ""
    model_dir = args.model_name.replace("/", "_")
    os.makedirs(f"results/{model_dir}", exist_ok=True)
    results_fname = f"results/{model_dir}/{model_dir}_{q_type}_{prompt_type}-prompt_{explanation}_{args.num_few_shot}-shot_score_{score:.4f}_{len(data)}.csv"
    data.to_csv(results_fname, index=False)
    logger.info(f"Results saved to: {results_fname}")
    return results_fname


def post_process_output(model_output: str) -> str:
    cleaned_output = [text[0] for text in model_output]
    return cleaned_output


def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content
