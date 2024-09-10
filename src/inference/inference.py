import os
import logging
from typing import List, Tuple
from tqdm import tqdm
import torch
import pandas as pd
import traceback

from src.utils.utils import logging_cuda_memory_usage


logger = logging.getLogger(__name__)


def infer(prompt, model, **kwargs):
    pred = model.predict(prompt)
    return pred


def run_inference(model, data, use_cuda=False) -> Tuple[List[str], List[str]]:
    outputs = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Running Inference"):
        try:
            output = infer(row["model_prompt"], model)
        except Exception:
            print(traceback.format_exc())
            output = ""
        outputs.append(output)
    if use_cuda:
        logging_cuda_memory_usage()
        torch.cuda.empty_cache()

    return outputs
