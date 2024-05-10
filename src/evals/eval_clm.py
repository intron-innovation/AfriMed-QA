
import os
import logging
from tqdm import tqdm
from src.utils.utils import (
    parse_arguments,
    _orange, _blue, _purple,
    compute_score,
    patch_open,
    mcq_eval, 
    saq_eval,
    consumer_queries_eval
)
from src.utils.prepare_data import prep_data
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import gc

import pynvml
pynvml.nvmlInit()

logger = logging.getLogger(__name__)


def logging_cuda_memory_usage():
    logger.info("******** Memory usage ********")
    n_gpus = pynvml.nvmlDeviceGetCount()
    for i in range(n_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info("GPU {}: {:.2f} GB / {:.2f} GB".format(i, meminfo.used / 1024 ** 3, meminfo.total / 1024 ** 3))


def eval(args, row, model, toker, **kwargs):
    if args.q_type == "mcq":
        pred = mcq_eval(args, row, model, toker, **kwargs)
        return pred
    elif args.q_type == "saq":
        pred = saq_eval(args, row, model, toker, **kwargs)
        return pred
    elif args.q_type == "saq":
        pred = consumer_queries_eval(args, row, model, toker, **kwargs)
        return pred



def main():
    patch_open()

    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_path, use_fast=False,
        add_bos_token=False, add_eos_token=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        device_map='auto',
        use_safetensors=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    if args.q_type !="mcq":
        model = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    logging_cuda_memory_usage()
    logger.info(_blue(f"Preparing: data"))
    data = prep_data(args)
    results = []

    logger.info(_blue(f"Evaluation started: data"))
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Running eval") :
        result = eval(args, row, model, tokenizer)
        results.append(result)
    
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(_orange(f"Evaluation completed: {args.q_type}"))

    data['preds'] = results
    data, score = compute_score(args, data)
    file_name = os.path.basename(args.data_path)
    results_fname = f"results/{file_name.split('.csv')[0]}_{args.model_name.replace('/', '_')}_{args.q_type}_score-{score}_{len(data)}.csv"
    logger.info(f"Socre is {_blue(score)}")

    data.to_csv(results_fname, index=False)
    logger.info(f"Results saved to: {results_fname}")

    logging_cuda_memory_usage()




if __name__ == "__main__":
    main()