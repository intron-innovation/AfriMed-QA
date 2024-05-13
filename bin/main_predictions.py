import os
import logging
import gc
from src.utils.utils import (
    parse_arguments,
    _orange, _blue, _purple,
    compute_score,
    patch_open
)
from src.utils.prepare_data import prep_data
import torch
from src.utils.utils import  write_results
from src.models.models import Ph3
from src.inference.inference import run_inference

logger = logging.getLogger(__name__)

def main():
    patch_open()

    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    
    args = parse_arguments()
    
    gc.collect()
    torch.cuda.empty_cache()

    data = prep_data(args)
    model = Ph3(args) # please define your own model file
    logger.info(_blue(f"Running predictions for {args.q_type} started"))
    outputs = run_inference(args, model, data) #this should result anad the 
    logger.info(_orange(f"Running predictions for {args.q_type} completed"))
    data['output'] = outputs
    data['preds']  = results
    data, eval_scores = compute_score(args, data) #returns a tuple 
    logger.info(f"Socre is {_blue(eval_scores)}")
    write_results(args=args, data=data, score=eval_scores)

if __name__ == "__main__":
    main()