import os
import logging
import gc
from src.utils.utils import (
    parse_arguments,
    _orange, _blue, _purple,
    patch_open
)
from src.utils.prepare_data import prep_data
import torch
from src.utils.utils import  write_results, post_process_output
from src.models.phi3 import Phi3
from src.inference.inference import run_inference
from src.evals.evaluate import compute_score
from transformers import set_seed

logger = logging.getLogger(__name__)
set_seed(42)

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
    model = Phi3(args.pretrained_model_path) # please define your own model file
    logger.info(_blue(f"Running predictions for {args.q_type} started"))
    outputs = run_inference(model, data) #this should result a list of predictions
    logger.info(_orange(f"Running predictions for {args.q_type} completed"))
    data['outputs'] = outputs
    if args.q_type =="mcq":
        options_from_output = post_process_output(outputs) #edit the post_processing fxn accordingly
        data['preds']  = options_from_output
    (data, BERTScore_Precision, BERTScore_Recall, BERTScore_F1, rg1, rg2, rl,  accuracy) = compute_score(args.q_type, data) #returns a tuple 
    logger.info(f"Socre is {_blue(rl)}")
    write_results(args=args, data=data, score=rl)

if __name__ == "__main__":
    main()