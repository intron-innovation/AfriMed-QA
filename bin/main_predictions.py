import os
import sys

parent_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_folder_path)

import logging
import gc
import torch
from src.utils.utils import (
    parse_arguments,
    _orange, _blue, _purple,
    patch_open
)
from src.utils.prepare_data import prep_data
from src.utils.utils import write_results, post_process_output
from src.models.phi3 import Phi3
from src.models.openai import OpenAIModel
from src.models.biomistral import BioMistral
from src.models.medalpaca import MedAlpaca
from src.models.medllama import MedLlama
from src.inference.inference import run_inference
from src.evals.evaluate import compute_score

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

    logger.info(f"Preprocessing data at {args.data_path}")
    data = prep_data(args)

    logger.info(f"Loading model from {args.pretrained_model_path}")
    # please define your own model class
    if "gpt" in args.pretrained_model_path:
        model = OpenAIModel(args.pretrained_model_path)
    elif "phi3" in args.pretrained_model_path:
        model = Phi3(args.pretrained_model_path)
    elif "Mistral" in args.pretrained_model_path:
        model = BioMistral(args.pretrained_model_path)
    elif "Alpaca" in args.pretrained_model_path:
        model = MedAlpaca(args.pretrained_model_path)
    elif "Llama" in args.pretrained_model_path:
        model = MedLlama(args.pretrained_model_path)
    else:
        raise NotImplementedError(f"No model class defined for {args.pretrained_model_path}")
    logger.info("Model loaded successfully")

    logger.info(_blue(f"Running predictions for {args.q_type} started"))
    outputs = run_inference(model, data, args.use_cuda)  # this should result a list of predictions
    logger.info(_orange(f"Running predictions for {args.q_type} completed"))
    data['outputs'] = outputs
    if args.q_type == "mcq":
        options_from_output = post_process_output(outputs)  # edit the post_processing fxn accordingly
        data['preds'] = options_from_output
    (data, BERTScore_Precision, BERTScore_Recall, BERTScore_F1, rg1, rg2, rl, accuracy) = compute_score(args.q_type,
                                                                                                        data)  # returns a tuple
    logger.info(f"Score is {_blue(rl)}")
    write_results(args=args, data=data, score=rl)


if __name__ == "__main__":
    main()
