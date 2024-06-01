import errno
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import ast
from src.utils.utils import read_txt_file


def prep_mcqs_options(row):
    row = row[
        ~row.isna()
    ]  # this line drops the empty options if there are just 2 valid options like true/false
    question = "###Question: " + row["question"]
    options = dict(row.drop(["question", "sample_id", "answer", "rationale", "options_len"]))
    formatted_options = ""
    for key, value in options.items():
        formatted_options += f"{key}.  {value}\n"
    answer = f"""###Answer: {row['answer']} \n###Rationale: {row['rationale']}"""
    return question, formatted_options, answer


def transform_medqa(data):
    questions = []
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Preprocessing the data"):
        formatted_options = ast.literal_eval(row["options"])
        transformed_row = {
            "sample_id": row["sample_id"],
            "question": row["question"],
            "rationale": row["question"],
            "options_len": len(formatted_options),
            "answer": row["correct_answer"],
            **formatted_options,
        }
        questions.append(transformed_row)
    return questions
 

def transform_mcqs(args, data):
    if 'MedQA' in args.data_path:
        questions = transform_medqa(data)
    else:
        questions = []
        for index, row in tqdm(data.iterrows(), total=len(data), desc="Preprocessing the data"):
            options = json.loads(row["answer_options"])
            option_keys = [
                f"option{i + 1}" for i in range(5)
            ]  # Adjust number of options if needed
            answer_labels = "ABCDE"

            # Extract and label options
            formatted_options = {
                label: options.get(key, "")
                for label, key in zip(answer_labels, option_keys)
                if key in options
            }
            keys = list(formatted_options.keys())
            for key in keys:
                if formatted_options[key].upper() == "N/A":
                    formatted_options.pop(key)
            # Determine the correct answer label
            options_len = len(formatted_options)
            correct_answer_label = answer_labels[option_keys.index(row["correct_answer"])]
            transformed_row = {
                "sample_id": row["sample_id"],
                "question": row["question"],
                "rationale": row["answer_rationale"],
                "options_len": options_len,
                "answer": correct_answer_label,
                **formatted_options,
            }
            questions.append(transformed_row)

    transformed_data = pd.DataFrame(questions)
    if args.prompt_file_path != None:
        user_msg = read_txt_file(args.prompt_file_path)

        data_w_prompt = []
        for index, row in tqdm(transformed_data.iterrows(), total=len(transformed_data), desc="creating prompts"):
            question, formatted_options, _ = prep_mcqs_options(row)
            sys_msg = user_msg + "\n\n"

            if args.num_few_shot > 0:
                try:
                    other_indices = transformed_data[
                        transformed_data['options_len'] == row['options_len']].index.difference([index]).tolist()
                    random_indices = np.random.choice(other_indices, size=args.num_few_shot, replace=False)
                    few_shots = transformed_data.iloc[random_indices]

                except:
                    other_indices = transformed_data[transformed_data['options_len'] == 4].index.difference(
                        [index]).tolist()
                    random_indices = np.random.choice(other_indices, size=args.num_few_shot, replace=False)
                    few_shots = transformed_data.iloc[random_indices]

                sys_msg += "Here are some examples and then answer the last question:" + "\n\n"

                for _, s in few_shots.iterrows():
                    sq, sf, sa = prep_mcqs_options(s)
                    sys_msg += sq + "\n" + sf + "\n" + sa + "\n\n"
                sys_msg += sys_msg + "\n\n"
            final_prompt = sys_msg + question + "\n" 
            final_prompt = final_prompt + "###Options:" + "\n" + formatted_options

            final_prompt += '\n' + "###Answer:"
            row['model_prompt'] = final_prompt
            data_w_prompt.append(row)

    else:
        Exception("Prompt file not found")

    transformed_data = pd.DataFrame(data_w_prompt)

    return transformed_data


def transform_saqs(args, data):
    questions = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Preprocessing the data"):
        transformed_row = {
            "sample_id": row["sample_id"],
            "question": row["question"],
            "rationale": row["answer_rationale"],
        }
        questions.append(transformed_row)

    transformed_data = pd.DataFrame(questions)

    if args.prompt_file_path != None:
        user_msg = read_txt_file(args.prompt_file_path)

        data_w_prompt = []
        for index, row in tqdm(transformed_data.iterrows(), total=len(transformed_data), desc="creating prompts"):
            question = "###Question: " + row["question"]
            sys_msg = user_msg + "\n\n"

            if args.num_few_shot > 0:
                other_indices = transformed_data.index.difference([index]).tolist()
                random_indices = np.random.choice(other_indices, size=args.num_few_shot, replace=False)

                few_shots = transformed_data.iloc[random_indices]

                sys_msg += "Here are some examples and then answer the last question:" + "\n\n"

                for _, s in few_shots.iterrows():
                    squestion = "###Question: " + s["question"]
                    srationale = "###Rationale: " + s['rationale']
                    sys_msg += squestion + "\n" + srationale + "\n\n"
                sys_msg += sys_msg + "\n\n"
            final_prompt = sys_msg + question
            final_prompt += '\n' + "###Answer:"
            row['model_prompt'] = final_prompt

            data_w_prompt.append(row)

    else:
        Exception("Prompt file not found")

    transformed_data = pd.DataFrame(data_w_prompt)
    return transformed_data


def transform_consumer_queries(args, data):
    questions = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Preprocessing the data"):
        transformed_row = {
            "sample_id": row["sample_id"],
            "prompt": row["prompt"],
            "question": row["question"],
            "rationale": "" if type(row["answer_rationale"]) == float else row["answer_rationale"],

        }
        questions.append(transformed_row)

    transformed_data = pd.DataFrame(questions)

    if args.prompt_file_path != None:
        user_msg = read_txt_file(args.prompt_file_path)

        data_w_prompt = []
        for index, row in tqdm(transformed_data.iterrows(), total=len(transformed_data), desc="creating prompts"):
            question_prompt = "###Prompt: " + row["prompt"]
            question = "###Question: " + row["question"]
            sys_msg = user_msg + "\n\n"

            if args.num_few_shot > 0:
                other_indices = transformed_data.index.difference([index]).tolist()
                random_indices = np.random.choice(other_indices, size=args.num_few_shot, replace=False)
                few_shots = transformed_data.iloc[random_indices]

                sys_msg += "Here are some examples and then answer the last question:" + "\n\n"

                for _, s in few_shots.iterrows():
                    squestion_prompt = "###Prompt: " + s["prompt"]
                    squestion = "###Question: " + s["question"]
                    srationale = "###Rationale: " + s['rationale']
                    sys_msg += squestion_prompt + "" + squestion + "\n" + srationale + "\n\n"
                sys_msg += sys_msg + "\n\n"
            final_prompt = question_prompt + "\n" + question
            final_prompt += '\n' + "###Answer:"
            row['model_prompt'] = final_prompt

            data_w_prompt.append(row)

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.prompt_file_path)

    transformed_data = pd.DataFrame(data_w_prompt)
    return transformed_data


transformation_types = {
    "mcq": transform_mcqs,
    "saq": transform_saqs,
    "consumer_queries": transform_consumer_queries,
}


def prep_data(args) -> pd.DataFrame:
    data = pd.read_csv(args.data_path)
    data = data.replace("N/A", np.nan)
    if args.q_type in transformation_types.keys():
        data = (
            data[data["question_type"] == args.q_type.strip()]
                .copy()
                .reset_index(drop=True)
        )
        if args.q_type == "mcq":
            data["correct_answer"] = data["correct_answer"].str.split(",").str[0]
        if 'MedQA' in args.data_path and args.q_type != 'mcq':
            raise Exception("Please provide a valid question type for this dataset.")
        data = transformation_types[args.q_type.strip()](args, data)
    else:
        Exception(
            f"The question type `{args.q_type}` is invalid for this dataset. "
            f"Please provide a valid question type for this dataset. Valid question types are {transformation_types.keys()} for AfriMed-QA and mcq for MedQA."
        )
    return data
