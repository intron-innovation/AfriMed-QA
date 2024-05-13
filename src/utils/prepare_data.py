import pandas as pd
import json
from tqdm import tqdm


def transform_mcqs(data):
    questions = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Preprocessing the data"):
        options = json.loads(row["answer_options"])
        option_keys = [
            f"option{i+1}" for i in range(5)
        ]  # Adjust number of options if needed
        answer_labels = "ABCDE"

        # Extract and label options
        formatted_options = {
            label: options.get(key, "")
            for label, key in zip(answer_labels, option_keys)
            if key in options
        }

        # Determine the correct answer label
        correct_answer_label = answer_labels[option_keys.index(row["correct_answer"])]

        # Construct the final row for DataFrame
        transformed_row = {
            "sample_id": row["sample_id"],
            "question": row["question"],
            "rationale": row["answer_rationale"],
            "answer": correct_answer_label,
            **formatted_options,
        }
        questions.append(transformed_row)

    transformed_data = pd.DataFrame(questions)
    return transformed_data


def transform_saqs(data):
    questions = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Preprocessing the data"):
        transformed_row = {
            "sample_id": row["sample_id"],
            "question": row["question"],
            "rationale": row["answer_rationale"],
        }
        questions.append(transformed_row)

    transformed_data = pd.DataFrame(questions)
    return transformed_data


def transform_consumer_queries(data):
    questions = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Preprocessing the data"):
        transformed_row = {
            "sample_id": row["sample_id"],
            "prompt": row["prompt"],
            "question": row["question"],
            "rationale": row["answer_rationale"],

        }
        questions.append(transformed_row)

    transformed_data = pd.DataFrame(questions)
    return transformed_data


transformation_types = {
    "mcq": transform_mcqs,
    "saq": transform_saqs,
    "consumer_queries": transform_consumer_queries,
}


def prep_data(args) -> pd.DataFrame:
    data = pd.read_csv(args.data_path)
    if args.q_type in transformation_types.keys():
        data = (
            data[data["question_type"] == args.q_type.strip()]
            .copy()
            .reset_index(drop=True)
        )
        if args.q_type == "mcq":
            data["correct_answer"] = data["correct_answer"].str.split(",").str[0]
        data = transformation_types[args.q_type.strip()](data)

    else:
        Exception(
            f"The question type `{args.q_type}` is invalid. Please provide a valid question type. Valid question types are {transformation_types.keys()}"
        )
    return data
