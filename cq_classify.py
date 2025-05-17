


import time
import json
import pandas as pd
from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

# Model configuration
model_name = 'qwen2.5-32b'

def predict_with_backoff(
    question: str,
    project="991221573547",
    endpoint_id="6895772991062278144",
    location="us-central1",
    api_endpoint="us-central1-aiplatform.googleapis.com",
    max_retries=5
):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)



    instances = {"prompt": question,"max_tokens":512}
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )

    retries = 0
    while retries < max_retries:
        try:
            response = client.predict(
                endpoint=endpoint, instances=instances, parameters=parameters
            )
            return response.predictions[0].split('Output:')[-1]
        except Exception as e:
            wait_time = 2 ** retries
            print(f"Error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception(f"Failed to generate response after {max_retries} retries.")



# Read and preprocess data
df = pd.read_csv('/mnt/external_aka/data/aka/AfriMed-QA/cq_quest.csv')
count = len(df)
questions = df['question'].tolist()
scenarios = df['prompt'].tolist()



predictions_ = []
fps = []


for i in range(len(df)):

    scenario = scenarios[i]
    question = questions[i]

    prompt = f'''
    You are a classifier that reads a medical scenario and a question, and decides whether the question is actually about the scenario.

    For each input, output exactly one of:
    — “Yes”   (the question asks something about the scenario)
    — “No”    (the question is off-topic or unrelated)

    Always respond with only “Yes” or “No”.

    ---

    Example 1:
    Scenario:
    “Your female classmate complains of persistent cough, night sweats and thinks she has Tuberculosis and is going to visit the nearest doctor.”

    Question:
    “what is tuberculosis? How do i know if i have tuberculosis? What are the symptoms of tuberculosis? Can tuberculosis be cured?”

    Answer:
    Yes

    ---

    Example 2:
    Scenario:
    “A patient has just undergone knee surgery and is recovering in the hospital.”

    Question:
    “What are the signs of malaria and how is it treated?”

    Answer:
    No

    ---

    Now classify this:

    Scenario:
    “{scenario}”

    Question:
    “{question}”

    Answer:

    '''
   

    try:
        pred = predict_with_backoff(prompt)
        pr = prompt

        fps.append(pr)
        predictions_.append(pred)
        print(f"Processed: {i + 1}/{count}")
        print(f"Prompt: {pr}")
        print(predictions_[i])
    except Exception as e:
        print(f"Skipping question at index {i}: {e}")
        predictions_.append(None)
        fps.append(None)  # Append None if all retries fail

    # Save progress every 500 predictions
    if (i + 1) % 20 == 0 or (i + 1) == count:  # Save at multiples of 500 or final prediction

        df.loc[:i, 'class_prompt'] = fps
        df.loc[:i, 'class'] = predictions_  # Update DataFrame with current predictions
        df.to_csv(f"cq_class_progress.csv", index=False)
        print(f"Saved progress up to index {i + 1}.")

# Final save
csv_path = f"cq_class.csv"
df.to_csv(csv_path, index=False)
print(f"Final results saved to {csv_path}.")





