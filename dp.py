from openai import OpenAI
from pydantic import BaseModel
from typing import List
import json
import pandas as pd
import time
import numpy as np
import os


api_key = "deepseek_api_key"  # Replace with your actual DeepSeek API key
model_name = "deepseek-reasoner"



def gpt4_quest(prompt, model_name):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    start_time = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a skillful expert medical assistant"},
            {"role": "user", "content": prompt},
        ],
        #max_tokens=500,
        stream=False
    )
    end_time = time.time()
    inference_time = end_time - start_time

    content = response.choices[0].message.content.strip()
    reasoning_content = response.choices[0].message.reasoning_content

    return content, reasoning_content, inference_time


df = pd.read_csv('/home/chinemelu/AfriMed-QA/data/expert_questions.csv')
# df2 = preproccess(df)
count = len(df)
questions = df['model_prompt'].tolist()
predictions_ = []
 
reasoning_ = []
inference_times = []

for i in range(len(df)):
    try:
        pred, reas, inf_time = gpt4_quest(questions[i], model_name)
        predictions_.append(pred)
        reasoning_.append(reas)
    
        inference_times.append(inf_time)
        avr_inf = np.mean(inference_times)
        total_inf = sum(inference_times)
        remaining = (count - (i + 1)) * avr_inf
        print(f"Processed: {i + 1}/{count} | Inference time: {inf_time:.2f} seconds , run for: {(total_inf/60):.2f}/{(remaining/60):.2f} minutes")
        print(f"result: {pred}")
        #print(f"reasoning: {reas}")
    except Exception as e:
        print(f"Skipping question at index {i}: {e}")
        predictions_.append(None)
        inference_times.append(None)
        reasoning_.append(None)


    # Save progress every 50 predictions or at the end
    if (i + 1) % 50 == 0 or (i + 1) == count:
        df.loc[:i, 'outputs'] = predictions_
        df.loc[:i, 'inference_time'] = inference_times
        df.loc[:i, 'reasoning'] = reasoning_
        df.to_csv(f"/home/chinemelu/AfriMed-QA/results/deepseek-R1/afrimed_deepseek_R1_progress_{i+1}.csv", index=False)
        print(f"Saved progress up to index {i + 1}.")

# Final save
csv_path = f"/home/chinemelu/AfriMed-QA/results/deepseek-R1/afrimed_deepseek_R1_{i+1}.csv"
df.to_csv(csv_path, index=False)
print(f"Final results saved to {csv_path}.")