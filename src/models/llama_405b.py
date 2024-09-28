import os
import requests
import json
import ast
import time


class Llama405B:
    def __init__(self, model_name):
        self.endpoint = "us-central1-aiplatform.googleapis.com"
        self.region = "us-central1"
        self.project_id = "afrimed-qa"
        self.model_name = model_name
        self.url = f"https://{self.endpoint}/v1beta1/projects/{self.project_id}/locations/{self.region}/endpoints/openapi/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.get_access_token()}",
            "Content-Type": "application/json",
        }
        self.system_prompt = "You are a skillful expert medical assistant"

    def get_access_token(self):
        return os.popen("gcloud auth print-access-token").read().strip()

    def predict(self, prompt) -> str:
        payload = {
            "model": self.model_name,
            "stream": False,
            "max_tokens": 300,
            "top_p": 1,
            "top_k": 1,
            "num_return_sequences": 1,
            "messages": [{"role": "user", "content": prompt}],
        }
        output = ""
        counter = 0
        while len(output) < 1:
            response = requests.post(self.url, headers=self.headers, json=payload)

            if response.status_code == 200:
                try:
                    response = json.loads(response.text)
                    output = response["choices"][0]["message"]["content"].strip()
                except:
                    response = ast.literal_eval(response.text)
                    output = response["choices"][0]["message"]["content"].strip()

            else:
                time.sleep(60)
            if counter == 5:
                break
            counter += 1

        return output

    def extract_mcq_answer(self, raw_text_model_output_list):
        cleaned_output = [
            text[0] if len(text) >= 1 else " " for text in raw_text_model_output_list
        ]
        return cleaned_output
