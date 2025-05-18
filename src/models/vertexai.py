import re
import traceback
from src.models.models import Model
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.cloud import aiplatform
from typing import Dict, List, Union
from huggingface_hub import login
 
login(token=hf_token)

class VertexAIModel(Model):
    def __init__(
        self,
        explanation,
        project="991221573547",
        endpoint_id="1279897505428930560",
        location="us-west1",
        api_endpoint="us-west1-aiplatform.googleapis.com",
        **kwargs
    ):
        super().__init__("vertex_ai", **kwargs)
        self.explanation = explanation
        self.project = project
        self.endpoint_id = endpoint_id
        self.location = location
        self.api_endpoint = api_endpoint
        self.client_options = {"api_endpoint": self.api_endpoint}
        self.client = aiplatform.gapic.PredictionServiceClient(
            client_options=self.client_options
        )
        self.system_prompt = "You are a skillful expert medical assistant"
        self.pattern1 = re.compile(
            r"([\w\d\s]+)?\n?([#\w\s\*\:]+)?\s{0,2}\(?([A-E])\)?\.?\s*\n?\s*(.+)"
        )
        self.pattern2 = re.compile(
            r"([\w\d\s]+)?\n?Option?\s{0,2}\(?([A-E])\)?\.?\s*\n?\s*:\s*(.+)"
        )

    def predict(self, prompt: str) -> str:
        try:
            # Combine system prompt and user prompt if desired
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            
            # Prepare the instance with the full prompt and token limit.
            #instance_dict = {"prompt": full_prompt, "max_tokens": 512, "temperature":1}
            instance_dict = {"inputs": full_prompt, "max_tokens": 512,  "temperature":1}

            instances = [instance_dict]
            # Convert each instance dictionary to a Protobuf Value.
            instances = [json_format.ParseDict(instance, Value()) for instance in instances]

            # Prepare parameters (empty for now)
            parameters_dict = {}
            parameters = json_format.ParseDict(parameters_dict, Value())

            # Construct the fully qualified endpoint resource path.
            endpoint = self.client.endpoint_path(
                project=self.project, location=self.location, endpoint=self.endpoint_id
            )

            # Make the prediction request.
            response = self.client.predict(
                endpoint=endpoint, instances=instances, parameters=parameters
            )
            # Process the response by splitting on 'Output:' to extract the generated text.
            output = response.predictions[0].split("Output:")[-1]
            
            # If explanation is disabled, remove any extraneous parts of the output.
            if not self.explanation:
                if "Prompt:" in output:
                    output = output.split("Prompt:")[0]
                if "Question:" in output:
                    output = output.split("Question:")[0]
                output = output.replace("###", "")
            return output
        except Exception:
            print("Error during predict:")
            print(traceback.format_exc())
            return ""

    def extract_mcq_answer(self, raw_text_model_output_list):
        if self.explanation:
            print("raw_text_model_output_list", raw_text_model_output_list) 
            cleaned_output = [
                self.pattern_match(text) for text in raw_text_model_output_list
            ]
        else:
            cleaned_output = [text[0] for text in raw_text_model_output_list]
        return cleaned_output

    def pattern_match(self, text, n=40):
        try:
            match = self.pattern1.match(text[:n])
            if match is not None:
                return match.groups()[2]
        except Exception:
            print(text[:n])
            print(traceback.format_exc())
        try:
            return self.pattern2.match(text[:n]).groups()[2]
        except Exception:
            print(text[:n])
            print(traceback.format_exc())
            return text[:n]
