import re
import traceback
from src.models.models import Model
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.cloud import aiplatform
from typing import Dict, List, Union
from huggingface_hub import login
hf_token = "hf_token"
login(token=hf_token)

project="991221573547"
endpoint_id="4675213601257029632"
location="europe-west4"
api_endpoint="europe-west4-aiplatform.googleapis.com"

def endpoint_predict_sample(
    project: str, location: str, instances: list, endpoint: str
):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(endpoint)

    prediction = endpoint.predict(instances=instances)
    print(prediction)
    return prediction


endpoint_predict_sample(
    project=project,
    location=location,
    instances=[{"prompt": "What is malaria?"}],
    endpoint=endpoint_id
)