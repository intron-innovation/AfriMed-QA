import torch
from src.models.models import Model
from transformers import AutoTokenizer, AutoModel, pipeline


class MedAlpaca(Model):
    def __init__(self, pretrained_model_path):
        super().__init__(pretrained_model_path)
        self.model = pipeline("text-generation", model=pretrained_model_path, tokenizer=pretrained_model_path)

    def predict(self, prompt) -> str:
        prompt = f"Context: \n\nQuestion: {prompt}\n\nAnswer: "

        output = self.model(prompt)
        output = output[0]["generated_text"]
        return output

    def post_process(self, raw_text_output):
        return raw_text_output
