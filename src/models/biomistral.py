import torch
from src.models.models import Model
from transformers import AutoTokenizer, AutoModel, pipeline


class BioMistral(Model):
    def __init__(self, pretrained_model_path):
        super().__init__(pretrained_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.model = AutoModel.from_pretrained(pretrained_model_path)
        self.model = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )

    def predict(self, prompt) -> str:
        prompt = [{"role": "user", "content": prompt}]

        output = self.model(prompt)
        output = output[0]["generated_text"]
        print(output)
        return output

    def post_process(self, raw_text_output):
        return raw_text_output
