import torch
from src.models.models import Model
from transformers import AutoTokenizer, AutoModel, pipeline


class MedLlama(Model):
    def __init__(self, pretrained_model_path):
        super().__init__(pretrained_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.model = AutoModel.from_pretrained(pretrained_model_path)
        self.model = pipeline("text-generation", model=model, torch_dtype=torch.float16, device_map="auto")

    def predict(self, prompt) -> str:
        prompt = [{"role": "user", "content": prompt}]
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        output = self.model(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        output = output[0]["generated_text"]
        return output

    def post_process(self, raw_text_output):
        return raw_text_output
