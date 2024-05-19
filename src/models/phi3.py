import torch
from src.models.models import Model


class Phi3(Model):
    def __init__(self, pretrained_model_path, **kwargs):
        super().__init__(pretrained_model_path, **kwargs)

        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path,
            use_fast=False,
            add_bos_token=False,
            add_eos_token=False,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            device_map="auto",
            use_safetensors=True,
            torch_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16,
            trust_remote_code=True,
        )
        self.model = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer,
        )
        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
        }

    def predict(self, prompt) -> str:
        prompt = [{"role": "user", "content": prompt}]

        output = self.model(prompt, **self.generation_args)
        output = output[0]["generated_text"].replace("<|end|>", "").strip(" ")
        return output

    def extract_mcq_answer(self, raw_text_model_output_list):
        cleaned_output = [text[0] for text in raw_text_model_output_list]
        return cleaned_output
