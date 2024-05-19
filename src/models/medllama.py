import torch
from src.models.models import Model
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class MedLlama(Model):
    def __init__(self, pretrained_model_path):
        super().__init__(pretrained_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
        	pretrained_model_path, add_bos_token=False, add_eos_token=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
        	pretrained_model_path, device_map="auto",
        	torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        self.model = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)


    def predict(self, prompt) -> str:
        messages = [{"role": "user", "content": prompt}]

        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).to(self.device)
        output = self.model(inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        output = output[0]["generated_text"]
        print(output)
        return output

    def post_process(self, raw_text_output):
        return raw_text_output
