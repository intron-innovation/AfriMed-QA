import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Model:
    def __inti__(self, args, **kwargs):
        pass

    def predict(self, data) -> str:
        pass


class Ph3(Model):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_path, use_fast=False,
            add_bos_token=False, add_eos_token=False,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_path,
            device_map='auto',
            use_safetensors=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        self.model = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
            

    def predict(self, prompt) -> str:
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        output = self.model(prompt, **generation_args)
        return output[0]['generated_text']