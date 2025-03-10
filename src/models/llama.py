import torch
from src.models.models import Model


class Llama(Model):
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
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.torch_dtype = torch.float16 if torch.cuda.is_bf16_supported() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            device_map="auto",
            use_safetensors=True,
            torch_dtype=self.torch_dtype, 
            attn_implementation="flash_attention_2" if self.torch_dtype == torch.float16 else "sdpa",
            trust_remote_code=True,
            
        )
        self.model = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.generation_args = {
            "max_new_tokens": 300,
            "return_full_text": False,
        }

    def predict(self, prompt) -> str:
        output = self.model(prompt, **self.generation_args)
        output = output[0]["generated_text"].strip().strip("\n\n")
        if "Prompt:" in output:
            output = output.split("Prompt:")[0]
        if "Question:" in output:
            output = output.split("Question:")[0]
        output = output.replace("###", "")
        return output

    def extract_mcq_answer(self, raw_text_model_output_list):
        cleaned_output = [text[0] for text in raw_text_model_output_list]
        return cleaned_output
