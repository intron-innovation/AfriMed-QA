import torch
from src.models.models import Model
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class BioMistral(Model):
    def __init__(self, pretrained_model_path):
        super().__init__(pretrained_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
        	pretrained_model_path, add_bos_token=False, add_eos_token=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, device_map="auto")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def predict(self, prompt) -> str:
        messages = [{"role": "user", "content": prompt}]

        model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=100, do_sample=True)
        output = self.tokenizer.batch_decode(generated_ids)[0]
        output = output.replace("[INST] ", "").replace("[/INST] ", "").replace("<s>", "").replace("</s>", "").strip(" ")
        if ouput[0]==" ": output = output[1:]
        print(output)
        return output

    def post_process(self, raw_text_output):
        return raw_text_output
