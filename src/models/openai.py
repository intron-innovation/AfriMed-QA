import os
import re
from openai import OpenAI
import traceback

from src.models.models import Model


class OpenAIModel(Model):
    def __init__(self, model_name, explanation **kwargs):
        super().__init__(model_name, **kwargs)
        from src.models.models import Model

        self.model_name = model_name
        self.explanation = explanation
        self.client = OpenAI(api_key=os.environ["OPENAI_KEY"])
        self.system_prompt = "You are a skillful expert medical assistant"
        self.pattern1 = re.compile(
            r"([\w\d\s]+)?\n?([#\w\s\*\:]+)?\s{0,2}\(?([A-E])\)?\.?\s*\n?\s*(.+)"
        )
        self.pattern2 = re.compile(
            r"([\w\d\s]+)?\n?Option?\s{0,2}\(?([A-E])\)?\.?\s*\n?\s*:\s*(.+)"
        )

    def predict(self, prompt) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,  # "gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        output = completion.choices[0].message.content
        if self.explantion == False:
            if "Prompt:" in output:
                output = output.split("Prompt:")[0]
            if "Question:" in output:
                output = output.split("Question:")[0]
            output = output.replace("###", "")
        return output
    
        

    def extract_mcq_answer(self, raw_text_model_output_list):
        if self.explanation:
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
