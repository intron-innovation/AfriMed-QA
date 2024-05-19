import os
import re
from openai import OpenAI

from src.models.models import Model


class OpenAIModel(Model):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        from src.models.models import Model

        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ['OPENAI_KEY'])
        self.system_prompt = 'You are a skillful expert medical assistant'
        self.pattern = re.compile(r"([\w\d\s]+)?\n?([#\w\s\*\:]+)?\s{0,2}([A-E])\.\s(.+)")

    def predict(self, prompt) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,  # "gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    def extract_mcq_answer(self, raw_text_model_output_list):
        cleaned_output = [self.pattern_match(text) for text in raw_text_model_output_list]
        return cleaned_output

    def pattern_match(self, text, n=40):
        try:
            return self.pattern.match(text[:n]).groups()[2]
        except Exception as e:
            return text[:n]
