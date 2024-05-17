import os
from openai import OpenAI

from src.models.models import Model


class OpenAIModel(Model):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        from src.models.models import Model

        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ['OPENAI_KEY'])
        self.system_prompt = 'You are a skillful expert medical assistant'

    def predict(self, prompt) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,  # "gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    def post_process(self, raw_text_output):
        # The correct option is B.
        # Answer: B.
        # is option E.
        # Correct option: C.
        # The most likely diagnosis in this scenario is D.

        return raw_text_output
