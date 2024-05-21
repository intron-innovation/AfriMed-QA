import os
import re
import anthropic

from src.models.models import Model


class ClaudeModel(Model):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        from src.models.models import Model

        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_KEY'])
        self.system_prompt = 'You are a skillful expert medical assistant'
        self.pattern1 = re.compile(r"([\w\d\s]+)?\n?([#\w\s\*\:]+)?\s{0,2}\(?([A-E])\)?\.?\s(.+)")
        self.pattern2 = re.compile(r"([\w\d\s]+)?\n?Option?\s{0,2}\(?([A-E])\)?\.?\:?\s(.+)")

    def predict(self, prompt) -> str:
        message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    temperature=0.2,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}])
        return message.content[0].text

    def extract_mcq_answer(self, raw_text_model_output_list):
        cleaned_output = [self.pattern_match(text) for text in raw_text_model_output_list]
        return cleaned_output

    def pattern_match(self, text, n=40):
        try:
            return self.pattern1.match(text[:n]).groups()[2]
        except Exception:
            return self.pattern2.match(text[:n]).groups()[2]
        except Exception:
            return text[:n]
