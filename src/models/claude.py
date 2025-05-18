import os
import re
import anthropic
import traceback

from src.models.models import Model

class ClaudeModel(Model):
    def __init__(self, model_name, explanation, **kwargs):
        super().__init__(model_name, **kwargs)
        from src.models.models import Model

        self.model_name = model_name
        self.explanation = explanation
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_KEY"])
        self.system_prompt = "You are a skillful expert medical assistant"
        self.pattern1 = re.compile(
            r"([\w\d\s]+)?\n?([#\w\s\*\:]+)?\s{0,2}\(?([A-E])\)?\.?\s*\n?\s*(.+)"
        )
        self.pattern2 = re.compile(
            r"([\w\d\s]+)?\n?Option?\s{0,2}\(?([A-E])\)?\.?\s*\n?\s*:\s*(.+)"
        )

    def predict(self, prompt) -> str:
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=1,
            system=self.system_prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        output = message.content[0].text
        if self.explanation == False:
            if "Prompt:" in output:
                output = output.split("Prompt:")[0]
            if "Question:" in output:
                output = output.split("Question:")[0]
            output = output.replace("###", "")
        return output

    def extract_mcq_answer(self, raw_text_model_output_list):
        cleaned_res = []
        if self.explanation:
            cleaned_output = [
                self.pattern_match(text) for text in raw_text_model_output_list
            ]
        else:
            for text in raw_text_model_output_list:
                if "The correct answer is" in text:
                    ans = text.split("The correct answer is")[1].split(".")[0].strip()
                elif "The correct option is" in text:
                    ans = text.split("The correct option is")[1].split(".")[0].strip()
                elif "The answer is" in text:
                    ans = text.split("The answer is")[1].split(".")[0].strip()
                elif "Answer:" in text:
                    ans = text.split("Answer:")[1].split(".")[0].strip()
                elif "I choose option" in text:
                    ans = text.split("I choose option")[1].split(".")[0].strip()
                
                else:
                    ans = text[0]
                ans = ans.upper()
                cleaned_res.append(ans)
            cleaned_texts = [re.sub(r'[?;:.]', '', text).strip() for text in cleaned_res]
        return cleaned_texts
            #cleaned_output = [text[0] for text in raw_text_model_output_list]
        
        #return cleaned_output
        

    def pattern_match(self, text, n=40):
        try:
            return self.pattern1.match(text[:n]).groups()[2]
        except Exception:
            print(text[:n])
            print(traceback.format_exc())
        try:
            return self.pattern2.match(text[:n]).groups()[2]
        except Exception:
            print(text[:n])
            print(traceback.format_exc())
            return text[:n]

