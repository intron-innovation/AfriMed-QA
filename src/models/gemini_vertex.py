import traceback
import vertexai
from vertexai.generative_models import GenerativeModel
from src.models.models import Model

class GeminiVertexAIModel(Model):
    def __init__(
        self,
        explanation,
        project="afrimed-qa",
        location="us-central1",
        model_name="gemini-2.5-pro-preview-05-06",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.explanation = explanation
        self.project = project
        self.location = location
        self.model_name = model_name
        
        # Initialize Vertex AI
        vertexai.init(project=self.project, location=self.location)
        self.model = GenerativeModel(self.model_name)
        
        self.system_prompt = "You are a skillful expert medical assistant"
    
    def predict(self, prompt: str) -> str:
        try:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            generation_config = {"max_output_tokens": 1024, "temperature": 1}
            
            responses = self.model.generate_content(
                [full_prompt], generation_config=generation_config, stream=True
            )
            
            output = "".join(response.text for response in responses)
            
            # If explanation is disabled, clean the output
            if not self.explanation:
                if "Prompt:" in output:
                    output = output.split("Prompt:")[0]
                if "Question:" in output:
                    output = output.split("Question:")[0]
                output = output.replace("###", "")
            
            return output
        except Exception:
            print("Error during predict:")
            print(traceback.format_exc())
            return ""
