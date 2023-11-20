import vertexai
from vertexai.preview.language_models import TextGenerationModel
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import config
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.PATH_TO_GCP_CREDS

vertexai.init(project="llm-grammar", location="us-east4")

model = TextGenerationModel.from_pretrained("text-bison@001")
response = model.predict(
    "Write a text about dancing on CEFR level A1.",
    temperature=0.2,
    max_output_tokens=512,
    top_k=40,
    top_p=0.8,
)
print(f"Response from Model: {response.text}")