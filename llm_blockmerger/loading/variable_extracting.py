"""
Placeholder file for future implementation of variable extraction
Will be combined with blockloading.py to determine the variables of each notebook
and a short description for each, which will be later used to align the variables
with the same semantics.
"""


import torch, textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings, logs and progress bars (UserWarning: 1Torch was not compiled with flash attention.)
import warnings
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()
warnings.filterwarnings("ignore")

class LanguageModel:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def answer(self, prompt=''):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=200)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


llm = LanguageModel()
response = llm.answer('x = 0 \n y = x \n print(y) \n What are the variables in this python script?')
print(textwrap.fill(response, 80))