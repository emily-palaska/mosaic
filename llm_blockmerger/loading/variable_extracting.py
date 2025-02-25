import torch, textwrap
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Suppress warnings, logs and progress bars (UserWarning: 1Torch was not compiled with flash attention.)
import warnings
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()
#warnings.filterwarnings("ignore")

class Llama:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        #self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)

    def answer(self, prompt=''):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=200)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


llm = Llama()
with open("prompt.txt", "r", encoding="utf-8") as file:
    text = file.read()

print(textwrap.fill(f'PROMPT: {text}', 100))


response = llm.answer(text)
print(textwrap.fill(f'LLAMA: {response}', 100))