from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

class LLM:
    def __init__(self, task, name=None, verbose=False):
        self.task = task
        self.verbose = verbose
        self._set_verbosity()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if task == 'question':
            self.name = name if name else 'meta-llama/Llama-3.2-3B'
            self.tokenizer = AutoTokenizer.from_pretrained(self.name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.name,
                torch_dtype=torch.float16
            ).to(self.device)
        elif task == 'embedding':
            self.name = name if name else 'sentence-transformers/all-MiniLM-L6-v2'
            self.model = SentenceTransformer(self.name)
        else:
            raise TypeError("Task must be either 'question' or 'embedding'")

    def _set_verbosity(self):
        # Suppress warnings, logs and progress bars
        # (UserWarning: Torch was not compiled with flash attention.)
        if not self.verbose:
            from transformers.utils.logging import disable_progress_bar
            disable_progress_bar()
            import warnings
            warnings.filterwarnings("ignore")

    def encode(self, labels):
        if self.task != 'embedding': return None
        if isinstance(labels, str): labels = [labels]
        return self.model.encode(labels)

    def answer(self, prompt, max_new_tokens=20):
        if self.task != 'question': return None
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.pad_token_id
        )
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if self.verbose: print('\nMODEL OUTPUT:\n', output_text)
        return output_text


def code_prompt(script: str):
    return f"""
Write a Python program that implements the following query. Make it functional and include all the necessary functions.

'''{script}'''

Code:

"""


def separate_code(output: str):
    if (position := output.find('Code:')) != -1:
        return output[position + len('Code:\n'):]
    return output
