from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

class LLM:
    def __init__(self, task, model_name=None, verbose=False):
        self.task = task
        self.verbose = verbose
        self._set_verbosity()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if task == 'question':
            self.model_name = model_name if model_name else 'meta-llama/Llama-3.2-3B'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            ).to(self.device)
        elif task == 'embedding':
            self.model_name = model_name if model_name else 'sentence-transformers/all-MiniLM-L6-v2'
            self.model = SentenceTransformer(self.model_name)
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

    def answer(self, prompt):
        if self.task != 'question': return None
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            min_length=len(prompt.split()) * 2,
            max_length=len(prompt.split()) * 2 + 100,
            temperature=0.3,
            top_p=0.9,
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.pad_token_id
        )
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if self.verbose: print('\nLLAMA OUTPUT:\n', output_text)
        return output_text