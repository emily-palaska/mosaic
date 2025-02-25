from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, ast, textwrap

def extract_variables(script, model=None):
    if model is None:
        return _ast_extraction(script)
    else:
        prompt = _create_prompt(script)
        output_text = model.answer(prompt)
        return _separate_output(output_text)

class Llama:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B", verbose=False):
        self.verbose = verbose
        self._set_verbosity()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

    def _set_verbosity(self):
        # Suppress warnings, logs and progress bars (UserWarning: 1Torch was not compiled with flash attention.)
        if not self.verbose:
            from transformers.utils.logging import disable_progress_bar
            disable_progress_bar()
            import warnings
            warnings.filterwarnings("ignore")

    def answer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            min_length=len(prompt.split()) * 2,
            max_length=len(prompt.split()) * 2 + 100,
            temperature=0.3,  # Slightly higher for creativity
            top_p=0.9,  # Nucleus sampling for better results
            attention_mask=inputs["attention_mask"],  # Explicitly pass attention mask
            pad_token_id=self.tokenizer.pad_token_id
        )
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if self.verbose: print('\nLLAMA OUTPUT:\n', output_text)
        return output_text

def _ast_extraction(script=''):
    tree = ast.parse(script)
    variables = set()

    def handle_function(curr_node):
        for arg in curr_node.args.args:
            variables.add(arg.arg)

    def handle_loop(curr_node):
        if isinstance(curr_node.target, ast.Name):
            variables.add(curr_node.target.id)
        elif isinstance(curr_node.target, ast.Tuple):
            handle_tuple(curr_node)

    def handle_tuple(curr_node):
        for target in curr_node.targets:
            if isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        variables.add(elt.id)

    def visit_node(curr_node):
        match curr_node:
            case ast.Name(ctx=ast.Store()):
                variables.add(curr_node.id)
            case ast.FunctionDef():
                handle_function(curr_node)
            case ast.For():
                handle_loop(curr_node)
            case ast.Assign():
                handle_tuple(curr_node)

    for node in ast.walk(tree):
        visit_node(node)

    return sorted(list(variables))

def _create_prompt(script=''):
    return f"""
Analyze the following Python script and create a list of all the variables you can find. 
Return only the list of variable names.
Note that they should be separated by commas (,).

Script:
{textwrap.indent(script, '\t')}

Output:
"""

def _separate_output(output_text):
    if "Output:" in output_text:
        variables_str = output_text.split("Output:")[1].strip()
        variables_str = variables_str.split("\n")[0].strip()
        variables = {
            var.strip().replace("[", "").replace("]", "")
            for var in variables_str.split(",")
        }
        variables.discard("")
    else:
        variables = set()
    return sorted(variables)


