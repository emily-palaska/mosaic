import textwrap
from llm_blockmerger.core.utils import ast_extraction
from llm_blockmerger.core.managers import concatenate_block

def extract_notebook_variables(block_manager, model, empty=False):
    if empty:
        block_manager.set(variables=[[] for _ in range(len(block_manager))],
                          var_descriptions=[[] for _ in range(len(block_manager))])
        return

    # todo update d if v is already in the set
    notebook_variables = set()
    for block in block_manager.blocks:
        try:
            block_variables = _extract_block_variables(concatenate_block(block))
            block_descriptions = _extract_block_descriptions(block_variables, concatenate_block(block), model)
            notebook_variables.update((v, d) for v, d in zip(block_variables, block_descriptions))
        except IndentationError: continue
        except Exception: raise
    separated_variables, separated_descriptions = _separate_variables_per_block(block_manager.blocks, notebook_variables)
    block_manager.set(variables=separated_variables, var_descriptions=separated_descriptions)

def _separate_variables_per_block(blocks, notebook_variables):
    # todo try reg ex
    separated_variables = []
    separated_descriptions = []
    for block in blocks:
        block_variables, block_descriptions = [], []
        for variable, description in notebook_variables:
            if variable in concatenate_block(block):
                block_variables.append(variable)
                block_descriptions.append(description)
        separated_variables.append(block_variables)
        separated_descriptions.append(block_descriptions)
    return separated_variables, separated_descriptions

def _extract_block_variables(script, model=None):
    if model is None:
        return ast_extraction(script)
    output_text = model.answer(_create_variable_extraction_prompt(script))
    return _separate_variable_string(output_text)

def _extract_block_descriptions(variables, script, model):
    descriptions = []
    for variable in variables:
        output_text = model.answer(_create_variable_description_prompt(variable,script))
        descriptions.append(_separate_description_string(output_text))
    return descriptions

def _create_variable_extraction_prompt(script=''):
    return f"""
Analyze the following Python script and create a list of all the variables you can find. 
Return only the list of variable names.
Note that they should be separated by commas (,).

Script:
{textwrap.indent(script, '\t')}

Output:
"""

def _create_variable_description_prompt(variable, script=''):
    return f"""
Given the following Python script provide a small description of the variable {variable}:.
Explain its role and significance within the script.

Script:
{textwrap.indent(script, '\t')}

Output:
Small description of variable {variable}:
"""

def _separate_variable_string(output_text):
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

def _separate_description_string(output_text):
    if "Output:" in output_text:
        description_str = output_text.split("Output:")[1].strip()
        description_str = description_str.split(":")[1].strip()
        description_str = description_str.split("\n")[0].strip()
    else:
        description_str = "LLM failed to generate description that fits the predefined structure."
    return description_str

def main():
    from llm_blockmerger.core.models import LLM
    #model_name = 'huggyllama/llama-7b'
    model_name = "meta-llama/Llama-3.2-3B"
    llama = LLM(task='question', model_name=model_name, verbose=False)

    blocks = [['x = 1', 'y = 2'], ['z = x + y']]
    labels = ['Simple addition code']

    from llm_blockmerger.load.block_loading import CodeBlocksManager
    demo_manager = CodeBlocksManager(blocks=blocks, labels=labels)
    extract_notebook_variables(demo_manager, llama)

    for block_variables, block_descriptions in zip(demo_manager.variables, demo_manager.var_descriptions):
        for v, d in zip(block_variables, block_descriptions):
            print(f'{v}: {d}')
        print('-'*40)

if __name__ == '__main__':
    main()