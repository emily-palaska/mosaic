import textwrap
from llm_blockmerger.core import (
    ast_extraction,
    var_separation,
    parse_vars,
    parse_desc
)

def extract_notebook_variables(block_manager, model, empty=False):
    if empty:
        block_manager.set(variables_dict=[{} for _ in range(len(block_manager))])
        return

    notebook_variables = set()
    for block in block_manager.blocks:
        try:
            block_variables = _extract_block_variables(block)
            block_descriptions = _extract_block_descriptions(block_variables, block, model)
            notebook_variables.update((v, d) for v, d in zip(block_variables, block_descriptions))
        except IndentationError: continue
        except Exception: raise
    variable_dictionaries = var_separation(block_manager.blocks, notebook_variables)
    block_manager.set(variable_dictionaries=variable_dictionaries)

def _extract_block_variables(script, model=None):
    if model is None:
        return ast_extraction(script)
    output_text = model.answer(_create_variable_extraction_prompt(script))
    return parse_vars(output_text)

def _extract_block_descriptions(variables, script, model):
    descriptions = []
    for variable in variables:
        output_text = model.answer(_create_variable_description_prompt(variable,script))
        descriptions.append(parse_desc(output_text))
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
Given the following Python script provide a small description of the variable {variable}.
Explain its role and significance within the script.

Script:
{textwrap.indent(script, '\t')}

Output:
Small description of variable {variable}:
"""

