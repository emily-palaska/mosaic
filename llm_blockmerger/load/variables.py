from textwrap import indent
from llm_blockmerger.core import ast_extraction, var_separation, parse_vars, parse_desc

def nb_variables(manager, model, empty=False):
    if empty:
        manager.set()
        return

    nb_vars = set()
    for block in manager.blocks:
        try:
            block_vars = _block_vars(block)
            block_descs = _block_descs(block_vars, block, model)
            nb_vars.update((v, d) for v, d in zip(block_vars, block_descs))
        except IndentationError: continue
        except Exception: raise
    var_dicts = var_separation(manager.blocks, nb_vars)
    manager.set(var_dicts=var_dicts)

def _block_vars(script, model=None):
    if model is None:
        return ast_extraction(script)
    output = model.answer(_var_prompt(script))
    return parse_vars(output)

def _block_descs(variables, script, model):
    descriptions = []
    for variable in variables:
        output = model.answer(_desc_prompt(variable, script))
        descriptions.append(parse_desc(output))
    return descriptions

def _var_prompt(script=''):
    return f"""
Analyze the following Python script and create a list of all the variables you can find. 
Return only the list of variable names.
Note that they should be separated by commas (,).

Script:
{indent(script, '\t')}

Output:
"""

def _desc_prompt(variable, script=''):
    return f"""
Given the following Python script provide a small description of the variable {variable}.
Explain its role and significance within the script.

Script:
{indent(script, '\t')}

Output:
Small description of variable {variable}:
"""

