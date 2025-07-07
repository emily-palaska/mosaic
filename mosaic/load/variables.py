from re import search
from textwrap import indent
from mosaic.core import ast_extraction

def nb_variables(manager, model, empty=False):
    if empty:
        manager.set(var_dicts=[dict() for _ in manager.blocks])
        return

    nb_vars = set()
    for block in manager.blocks:
        try:
            block_vars = _block_vars(block)
            block_descs = _block_descs(block_vars, block, model)
            nb_vars.update((v, d) for v, d in zip(block_vars, block_descs))
        except IndentationError: continue
        except Exception: raise
    var_dicts = _var_separation(manager.blocks, nb_vars)
    manager.set(var_dicts=var_dicts)


def _block_vars(script, model=None):
    if model is None:
        return ast_extraction(script)
    output = model.answer(_var_prompt(script))
    return _parse_vars(output)


def _block_descs(variables, script, model):
    descriptions = []
    for variable in variables:
        output = model.answer(_desc_prompt(variable, script))
        descriptions.append(_parse_desc(output))
    return descriptions


def _var_separation(blocks, nb_vars):
    var_dicts = []
    for block in blocks:
        block_dict = {}
        for var, desc in nb_vars:
            if search(rf'\b{var}\b', block):
                block_dict[var] = desc
        var_dicts.append(block_dict)
    return var_dicts


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


def _parse_vars(output):
    if "Output:" in output:
        var_str = output.split("Output:")[1].strip()
        var_str = var_str.split("\n")[0].strip()
        var_set = {
            var.strip().replace("[", "").replace("]", "")
            for var in var_str.split(",")
        }
        var_set.discard("")
    else:
        var_set = set()
    return sorted(var_set)


def _parse_desc(output):
    if "Output:" in output:
        description_str = output.split("Output:")[1].strip()
        description_str = description_str.split(":")[1].strip()
        description_str = description_str.split("\n")[0].strip()
    else:
        description_str = "LLM failed to generate description that fits the predefined structure."
    return description_str
