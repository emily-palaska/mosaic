from re import match, escape
from llm_blockmerger.core import dedent_blocks, separate_lines


def cell_content(nb):
    cells, acc_md = [], []
    curr_md = ''
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'markdown':
            curr_md += ''.join(cell['source'])

        elif cell['cell_type'] == 'code':
            cells.append(separate_lines(cell.get('source', "")))
            if curr_md:  md = curr_md.replace('#', '')
            elif acc_md: md = acc_md[-1]
            else: md = ''
            acc_md.append(md)
            curr_md = ''
    return cells, acc_md


def generate_blocks(cells, acc_md):
    assert len(cells) == len(acc_md), f"Invalid lengths {len(cells)} != {len(acc_md)}"
    blocks, labels = [], []

    for md, cell in zip(acc_md, cells):
        cell = reposition_comments(cell)
        md_prefix = f'CONTEXT: {md}\nCOMMENT: ' if md else 'COMMENT: '
        sections = _split_sections(cell)

        for func, lines in sections:
            sec_blocks, sec_labels = func(lines, md_prefix)
            if len(sec_blocks) == 0: continue
            blocks.extend(sec_blocks if isinstance(sec_blocks, list) else [sec_blocks])
            labels.extend(sec_labels)
    assert len(blocks) == len(labels), f"Invalid lengths {len(blocks)} != {len(labels)}"
    for block in blocks: assert isinstance(block, str), f"Invalid block type: {type(block)}"
    return blocks, labels

def _split_sections(cell):
    sections, curr_sec, types_stack = [], [], []
    types = {'func': func_section, 'main': main_section}

    for line in cell:
        stripped = line.strip()
        if not stripped: continue
        curr_ind = len(line) - len(line.lstrip())

        while types_stack and curr_ind <= types_stack[-1]['indent']:
            sections.append((types['func'], types_stack.pop()['lines']))
            curr_sec = []
            if types_stack: break

        if stripped.startswith(('def ', 'class ')):
            if curr_sec:
                sections.append((types['main'], curr_sec))
                curr_sec = []

            types_stack.append({'indent': curr_ind, 'lines': [line]})
            continue

        if types_stack: types_stack[-1]['lines'].append(line)
        else: curr_sec.append(line)

    while types_stack: sections.append((types['func'], types_stack.pop()['lines']))
    if curr_sec: sections.append((types['main'], curr_sec))
    return sections


def func_section(lines, prefix):
    block_lines, comment_lines = [], []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'): comment_lines.append(stripped[1:].strip())
        else: block_lines.append(line)

    return [''.join(block_lines)], [prefix + ' '.join(comment_lines)]


def main_section(lines, prefix):
    blocks, labels = [], []
    curr_block, curr_comments = [], []

    for line in lines:
        stripped = line.strip()

        if '#' in line and not stripped.startswith('#'):
            block, label, stripped, line = side_comment(line, prefix)
            if block:
                blocks.append(block)
                labels.append(label)

        if stripped.startswith('#'):
            if curr_block:
                blocks.append('\n'.join(curr_block))
                labels.append(prefix + '\n'.join(curr_comments))
                curr_block, curr_comments = [], []
            curr_comments.append(stripped[1:].strip())
        else:
            curr_block.append(line)

    if curr_block:
        blocks.append('\n'.join(curr_block))
        labels.append(prefix + '\n'.join(curr_comments))
    return blocks, labels


def side_comment(line, prefix):
    code, comment = hash_split(line)
    block, label = '', ''
    if code.strip() and comment:
        block = code.rstrip()
        label = prefix + comment.strip()
    line = code.rstrip() + '\n'
    stripped = line.strip()
    return block, label, stripped, line


def hash_split(line):
    # This regex splits on '#' only if it's not inside quotes
    import re
    parts = re.split(r'#(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)', line, maxsplit=1)
    comment = parts[1].strip() if len(parts) > 1 else ''
    return parts[0], comment


def reposition_comments(cell: list):
    script = '\n'.join(cell)
    lines, out_lines, i = script.splitlines(), [], 0
    block_keywords = {'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with', 'def', 'class'}

    while i < len(lines):
        line = lines[i]
        out_lines.append(line)

        # Match indent-introducing lines
        if m := match(r'^(\s*)(%s)\b.*:\s*$' % '|'.join(block_keywords), line):
            indent = m.group(1)
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if match(r'^' + escape(indent) + r'[ \t]+#', next_line):
                    out_lines.pop()
                    out_lines.append(next_line[4:])
                    out_lines.append(line)
                    i += 1
        i += 1
    return out_lines
