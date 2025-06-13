from llm_blockmerger.core import dedent_blocks

def cell_content(nb):
    lines, acc_md = [], []
    curr_md = ''

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'markdown':
            curr_md += ''.join(cell['source'])

        elif cell['cell_type'] == 'code':
            lines.append(cell['source'])

            if curr_md:  md = curr_md.replace('#', '')
            elif acc_md: md = acc_md[-1]
            else: md = ''
            acc_md.append(md)
            curr_md = ''

    return lines, acc_md


def generate_blocks(lines, acc_md):
    blocks, labels = [], []

    for md, section in zip(acc_md, lines):
        md_prefix = f'CONTEXT: {md}\nCOMMENT: ' if md else 'COMMENT: '
        sections = _split_sections(section)

        for func, lines in sections:
            sec_blocks, sec_labels = func(lines, md_prefix)
            sec_blocks = dedent_blocks(sec_blocks)
            blocks.extend(sec_blocks)
            labels.extend(sec_labels)
    return blocks, labels


def _split_sections(section):
    sections, curr_sec, types_stack = [], [], []
    types = {
        'func': func_section,
        'main': main_section
    }

    for line in section:
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

            types_stack.append({
                'indent': curr_ind,
                'lines': [line]
            })
            continue

        if types_stack: types_stack[-1]['lines'].append(line)
        else: curr_sec.append(line)

    while types_stack: sections.append((types['func'], types_stack.pop()['lines']))
    if curr_sec: sections.append((types['main'], curr_sec))
    return sections


def func_section(lines, md_prefix):
    block_lines, comment_lines = [], []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'): comment_lines.append(stripped[1:].strip())
        else: block_lines.append(line)

    return [''.join(block_lines)], [md_prefix + ' '.join(comment_lines)]


def main_section(sec_lines, md_prefix):
    blocks, labels = [], []
    curr_block, curr_comments = [], []

    for line in sec_lines:
        stripped = line.strip()

        # Handle side comments
        if '#' in line and not stripped.startswith('#'):
            code_part, comment_part = line.split('#', 1)
            if code_part.strip():
                blocks.append(code_part.rstrip())
                labels.append(md_prefix + comment_part.strip())
            line = code_part.rstrip() + '\n'
            stripped = line.strip()

        # Handle full-line comments
        if stripped.startswith('#'):
            if curr_block:
                blocks.append(''.join(curr_block))
                labels.append(md_prefix + '\n'.join(curr_comments))
                curr_block, curr_comments = [], []
            curr_comments.append(stripped[1:].strip())
        else:
            curr_block.append(line)

    if curr_block:
        blocks.append(''.join(curr_block))
        labels.append(md_prefix + '\n'.join(curr_comments))
    return blocks, labels

