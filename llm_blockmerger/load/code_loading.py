def _extract_cell_content(notebook):
    code_lines, accumulated_markdown = [], []
    cell_markdown = ''

    for cell in notebook.get('cells', []):
        if cell['cell_type'] == 'markdown':
            cell_markdown += ''.join(cell['source'])
        elif cell['cell_type'] == 'code':
            code_lines.append(cell['source'])
            accumulated_markdown.append(cell_markdown.replace('#', '')  if cell_markdown else accumulated_markdown[-1])
            cell_markdown = ''
    return code_lines, accumulated_markdown


def _preprocess_code_lines(code_lines, accumulated_markdown):
    blocks, labels = [], []

    for md, code_section in zip(accumulated_markdown, code_lines):
        md_prefix = f'CONTEXT: {md}\nCOMMENT: ' if md else 'COMMENT: '
        sections = _split_into_sections(code_section)

        for section_function, section_lines in sections:
            section_blocks, section_labels = section_function(section_lines, md_prefix)

            blocks.extend(section_blocks)
            labels.extend(section_labels)
    return blocks, labels


def _split_into_sections(code_section):
    sections, current_section, function_stack = [], [], []
    section_types = {
        'func': _process_function_section,
        'non-func': _process_non_function_section
    }

    for line in code_section:
        stripped = line.strip()
        if not stripped: continue
        current_indent = len(line) - len(line.lstrip())

        while function_stack and current_indent <= function_stack[-1]['indent']:
            sections.append((section_types['func'], function_stack.pop()['lines']))
            current_section = []
            if function_stack: break

        if stripped.startswith(('def ', 'class ')):
            if current_section:
                sections.append((section_types['non-func'], current_section))
                current_section = []

            function_stack.append({
                'indent': current_indent,
                'lines': [line]
            })
            continue

        if function_stack: function_stack[-1]['lines'].append(line)
        else: current_section.append(line)

    while function_stack: sections.append((section_types['func'], function_stack.pop()['lines']))
    if current_section: sections.append((section_types['non-func'], current_section))
    return sections


def _process_function_section(section_lines, md_prefix):
    block_lines, comment_lines = [], []

    for line in section_lines:
        stripped = line.strip()
        if stripped.startswith('#'): comment_lines.append(stripped[1:].strip())
        else: block_lines.append(line)

    return [''.join(block_lines)], [md_prefix + ' '.join(comment_lines)]


def _process_non_function_section(section_lines, md_prefix):
    blocks, labels = [], []
    current_block, current_comments = [], []

    for line in section_lines:
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
            if current_block:
                blocks.append(''.join(current_block))
                labels.append(md_prefix + '\n'.join(current_comments))
                current_block = []
                current_comments = []
            current_comments.append(stripped[1:].strip())
        else:
            current_block.append(line)

    if current_block:
        blocks.append(''.join(current_block))
        labels.append(md_prefix + '\n'.join(current_comments))

    return blocks, labels