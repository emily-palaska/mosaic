import json

class BlockLoader:
    def __init__(self, notebook_paths):
        self.notebook_paths = notebook_paths
        self.notebook_data = []
        self.code_lines = []
        self.blocks = []
        self.labels = []

        self.load_notebooks()
        self.preprocess_blocks()

    def load_notebooks(self):
        for path in self.notebook_paths:
            # keep the raw json format of every notebook
            with open(path, 'r') as file:
                notebook_data = json.load(file)
            self.notebook_data.append(notebook_data)

            # keep code cells
            for cell in notebook_data['cells']:
                if cell['cell_type'] == 'code':
                    self.code_lines.append(cell['source'])

    def preprocess_blocks(self):
        """
        Extract blocks with their respective label (comment), discard uncommented blocks
        """
        for code in self.code_lines:
            found_comment = False
            current_block = []
            current_label = ''

            for line in code:
                if '#' in line and not line.startswith('#'): # case of comment next to line
                    before_hash, after_hash = line.split('#', 1)
                    self.blocks.append(before_hash)
                    self.labels.append(after_hash)
                elif line.startswith('#'): # case of large block (multiple lines)
                    if found_comment: # when there is a block already in progress
                        self.blocks.append(current_block)
                        self.labels.append(current_label)
                    else: # when it is the first large block found
                        found_comment = True

                    # empty current block and update current labels
                    _, current_label = line.split('#', 1)
                    current_block = []

                elif not '#' in line and found_comment: # case of line with no comment
                    current_block.append(line)

            # when block is over check if there are elements in the current block
            if current_block:
                self.blocks.append(current_block)
                self.labels.append(current_label)