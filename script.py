#!/usr/bin/env python3
import re
import sys
from collections import defaultdict

def extract_file_content(filename):
    """Extract content from the CLAUDE_chatlog.txt file."""
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    return content

def extract_code_blocks(content):
    """Extract Python code blocks from the chat log."""
    # Extract Python code blocks with ```python syntax
    code_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)
    python_blocks = [(m.group(1), content[:m.start()].count('\n') + 1) for m in code_pattern.finditer(content)]
    
    # Extract indented Python code blocks (without backticks)
    indented_pattern = re.compile(r'((?:^[^\S\n]*class\s+\w+.*?:.*?(?:\n[^\S\n]+.*?)*)|(?:^[^\S\n]*def\s+\w+.*?:.*?(?:\n[^\S\n]+.*?)*))', re.DOTALL | re.MULTILINE)
    indented_blocks = [(m.group(1), content[:m.start()].count('\n') + 1) for m in indented_pattern.finditer(content)]
    
    # Extract Python code introduced by variable assignment (no backticks)
    assignment_pattern = re.compile(r'((?:^[^\S\n]*\w+\s*=\s*.*?(?:\n[^\S\n]+.*?)*)|(?:^[^\S\n]*import\s+.*?(?:\n[^\S\n]+.*?)*))', re.DOTALL | re.MULTILINE)
    assignment_blocks = [(m.group(1), content[:m.start()].count('\n') + 1) for m in assignment_pattern.finditer(content)]
    
    # Combine and sort by line number (descending)
    all_blocks = python_blocks + indented_blocks + assignment_blocks
    all_blocks.sort(key=lambda x: x[1], reverse=True)
    return all_blocks

def extract_yaml_blocks(content):
    """Extract YAML code blocks from the chat log."""
    # Extract YAML blocks with ```yaml syntax
    yaml_pattern = re.compile(r'```ya?ml\s*(.*?)\s*```', re.DOTALL)
    yaml_blocks = [(m.group(1), content[:m.start()].count('\n') + 1) for m in yaml_pattern.finditer(content)]
    
    # Extract YAML-like indented blocks without backticks
    indented_yaml_pattern = re.compile(r'(?:^[^\S\n]*\w+:\s*(?:\n[^\S\n]+\w+:.*?)*)', re.DOTALL | re.MULTILINE)
    indented_blocks = [(m.group(0), content[:m.start()].count('\n') + 1) for m in indented_yaml_pattern.finditer(content)]
    
    # Combine and sort by line number (descending)
    all_blocks = yaml_blocks + indented_blocks
    all_blocks.sort(key=lambda x: x[1], reverse=True)
    return all_blocks

def classify_code_blocks(blocks, content):
    """Classify code blocks based on their content and context."""
    classified_blocks = []
    
    for block, line_num in blocks:
        # Skip very short blocks
        if len(block.strip().split('\n')) < 2:
            continue
            
        # Look for filename in the block or preceding context
        block_first_lines = block.strip().split('\n')[:5]
        block_context = content.split('\n')[max(0, line_num-10):line_num]
        
        file_name = "unknown.py"
        
        # Check for explicit filename in block
        for line in block_first_lines:
            file_match = re.search(r'# (\S+\.py)', line)
            if file_match:
                file_name = file_match.group(1)
                break
        
        # Check for explicit filename in context
        if file_name == "unknown.py":
            for line in reversed(block_context):
                file_match = re.search(r'# (\S+\.py)', line)
                if file_match:
                    file_name = file_match.group(1)
                    break
                    
                # Also check for "File: path/to/file.py" format
                file_match = re.search(r'File:\s*(\S+\.py)', line)
                if file_match:
                    file_name = file_match.group(1)
                    break
        
        # Check for class names in the block
        classes = []
        for line in block.split('\n'):
            class_match = re.match(r'\s*class\s+(\w+)', line)
            if class_match:
                classes.append(class_match.group(1))
        
        # Check for context clues in the surrounding text
        if file_name == "unknown.py" and (len(block_context) > 0):
            context_text = ' '.join(block_context)
            
            # Look for AIS fuser references
            if re.search(r'ais[\s_-]*fuser', context_text.lower()):
                file_name = "ais_fuser.py"
            # Look for transformer references
            elif re.search(r'tr?aisformer', context_text.lower()):
                file_name = "traisformer.py"
            # Look for processor references
            elif re.search(r'processor', context_text.lower()):
                if re.search(r'graph', context_text.lower()):
                    file_name = "graph_processor.py"
                else:
                    file_name = "ais_processor.py"
            # Look for datamodule references
            elif re.search(r'datamodule', context_text.lower()):
                file_name = "datamodule.py"
            # Look for evaluation references
            elif re.search(r'evaluation', context_text.lower()):
                file_name = "evaluation.py"
            # Look for sweep references
            elif re.search(r'sweep', context_text.lower()):
                file_name = "sweep_runner.py"
            # Look for train references
            elif re.search(r'train', context_text.lower()):
                file_name = "train.py"
            # Look for util references
            elif re.search(r'util', context_text.lower()):
                if re.search(r'maritime', context_text.lower()):
                    file_name = "maritime_utils.py"
                elif re.search(r'metric', context_text.lower()):
                    file_name = "metrics.py"
                elif re.search(r'visual', context_text.lower()):
                    file_name = "visualization.py"
        
        # Classify based on content if still unknown
        if file_name == "unknown.py":
            if any(c for c in classes if 'AISFuser' in c):
                file_name = "ais_fuser.py"
            elif any(c for c in classes if 'TrAISformer' in c):
                file_name = "traisformer.py"
            elif any(c for c in classes if 'Processor' in c) or 'process' in block.lower():
                if 'graph' in block.lower():
                    file_name = "graph_processor.py"
                else:
                    file_name = "ais_processor.py"
            elif any(c for c in classes if 'DataModule' in c) or 'datamodule' in block.lower():
                file_name = "datamodule.py"
        
        classified_blocks.append((file_name, line_num, block, classes))
    
    return classified_blocks

def classify_yaml_blocks(blocks, content):
    """Classify YAML blocks based on their content and context."""
    classified_blocks = []
    
    for block, line_num in blocks:
        # Skip very short blocks
        if len(block.strip().split('\n')) < 2:
            continue
            
        # Look for filename in context
        block_context = content.split('\n')[max(0, line_num-10):line_num]
        
        file_name = "unknown.yaml"
        
        # Check for explicit filename in context
        for line in reversed(block_context):
            file_match = re.search(r'# (\S+\.ya?ml)', line)
            if file_match:
                file_name = file_match.group(1)
                break
                
            # Also check for "File: path/to/file.yaml" format
            file_match = re.search(r'File:\s*(\S+\.ya?ml)', line)
            if file_match:
                file_name = file_match.group(1)
                break
        
        # Check for context clues in the surrounding text
        if file_name == "unknown.yaml":
            context_text = ' '.join(block_context)
            
            # Look for configuration references
            if re.search(r'ais[\s_-]*fuser', context_text.lower()):
                file_name = "ais_fuser.yaml"
            elif re.search(r'tr?aisformer', context_text.lower()):
                file_name = "traisformer.yaml"
            elif re.search(r'ais[\s_-]*processed', context_text.lower()):
                file_name = "ais_processed.yaml"
            elif re.search(r'sweep', context_text.lower()):
                if re.search(r'ais[\s_-]*fuser', context_text.lower()):
                    file_name = "ais_fuser_sweep.yaml"
                else:
                    file_name = "sweep.yaml"
            elif re.search(r'base', context_text.lower()):
                file_name = "base.yaml"
            elif re.search(r'arch[\s_-]*search', context_text.lower()):
                file_name = "arch_search.yaml"
        
        # Check content for clues
        if file_name == "unknown.yaml":
            content_text = block.lower()
            if 'ais_fuser' in content_text or '_target_: models.aisfuser' in content_text:
                file_name = "ais_fuser.yaml"
            elif 'traisformer' in content_text or '_target_: models.traisformer' in content_text:
                file_name = "traisformer.yaml"
            elif 'ais_processed' in content_text:
                file_name = "ais_processed.yaml"
            elif 'sweep' in content_text:
                if 'ais_fuser' in content_text:
                    file_name = "ais_fuser_sweep.yaml"
                else:
                    file_name = "sweep.yaml"
            elif 'base' in content_text:
                file_name = "base.yaml"
            elif 'arch_search' in content_text:
                file_name = "arch_search.yaml"
        
        classified_blocks.append((file_name, line_num, block))
    
    return classified_blocks

def map_to_empty_files(py_blocks, yaml_blocks, empty_files):
    """Match extracted code blocks to empty files that need to be filled."""
    empty_file_map = {}
    
    for empty_file in empty_files:
        base_name = empty_file.split('/')[-1]
        # Skip __init__.py files
        if base_name == "__init__.py":
            continue
        
        matches = []
        
        # For Python files
        if base_name.endswith('.py'):
            for file_name, line_num, code, _ in py_blocks:
                if base_name == file_name or base_name in file_name or file_name in empty_file:
                    matches.append((file_name, line_num, code))
        
        # For YAML files
        elif base_name.endswith('.yaml'):
            for file_name, line_num, yaml_content in yaml_blocks:
                if base_name == file_name or base_name in file_name or file_name in empty_file:
                    matches.append((file_name, line_num, yaml_content))
        
        if matches:
            # Sort by line number (descending) to get the latest version
            matches.sort(key=lambda x: x[1], reverse=True)
            empty_file_map[empty_file] = matches[0]
    
    return empty_file_map

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <chatlog_file> [empty_files_list]")
        sys.exit(1)
    
    chatlog_file = sys.argv[1]
    content = extract_file_content(chatlog_file)
    
    # Extract code blocks
    code_blocks = extract_code_blocks(content)
    yaml_blocks = extract_yaml_blocks(content)
    
    # Classify the blocks
    py_blocks = classify_code_blocks(code_blocks, content)
    yaml_blocks = classify_yaml_blocks(yaml_blocks, content)
    
    # Print classified blocks
    print("\n=== PYTHON CODE BLOCKS ===")
    for file_name, line_num, _, classes in py_blocks:
        if classes:
            print(f"Line {line_num}: {file_name} containing classes: {', '.join(classes)}")
        else:
            print(f"Line {line_num}: {file_name}")
    
    print("\n=== YAML BLOCKS ===")
    for file_name, line_num, _ in yaml_blocks:
        print(f"Line {line_num}: {file_name}")
    
    # If empty files list is provided, map the extracted code to the empty files
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'r') as f:
            empty_files = [line.strip() for line in f if line.strip()]
        
        file_map = map_to_empty_files(py_blocks, yaml_blocks, empty_files)
        
        print("\n=== MAPPED TO EMPTY FILES ===")
        for empty_file, (found_file, line_num, content) in file_map.items():
            print(f"{empty_file} -> {found_file} (Line {line_num})")
            
            # Show a preview of the content
            preview = content[:100] + "..." if len(content) > 100 else content
            preview = preview.replace('\n', ' ')
            print(f"  Preview: {preview}")
            
            # Ask if we should populate the file
            if len(sys.argv) < 4 or sys.argv[3] != "--auto":
                response = input(f"Populate {empty_file} with this content? (y/n): ")
                if response.lower() == 'y':
                    # Create the directory if it doesn't exist
                    import os
                    os.makedirs(os.path.dirname(empty_file), exist_ok=True)
                    
                    # Write the content to the file
                    with open(empty_file, 'w') as f:
                        f.write(content)
                    print(f"  Created: {empty_file}")
            else:
                # Auto mode - populate all files
                import os
                os.makedirs(os.path.dirname(empty_file), exist_ok=True)
                
                # Write the content to the file
                with open(empty_file, 'w') as f:
                    f.write(content)
                print(f"  Created: {empty_file}")

if __name__ == "__main__":
    main()