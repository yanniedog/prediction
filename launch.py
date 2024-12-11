# launch.py
import os
import sys

def process_python_files(directory, exclude_file=None, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith('.py') and file != 'repair-remarks.py':
                file_path = os.path.join(root, file)
                if exclude_file and file_path == exclude_file:
                    continue
                process_file(file_path, file)

def process_file(file_path, filename):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        correct_comment = f'
        
        modified_lines = []
        found_correct_comment = False
        
        for line in lines:
            stripped_line = line.strip()
            if stripped_line == correct_comment:
                if not found_correct_comment:
                    modified_lines.append(line)
                    found_correct_comment = True
            else:
                if '
                    code_part, _, _ = line.partition('
                    if code_part.strip():
                        modified_lines.append(code_part.rstrip() + '\n')
                else:
                    modified_lines.append(line)
        
        if not found_correct_comment:
            modified_lines.insert(0, f'{correct_comment}\n')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)
        print(f"Modified {file_path}: Added/Updated correct comment and removed in-line remarks.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    script_path = sys.argv[0]
    directories = [os.getcwd(), os.path.join(os.getcwd(), 'scripts')]
    exclude_dirs = ['.venv', 'venv', '_pycache_']
    for dir_path in directories:
        if os.path.exists(dir_path):
            process_python_files(dir_path, exclude_file=script_path, exclude_dirs=exclude_dirs)
        else:
            print(f"Directory {dir_path} does not exist.")

if __name__ == "__main__":
    main()