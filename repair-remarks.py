# repair-remarks.py
import os
import sys

def process_python_files(directory, exclude_file=None, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if exclude_file and file_path == exclude_file:
                    continue
                process_file(file_path, file)

def process_file(file_path, filename):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        correct_comment = f'# {filename}'
        
        if lines and lines[0].strip() == correct_comment:
            modified_lines = [line for line in lines if not (line.strip().startswith('#') and line.strip() != correct_comment)]
        else:
            modified_lines = [f'{correct_comment}\n']
            modified_lines.extend([line for line in lines if not (line.strip().startswith('#') and line.strip() != correct_comment)])
        
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