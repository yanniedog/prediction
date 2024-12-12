# repair-remarks.py
import os, sys
def process_python_files(d, ex_file=None, ex_dirs=None):
    ex_dirs = ex_dirs or []
    for root, dirs, files in os.walk(d):
        dirs[:] = [d for d in dirs if d not in ex_dirs]
        for file in files:
            if file.endswith('.py') and file != 'repair-remarks.py':
                path = os.path.join(root, file)
                if ex_file and path == ex_file: continue
                process_file(path, file)
def process_file(path, name):
    try:
        with open(path, 'r', encoding='utf-8') as f: lines = f.readlines()
        correct_comment = f'# {name}'
        mod_lines, found_correct_comment, modified = [], False, False
        for line in lines:
            if line.strip() == correct_comment:
                found_correct_comment = True
            mod_lines.append(line)
        if not found_correct_comment:
            mod_lines.insert(0, f'{correct_comment}\n')
            modified = True
        if modified:
            with open(path, 'w', encoding='utf-8') as f: f.writelines(mod_lines)
            print(f"Modified {path}: Added correct comment.")
        else:
            print(f"Checked {path}: No changes needed.")
    except Exception as e: print(f"Error processing {path}: {e}")
def main():
    dirs, exclude = [os.getcwd(), os.path.join(os.getcwd(), 'scripts')], ['.venv', 'venv', '_pycache_']
    for d in dirs:
        if os.path.exists(d): process_python_files(d, ex_file=sys.argv[0], ex_dirs=exclude)
        else: print(f"Directory {d} does not exist.")
if __name__ == "__main__": main()
