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
        correct_comment, mod_lines, found, inline, stand, dup = f'# {name}', [], False, 0, 0, 0
        for line in lines:
            strip = line.strip()
            if strip == correct_comment:
                if not found: mod_lines.append(line); found = True
                else: dup += 1
            elif strip.startswith('#'): stand += 1
            else:
                if '#' in line:
                    code, _, comment = line.partition('#')
                    if code.strip():
                        mod_lines.append(code.rstrip() + '\n')
                        if comment.strip(): inline += 1
                    else: mod_lines.append(line)
                else: mod_lines.append(line)
        if not found: mod_lines.insert(0, f'{correct_comment}\n')
        if inline or stand or dup:
            with open(path, 'w', encoding='utf-8') as f: f.writelines(mod_lines)
        print(f"Modified {path}: " + (f"{'Updated' if found else 'Added'} correct comment, Removed {inline} inline, {stand} standalone, {dup} duplicate comments." if inline or stand or dup else "No changes made."))
    except Exception as e: print(f"Error processing {path}: {e}")
def main():
    dirs, exclude = [os.getcwd(), os.path.join(os.getcwd(), 'scripts')], ['.venv', 'venv', '_pycache_']
    for d in dirs:
        if os.path.exists(d): process_python_files(d, ex_file=sys.argv[0], ex_dirs=exclude)
        else: print(f"Directory {d} does not exist.")
if __name__ == "__main__": main()
