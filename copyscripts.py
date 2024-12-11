# copyscripts.py
# copyscripts.py
import os
import sys
import argparse
from datetime import datetime
import shutil
from collections import defaultdict
import chardet  # For encoding detection
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate a .GPT file containing contents of specified scripts."
    )
    parser.add_argument(
        '-e', '--extensions',
        nargs='+',
        help='Additional file extensions to include (e.g., -e txt md)'
    )
    parser.add_argument(
        '-f', '--folders',
        nargs='+',
        help='Additional subdirectories to search for files (e.g., -f utils helpers)'
    )
    return parser.parse_args()
def get_current_directory():
    return os.getcwd()
def get_directory_name(path):
    return os.path.basename(os.path.normpath(path))
def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")
def backup_existing_gpt_files(current_dir, work_dir_name):
    backup_base = r'C:\code\backups'
    backup_dir = os.path.join(backup_base, work_dir_name, 'copyscript-backups')
    try:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"Backup directory ensured at '{backup_dir}'.")
    except Exception as e:
        print(f"Error creating backup directory '{backup_dir}': {e}")
        sys.exit(1)
    for item in os.listdir(current_dir):
        if item.lower().endswith('.gpt'):
            original_path = os.path.join(current_dir, item)
            bak_filename = os.path.splitext(item)[0] + '.GPTBAK'
            backup_path = os.path.join(backup_dir, bak_filename)
            try:
                os.rename(original_path, os.path.join(current_dir, bak_filename))
                print(f"Renamed '{item}' to '{bak_filename}'.")
                shutil.move(os.path.join(current_dir, bak_filename), backup_path)
                print(f"Moved '{bak_filename}' to '{backup_path}'.")
            except Exception as e:
                print(f"Error backing up '{item}': {e}")
                continue
def collect_files(base_dirs, extensions, excluded_filenames, exclude_subdirs_map, always_excluded_subdirs):
    filename_map = defaultdict(list)
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: The directory '{base_dir}' does not exist and will be skipped.")
            continue
        exclude_subdirs = exclude_subdirs_map.get(base_dir, []) + always_excluded_subdirs
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d not in exclude_subdirs and not d.startswith('.')]
            for file in files:
                if file.startswith('.'):
                    continue
                if file.lower() in excluded_filenames:
                    continue
                if any(file.lower().endswith(ext.lower()) for ext in extensions) or file.lower() == 'requirements.txt':
                    full_path = os.path.join(root, file)
                    filename_map[file.lower()].append(full_path)
    return filename_map
def alert_duplicate_filenames(duplicate_files):
    print("\n=== Duplicate Filenames Detected ===\n")
    for filename, paths in duplicate_files.items():
        print(f"Duplicate Filename: {filename}")
        print("-----------------------------------")
        for idx, path in enumerate(paths, start=1):
            try:
                size = os.path.getsize(path)
                ctime = datetime.fromtimestamp(os.path.getctime(path)).strftime('%Y-%m-%d %H:%M:%S')
                mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    num_lines = len(lines)
            except Exception as e:
                print(f"Error retrieving information for '{path}': {e}")
                size = 'N/A'
                ctime = 'N/A'
                mtime = 'N/A'
                num_lines = 'N/A'
            print(f"File {idx}:")
            print(f"  Path           : {path}")
            print(f"  Size           : {size} bytes")
            print(f"  Created        : {ctime}")
            print(f"  Last Modified  : {mtime}")
            print(f"  Lines of Code  : {num_lines}")
            print()
        print("-----------------------------------\n")
    print("Please resolve duplicate filenames to ensure each script is uniquely identified.\n")
def read_file_contents(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        content = raw_data.decode(encoding, errors='replace')
        return content.replace('\r\n', '\n')  # Standardize line endings
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return "[Error reading file]"
def read_single_log_file(log_files):
    if len(log_files) == 1:
        return read_file_contents(log_files[0])
    return None
def generate_output(collected_files, log_content=None):
    header = (
        "I encounter the following error when running my script. Below, I’ve included the output I received, followed by all the scripts in my project. Please provide a complete and working fix for any scripts requiring revision. Do not truncate or omit any code; provide full, functional, and production-ready revisions. Ensure all code you provide is complete, error-free, and ready for deployment, with no placeholders, hypothetical examples, or omissions.\n\n"
    )
    error_header = ""
    if log_content:
        error_header += (
            f"See the error I receive here:\n\n====================\n\n{log_content}\n\n====================\n\n"
        )
    error_header += "I've listed all the scripts in this project here:\n\n====================\n\n"
    sections = [header + error_header]
    for idx, (filename, path) in enumerate(collected_files, start=1):
        relative_path = os.path.relpath(path, start=current_dir)
        location = (
            "located in the 'scripts' subdirectory"
            if os.path.commonpath([path, scripts_dir]) == scripts_dir
            else "located in the working directory"
        )
        file_contents = read_file_contents(path)
        section = (
            f"{idx}) {os.path.basename(relative_path)} ({location}):\n"
            f"{file_contents}"
            f"\n===================="
        )
        sections.append(section)
    content = '\n'.join(sections)
    return content
def write_output_file(output_path, content):
    try:
        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        print(f"Successfully created '{output_path}'.")
    except Exception as e:
        print(f"Error writing to file '{output_path}': {e}")
if __name__ == "__main__":
    current_dir = get_current_directory()
    work_dir_name = get_directory_name(current_dir)
    scripts_dir = os.path.join(current_dir, "scripts")
    backup_existing_gpt_files(current_dir, work_dir_name)
    args = parse_arguments()
    timestamp = get_timestamp()
    output_filename = f"{work_dir_name}-{timestamp}.GPT"
    output_path = os.path.join(current_dir, output_filename)
    script_filename = os.path.basename(__file__).lower()
    excluded_filenames = {script_filename, 'parsetab.py', 'copyscripts.py', 'repair-remarks.py', 'cspell.json'}
    extensions = ['.py', '.ps']  # Include .ps files
    if args.extensions:
        additional_ext = [ext.lower() if ext.startswith('.') else f".{ext.lower()}" for ext in args.extensions]
        extensions.extend(additional_ext)
    base_dirs = [current_dir, scripts_dir]
    base_dirs = list(dict.fromkeys(base_dirs))
    if args.folders:
        for folder in args.folders:
            additional_dir = os.path.join(current_dir, folder)
            if additional_dir not in base_dirs:
                base_dirs.append(additional_dir)
    exclude_subdirs_map = {
        current_dir: ['scripts']
    }
    always_excluded_subdirs = ['venv', '.venv']
    filename_map = collect_files(base_dirs, extensions, excluded_filenames, exclude_subdirs_map, always_excluded_subdirs)
    duplicate_filenames = {fname: paths for fname, paths in filename_map.items() if len(paths) > 1}
    if duplicate_filenames:
        alert_duplicate_filenames(duplicate_filenames)
        for fname in duplicate_filenames.keys():
            del filename_map[fname]
    unique_files = [(fname, paths[0]) for fname, paths in filename_map.items()]
    if not unique_files:
        print("No files found matching the specified criteria or all have duplicates.")
        sys.exit(0)
    log_files = [os.path.join(current_dir, f) for f in os.listdir(current_dir) if f.endswith('.log')]
    log_content = read_single_log_file(log_files)
    output_content = generate_output(unique_files, log_content)
    write_output_file(output_path, output_content)