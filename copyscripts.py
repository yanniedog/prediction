# copyscripts.py
import os
import sys
import argparse
from datetime import datetime
import shutil
from collections import defaultdict
import chardet


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a .GPT file containing contents of specified scripts.")
    parser.add_argument('-e', '--extensions', nargs='+', help='Additional file extensions to include (e.g., -e txt md)')
    parser.add_argument('-f', '--folders', nargs='+', help='Additional subdirectories to search for files (e.g., -f utils helpers)')
    return parser.parse_args()


def get_current_directory():
    return os.getcwd()


def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def backup_existing_gpt_files(current_dir, work_dir_name):
    backup_base = r'C:\code\backups'
    backup_dir = os.path.join(backup_base, work_dir_name, 'copyscript-backups')
    try:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"Backup directory ensured at '{backup_dir}'.")
    except Exception as e:
        print(f"Error creating directory '{backup_dir}': {e}")
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
                if file.startswith('.') or file.lower() in excluded_filenames:
                    continue
                if any(file.lower().endswith(ext.lower()) for ext in extensions) or file.lower() == 'requirements.txt':
                    filename_map[file.lower()].append(os.path.join(root, file))
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
                    num_lines = len(f.readlines())
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
        return raw_data.decode(encoding, errors='replace').replace('\r\n', '\n')
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return "[Error reading file]"


def extract_relevant_log_section(log_content):
    """
    Extracts the log section starting three lines above the first occurrence
    of 'ERROR' or 'Traceback' (whichever comes first) and continues to the end.
    """
    lines = log_content.split('\n')
    error_indices = [i for i, line in enumerate(lines) if 'ERROR' in line]
    traceback_indices = [i for i, line in enumerate(lines) if 'Traceback' in line]

    first_error = error_indices[0] if error_indices else None
    first_traceback = traceback_indices[0] if traceback_indices else None

    # Determine which comes first
    if first_error is not None and (first_traceback is None or first_error < first_traceback):
        start_index = max(first_error - 3, 0)
    elif first_traceback is not None:
        start_index = max(first_traceback - 3, 0)
    else:
        # If neither ERROR nor Traceback is found, return the entire log
        return log_content

    # Join the lines from start_index to the end
    relevant_log = '\n'.join(lines[start_index:])
    return relevant_log


def generate_output(collected_files, log_content=None):
    header = (
        "I’m encountering an error in my script, and I’ve included the output along with all related project scripts below. "
        "Please review and provide a complete fix. Ensure your response includes a fully functional, error-free, and "
        "deployment-ready script, with no placeholders or omissions. Additionally, any revised code should be as compact as "
        "possible, without remarks or docstrings, while maintaining full functionality, compatibility, and interoperability. "
        "If no changes are needed, there’s no need to include the script in your response.\n\n"
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
        location = "located in the 'scripts' subdirectory" if os.path.commonpath([path, scripts_dir]) == scripts_dir else "located in the working directory"
        file_contents = read_file_contents(path)
        section = f"{idx}) {os.path.basename(relative_path)} ({location}):\n{file_contents}\n===================="
        sections.append(section)
    return '\n'.join(sections)


def write_output_file(output_path, content):
    try:
        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        print(f"Successfully created '{output_path}'.")
    except Exception as e:
        print(f"Error writing to file '{output_path}': {e}")


if __name__ == "__main__":
    current_dir = get_current_directory()
    work_dir_name = os.path.basename(os.path.normpath(current_dir))
    scripts_dir = os.path.join(current_dir, "scripts")
    backup_existing_gpt_files(current_dir, work_dir_name)
    args = parse_arguments()
    timestamp = get_timestamp()
    output_filename = f"{work_dir_name}-{timestamp}.GPT"
    output_path = os.path.join(current_dir, output_filename)
    script_filename = os.path.basename(__file__).lower()
    excluded_filenames = {script_filename, 'parsetab.py', 'copyscripts.py', 'repair-remarks.py', 'cspell.json', 'revert-to-gpt-scripts.py'}
    extensions = ['.py', '.ps']
    if args.extensions:
        extensions.extend([ext.lower() if ext.startswith('.') else f".{ext.lower()}" for ext in args.extensions])
    base_dirs = [current_dir, scripts_dir]
    base_dirs = list(dict.fromkeys(base_dirs))
    if args.folders:
        base_dirs.extend([os.path.join(current_dir, folder) for folder in args.folders if os.path.join(current_dir, folder) not in base_dirs])
    exclude_subdirs_map = {current_dir: ['scripts']}
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
    print(f"Log files found: {log_files}")
    log_content = None
    if log_files:
        full_log_content = read_file_contents(log_files[0])
        log_content = extract_relevant_log_section(full_log_content)

    output_content = generate_output(unique_files, log_content)
    write_output_file(output_path, output_content)
