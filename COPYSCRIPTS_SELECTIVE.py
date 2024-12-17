# COPYSCRIPTS_SELECTIVE.py
import os
import sys
import argparse
from datetime import datetime
import shutil
from collections import defaultdict
import chardet

def run_repair_remarks():
    repair_script = os.path.join(os.getcwd(), "repair-remarks.py")
    if os.path.exists(repair_script):
        try:
            os.system(f"python {repair_script}")
        except Exception:
            sys.exit(1)

run_repair_remarks()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a .SELECTIVE file containing contents of specified scripts.")
    parser.add_argument('-e', '--extensions', nargs='+', help='Additional file extensions to include (e.g., -e txt md)')
    parser.add_argument('-f', '--folders', nargs='+', help='Additional subdirectories to search for files (e.g., -f utils helpers)')
    return parser.parse_args()

def get_current_directory():
    return os.getcwd()

def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def backup_existing_selective_files(current_dir, work_dir_name):
    backup_base = r'C:\code\backups'
    backup_dir = os.path.join(backup_base, work_dir_name, 'copyscript-backups')
    os.makedirs(backup_dir, exist_ok=True)
    for item in os.listdir(current_dir):
        if item.lower().endswith('.selective'):
            original_path = os.path.join(current_dir, item)
            bak_filename = os.path.splitext(item)[0] + '.SELECTIVEBAK'
            backup_path = os.path.join(backup_dir, bak_filename)
            try:
                os.rename(original_path, os.path.join(current_dir, bak_filename))
                shutil.move(os.path.join(current_dir, bak_filename), backup_path)
            except Exception:
                continue

def collect_files(base_dirs, extensions, excluded_filenames, exclude_subdirs_map, always_excluded_subdirs):
    filename_map = defaultdict(list)
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
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
    for filename, paths in duplicate_files.items():
        pass

def read_file_contents(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        encoding = chardet.detect(raw_data)['encoding']
        return raw_data.decode(encoding, errors='replace').replace('\r\n', '\n')
    except:
        return "[Error reading file]"

def extract_relevant_log_section(log_content):
    lines = log_content.split('\n')
    error_indices = [i for i, line in enumerate(lines) if 'ERROR' in line]
    traceback_indices = [i for i, line in enumerate(lines) if 'Traceback' in line]
    first_error = error_indices[0] if error_indices else None
    first_traceback = traceback_indices[0] if traceback_indices else None
    start_index = max(first_error - 8, 0) if first_error is not None else max(first_traceback - 8, 0) if first_traceback is not None else 0
    return '\n'.join(lines[start_index:])

def generate_output(collected_files, log_content=None):
    header = ("I’m encountering an error in my script, and I’ve included the output along with all related project scripts below. "
              "Please review and provide a complete fix. Ensure your response includes a fully functional, error-free, and "
              "deployment-ready script, with no placeholders or omissions. Additionally, any revised code should be as compact as "
              "possible, without remarks or docstrings, while maintaining full functionality, compatibility, and interoperability. "
              "If no changes are needed, there’s no need to include the script in your response.\n\n")
    error_header = ""
    if log_content:
        error_header += (f"See the error I receive here:\n\n====================\n\n{log_content}\n\n====================\n\n")
    error_header += "I've listed all the scripts in this project here:\n\n====================\n\n"
    sections = [header + error_header]
    for filename, path in collected_files.items():
        sections.append(f"{filename}:\n{read_file_contents(path[0])}\n====================")
    return '\n'.join(sections)

def write_output_file(output_path, content):
    try:
        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
    except:
        pass

if __name__ == "__main__":
    current_dir = get_current_directory()
    work_dir_name = os.path.basename(os.path.normpath(current_dir))
    scripts_dir = os.path.join(current_dir, "scripts")
    backup_existing_selective_files(current_dir, work_dir_name)
    args = parse_arguments()
    timestamp = get_timestamp()
    output_filename = f"{work_dir_name}-{timestamp}.SELECTIVE"
    output_path = os.path.join(current_dir, output_filename)
    script_filename = os.path.basename(__file__).lower()
    excluded_filenames = {script_filename, 'parsetab.py', 'backup_cleanup.py', 'linear_regression.py', 'advanced_analysis.py', 'binance_historical_data_downloader.py', 'generate_heatmaps.py', 'backup_utils.py', 'copyscripts.py', 'COPYSCRIPTS_SELECTIVE.py', 'restore_backup.py', 'repair-remarks.py', 'cspell.json', 'REVERT-to-gpt-scripts.py', 'test_indicators.py', 'tweak_indicators.py', 'tweak_params.py', 'tweak_trials.py'}
    extensions = ['.py', '.ps', '.json']
    if args.extensions:
        extensions.extend([ext.lower() if ext.startswith('.') else f".{ext.lower()}" for ext in args.extensions])
    base_dirs = [current_dir, scripts_dir]
    if args.folders:
        base_dirs.extend([os.path.join(current_dir, folder) for folder in args.folders])
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
        sys.exit(0)
    log_files = [os.path.join(current_dir, f) for f in os.listdir(current_dir) if f.endswith('.log')]
    log_content = extract_relevant_log_section(read_file_contents(log_files[0])) if log_files else None
    output_content = generate_output(filename_map, log_content)
    write_output_file(output_path, output_content)