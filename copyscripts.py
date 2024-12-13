# copyscripts.py
import os, sys, argparse, shutil
from datetime import datetime
from collections import defaultdict
import chardet

def run_repair_remarks():
    script = os.path.join(os.getcwd(), "repair-remarks.py")
    if os.path.exists(script):
        try: os.system(f"python {script}")
        except Exception as e: print(f"Error running repair_remarks.py: {e}"); sys.exit(1)

run_repair_remarks()

def parse_arguments():
    p = argparse.ArgumentParser(description="Generate .GPT file containing script contents.")
    p.add_argument('-e', '--extensions', nargs='+', help='Additional file extensions (e.g., -e txt md)')
    p.add_argument('-f', '--folders', nargs='+', help='Subdirectories to search (e.g., -f utils helpers)')
    return p.parse_args()

def get_current_directory():
    return os.getcwd()

def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def backup_existing_gpt_files(cdir, wname):
    bdir = os.path.join(r'C:\code\backups', wname, 'copyscript-backups')
    os.makedirs(bdir, exist_ok=True)
    for item in os.listdir(cdir):
        if item.lower().endswith('.gpt'):
            opath = os.path.join(cdir, item)
            bname = os.path.splitext(item)[0] + '.GPTBAK'
            shutil.move(opath, os.path.join(bdir, bname))

def collect_files(bdirs, exts, excl_files, excl_dirs_map, always_excl):
    fmap = defaultdict(list)
    for bdir in bdirs:
        if not os.path.exists(bdir): continue
        excl_dirs = excl_dirs_map.get(bdir, []) + always_excl
        for root, dirs, files in os.walk(bdir):
            dirs[:] = [d for d in dirs if d not in excl_dirs and not d.startswith('.')]
            for file in files:
                if file.startswith('.') or file.lower() in excl_files: continue
                if file.lower() == 'indicator_params.json' or any(file.lower().endswith(ext) for ext in exts):
                    fmap[file.lower()].append(os.path.join(root, file))
    return fmap

def alert_duplicate_filenames(dups):
    for fname, paths in dups.items():
        print(f"Duplicate: {fname}")
        for p in paths: print(f"  {p}")

def read_file_contents(fp):
    try:
        with open(fp, 'rb') as f: raw = f.read()
        return raw.decode(chardet.detect(raw)['encoding'], errors='replace').replace('\r\n', '\n')
    except Exception: return "[Error reading file]"

def extract_relevant_log_section(log_content):
    lines = log_content.split('\n')
    eidx = next((i for i, line in enumerate(lines) if 'ERROR' in line), None)
    tidx = next((i for i, line in enumerate(lines) if 'Traceback' in line), None)
    start = max((eidx or tidx or 0) - 8, 0)
    return '\n'.join(lines[start:])

def generate_output(files, logs=None):
    h = (
        "I’m encountering an error in my script, and I’ve included the output along with all related project scripts below. "
        "Please review and provide a complete fix. Ensure your response includes a fully functional, error-free, and "
        "deployment-ready script, with no placeholders or omissions. Additionally, any revised code should be as compact as "
        "possible, without remarks or docstrings, while maintaining full functionality, compatibility, and interoperability. "
        "If no changes are needed, there’s no need to include the script in your response.\n\n"
    )
    log_h = f"See the error I receive here:\n\n====================\n\n{logs}\n\n====================\n\n" if logs else ""
    out = [h + log_h + "I've listed all the scripts in this project here:\n\n====================\n\n"]
    for idx, (fname, path) in enumerate(files, 1):
        out.append(f"{idx}) {fname}:\n{read_file_contents(path)}\n====================")
    return '\n'.join(out)

def write_output_file(path, content):
    try:
        with open(path, 'w', encoding='utf-8') as f: f.write(content)
    except Exception as e: print(f"Error writing '{path}': {e}")

if __name__ == "__main__":
    cdir = get_current_directory()
    wname = os.path.basename(cdir)
    args = parse_arguments()
    backup_existing_gpt_files(cdir, wname)
    exts = ['.py', '.ps']
    if args.extensions: exts.extend([f".{x.lstrip('.')}" for x in args.extensions])
    bdirs = list(dict.fromkeys([cdir, os.path.join(cdir, "scripts")] + [os.path.join(cdir, f) for f in (args.folders or [])]))
    fnames = {os.path.basename(__file__).lower(), 'parsetab.py', 'copyscripts.py', 'repair-remarks.py', 'cspell.json', 'revert-to-gpt-scripts.py'}
    excl_dirs_map, always_excl = {cdir: ['scripts']}, ['venv', '.venv']
    fmap = collect_files(bdirs, exts, fnames, excl_dirs_map, always_excl)
    dups = {f: p for f, p in fmap.items() if len(p) > 1}
    if dups: alert_duplicate_filenames(dups)
    unique_files = [(f, p[0]) for f, p in fmap.items() if len(p) == 1]
    if not unique_files: print("No files found."); sys.exit(0)
    logs = [read_file_contents(os.path.join(cdir, f)) for f in os.listdir(cdir) if f.endswith('.log')]
    output = generate_output(unique_files, extract_relevant_log_section(logs[0]) if logs else None)
    write_output_file(os.path.join(cdir, f"{wname}-{get_timestamp()}.GPT"), output)
