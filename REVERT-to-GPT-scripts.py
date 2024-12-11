import os
import sys
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Files to exclude from management
EXCLUDED_FILES = {
    '.gitignore',
    'copyscripts.py',
    'repair-remarks.py',
    'REVERT-to-GPT-scripts.py'
}

def list_gpt_files(directory: Path) -> list:
    """
    Lists all .GPT files in the specified directory.

    Parameters:
    - directory: Path object of the directory to search.

    Returns:
    - List of Path objects for each .GPT file found.
    """
    gpt_files = list(directory.glob('*.GPT'))
    return gpt_files

def prompt_user_to_select_gpt(gpt_files: list) -> Path:
    """
    Prompts the user to select a GPT file from the list.

    Parameters:
    - gpt_files: List of Path objects for GPT files.

    Returns:
    - Path object of the selected GPT file.
    """
    if not gpt_files:
        logging.error("No .GPT files found in the working directory.")
        sys.exit(1)
    
    if len(gpt_files) == 1:
        selected = gpt_files[0]
        logging.info(f"Only one GPT file found: '{selected.name}'. Selecting it by default.")
        return selected
    
    logging.info("Multiple GPT files found:")
    for idx, gpt in enumerate(gpt_files, start=1):
        logging.info(f"{idx}) {gpt.name}")
    
    while True:
        try:
            choice = input(f"Enter the number of the GPT file you wish to restore (1-{len(gpt_files)}): ").strip()
            if not choice.isdigit():
                logging.warning("Invalid input. Please enter a number.")
                continue
            choice = int(choice)
            if 1 <= choice <= len(gpt_files):
                selected = gpt_files[choice - 1]
                logging.info(f"Selected GPT file: '{selected.name}'.")
                return selected
            else:
                logging.warning(f"Please enter a number between 1 and {len(gpt_files)}.")
        except Exception as e:
            logging.error(f"Error during selection: {e}")

def confirm_overwrite() -> None:
    """
    Prompts the user to confirm overwriting scripts by typing "I am sure".
    Repeats until the user inputs the exact phrase.
    """
    prompt = 'Are you sure you want to overwrite the existing scripts with the ones in the selected GPT file? Type "I am sure" to proceed: '
    while True:
        confirmation = input(prompt).strip()
        if confirmation == "I am sure":
            logging.info("Confirmation received. Proceeding with script replacement.")
            break
        else:
            logging.warning('Confirmation failed. Please type "I am sure" to proceed.')

def parse_gpt_file(gpt_filepath: Path):
    """
    Parses the GPT file and extracts scripts with their filenames and target directories.

    Parameters:
    - gpt_filepath: Path object pointing to the GPT file.

    Returns:
    - List of dictionaries with 'filename', 'directory', and 'content' keys.
    """
    if not gpt_filepath.exists():
        logging.error(f"GPT file '{gpt_filepath}' does not exist.")
        sys.exit(1)
    
    with gpt_filepath.open('r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content into sections based on separators (5 or more equal signs)
    sections = re.split(r'={5,}', content)
    
    scripts = []
    
    # Pattern to match script headers, e.g.,
    # 1) launch.py (located in the working directory):
    # or
    # 2) backup_cleanup.py (located in the 'scripts' subdirectory):
    script_header_pattern = re.compile(
        r'^\s*\d+\)\s+([^\s]+)\s+\(located in the (working directory|\'([^\']+)\' subdirectory)\):\s*\n',
        re.MULTILINE
    )
    
    for section in sections:
        match = script_header_pattern.search(section)
        if match:
            filename = match.group(1).strip()
            location = match.group(2).strip()
            subdirectory = match.group(3)  # This will be None if location is 'working directory'
            
            if filename in EXCLUDED_FILES:
                logging.info(f"Skipping excluded file: '{filename}'.")
                continue
            
            if subdirectory:
                target_dir = Path(subdirectory)
            else:
                target_dir = Path.cwd()
            
            # Extract the script content after the header line
            # Find the position where the match ends
            start_pos = match.end()
            script_content = section[start_pos:].strip()
            
            # Remove any leading/trailing delimiters or newlines
            script_content = script_content.strip('`').strip()
            
            scripts.append({
                'filename': filename,
                'directory': target_dir,
                'content': script_content
            })
        else:
            continue  # Non-script sections are ignored
    
    return scripts

def replace_scripts(scripts: list):
    """
    Replaces existing scripts with the provided scripts.

    Parameters:
    - scripts: List of dictionaries with 'filename', 'directory', and 'content' keys.
    """
    for script in scripts:
        target_dir = script['directory']
        filename = script['filename']
        content = script['content']
        
        # Determine the absolute path
        if not target_dir.is_absolute():
            target_dir = Path.cwd() / target_dir
        
        # Exclude certain files
        if filename in EXCLUDED_FILES:
            logging.info(f"Skipping excluded file: '{filename}'.")
            continue
        
        # Ensure target directory exists
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Ensured directory exists: {target_dir}")
        except Exception as e:
            logging.error(f"Failed to create directory '{target_dir}': {e}")
            continue
        
        target_file = target_dir / filename
        
        # Delete existing script if it exists
        try:
            if target_file.exists():
                target_file.unlink()
                logging.info(f"Deleted existing script: {target_file}")
        except Exception as e:
            logging.error(f"Failed to delete existing script '{target_file}': {e}")
            continue
        
        # Write new script content
        try:
            with target_file.open('w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"Created/updated script: {target_file}")
        except Exception as e:
            logging.error(f"Failed to write to '{target_file}': {e}")

def main():
    """
    Main function to execute the script replacement.
    """
    working_dir = Path.cwd()
    gpt_files = list_gpt_files(working_dir)
    
    selected_gpt = prompt_user_to_select_gpt(gpt_files)
    
    confirm_overwrite()
    
    logging.info(f"Parsing GPT file: {selected_gpt}")
    
    scripts = parse_gpt_file(selected_gpt)
    
    if not scripts:
        logging.warning("No valid scripts found in the GPT file.")
        sys.exit(0)
    
    logging.info(f"Found {len(scripts)} scripts to replace.")
    
    replace_scripts(scripts)
    
    logging.info("Script replacement completed successfully.")

if __name__ == "__main__":
    main()
