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

# Directories to manage
MANAGED_DIRECTORIES = [
    Path.cwd(),
    Path.cwd() / 'scripts'
]

def list_gptbak_files(directory: Path) -> list:
    """
    Lists all .GPTBAK files in the specified directory.

    Parameters:
    - directory: Path object of the directory to search.

    Returns:
    - List of Path objects for each .GPTBAK file found.
    """
    gptbak_files = list(directory.glob('*.GPTBAK'))
    return gptbak_files

def prompt_user_to_select_gptbak(gptbak_files: list) -> Path:
    """
    Prompts the user to select a GPTBAK file from the list.

    Parameters:
    - gptbak_files: List of Path objects for GPTBAK files.

    Returns:
    - Path object of the selected GPTBAK file.
    """
    if not gptbak_files:
        logging.error("No .GPTBAK files found in the working directory.")
        sys.exit(1)
    
    if len(gptbak_files) == 1:
        selected = gptbak_files[0]
        logging.info(f"Only one GPTBAK file found: '{selected.name}'. Selecting it by default.")
        return selected
    
    logging.info("Multiple GPTBAK files found:")
    for idx, gptbak in enumerate(gptbak_files, start=1):
        logging.info(f"{idx}) {gptbak.name}")
    
    while True:
        try:
            choice = input(f"Enter the number of the GPTBAK file you wish to restore (1-{len(gptbak_files)}): ").strip()
            if not choice.isdigit():
                logging.warning("Invalid input. Please enter a number.")
                continue
            choice = int(choice)
            if 1 <= choice <= len(gptbak_files):
                selected = gptbak_files[choice - 1]
                logging.info(f"Selected GPTBAK file: '{selected.name}'.")
                return selected
            else:
                logging.warning(f"Please enter a number between 1 and {len(gptbak_files)}.")
        except Exception as e:
            logging.error(f"Error during selection: {e}")

def confirm_overwrite() -> None:
    """
    Prompts the user to confirm overwriting scripts by typing "I am sure".
    Repeats until the user inputs the exact phrase.
    """
    prompt = 'Are you sure you want to overwrite the existing scripts with the ones in the selected GPTBAK file? Type "I am sure" to proceed: '
    while True:
        confirmation = input(prompt).strip()
        if confirmation == "I am sure":
            logging.info("Confirmation received. Proceeding with script replacement.")
            break
        else:
            logging.warning('Confirmation failed. Please type "I am sure" to proceed.')

def parse_gptbak_file(gptbak_filepath: Path):
    """
    Parses the GPTBAK file and extracts scripts with their filenames and target directories.

    Parameters:
    - gptbak_filepath: Path object pointing to the GPTBAK file.

    Returns:
    - List of dictionaries with 'filename', 'directory', and 'content' keys.
    """
    if not gptbak_filepath.exists():
        logging.error(f"GPTBAK file '{gptbak_filepath}' does not exist.")
        sys.exit(1)
    
    with gptbak_filepath.open('r', encoding='utf-8') as file:
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

def find_existing_scripts(managed_dirs: list, filename: str) -> list:
    """
    Searches only the managed directories for existing scripts with the given filename.

    Parameters:
    - managed_dirs: List of Path objects representing directories to search.
    - filename: Name of the script file to search for.

    Returns:
    - List of Path objects where the script is found.
    """
    existing_scripts = []
    for directory in managed_dirs:
        # Search only in the specified directory, not recursively
        for file_path in directory.glob(filename):
            if file_path.name in EXCLUDED_FILES:
                continue
            existing_scripts.append(file_path)
    return existing_scripts

def replace_scripts(scripts: list, managed_dirs: list):
    """
    Replaces existing scripts with the provided scripts.

    Parameters:
    - scripts: List of dictionaries with 'filename', 'directory', and 'content' keys.
    - managed_dirs: List of Path objects representing directories to manage.
    """
    for script in scripts:
        filename = script['filename']
        target_dir = script['directory']
        content = script['content']
        
        # Ensure target directory is absolute
        if not target_dir.is_absolute():
            # Determine if the directory is relative to the working directory or the 'scripts' subdirectory
            # Since managed_dirs includes both, we assume subdirectories are relative to the working directory
            target_dir = Path.cwd() / target_dir
        
        # Ensure target directory exists
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Ensured directory exists: {target_dir}")
        except Exception as e:
            logging.error(f"Failed to create directory '{target_dir}': {e}")
            continue
        
        # Find and delete existing scripts with the same filename within managed directories
        existing_scripts = find_existing_scripts(managed_dirs, filename)
        for existing_script in existing_scripts:
            try:
                existing_script.unlink()
                logging.info(f"Deleted existing script: {existing_script}")
            except Exception as e:
                logging.error(f"Failed to delete '{existing_script}': {e}")
        
        # Define the path for the new script
        new_script_path = target_dir / filename
        
        # Write the new script content
        try:
            with new_script_path.open('w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"Created/Updated script: {new_script_path}")
        except Exception as e:
            logging.error(f"Failed to write to '{new_script_path}': {e}")

def main():
    """
    Main function to execute the script restoration.
    """
    working_dir = Path.cwd()
    scripts_dir = working_dir / 'scripts'
    managed_dirs = [working_dir, scripts_dir]
    
    gptbak_files = list_gptbak_files(working_dir)
    
    selected_gptbak = prompt_user_to_select_gptbak(gptbak_files)
    
    confirm_overwrite()
    
    logging.info(f"Parsing GPTBAK file: {selected_gptbak}")
    
    scripts = parse_gptbak_file(selected_gptbak)
    
    if not scripts:
        logging.warning("No valid scripts found in the GPTBAK file.")
        sys.exit(0)
    
    logging.info(f"Found {len(scripts)} scripts to replace.")
    
    replace_scripts(scripts, managed_dirs)
    
    logging.info("Script restoration completed successfully.")

if __name__ == "__main__":
    main()
