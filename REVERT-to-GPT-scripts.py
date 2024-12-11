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
    
    # Split the content into sections based on separators
    sections = re.split(r'={5,}', content)
    
    scripts = []
    
    script_pattern = re.compile(
        r'^\s*\d+\)\s+([^\s]+)\s+\(located in the (working directory|\'([^\']+)\' subdirectory)\):\s*\n', 
        re.MULTILINE
    )
    
    for section in sections:
        match = script_pattern.search(section)
        if match:
            filename = match.group(1).strip()
            location = match.group(2).strip()
            subdirectory = match.group(3)  # This will be None if location is 'working directory'
            
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

def replace_scripts(scripts):
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
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_file = target_dir / filename
        
        try:
            with target_file.open('w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"Replaced script: {target_file}")
        except Exception as e:
            logging.error(f"Failed to write to {target_file}: {e}")

def main():
    """
    Main function to execute the script replacement.
    """
    # Determine the GPT file path
    if len(sys.argv) > 1:
        gpt_filename = sys.argv[1]
    else:
        gpt_filename = 'scripts.gpt'  # Default GPT filename
    
    gpt_filepath = Path.cwd() / gpt_filename
    
    logging.info(f"Parsing GPT file: {gpt_filepath}")
    
    scripts = parse_gpt_file(gpt_filepath)
    
    if not scripts:
        logging.warning("No scripts found in the GPT file.")
        sys.exit(0)
    
    logging.info(f"Found {len(scripts)} scripts to replace.")
    
    replace_scripts(scripts)
    
    logging.info("All scripts have been replaced successfully.")

if __name__ == "__main__":
    main()
