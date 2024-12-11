# repair-filename-remarks.py
import os
import sys

def process_python_files(directory, exclude_file=None, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        # Exclude directories whose names are in exclude_dirs
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                # Exclude the script itself if specified
                if exclude_file and file_path == exclude_file:
                    continue
                process_file(file_path, file)

def process_file(file_path, filename):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Read first three lines
        first_three = lines[:3]
        # Check if the correct comment is already present
        correct_comment = f'# {filename}'
        if any(line.strip().startswith(correct_comment) for line in first_three):
            print(f"Correct comment already present in {file_path}. No changes made.")
            return

        # Check for lines in the first three that end with '.py'
        # and are not the correct comment
        modified_lines = []
        found_py_line = False
        for line in first_three:
            stripped_line = line.strip()
            if stripped_line.endswith('.py'):
                # Check if it's the correct comment
                if not stripped_line.startswith(correct_comment):
                    # Skip this line (remove it)
                    found_py_line = True
                    continue
            modified_lines.append(line)
        # Add any remaining lines
        modified_lines += lines[3:]

        # If a .py line was found in the first three, insert the correct comment
        if found_py_line:
            modified_lines.insert(0, f'{correct_comment}\n')
            print(f"Modified {file_path}: Removed incorrect .py comment and added correct one.")
        else:
            # If correct comment not present, add it at the beginning
            modified_lines.insert(0, f'{correct_comment}\n')
            print(f"Modified {file_path}: Added correct comment.")

        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    # Get the script's own file path to exclude it from processing
    script_path = sys.argv[0]
    # List of directories to search
    directories = [os.getcwd(), os.path.join(os.getcwd(), 'scripts')]
    # List of directories to exclude
    exclude_dirs = ['.venv']
    for dir_path in directories:
        if os.path.exists(dir_path):
            process_python_files(dir_path, exclude_file=script_path, exclude_dirs=exclude_dirs)
        else:
            print(f"Directory {dir_path} does not exist.")

if __name__ == "__main__":
    main()