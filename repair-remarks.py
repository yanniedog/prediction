# repair-remarks.py
import os
import sys
import tokenize
from io import StringIO

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
            content = f.read()
        # Identify the filename remark
        correct_comment = f'# {filename}'
        lines = content.splitlines()
        # Check if the correct comment is the first line
        if lines and lines[0].strip() == correct_comment:
            print(f"Correct comment already present in {file_path}. No changes made.")
            return
        # Read tokens and rebuild the file without comments
        tokens = []
        filename_remark = None
        with tokenize.open(file_path) as f:
            for token in tokenize.generate_tokens(f.readline):
                if token.type == tokenize.COMMENT:
                    if token.start[0] == 1 and token.string.strip() == correct_comment:
                        filename_remark = token
                    continue  # Skip comments
                tokens.append(token)
        # Add the filename remark at the beginning if not present
        if filename_remark is None:
            # Insert the filename remark as the first token
            tokens.insert(0, tokenize.TokenInfo(type=tokenize.COMMENT, string=' #' + filename, start=(1,0), end=(1,len(correct_comment)+1), line=correct_comment + '\n'))
        # Rebuild the file content from tokens
        new_content = tokenize.untokenize(tokens).decode('utf-8')
        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Modified {file_path}: Removed comments and added correct filename remark.")
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