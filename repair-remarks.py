# repair-remarks.py
import os
import sys
import tokenize
from io import StringIO

def process_python_files(directory, excluded_files=None):
    if excluded_files is None:
        excluded_files = []
    for root, dirs, files in os.walk(directory):
        if root != directory:
            continue
        for file in files:
            if file.endswith('.py') and file not in excluded_files:
                file_path = os.path.join(root, file)
                yield file_path, file

def process_file(file_path, filename):
    try:
        # Read original content
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
            original_length = len(original_content)
        
        # Token processing
        with tokenize.open(file_path) as f:
            tokens = list(tokenize.generate_tokens(f.readline))
        
        correct_comment = f'# {filename}'
        new_tokens = []
        encoding_declared = False
        filename_remark_added = False
        
        for token in tokens:
            if token.type == tokenize.COMMENT:
                if not encoding_declared:
                    # Preserve encoding declaration
                    new_tokens.append(token)
                    encoding_declared = True
                    if token.string.strip() == correct_comment:
                        filename_remark_added = True
                elif not filename_remark_added:
                    # Check if the comment is the filename remark
                    if token.string.strip() == correct_comment:
                        new_tokens.append(token)
                        filename_remark_added = True
                    # Skip other comments
                    continue
                else:
                    # Skip additional comments
                    continue
            else:
                new_tokens.append(token)
        
        # Insert filename remark if not added
        if not filename_remark_added:
            if encoding_declared:
                insert_pos = 1
            else:
                insert_pos = 0
            filename_token = tokenize.TokenInfo(
                type=tokenize.COMMENT,
                string=f'#{correct_comment}',
                start=(1, 0),
                end=(1, len(correct_comment)+1),
                line=f'{correct_comment}\n'
            )
            new_tokens.insert(insert_pos, filename_token)
        
        # Rebuild the file content
        content = tokenize.untokenize(new_tokens)
        if isinstance(content, bytes):
            new_content = content.decode('utf-8')
        else:
            new_content = content
        
        # Write the new content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        new_length = len(new_content)
        characters_deleted = original_length - new_length
        print(f"Deleted {characters_deleted} characters from {file_path}.")
        return characters_deleted
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        print(f"Exception type: {type(e)}, Exception args: {e.args}")
        return 0

def main():
    script_path = os.path.abspath(sys.argv[0])
    excluded_files = ['copyscripts.py', 'repair-remarks.py']
    directories = [os.getcwd()]
    scripts_dir = os.path.join(os.getcwd(), 'scripts')
    if os.path.isdir(scripts_dir):
        directories.append(scripts_dir)
    
    total_characters_deleted = 0
    for dir_path in directories:
        for file_path, filename in process_python_files(dir_path, excluded_files):
            # Exclude the script itself
            if os.path.abspath(file_path) == script_path:
                continue
            deleted = process_file(file_path, filename)
            total_characters_deleted += deleted
    
    print(f"Total characters deleted: {total_characters_deleted}")

if __name__ == "__main__":
    main()