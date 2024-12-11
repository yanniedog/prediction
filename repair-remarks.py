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
        
        new_tokens = []
        previous_token_type = None
        in_comment = False
        in_string = False
        skip_newline = False

        for token in tokens:
            if token.type == tokenize.COMMENT:
                in_comment = True
            elif token.type == tokenize.STRING:
                in_string = True
            elif token.type == tokenize.NEWLINE:
                if previous_token_type == tokenize.NEWLINE and not in_comment and not in_string:
                    # Skip blank line in code
                    skip_newline = True
                    continue
                elif skip_newline:
                    # If we skipped a newline, add one back to maintain line structure
                    new_tokens.append(tokenize.TokenInfo(tokenize.NEWLINE, '\n', token.start, token.end, token.line))
                    skip_newline = False
            elif token.type == tokenize.INDENT or token.type == tokenize.DEDENT:
                # Manage indentation
                pass
            # Reset flags for non-string/comment tokens
            if token.type not in (tokenize.STRING, tokenize.COMMENT):
                in_comment = False
                in_string = False
            if not skip_newline:
                new_tokens.append(token)
            previous_token_type = token.type
        
        # Rebuild the file content
        content = tokenize.untokenize(new_tokens)
        if isinstance(content, bytes):
            new_content = content.decode('utf-8')
        else:
            new_content = content
        
        # Write the new content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        # Verify the file was written correctly
        with open(file_path, 'r', encoding='utf-8') as f:
            content_after = f.read()
        assert content_after == new_content, "File content was not written correctly"
        
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