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
                process_file(file_path, file)

def process_file(file_path, filename):
    try:
        with tokenize.open(file_path) as f:
            tokens = list(tokenize.generate_tokens(f.readline))
        
        correct_comment = f'# {filename}'
        if tokens and tokens[0].type == tokenize.COMMENT and tokens[0].string.strip() == correct_comment:
            new_tokens = [token for token in tokens if token.type != tokenize.COMMENT or token == tokens[0]]
            if new_tokens != tokens:
                new_content = tokenize.untokenize(new_tokens).decode('utf-8')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Modified {file_path}: Removed extra comments.")
            else:
                print(f"No changes needed in {file_path}.")
            return
        new_tokens = [token for token in tokens if token.type != tokenize.COMMENT]
        filename_token = tokenize.TokenInfo(type=tokenize.COMMENT, string=f'#{correct_comment}', start=(1, 0), end=(1, len(correct_comment)+1), line=f'{correct_comment}\n')
        new_tokens.insert(0, filename_token)
        new_content = tokenize.untokenize(new_tokens).decode('utf-8')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Modified {file_path}: Added correct filename remark and removed comments.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        print(f"Exception type: {type(e)}, Exception args: {e.args}")

def main():
    script_path = sys.argv[0]
    excluded_files = ['copyscripts.py', 'repair-remarks.py']
    directories = [os.getcwd()]
    scripts_dir = os.path.join(os.getcwd(), 'scripts')
    if os.path.isdir(scripts_dir):
        directories.append(scripts_dir)
    for dir_path in directories:
        process_python_files(dir_path, excluded_files)

if __name__ == "__main__":
    main()