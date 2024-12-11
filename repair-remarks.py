import os
import sys

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
            lines = f.readlines()
        
        new_lines = []
        previous_line_blank = False

        for line in lines:
            stripped_line = line.strip()
            if stripped_line:
                new_lines.append(line)
                previous_line_blank = False
            else:
                if not previous_line_blank:
                    new_lines.append('\n')
                    previous_line_blank = True
        
        # Ensure no trailing newline
        if new_lines and new_lines[-1] == '\n':
            new_lines.pop()
        
        # Ensure no blank lines between code blocks
        final_lines = []
        previous_line_blank = False

        for line in new_lines:
            stripped_line = line.strip()
            if stripped_line:
                final_lines.append(line)
                previous_line_blank = False
            else:
                if not previous_line_blank and final_lines and final_lines[-1] != '\n':
                    final_lines.append('\n')
                    previous_line_blank = True
        
        new_content = ''.join(final_lines)
        original_content = ''.join(lines)
        original_length = len(original_content)
        new_length = len(new_content)
        
        # Write the new content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        # Verify the file was written correctly
        with open(file_path, 'r', encoding='utf-8') as f:
            content_after = f.read()
        assert content_after == new_content, "File content was not written correctly"
        
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