import os
import sys

def process_python_files(directory, exclude_file=None, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith('.py') and file != 'repair-remarks.py':
                file_path = os.path.join(root, file)
                if exclude_file and file_path == exclude_file:
                    continue
                print(f"Processing file: {file_path}")  # Debug print
                process_file(file_path, file)

def process_file(file_path, filename):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        correct_comment = f'# {filename}'
        print(f"Expected comment: {correct_comment}")  # Debug print
        
        # Remove any in-line comments that are not the correct comment
        modified_lines = []
        found_correct_comment = False
        in_line_comments_removed = 0
        standalone_comments_removed = 0
        duplicate_correct_comments_removed = 0
        
        for line in lines:
            stripped_line = line.strip()
            if stripped_line == correct_comment:
                if not found_correct_comment:
                    modified_lines.append(line)
                    found_correct_comment = True
                else:
                    duplicate_correct_comments_removed += 1
            elif stripped_line.startswith('#'):
                # Remove standalone comments
                standalone_comments_removed += 1
            else:
                # Remove in-line comments
                if '#' in line:
                    code_part, _, comment_part = line.partition('#')
                    if code_part.strip():  # Ensure there is code before the comment
                        modified_lines.append(code_part.rstrip() + '\n')
                        if comment_part.strip():
                            in_line_comments_removed += 1
                    else:
                        modified_lines.append(line)
                else:
                    modified_lines.append(line)
        
        # Ensure there is exactly one correct comment at the top
        if not found_correct_comment:
            modified_lines.insert(0, f'{correct_comment}\n')
        
        # Write modified lines back to the file if any changes were made
        modified = in_line_comments_removed > 0 or standalone_comments_removed > 0 or duplicate_correct_comments_removed > 0
        if not found_correct_comment:
            modified = True
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(modified_lines)
        
        # Report what was done
        report = f"Modified {file_path}: "
        if not modified:
            report += "No changes made."
        else:
            actions = []
            if found_correct_comment:
                actions.append(f"Updated correct comment")
            else:
                actions.append(f"Added correct comment")
            if in_line_comments_removed > 0:
                actions.append(f"Removed {in_line_comments_removed} in-line comments")
            if standalone_comments_removed > 0:
                actions.append(f"Removed {standalone_comments_removed} standalone comments")
            if duplicate_correct_comments_removed > 0:
                actions.append(f"Removed {duplicate_correct_comments_removed} duplicate correct comments")
            report += " and ".join(actions) + "."
        
        print(report)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    script_path = sys.argv[0]
    directories = [os.getcwd(), os.path.join(os.getcwd(), 'scripts')]
    exclude_dirs = ['.venv', 'venv', '_pycache_']
    for dir_path in directories:
        if os.path.exists(dir_path):
            print(f"Processing directory: {dir_path}")  # Debug print
            process_python_files(dir_path, exclude_file=script_path, exclude_dirs=exclude_dirs)
        else:
            print(f"Directory {dir_path} does not exist.")

if __name__ == "__main__":
    main()