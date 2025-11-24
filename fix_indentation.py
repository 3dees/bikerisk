"""Quick script to fix indentation in group_clauses_by_category_then_similarity."""

# Read the file
with open('harmonization/grouping.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the function and fix indentation in its body
in_function = False
function_start = -1
docstring_done = False

for i, line in enumerate(lines):
    if 'def group_clauses_by_category_then_similarity(' in line:
        in_function = True
        function_start = i
        docstring_done = False
        print(f"Found function at line {i+1}")
        continue
    
    if in_function and not docstring_done:
        # Check if we're past the docstring
        if '"""' in line and i > function_start + 3:
            docstring_done = True
            print(f"Docstring ends at line {i+1}")
        continue
    
    if in_function and docstring_done:
        # We're in the function body
        # Check if this is the next function
        if line.strip() and not line[0].isspace() and 'def ' in line:
            print(f"Function ends at line {i+1}")
            break
        
        # Fix lines that have 8 spaces of indentation (should be 4)
        if line.startswith('        ') and not line.startswith('         '):
            # Has exactly 8 spaces - reduce to 4
            lines[i] = line[4:]
            print(f"Fixed line {i+1}: {line[:20].strip()}...")

# Write back
with open('harmonization/grouping.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Done!")
