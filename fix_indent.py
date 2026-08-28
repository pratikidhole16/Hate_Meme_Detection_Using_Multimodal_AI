with open('explainability.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and fix the problematic lines
fixed_lines = []
for i, line in enumerate(lines):
    # Remove lines with just empty def statement
    if line.strip() == "def _get_reasoning(self, predicted_class: int):":
        # Skip it if it's followed by another def
        if i + 1 < len(lines) and "def _get_reasoning" in lines[i + 1]:
            continue
    fixed_lines.append(line)

with open('explainability.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("Fixed indentation issues")
