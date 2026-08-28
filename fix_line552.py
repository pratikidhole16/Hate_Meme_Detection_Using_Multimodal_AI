with open('explainability.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find line 552 (index 551) and fix it
for i in range(len(lines)):
    if i == 551:  # Line 552 (0-indexed)
        if lines[i].startswith('        def _get_reasoning'):
            lines[i] = '    def _get_reasoning' + lines[i][20:]
            print(f"Fixed line {i+1}: '{lines[i].rstrip()}'")

with open('explainability.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed!")
