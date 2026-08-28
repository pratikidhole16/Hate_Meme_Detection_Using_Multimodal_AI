import re

# Read the file
with open('explainability.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and remove the duplicate empty function definition
# Pattern: "def _get_reasoning(self, predicted_class: int):" followed by blank line
pattern = r'def _get_reasoning\(self, predicted_class: int\):\n\s*\n'
content = re.sub(pattern, '', content, count=1)

# Write back
with open('explainability.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed duplicate function definition")
