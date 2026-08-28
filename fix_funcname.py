with open('explainability.py', 'r') as f:
    content = f.read()

# Fix the corrupted function name
content = content.replace('def _get_reasoningsoning(', 'def _get_reasoning(')

with open('explainability.py', 'w') as f:
    f.write(content)

print("Function name fixed!")
