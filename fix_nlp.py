with open('explainability.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('explainability_nlp_fix.py', 'r', encoding='utf-8') as f:
    new_func = f.read()

# Replace lines 378-461 with new function
new_lines = lines[:377] + [new_func + '\n'] + lines[461:]

with open('explainability.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('NLP reasoning function fixed successfully!')
