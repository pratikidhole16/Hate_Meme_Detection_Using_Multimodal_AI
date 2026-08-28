with open('explainability.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the problematic section and replace it
old_section = '''        result = "\n".join(reasoning_parts)
        logger.info(f"===== NLP REASONING END =====\n{result}\n")
        return result
    
        def _get_reasoning(self, predicted_class: int):
        """Fallback reasoning for prediction"""
        if predicted_class == 1:'''

new_section = '''        result = "\n".join(reasoning_parts)
        logger.info(f"===== NLP REASONING END =====\n{result}\n")
        return result
    
    def _get_reasoning(self, predicted_class: int):
        """Fallback reasoning for prediction"""
        if predicted_class == 1:'''

content = content.replace(old_section, new_section)

with open('explainability.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Indentation fixed!")
