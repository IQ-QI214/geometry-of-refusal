"""
Unit test for T2: verify Llama3 chat template has no spurious leading quote.
Run: python experiments/repro_arditi_wollschlager/test_t2_llama_fix.py
"""
import sys, os

# Read the templates directly from the file to avoid import issues
def read_template_from_source(filepath, var_name):
    """Extract template value from Python source file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith(f'{var_name} = '):
            # Start of the template definition
            result = []
            j = i
            # Extract the full template (may span multiple lines)
            while j < len(lines):
                result.append(lines[j])
                if j > i and lines[j].strip().endswith('"""'):
                    break
                j += 1

            # Parse the template
            full_def = ''.join(result)
            # Use exec to evaluate the template assignment
            local_ns = {}
            exec(full_def, {}, local_ns)
            return local_ns[var_name]

    raise ValueError(f"Could not find {var_name} in {filepath}")

llama3_model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../refusal_direction/pipeline/model_utils/llama3_model.py"
)

LLAMA3_CHAT_TEMPLATE = read_template_from_source(llama3_model_path, 'LLAMA3_CHAT_TEMPLATE')
LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM = read_template_from_source(llama3_model_path, 'LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM')

# Test 1: No spurious leading quote
assert not LLAMA3_CHAT_TEMPLATE.startswith('"'), \
    f'LLAMA3_CHAT_TEMPLATE starts with spurious quote: {repr(LLAMA3_CHAT_TEMPLATE[:30])}'
print("[PASS] LLAMA3_CHAT_TEMPLATE: no spurious leading quote")

assert not LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM.startswith('"'), \
    f'LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM starts with spurious quote: {repr(LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM[:30])}'
print("[PASS] LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM: no spurious leading quote")

# Test 2: Templates start correctly
assert LLAMA3_CHAT_TEMPLATE.startswith('<|begin_of_text|>'), \
    f'Template should start with <|begin_of_text|>, got: {repr(LLAMA3_CHAT_TEMPLATE[:40])}'
print("[PASS] LLAMA3_CHAT_TEMPLATE starts with <|begin_of_text|>")

assert LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM.startswith('<|begin_of_text|>'), \
    f'WITH_SYSTEM template should start with <|begin_of_text|>, got: {repr(LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM[:40])}'
print("[PASS] LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM starts with <|begin_of_text|>")

# Test 3: Template content is preserved
assert '<|start_header_id|>user<|end_header_id|>' in LLAMA3_CHAT_TEMPLATE
assert '{instruction}' in LLAMA3_CHAT_TEMPLATE
print("[PASS] LLAMA3_CHAT_TEMPLATE has expected content")

assert '<|start_header_id|>system<|end_header_id|>' in LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM
assert '{system_prompt}' in LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM
print("[PASS] LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM has expected content")

print("\nAll T2 tests passed.")
