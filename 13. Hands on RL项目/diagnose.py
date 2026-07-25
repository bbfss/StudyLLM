import json

with open('20 OPD.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Total cells: {len(nb["cells"])}')
print()

for i, c in enumerate(nb['cells']):
    source = c.get('source', [])
    cell_type = c.get('cell_type', 'unknown')
    
    if source:
        # Get first line
        if isinstance(source, list):
            first_line = ''.join(source[:5]).strip()[:80]
        else:
            first_line = str(source)[:80].strip()
        
        # Check if cell has no markdown header
        full_content = ''.join(source) if isinstance(source, list) else source
        has_header = full_content.startswith('#')
        
        print(f'Cell {i}: {cell_type:10} | Header: {str(has_header):5} | {first_line}')
    else:
        print(f'Cell {i}: {cell_type:10} | [empty]')

print("\n=== Looking for markdown cells without proper headers ===")
for i, c in enumerate(nb['cells']):
    if c.get('cell_type') == 'markdown':
        source = c.get('source', [])
        if source:
            full_content = ''.join(source) if isinstance(source, list) else source
            if not full_content.strip().startswith('#') and full_content.strip():
                print(f"Cell {i}: Missing header")
                print(f"  First 100 chars: {full_content[:100]}")
