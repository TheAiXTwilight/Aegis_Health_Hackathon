#!/usr/bin/env python3
"""Remove trailing '====...' separator lines left by the export parser."""
from pathlib import Path
import re

for p in Path('/home/user').rglob('*.py'):
    content = p.read_text(encoding='utf-8')
    cleaned = re.sub(r'\n+={20,}\s*$', '', content)
    if cleaned != content:
        p.write_text(cleaned, encoding='utf-8')
        print(f'cleaned {p}')

print('done')
