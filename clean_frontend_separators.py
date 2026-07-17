#!/usr/bin/env python3
"""Remove trailing '====...' separator lines left by the export parser in frontend files."""
from pathlib import Path
import re

for ext in ('*.jsx', '*.js', '*.css', '*.json', '*.md', '*.html', '*.yml', '*.sh', '*.txt'):
    for p in Path('/home/user/frontend').rglob(ext):
        content = p.read_text(encoding='utf-8')
        cleaned = re.sub(r'\n+={20,}\s*$', '', content)
        if cleaned != content:
            p.write_text(cleaned, encoding='utf-8')
            print(f'cleaned {p}')

print('done')
