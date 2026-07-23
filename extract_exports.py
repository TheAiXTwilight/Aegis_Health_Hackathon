#!/usr/bin/env python3
"""
Parse the Backend_txt_generated.txt / Frontend_txt_generated.txt text exports
and reconstruct the actual project files in the workspace.

- Skips __MACOSX metadata files and binary / DS_Store content.
- Keeps directory structure relative to the project root.
"""
from __future__ import annotations

import re
from pathlib import Path


EXPORTS = ["uploads/Backend_txt_generated.txt", "uploads/Frontend_txt_generated.txt"]
SKIP_PREFIXES = (
    "__MACOSX",
    ".DS_Store",
    "._",
    "__pycache__",
)
SKIP_SUFFIXES = (
    ".pyc",
    ".pyo",
)


def is_skip_path(path: str) -> bool:
    p = path.replace("\\", "/")
    parts = p.split("/")
    if any(part.startswith(SKIP_PREFIXES) or part.endswith(SKIP_SUFFIXES) for part in parts):
        return True
    return False


def extract_file(path: str, text: str, start: int, end: int) -> tuple[str, int, int] | None:
    """Extract the file path and content span for one file section."""
    # Find the header: "FILE: <path>"
    m = re.search(r"^FILE: (.+)$", text[start:end], re.MULTILINE)
    if not m:
        return None
    file_path = m.group(1).strip()

    # Find the start of actual source code after "Source Code" or similar
    code_start_mark = re.search(r"\nSource Code\n-+\n", text[start:end])
    summary_start_mark = re.search(r"\n(?:Frontend Summary|Backend Summary)\n-+-\nSource Code\n-+-\n", text[start:end])

    if summary_start_mark:
        content_begin = start + summary_start_mark.end()
    elif code_start_mark:
        content_begin = start + code_start_mark.end()
    else:
        # No recognizable content marker; skip this file
        return None

    content_end = end
    # Trim trailing blank line right before the end separator
    while content_end > content_begin and text[content_end - 1] == "\n":
        content_end -= 1

    content = text[content_begin:content_end]
    return file_path, content_begin, content_end - content_begin


def extract_all(export_path: Path) -> dict[str, str]:
    raw = export_path.read_bytes()
    # Drop null bytes (binary metadata) and decode the rest as best-effort
    text = raw.replace(b"\x00", b"").decode("utf-8", errors="ignore")

    files: dict[str, str] = {}
    # Find every "FILE: " header line
    for m in re.finditer(r"^FILE: .+$", text, re.MULTILINE):
        start = m.start()
        # End of this section is start of next FILE: or end of file
        next_m = re.search(r"^FILE: .+$", text[m.end():], re.MULTILINE)
        end = m.end() + next_m.start() if next_m else len(text)

        parsed = extract_file("", text, start, end)
        if not parsed:
            continue
        file_path, content_begin, content_len = parsed
        content_end = content_begin + content_len
        content = text[content_begin:content_end]

        if is_skip_path(file_path):
            continue

        # Normalize line endings
        content = content.replace("\r\n", "\n")
        files[file_path] = content

    return files


def main() -> None:
    root = Path("/home/user")
    for export in EXPORTS:
        export_path = root / export
        if not export_path.exists():
            print(f"SKIP (not found): {export}")
            continue
        print(f"Extracting {export} ...")
        files = extract_all(export_path)
        for rel_path, content in files.items():
            out_path = root / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"  wrote {rel_path} ({len(content)} chars)")
        print(f"Done: {len(files)} files from {export}\n")


if __name__ == "__main__":
    main()
