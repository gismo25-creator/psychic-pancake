#!/usr/bin/env python3
"""
Auto-fix for a common copy/paste corruption in streamlit_app.py where a raw regex string
gets split across lines like: re.split(r"[<NEWLINE>... which causes:
SyntaxError: unterminated string literal

This script:
- Locates a broken 're.split(r"[' pattern that contains a newline before the closing quote
- Replaces the re.split(...) call used for pairs import parsing with a safe splitter:
    re.split(r"[\\s,;]+", <same_input_var>)

Usage (from your project root):
    python tools/apply_fix_unterminated_regex.py

It will create a backup: streamlit_app.py.bak
"""
import sys, re, shutil
from pathlib import Path

TARGET = Path("streamlit_app.py")

def main():
    if not TARGET.exists():
        print("ERROR: streamlit_app.py not found in current directory.")
        sys.exit(1)

    src = TARGET.read_text(encoding="utf-8", errors="replace")

    # Detect a broken raw regex string that starts with re.split(r"[ and then hits a newline.
    broken1 = re.compile(r"re\\.split\\(\\s*r\\\"\\[[\\r\\n]+", re.MULTILINE)
    broken2 = re.compile(r"re\\.split\\(\\s*r\\\"\\\\\\[[\\r\\n]+", re.MULTILINE)

    if not (broken1.search(src) or broken2.search(src)):
        print("No broken re.split(r\"[ ... newline ...) pattern found. Nothing to do.")
        return

    # Try to patch the specific imported=... list comprehension line.
    line_pat = re.compile(
        r"(imported\\s*=\\s*\\[.*?for\\s+x\\s+in\\s+)re\\.split\\(\\s*r\\\"\\\\?\\[.*?\\\"\\s*,\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\)(\\s*if\\s+x\\.strip\\(\\)\\s*\\])",
        re.DOTALL
    )

    m = line_pat.search(src)
    if m:
        prefix, var, suffix = m.group(1), m.group(2), m.group(3)
        patched = src[:m.start()] + f'{prefix}re.split(r"[\\\\s,;]+", {var}){suffix}' + src[m.end():]
    else:
        # Fallback: replace the first corrupted re.split(..., VAR) occurrence.
        any_pat = re.compile(r"re\\.split\\(\\s*r\\\"\\\\?\\[.*?\\\"\\s*,\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\)", re.DOTALL)
        m2 = any_pat.search(src)
        if not m2:
            print("Found a broken pattern, but could not safely rewrite it.")
            print('Manual fix: replace the broken re.split(...) with re.split(r"[\\\\s,;]+", <your_text_var>)')
            sys.exit(2)
        var = m2.group(1)
        patched = any_pat.sub(f're.split(r"[\\\\s,;]+", {var})', src, count=1)

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(patched, encoding="utf-8")
    print("OK: Patched streamlit_app.py")
    print(f"Backup written to: {backup}")

if __name__ == "__main__":
    main()
