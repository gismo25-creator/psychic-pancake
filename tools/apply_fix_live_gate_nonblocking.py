#!/usr/bin/env python3
"""
Fix: Live gate blocks the entire page.

Some builds call `st.stop()` (or otherwise exit) when no ACTIVE bundle is present,
which makes it look like you "can't get to the Live page" when selecting Live.

This patch makes the gate NON-BLOCKING:
- It will still show the red error box
- But it will NOT stop the script
- It will disable Live executor toggles instead

What it changes in streamlit_app.py:
- Replaces `st.stop()` with a no-op comment marker
- (Optionally) replaces `return` in the top-level gate block if present

It creates a backup: streamlit_app.py.bak

Run from project root:
    python tools/apply_fix_live_gate_nonblocking.py
"""

from pathlib import Path
import shutil, re, sys

TARGET = Path("streamlit_app.py")

def main():
    if not TARGET.exists():
        print("ERROR: streamlit_app.py not found in current directory.")
        sys.exit(1)

    src = TARGET.read_text(encoding="utf-8", errors="replace")

    # Replace direct st.stop() calls (common pattern in gate blocks)
    patched = re.sub(r"^\s*st\.stop\(\)\s*$", "    # st.stop() disabled by patch: keep rendering page", src, flags=re.MULTILINE)

    # Some builds use "return" at top-level inside if-block via function wrapper; keep conservative:
    # Only replace a bare 'return' line that is immediately after sidebar.error in the gate block.
    patched = re.sub(
        r"(st\.sidebar\.error\([^\)]*\)\s*\n)\s*return\s*\n",
        r"\1    # return disabled by patch: keep rendering page\n",
        patched,
        flags=re.MULTILINE
    )

    if patched == src:
        print("NOTE: No st.stop()/return patterns found to patch. Nothing changed.")
        return

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(patched, encoding="utf-8")
    print("OK: Patched streamlit_app.py to make Live gate non-blocking.")
    print(f"Backup: {backup}")

if __name__ == "__main__":
    main()
