"""Compile all Python files in repository to detect syntax errors."""
import os, sys, py_compile

ERRORS = []
ROOT = os.path.dirname(__file__)

for dirpath, _, filenames in os.walk(ROOT):
    for fn in filenames:
        if fn.endswith('.py') and fn not in {'compile_all.py'}:
            path = os.path.join(dirpath, fn)
            try:
                py_compile.compile(path, doraise=True)
            except Exception as e:
                ERRORS.append((path, repr(e)))

if ERRORS:
    print("Syntax errors detected:")
    for p, err in ERRORS:
        print(f"  {p}: {err}")
    sys.exit(1)
else:
    print("All Python files compiled successfully.")
