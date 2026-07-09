"""Quick AST syntax check for app.py."""
import sys, ast
p = r"d:\Codes\KUET PROJECTS\3 1 Embedded\DocsMaker\DocumentScanner\app.py"
with open(p, "rb") as f:
    data = f.read()
if data.startswith(b"\xef\xbb\xbf"):
    data = data[3:]
try:
    ast.parse(data.decode("utf-8"), filename=p)
    print("OK")
except SyntaxError as e:
    print("SYNTAX ERROR:", e)
    sys.exit(1)