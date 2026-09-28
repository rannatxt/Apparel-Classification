"""Helper: add sys.stdout.reconfigure(encoding='utf-8') to all pipeline scripts."""
import sys
from pathlib import Path

UTF8_SNIPPET = "import sys\nif hasattr(sys.stdout, 'reconfigure'):\n    sys.stdout.reconfigure(encoding='utf-8')\n"

scripts_dir = Path(__file__).parent.parent / "scripts"
top_level   = [
    Path(__file__).parent.parent / "prepare_dataset.py",
    Path(__file__).parent.parent / "validate_dataset.py",
]
all_scripts = list(scripts_dir.glob("*.py")) + top_level

for script_path in all_scripts:
    content = script_path.read_text(encoding="utf-8")
    if "reconfigure" in content:
        print(f"  OK (already has reconfigure): {script_path.name}")
        continue

    lines = content.split("\n")

    # Find end of module docstring (first """ block)
    insert_at = 0
    if lines and '"""' in lines[0]:
        for i in range(1, len(lines)):
            if '"""' in lines[i]:
                insert_at = i + 1
                break
    elif lines and "'''" in lines[0]:
        for i in range(1, len(lines)):
            if "'''" in lines[i]:
                insert_at = i + 1
                break

    inject_lines = ["", "import sys", "if hasattr(sys.stdout, 'reconfigure'):",
                    "    sys.stdout.reconfigure(encoding='utf-8')", ""]
    new_lines = lines[:insert_at] + inject_lines + lines[insert_at:]
    script_path.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"  Fixed: {script_path.name}")

print("\nDone.")
