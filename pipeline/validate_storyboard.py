"""Validates storyboard.json against schema/storyboard.schema.json.

Callable via CLI or imported by other pipeline modules.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.utils import validate_storyboard


def main() -> int:
    storyboard_file = os.path.join(REPO_ROOT, "storyboard.json")
    schema_file = os.path.join(REPO_ROOT, "schema", "storyboard.schema.json")

    print(f"Validating '{storyboard_file}' against '{schema_file}'...")
    try:
        data = validate_storyboard(storyboard_file, schema_file)
        shots_count = len(data.get("shots", []))
        title = data.get("project", {}).get("title", "Untitled")
        print(f"Validation successful! Project '{title}' has {shots_count} valid shots.")
        return 0
    except Exception as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
