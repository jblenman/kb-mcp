"""`python3 -m kb_mcp ...` runs the CLI in kb.py."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import kb  # noqa: E402

if __name__ == "__main__":
    sys.exit(kb.main())
