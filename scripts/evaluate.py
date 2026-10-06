"""运行检索对比实验。

用法：python scripts/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.evaluation.retrieval_cli import main as run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
