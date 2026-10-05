"""命令行运行评测：python -m app.evaluation"""

from __future__ import annotations

import sys

from app.config import get_settings
from app.core.container import build_container
from app.core.exceptions import AppError
from app.evaluation.runner import EvaluationRunner


def main() -> None:
    settings = get_settings()
    container = build_container(settings)
    try:
        report = EvaluationRunner.from_container(container).run()
    except AppError as exc:
        print(exc.message, file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        container.close()
    print(
        "评测完成 "
        f"文件={report.filename} "
        f"题数={report.question_count} "
        f"Recall@K={_fmt(report.recall_at_k)} "
        f"MRR@K={_fmt(report.mrr_at_k)} "
        f"关键词覆盖率={_fmt(report.keyword_coverage)} "
        f"拒答率={_fmt(report.abstention_rate)} "
        f"失败={report.failure_count}"
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
