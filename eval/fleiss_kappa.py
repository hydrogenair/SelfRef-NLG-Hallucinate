"""
Inter-rater agreement (Fleiss' Kappa) among human1, human2, human_final.

Used in detector selection (Part 1). Human rating CSVs are not bundled; set
HUMAN_RATINGS_DIR to a directory containing the three task CSV files.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RATER_COLS = ["human1", "human2", "human_final"]
TASK_FILES = [
    ("para", "para_50_all_new.csv"),
    ("sum", "sum_all_50.csv"),
    ("d2t", "d2t_all_50.csv"),
]


def normalize_label(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    label = str(value).strip().lower()
    if label in {"yes", "1", "true"}:
        return 1
    if label in {"no", "0", "false"}:
        return 0
    return None


def fleiss_kappa(counts: List[List[int]]) -> float:
    n_subjects = len(counts)
    n_raters = sum(counts[0])
    n_categories = len(counts[0])

    p_i = []
    for row in counts:
        p_i.append((sum(x * x for x in row) - n_raters) / (n_raters * (n_raters - 1)))
    p_bar = sum(p_i) / n_subjects

    col_totals = [sum(row[j] for row in counts) for j in range(n_categories)]
    total = n_subjects * n_raters
    p_j = [c / total for c in col_totals]
    p_e = sum(x * x for x in p_j)
    return (p_bar - p_e) / (1 - p_e)


def analyze_file(task: str, csv_path: Path) -> Dict:
    rows: List[List[int]] = []
    invalid = 0
    disagreements: List[Tuple[int, Dict[str, str]]] = []

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, 1):
            labels = {col: normalize_label(row[col]) for col in RATER_COLS}
            if any(v is None for v in labels.values()):
                invalid += 1
                continue
            values = [labels[col] for col in RATER_COLS]
            rows.append(values)
            if len(set(values)) > 1:
                disagreements.append(
                    (idx, {col: ("yes" if labels[col] else "no") for col in RATER_COLS})
                )

    n = len(rows)
    if n == 0:
        return {
            "task": task,
            "file": csv_path.name,
            "n_valid": 0,
            "full_agreement_rate": 0.0,
            "fleiss_kappa": 0.0,
            "n_disagreements": 0,
            "invalid_rows": invalid,
        }

    all_same = sum(1 for r in rows if r[0] == r[1] == r[2])
    pair_agree = {}
    for i, j, label in [
        (0, 1, "human1_vs_human2"),
        (0, 2, "human1_vs_human_final"),
        (1, 2, "human2_vs_human_final"),
    ]:
        pair_agree[label] = sum(1 for r in rows if r[i] == r[j]) / n

    yes_dist = Counter(sum(r) for r in rows)
    counts = [[3 - sum(r), sum(r)] for r in rows]
    kappa = fleiss_kappa(counts)

    logger.info(
        "Task=%s n_valid=%d kappa=%.4f full_agreement=%.1f%%",
        task,
        n,
        kappa,
        100.0 * all_same / n,
    )

    return {
        "task": task,
        "file": csv_path.name,
        "n_valid": n,
        "invalid_rows": invalid,
        "full_agreement_rate": all_same / n,
        "pair_agreement": pair_agree,
        "yes_vote_distribution": dict(yes_dist),
        "fleiss_kappa": kappa,
        "n_disagreements": len(disagreements),
        "disagreement_examples": disagreements[:10],
    }


def run_fleiss_analysis(ratings_dir: Path | None = None) -> List[Dict]:
    if ratings_dir is None:
        env = os.environ.get("HUMAN_RATINGS_DIR")
        if not env:
            raise RuntimeError("Set HUMAN_RATINGS_DIR to the directory with human rating CSVs")
        ratings_dir = Path(env)
    results = []
    for task, filename in TASK_FILES:
        path = ratings_dir / filename
        if not path.exists():
            logger.error("Missing human ratings file: %s", path.name)
            continue
        results.append(analyze_file(task, path))
    return results


def main() -> List[Dict]:
    return run_fleiss_analysis()


if __name__ == "__main__":
    from scripts.logging_config import configure_logging

    configure_logging()
    main()
