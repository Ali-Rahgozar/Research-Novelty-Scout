"""Retrieval eval: precision, recall and F1 of the papers the agent retrieves.

Papers are matched by exact OpenAlex ID against the gold set.

    python eval/run_eval.py
    python eval/run_eval.py --limit 3
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLD_PATH = Path(__file__).resolve().parent / "gold_set.json"


def prf(retrieved_ids, gold_ids):
    hits = len(retrieved_ids & gold_ids)
    precision = hits / len(retrieved_ids) if retrieved_ids else 0.0
    recall = hits / len(gold_ids) if gold_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N ideas.")
    args = parser.parse_args()

    from src.agent import run_agent_traced  # needs config.json

    cases = json.loads(GOLD_PATH.read_text(encoding="utf-8"))[: args.limit]
    rows = []

    print(f"{'idea':<30}{'precision':>10}{'recall':>9}{'f1':>7}")
    for case in cases:
        trace = run_agent_traced(case["idea"])
        retrieved = {p["id"] for p in trace["retrieved_papers"]}
        gold = {g["id"] for g in case["gold"]}
        scores = prf(retrieved, gold)
        rows.append(scores)
        print(f"{case['id']:<30}{scores[0]:>10.2f}{scores[1]:>9.2f}{scores[2]:>7.2f}", flush=True)

    n = len(rows)
    print(f"{'MEAN':<30}" + "".join(f"{sum(r[i] for r in rows) / n:>{w}.2f}" for i, w in enumerate((10, 9, 7))))


if __name__ == "__main__":
    main()