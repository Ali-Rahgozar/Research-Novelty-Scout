"""Retrieval eval: recall of the papers the agent retrieves.

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


def recall(retrieved_ids, gold_ids):
    hits = len(retrieved_ids & gold_ids)
    return hits / len(gold_ids) if gold_ids else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N ideas.")
    args = parser.parse_args()

    from src.agent import run_agent_traced  # needs config.json

    cases = json.loads(GOLD_PATH.read_text(encoding="utf-8"))[: args.limit]
    rows = []

    print(f"{'idea':<30}{'recall':>7}")
    for case in cases:
        trace = run_agent_traced(case["idea"])
        retrieved = {p["id"] for p in trace["retrieved_papers"]}
        gold = {g["id"] for g in case["gold"]}
        score = recall(retrieved, gold)
        rows.append(score)
        print(f"{case['id']:<30}{score:>7.2f}", flush=True)

    n = len(rows)
    print(f"{'MEAN':<30}{sum(rows) / n:>7.2f}")


if __name__ == "__main__":
    main()
