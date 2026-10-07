"""Retrieval eval for the novelty agent.

For each research idea in the gold set, run the agent, collect every unique
paper it retrieved across all of its searches, and compare that set against
the gold-standard papers for the idea.

    precision = retrieved papers that match a gold paper / all retrieved papers
    recall    = gold papers that were retrieved           / all gold papers

Usage (from the project root):
    python eval/run_eval.py
    python eval/run_eval.py --limit 3 --show-details
    python eval/run_eval.py --threshold 0.9 --temperature 0
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GOLD = Path(__file__).resolve().parent / "gold_set.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def normalize_title(title):
    title = (title or "").lower()
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def normalize_doi(doi):
    if not doi:
        return None
    doi = doi.lower().strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi or None


def normalize_openalex_id(oa_id):
    if not oa_id:
        return None
    return oa_id.rstrip("/").split("/")[-1].lower()


def is_match(paper, gold, threshold):
    """A retrieved paper matches a gold entry by OpenAlex ID, DOI, or title.

    Title matching is fuzzy because OpenAlex often holds several records for
    one paper (arXiv preprint vs. conference version) with slightly different
    titles, so exact-ID matching alone would undercount.
    """
    p_id = normalize_openalex_id(paper.get("id"))
    g_id = normalize_openalex_id(gold.get("openalex_id"))
    if p_id and g_id and p_id == g_id:
        return True

    p_doi = normalize_doi(paper.get("doi"))
    g_doi = normalize_doi(gold.get("doi"))
    if p_doi and g_doi and p_doi == g_doi:
        return True

    p_title = normalize_title(paper.get("title"))
    g_title = normalize_title(gold.get("title"))
    if not p_title or not g_title:
        return False
    return SequenceMatcher(None, p_title, g_title).ratio() >= threshold


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_case(retrieved, gold, threshold):
    """Score one idea. `retrieved` is a deduplicated list of paper dicts."""
    matched_gold = {}      # gold index -> the retrieved paper that matched it
    relevant_retrieved = []

    for paper in retrieved:
        for gi, g in enumerate(gold):
            if is_match(paper, g, threshold):
                relevant_retrieved.append(paper)
                matched_gold.setdefault(gi, paper)
                break

    n_retrieved = len(retrieved)
    n_relevant = len(relevant_retrieved)
    n_gold = len(gold)
    n_found = len(matched_gold)

    precision = n_relevant / n_retrieved if n_retrieved else 0.0
    recall = n_found / n_gold if n_gold else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "n_retrieved": n_retrieved,
        "n_relevant_retrieved": n_relevant,
        "n_gold": n_gold,
        "n_gold_found": n_found,
        "gold_found": [gold[i]["title"] for i in sorted(matched_gold)],
        "gold_missed": [
            g["title"] for i, g in enumerate(gold) if i not in matched_gold
        ],
        "retrieved_not_in_gold": [
            p.get("title") for p in retrieved if p not in relevant_retrieved
        ],
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate(scores):
    """Macro (mean of per-idea) and micro (pooled counts) averages."""
    total_retrieved = sum(s["n_retrieved"] for s in scores)
    total_relevant = sum(s["n_relevant_retrieved"] for s in scores)
    total_gold = sum(s["n_gold"] for s in scores)
    total_found = sum(s["n_gold_found"] for s in scores)

    micro_p = total_relevant / total_retrieved if total_retrieved else 0.0
    micro_r = total_found / total_gold if total_gold else 0.0

    return {
        "n_ideas": len(scores),
        "macro_precision": mean(s["precision"] for s in scores),
        "macro_recall": mean(s["recall"] for s in scores),
        "macro_f1": mean(s["f1"] for s in scores),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": f1(micro_p, micro_r),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_report(results, errors, show_details):
    print("\n" + "=" * 78)
    print(f"{'idea':<28}{'cat':<12}{'#ret':>5}{'P':>8}{'R':>8}{'F1':>8}{'#search':>9}")
    print("-" * 78)
    for r in results:
        s = r["score"]
        print(
            f"{r['id']:<28}{r['category']:<12}{s['n_retrieved']:>5}"
            f"{s['precision']:>8.2f}{s['recall']:>8.2f}{s['f1']:>8.2f}"
            f"{len(r['tool_calls']):>9}"
        )

    if show_details:
        print("\n" + "-" * 78)
        for r in results:
            s = r["score"]
            print(f"\n[{r['id']}]")
            print("  queries:")
            for call in r["tool_calls"]:
                print(f"    - {call['arguments']}  ({call['n_results']} results)")
            print(f"  gold found  : {s['gold_found']}")
            print(f"  gold MISSED : {s['gold_missed']}")
            print("  retrieved but not in gold (check: wrong, or gold incomplete?):")
            for title in s["retrieved_not_in_gold"]:
                print(f"    - {title}")

    scores = [r["score"] for r in results]
    if scores:
        agg = aggregate(scores)
        print("\n" + "=" * 78)
        print(f"OVERALL over {agg['n_ideas']} ideas")
        print(
            f"  macro  P={agg['macro_precision']:.3f}  "
            f"R={agg['macro_recall']:.3f}  F1={agg['macro_f1']:.3f}"
        )
        print(
            f"  micro  P={agg['micro_precision']:.3f}  "
            f"R={agg['micro_recall']:.3f}  F1={agg['micro_f1']:.3f}"
        )

        by_cat = defaultdict(list)
        for r in results:
            by_cat[r["category"]].append(r["score"])
        print("\nBy category (macro):")
        for cat, cat_scores in sorted(by_cat.items()):
            print(
                f"  {cat:<12} n={len(cat_scores)}  "
                f"P={mean(s['precision'] for s in cat_scores):.3f}  "
                f"R={mean(s['recall'] for s in cat_scores):.3f}"
            )

    if errors:
        print(f"\n{len(errors)} idea(s) failed and are excluded from the averages:")
        for e in errors:
            print(f"  - {e['id']}: {e['error']}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N ideas.")
    parser.add_argument("--threshold", type=float, default=0.88,
                        help="Fuzzy title match threshold (0-1).")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Pass a sampling temperature (some models reject it).")
    parser.add_argument("--show-details", action="store_true",
                        help="Print queries, missed gold, and unmatched retrievals.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write the JSON results.")
    args = parser.parse_args()

    # Imported here so the scoring functions can be used/tested without
    # needing config.json or network access.
    from src.agent import run_agent_traced

    cases = json.loads(args.gold.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    results, errors = [], []

    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['id']} ...", flush=True)
        try:
            trace = run_agent_traced(
                case["idea"],
                max_turns=args.max_turns,
                temperature=args.temperature,
            )
        except Exception as exc:  # keep going; one failure shouldn't kill the run
            errors.append({"id": case["id"], "error": repr(exc)})
            continue

        score = score_case(trace["retrieved_papers"], case["gold"], args.threshold)

        results.append({
            "id": case["id"],
            "category": case.get("category", "unknown"),
            "idea": case["idea"],
            "score": score,
            "tool_calls": trace["tool_calls"],
            "total_retrievals": trace["total_retrievals"],
            "hit_max_turns": trace["hit_max_turns"],
            "retrieved_titles": [p.get("title") for p in trace["retrieved_papers"]],
            "answer": trace["answer"],
        })

    print_report(results, errors, args.show_details)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "config": {
            "threshold": args.threshold,
            "max_turns": args.max_turns,
            "temperature": args.temperature,
            "gold_file": str(args.gold),
        },
        "aggregate": aggregate([r["score"] for r in results]) if results else None,
        "results": results,
        "errors": errors,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results written to {out}")


if __name__ == "__main__":
    main()
