#!/usr/bin/env python
"""Retrieval evaluation for demo1.

Measures retrieval separately from answer generation — the only way to tell
whether a bad answer came from bad retrieval or from bad generation.

Usage (from the repo root):

    python demo1_doc_intelligence/eval/run_eval.py
    python demo1_doc_intelligence/eval/run_eval.py --json /tmp/eval.json

The reranker is chosen by the container's memory ceiling (see retrieval.py);
`DEMO1_RERANK=on|off` pins it, so both measured paths can be compared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

DEMO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_ROOT / "src"))

import retrieval  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.yaml"


def load_golden():
    return yaml.safe_load(GOLDEN.read_text())["questions"]


def evaluate(questions):
    """Return one result row per question, with document- and page-level hits."""
    vectorstore = retrieval.load_vectorstore("en")

    rows = []
    for item in questions:
        chunks = retrieval.search_chunks(vectorstore, item["question"])
        got = [
            (c.metadata.get("source_file"), c.metadata.get("page")) for c in chunks
        ]

        # A question can legitimately be answered from more than one place —
        # "what does S-717 cover?" is answered as directly by the QRS scope
        # statement as by the TRS one. `also_accept` records alternatives that
        # were checked by hand against the document text.
        accepted = [
            (item["expected_file"], item.get("expected_pages") or [])
        ] + [
            (alt["file"], alt.get("pages") or [])
            for alt in item.get("also_accept") or []
        ]

        def matches(ranked, check_pages):
            for i, (f, p) in enumerate(ranked, 1):
                for accepted_file, accepted_pages in accepted:
                    if f != accepted_file:
                        continue
                    if check_pages and accepted_pages and p not in accepted_pages:
                        continue
                    return i
            return None

        doc_rank = matches([(f, None) for f, _ in got], check_pages=False)
        page_rank = matches(got, check_pages=True)

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_file": item["expected_file"],
                "expected_pages": item.get("expected_pages") or [],
                "accepted": accepted,
                "doc_hit": doc_rank is not None,
                "page_hit": page_rank is not None,
                "doc_rr": 1 / doc_rank if doc_rank else 0.0,
                "page_rr": 1 / page_rank if page_rank else 0.0,
                "retrieved": got,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="write raw per-question results here")
    parser.add_argument(
        "--answers",
        type=int,
        default=0,
        metavar="N",
        help="also run the full answer path on the first N questions and report "
        "citation groundedness (costs N API calls)",
    )
    args = parser.parse_args()

    questions = load_golden()
    rows = evaluate(questions)
    n = len(rows)

    def rate(key, top=None):
        hits = 0
        for r in rows:
            got = r["retrieved"][:top] if top else r["retrieved"]
            if key == "doc":
                ok = any(
                    f == accepted_file
                    for f, _ in got
                    for accepted_file, _pages in r["accepted"]
                )
            else:
                ok = any(
                    f == accepted_file
                    and (not accepted_pages or p in accepted_pages)
                    for f, p in got
                    for accepted_file, accepted_pages in r["accepted"]
                )
            hits += ok
        return hits / len(rows)

    print(f"questions              {n}")
    print(f"reranker               {retrieval.rerank_strategy()}")
    print(f"doc  hit-rate@6        {rate('doc', 6):.0%}")
    print(f"page hit-rate@6        {rate('page', 6):.0%}")
    print(f"doc  hit-rate@k        {rate('doc'):.0%}")
    print(f"page hit-rate@k        {rate('page'):.0%}")
    print(f"doc MRR                {sum(r['doc_rr'] for r in rows) / n:.3f}")
    print(f"page MRR               {sum(r['page_rr'] for r in rows) / n:.3f}")
    print()

    for r in rows:
        mark = "OK  " if r["page_hit"] else ("doc " if r["doc_hit"] else "MISS")
        print(f"  {mark} {r['id']:6s} {r['question'][:56]}")
        if not r["page_hit"]:
            print(f"         expected {r['expected_file']} {r['expected_pages']}")
            print(f"         got      {r['retrieved'][:3]}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nraw results -> {args.json}")

    if args.answers:
        report_groundedness(questions[: args.answers])


def report_groundedness(questions):
    """How often did the model try to cite something that does not exist?

    Because validate_claims discards bad citations before the user sees them,
    the interesting number is how much the guardrail had to catch.
    """
    import retrieval

    accepted_total = rejected_total = answered = 0
    print("\n=== answer groundedness ===")
    for item in questions:
        try:
            result = retrieval.ask(item["question"])
        except Exception as exc:                    # rate limits, network, ...
            print(f"  {item['id']:6s} ERROR {type(exc).__name__}: {str(exc)[:70]}")
            continue
        accepted_total += len(result["claims"])
        rejected_total += len(result["rejected"])
        answered += bool(result["claims"])
        print(
            f"  {item['id']:6s} claims {len(result['claims'])} "
            f"rejected {len(result['rejected'])}"
        )

    total = accepted_total + rejected_total
    print(f"\nquestions answered      {answered}/{len(questions)}")
    print(f"claims shown            {accepted_total}")
    print(f"fabrications caught     {rejected_total}")
    if total:
        print(f"citation validity       {accepted_total / total:.0%}")


if __name__ == "__main__":
    main()
