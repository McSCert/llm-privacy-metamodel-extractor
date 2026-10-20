#!/usr/bin/env python3
"""
diagnose_constraints.py — per-principle breakdown of Constraint_type errors.

Constraint_type is the largest-support concept scored (19) and sits at
75.0 / 63.2 / 68.6, with recall the weak side: 4 FP, 7 FN. evaluate.py reports
only aggregates, so this prints the gold set, the extracted set and the
per-principle difference, which is what a diagnosis needs.

Read-only. Touches nothing, runs in seconds.

    python3 diagnose_constraints.py
    python3 diagnose_constraints.py --repo data/model_repo.citations.db
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

import openpyxl

SEP = "|"
ABSENT = {"ABSENT", "", "UNCERTAIN", None}


def gold_constraints(path: Path, sheet: str | None = None) -> dict[str, set[str]]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet] if sheet else wb.active
    rows = list(ws.values)
    # The header row is the first row whose first cell is literally "principle".
    hdr_i = next(i for i, r in enumerate(rows)
                 if r and str(r[0]).strip().lower() == "principle")
    hdr = [str(c).strip() if c else "" for c in rows[hdr_i]]
    col = hdr.index("Constraint_type")
    pcol = hdr.index("principle")

    out: dict[str, set[str]] = {}
    for r in rows[hdr_i + 1:]:
        if not r or not r[pcol]:
            continue
        principle = str(r[pcol]).strip()
        cell = r[col]
        if cell is None or str(cell).strip() in ABSENT:
            out[principle] = set()
        else:
            out[principle] = {v.strip() for v in str(cell).split(SEP) if v.strip()}
    return out


def extracted_constraints(repo: Path) -> dict[str, set[str]]:
    con = sqlite3.connect(str(repo))
    rows = con.execute(
        "SELECT article_ref, statement_json FROM statements WHERE law='PIPEDA'"
    ).fetchall()
    con.close()

    out: dict[str, set[str]] = {}
    for article_ref, sj in rows:
        # article_ref looks like "4.3 Principle 3 - Consent"; key on the number.
        m = re.match(r"\s*(\d+(?:\.\d+)*)", article_ref or "")
        if not m:
            continue
        stmt = json.loads(sj)
        types = {
            c.get("type") for c in (stmt.get("constraints") or [])
            if c.get("type")
        }
        out[m.group(1)] = {t for t in types if t and t != "_Unset"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="evaluation/pipeda_gold_standard.xlsx")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--repo", default="data/model_repo.citations.db")
    a = ap.parse_args()

    gold = gold_constraints(Path(a.gold), a.sheet)
    got = extracted_constraints(Path(a.repo))

    def pkey(p: str) -> tuple:
        return tuple(int(x) for x in p.split("."))

    fn_counter: Counter = Counter()
    fp_counter: Counter = Counter()
    tp = fp = fn = 0

    print(f"gold : {a.gold}")
    print(f"repo : {a.repo}")
    print()
    print(f"{'Principle':<10} {'MISSED (FN)':<40} {'EXTRA (FP)':<34} ok")
    print("-" * 96)

    for p in sorted(gold, key=pkey):
        g = gold[p]
        e = got.get(p, set())
        missed = g - e
        extra = e - g
        hit = g & e
        tp += len(hit); fn += len(missed); fp += len(extra)
        fn_counter.update(missed)
        fp_counter.update(extra)
        if missed or extra:
            print(f"{p:<10} {('|'.join(sorted(missed)) or '-'):<40} "
                  f"{('|'.join(sorted(extra)) or '-'):<34} {len(hit)}")
        else:
            print(f"{p:<10} {'-':<40} {'-':<34} {len(hit)}  ✓")

    print("-" * 96)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"TP={tp}  FP={fp}  FN={fn}   "
          f"P={100*prec:.1f}%  R={100*rec:.1f}%  F1={100*f1:.1f}%")
    print()
    print("MISSED most often :", fn_counter.most_common() or "none")
    print("EXTRA most often  :", fp_counter.most_common() or "none")

    print()
    print("Principles where the model found NOTHING but gold expects something:")
    silent = [p for p in sorted(gold, key=pkey)
              if gold[p] and not got.get(p)]
    print("  ", silent or "none")


if __name__ == "__main__":
    main()
