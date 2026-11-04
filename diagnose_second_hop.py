"""
diagnose_second_hop.py — offline checks for second-hop retrieval. No LLM calls.

Builds every Pass-1 prompt the pipeline would send for PIPEDA Principles
4.1-4.10, exactly as stage_extract() does (same article list, same concept-tag
pre-filter, same retriever call, same prompt builder), and writes them to a
JSON file keyed by "<article_ref> | <concept>".

    python3 diagnose_second_hop.py --out prompts.off.json
    python3 diagnose_second_hop.py --second-hop --out prompts.on.json
    python3 diagnose_second_hop.py --compare prompts.off.json prompts.on.json

--compare prints which prompts changed and by how many characters, plus the
largest prompt in each file. Prompts are compared byte for byte.

The detected-reference table is printed whenever --second-hop is given.

Embedder: pass --embedder to use the real BM25 pickle. Without it the
retriever falls back to DB order. That only matters for 4.3 (two candidate
chunks, both returned at top_k=3, possibly in a different order), and it is
the same for both runs, so an off/on comparison is still valid.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from privacy_schema.prompts import build_concept_prompt          # noqa: E402
from rag_pipeline.retriever import Retriever                     # noqa: E402
from rag_pipeline.store import ChunkStore                        # noqa: E402
from run_pipeline import PASS1_CONCEPTS                          # noqa: E402

ARTICLES = ("4.1 Principle 1,4.2 Principle,4.3 Principle,4.4 Principle,"
            "4.5 Principle,4.6 Principle,4.7 Principle,4.8 Principle,"
            "4.9 Principle,4.10 Principle")
LAW = "PIPEDA"


def _filter() -> re.Pattern:
    # Same construction as run_pipeline.main(): substrings, case-insensitive.
    parts = [re.escape(p.strip()) for p in ARTICLES.split(",") if p.strip()]
    return re.compile("|".join(parts), re.IGNORECASE)


def build_prompts(db: Path, embedder: Path | None, top_k: int,
                  second_hop: bool) -> dict[str, str]:
    filt = _filter()
    kwargs = {"second_hop": True} if second_hop else {}
    retriever = Retriever(db, {LAW: embedder or Path("/nonexistent.pkl")},
                          **kwargs)
    out: dict[str, str] = {}
    with ChunkStore(db) as store:
        arts = store.list_articles(
            law=LAW, levels=["article", "section", "principle", "document"])
        for art in arts:
            ref = art["article_ref"]
            if not filt.search(ref):
                continue
            tags = art.get("concept_tags", [])
            for concept in PASS1_CONCEPTS:
                if concept not in tags:
                    continue
                rag = retriever.retrieve_for_prompt(
                    concept=concept, law=LAW, article_ref=ref, top_k=top_k)
                if rag.startswith("[No relevant"):
                    continue
                system, user = build_concept_prompt(concept, LAW, ref, rag)
                out[f"{ref} | {concept}"] = system + "\n\n" + user
    retriever.close()
    return out


def compare(a_path: Path, b_path: Path) -> int:
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    keys = sorted(set(a) | set(b))
    changed = [k for k in keys if a.get(k) != b.get(k)]
    print(f"{len(keys)} prompts; {len(keys) - len(changed)} identical; "
          f"{len(changed)} changed")
    for k in changed:
        la, lb = len(a.get(k, "")), len(b.get(k, ""))
        print(f"  {k:55s} {la:6d} -> {lb:6d} chars ({lb - la:+d})")
    for name, d in ((a_path.name, a), (b_path.name, b)):
        k = max(d, key=lambda x: len(d[x]))
        print(f"largest in {name}: {len(d[k])} chars  [{k}]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", default="data/chunks.db", type=Path)
    p.add_argument("--embedder", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--second-hop", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    args = p.parse_args()
    logging.basicConfig(level=logging.ERROR)

    if args.compare:
        return compare(*args.compare)
    if not args.out:
        p.error("--out is required unless --compare is given")

    if args.second_hop:
        from rag_pipeline.second_hop import reference_table
        with ChunkStore(args.db) as store:
            print(reference_table(store, LAW))

    prompts = build_prompts(args.db, args.embedder, args.top_k,
                            args.second_hop)
    args.out.write_text(json.dumps(prompts, indent=1, ensure_ascii=False))
    print(f"wrote {len(prompts)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
