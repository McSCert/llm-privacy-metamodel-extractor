#!/usr/bin/env python3
"""
apply_assemble_resume.py — stop a single timeout from discarding a whole run.

What happened on 2026-10-01: Pass-1 completed all ten PIPEDA principles over
33 minutes, Pass-2 stored three statements, then the fourth assembly call
(4.3 — the largest statement, eight concepts in one prompt) exceeded the
hardcoded 300-second socket timeout. The TimeoutError propagated out of
stage_assemble_and_store and killed the process. `extraction_results` lives
only in memory, so all 33 minutes of Pass-1 were lost and the run had to
start over.

Three independent problems, one patch:

  1. `timeout=300` is hardcoded and too low for Pass-2 on a local model.
     Pass-2 concatenates every Pass-1 concept into one prompt, so the biggest
     statement is the slowest call in the run — the opposite of what a
     uniform timeout assumes. Becomes --llm-timeout, default 900.

  2. `backend.call` in the Pass-2 retry loop sits OUTSIDE the try block, so a
     transient network error is not retried, it is fatal. It now behaves like
     any other assembly failure: retried, and on exhaustion the statement is
     skipped and the run continues.

  3. Pass-1 output is never persisted. It is now written to
     <repo>.pass1.json, and `--stage assemble` reads it back, so a lost
     Pass-2 costs minutes rather than the whole extraction.

Idempotent. Run from the repo root:

    python3 apply_assemble_resume.py
    git diff --stat
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path("run_pipeline.py")


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        die(f"{TARGET} not found — run this from the repo root.")

    src = TARGET.read_text()
    if "--llm-timeout" in src:
        print("Already applied (found --llm-timeout). Nothing to do.")
        return

    # ── 1. configurable timeout ──────────────────────────────────────────────
    old_init = """    def __init__(self, base_url: str, model: str):
        self.model       = model
        self._endpoint   = base_url.rstrip("/") + "/chat/completions"
        self._models_url = base_url.rstrip("/") + "/models"
"""
    if old_init not in src:
        die("could not find LocalBackend.__init__ to add the timeout to.")
    src = src.replace(old_init, """    def __init__(self, base_url: str, model: str, timeout: int = 900):
        self.model       = model
        self._endpoint   = base_url.rstrip("/") + "/chat/completions"
        self._models_url = base_url.rstrip("/") + "/models"
        # Pass-2 concatenates every Pass-1 concept into a single prompt, so the
        # richest statement is the slowest call in the run. A uniform timeout
        # sized for Pass-1 will reliably fail on exactly the statements that
        # matter most. 300s killed PIPEDA 4.3 (eight concepts).
        self._timeout    = timeout
""", 1)

    if "urllib.request.urlopen(req, timeout=300)" not in src:
        die("could not find the hardcoded timeout=300 call.")
    src = src.replace(
        "urllib.request.urlopen(req, timeout=300)",
        "urllib.request.urlopen(req, timeout=self._timeout)",
        1,
    )

    old_make = "        return LocalBackend(base_url=args.local_url, model=args.local_model)"
    if old_make not in src:
        die("could not find the LocalBackend instantiation.")
    src = src.replace(old_make, (
        "        return LocalBackend(\n"
        "            base_url=args.local_url, model=args.local_model,\n"
        "            timeout=args.llm_timeout,\n"
        "        )"
    ), 1)

    # ── 2. a network error must not be fatal in Pass-2 ───────────────────────
    old_call = """        # Pass-2 uses text-mode (schema=None) — structured output for Pass-2
        # is a planned future improvement.
        last_raw = backend.call(system, user, stats, schema=None, max_tokens=4096)
"""
    if old_call not in src:
        die("could not find the Pass-2 backend.call to guard.")
    src = src.replace(old_call, """        # Pass-2 uses text-mode (schema=None) — structured output for Pass-2
        # is a planned future improvement.
        #
        # Guarded: this call used to sit outside the try below, so a socket
        # timeout was fatal to the whole run rather than one retryable failure
        # for one statement. A transient network error is now treated like any
        # other assembly error — retried, and on exhaustion this statement is
        # skipped so the remaining articles still get assembled and stored.
        try:
            last_raw = backend.call(
                system, user, stats, schema=None, max_tokens=4096
            )
        except Exception as call_exc:
            last_errors = [f"backend call failed: {type(call_exc).__name__}: {call_exc}"]
            log.warning(
                f"  Pass-2 call failed @ {law}/{article_ref} "
                f"(attempt {attempt}): {type(call_exc).__name__}: {call_exc}"
            )
            continue
""", 1)

    # ── 3. persist Pass-1 and add --stage assemble ───────────────────────────
    anchor = "def _parse_law_files("
    if anchor not in src:
        die("could not find _parse_law_files to anchor the cache helpers on.")
    cache_helpers = '''def _pass1_cache_path(repo_path: Path) -> Path:
    """Where Pass-1 output is cached, derived from the repo db path."""
    return repo_path.with_suffix(repo_path.suffix + ".pass1.json")


def _save_pass1(extraction_results: dict, repo_path: Path) -> None:
    """
    Persist Pass-1 so a failed Pass-2 does not discard it.

    Pass-1 is the expensive half (~33 min for ten PIPEDA principles on a local
    model) and was previously held only in memory, so any exception in Pass-2
    threw all of it away.
    """
    path = _pass1_cache_path(repo_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(extraction_results, indent=2))
        n = sum(len(v) for v in extraction_results.values())
        log.info(f"  Pass-1 cached        : {path}  ({n} article(s))")
    except Exception as exc:
        log.warning(f"  Could not cache Pass-1 output: {exc}")


def _load_pass1(repo_path: Path) -> dict:
    """Read back a cached Pass-1 for --stage assemble."""
    path = _pass1_cache_path(repo_path)
    if not path.exists():
        log.error(
            f"No Pass-1 cache at {path}. "
            f"Run --stage extract first (it writes the cache)."
        )
        sys.exit(1)
    results = json.loads(path.read_text())
    n = sum(len(v) for v in results.values())
    log.info(f"  Pass-1 loaded        : {path}  ({n} article(s))")
    return results


'''
    src = src.replace(anchor, cache_helpers + anchor, 1)

    # --stage gains "assemble"
    old_stage = '"--stage", choices=["ingest", "extract", "analyse", "all"], default="all",'
    if old_stage not in src:
        die("could not find the --stage choices.")
    src = src.replace(
        old_stage,
        '"--stage", choices=["ingest", "extract", "assemble", "analyse", "all"],\n'
        '        default="all",',
        1,
    )

    # --llm-timeout argument, added next to the local-model argument.
    anchor_arg = '        "--stage", choices='
    src = src.replace(anchor_arg, (
        '        "--llm-timeout", type=int, default=900, metavar="SECONDS",\n'
        '        help=(\n'
        '            "Socket timeout for one LLM call (default: 900). "\n'
        '            "Pass-2 assembles every concept into one prompt, so the "\n'
        '            "largest statement is the slowest call in the run; 300 was "\n'
        '            "too low and killed PIPEDA 4.3 mid-run."\n'
        '        ),\n'
        '    )\n'
        '    p.add_argument(\n'
        + anchor_arg
    ), 1)

    # Wire the stages in main().
    old_main = """        stage_assemble_and_store(
            extraction_results = extraction_results,"""
    if old_main not in src:
        die("could not find the stage_assemble_and_store call in main().")
    src = src.replace(old_main, """        _save_pass1(extraction_results, repo_path)

        stage_assemble_and_store(
            extraction_results = extraction_results,""", 1)

    # A standalone assemble stage that reuses the cache.
    assemble_anchor = """    # ── Stage 5: Gap Analysis ─────────────────────────────────────────────────"""
    if assemble_anchor not in src:
        die("could not find the gap-analysis stage comment.")
    src = src.replace(assemble_anchor, """    # ── Stage 3-4 only: resume assembly from a cached Pass-1 ──────────────────
    if args.stage == "assemble":
        stage_assemble_and_store(
            extraction_results = _load_pass1(repo_path),
            repo_path          = repo_path,
            backend            = _make_backend(args),
            stats              = stats,
            max_retries        = args.max_retries,
            xmi_out_dir        = Path(args.xmi_out) if args.xmi_out else None,
        )

""" + assemble_anchor, 1)

    TARGET.write_text(src)
    print(f"✓ Patched {TARGET}")
    print("  - --llm-timeout (default 900, was a hardcoded 300)")
    print("  - Pass-2 backend.call guarded: a timeout costs one statement, not the run")
    print("  - Pass-1 cached to <repo>.pass1.json")
    print("  - --stage assemble resumes from that cache")
    print()
    print("Resume the crashed run without redoing Pass-1:")
    print("  python3 run_pipeline.py --stage assemble \\")
    print("    --backend local --local-model mistral-nemo \\")
    print("    --repo data/model_repo.citations.db")


if __name__ == "__main__":
    main()
