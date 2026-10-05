#!/usr/bin/env python3
"""
apply_prompt_budget.py — make silent prompt truncation visible.

THE RISK
--------
The payload sets temperature, seed, max_tokens and the schema, but never the
context window — Ollama chooses it. If a prompt exceeds that window, Ollama
truncates the prompt and answers anyway. Nothing in the response says so. The
run looks healthy and the damage shows up as poor extraction, which is
indistinguishable from the model simply being wrong.

That matters most for Pass-2, which concatenates every Pass-1 concept into one
prompt, so the richest statement produces the longest prompt in the run. PIPEDA
4.3 (eight concepts) is both the slowest call and the likeliest to clip.

WHAT THIS DOES — OBSERVATION ONLY
---------------------------------
Records the prompt_tokens Ollama reports for every call, tracks the maximum per
pass, and prints it in the summary. Flags any call whose prompt_tokens lands
exactly on a common context boundary (2048 / 4096 / 8192 / ...), which is the
fingerprint of clipping: a truncated prompt is exactly as long as the window.

Zero behaviour change. No request field is added or altered, so scored output
must be byte-identical. If it is not, that is a bug in this patch.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not set num_ctx. Ollama's OpenAI-compatible /v1/chat/completions
endpoint ignores an `options` block — that is native-API only — so a --num-ctx
flag here would be a placebo that reads as a fix. The real levers are the
OLLAMA_CONTEXT_LENGTH environment variable or `PARAMETER num_ctx` in a
Modelfile, and raising the window can change output (for the better, if
something was being clipped), so it belongs on the ladder as its own condition
with its own measurement — not smuggled in alongside a diagnostic.

Idempotent. Run from the repo root:

    python3 apply_prompt_budget.py
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
    if "max_prompt_tokens" in src:
        print("Already applied (found max_prompt_tokens). Nothing to do.")
        return

    # ── 1. stats fields ──────────────────────────────────────────────────────
    anchor = "    citation_flags:              dict = field(default_factory=dict)\n"
    if anchor not in src:
        die("could not find citation_flags — apply apply_citation_normalisation.py first.")
    src = src.replace(anchor, anchor + (
        "    max_prompt_tokens:           int = 0\n"
        "    prompt_token_samples:        list = field(default_factory=list)\n"
        "    suspected_truncations:       list = field(default_factory=list)\n"
    ), 1)

    # ── 2. the boundary table + recorder ─────────────────────────────────────
    helper_anchor = "class LLMBackend(ABC):"
    if helper_anchor not in src:
        die("could not find the LLMBackend class to anchor the recorder on.")
    recorder = '''# Context windows a runner is likely to be using. A truncated prompt is exactly
# as long as the window, so a prompt_tokens count landing precisely on one of
# these is the signature of clipping rather than a coincidence.
_CONTEXT_BOUNDARIES = (2048, 4096, 8192, 16384, 32768, 65536, 131072)


def _record_prompt_tokens(stats, n: int, label: str) -> None:
    """
    Track prompt size so silent truncation stops being silent.

    Ollama reports prompt_tokens as the number of tokens it actually processed.
    If it clipped the prompt to fit the context window, that count equals the
    window exactly — which is the only signal available, since neither the
    response nor the logs mention truncation.
    """
    if n <= 0:
        return
    stats.prompt_token_samples.append((label, n))
    if n > stats.max_prompt_tokens:
        stats.max_prompt_tokens = n
    if n in _CONTEXT_BOUNDARIES:
        stats.suspected_truncations.append((label, n))
        log.warning(
            f"  PROMPT {label}: prompt_tokens == {n}, exactly a common context "
            f"window. The prompt was probably TRUNCATED and the answer is "
            f"based on incomplete input. Raise the window via "
            f"OLLAMA_CONTEXT_LENGTH or a Modelfile and re-measure."
        )


'''
    src = src.replace(helper_anchor, recorder + helper_anchor, 1)

    # ── 3. record at the one place usage is read ─────────────────────────────
    old_usage = """                usage             = data.get("usage", {})
                stats.tokens_in  += usage.get("prompt_tokens",    0)
                stats.tokens_out += usage.get("completion_tokens", 0)
"""
    if old_usage not in src:
        die("could not find the usage accounting block in LocalBackend.call.")
    src = src.replace(old_usage, """                usage             = data.get("usage", {})
                stats.tokens_in  += usage.get("prompt_tokens",    0)
                stats.tokens_out += usage.get("completion_tokens", 0)

                # Pass-2 runs schema=None; Pass-1 always passes a schema. That
                # is enough to tell the two apart without threading a label
                # through every call site.
                _record_prompt_tokens(
                    stats,
                    usage.get("prompt_tokens", 0),
                    "pass-1" if schema is not None else "pass-2",
                )
""", 1)

    # ── 4. report it ─────────────────────────────────────────────────────────
    summary_anchor = """        if self.citation_flags:"""
    if summary_anchor not in src:
        die("could not find the citation summary block to anchor the report on.")
    report = '''        if self.max_prompt_tokens:
            per_pass: dict = {}
            for label, n in self.prompt_token_samples:
                per_pass[label] = max(per_pass.get(label, 0), n)
            detail = "  ".join(
                f"{label}={n}" for label, n in sorted(per_pass.items())
            )
            log.info(
                f"  Largest prompt       : {self.max_prompt_tokens} tokens  "
                f"({detail})"
            )
            if self.suspected_truncations:
                log.warning(
                    f"  TRUNCATION SUSPECTED : "
                    f"{len(self.suspected_truncations)} call(s) hit a context "
                    f"boundary exactly — output may be based on a clipped "
                    f"prompt. See the PROMPT warnings above."
                )
            else:
                log.info(
                    "  Truncation check     : no call landed on a context "
                    "boundary"
                )

'''
    src = src.replace(summary_anchor, report + summary_anchor, 1)

    TARGET.write_text(src)
    print(f"✓ Patched {TARGET}")
    print("  - PipelineStats: max_prompt_tokens / prompt_token_samples /")
    print("                   suspected_truncations")
    print("  - _record_prompt_tokens() + a context-boundary signature check")
    print("  - 'Largest prompt' and a truncation verdict in the summary")
    print()
    print("Observation only — no request field changed, so scored output must")
    print("be identical. Test against the cached Pass-1, no 40-minute redo:")
    print()
    print("  python3 run_pipeline.py --stage assemble \\")
    print("    --backend local --local-model mistral-nemo \\")
    print("    --repo data/model_repo.citations.db")


if __name__ == "__main__":
    main()