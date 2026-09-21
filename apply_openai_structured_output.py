#!/usr/bin/env python3
"""
apply_openai_structured_output.py — wire schema-constrained decoding into the
OpenAI backend.

Why this is needed
------------------
stage_extract calls  backend.call(system, user, stats, schema=validator).

  * LocalLLMBackend    injects response_format={"type":"json_schema", ...}
  * AnthropicBackend   injects the schema as a tool input_schema
  * OpenAIBackend      accepted **kwargs and SILENTLY DISCARDED the schema

So every GPT-4o run so far has been free-form text generation whose JSON
happened to parse, not schema-constrained decoding.  That matters before
A1: A1 changes the Pass-1 contract from one object to a list, and without
a schema on the wire, a malformed list would surface as an unexplained
failure instead of a schema rejection.

What this changes (run_pipeline.py, OpenAIBackend.call only)
------------------------------------------------------------
  * accept `schema` as a named parameter instead of swallowing it
  * build response_format from the Pydantic model, reusing _inline_refs()
    exactly as the local backend does
  * strict=False: OpenAI strict mode additionally requires
    additionalProperties=false and EVERY property listed as required, which
    the current models do not satisfy.  strict=False still constrains the
    output to the schema's shape and is the safe first step.
  * skip _strip_fences when a schema is active (same rule as local)
  * log system_fingerprint at debug level, so a later run can check whether
    the serving backend changed between runs

Run from the repo root:
    python apply_openai_structured_output.py            # check only
    python apply_openai_structured_output.py --write    # apply

All-or-nothing; idempotent (already-applied edits report ALREADY).
"""
import subprocess
import sys
from pathlib import Path

OLD_SIGNATURE = '''        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        for attempt in range(4):
            try:
                resp = self._client.chat.completions.create(
'''

OLD_TAIL = '''                stats.api_calls  += 1
                stats.tokens_in  += resp.usage.prompt_tokens
                stats.tokens_out += resp.usage.completion_tokens
                raw = resp.choices[0].message.content or ""
                return self._strip_fences(raw)
'''

NEW_SIGNATURE = '''        max_tokens: int = 2048,
        schema:     Optional[type] = None,
        **kwargs,
    ) -> str:
        # ── Inject JSON Schema into decoding constraints ───────────────────────
        # Previously `schema` fell into **kwargs and was discarded, so the
        # OpenAI arm ran unconstrained while the local arm did not.
        # strict=False: OpenAI strict mode also demands additionalProperties
        # =false and every property required, which these models do not meet.
        extra: dict = {}
        if schema is not None:
            json_schema = schema.model_json_schema()
            json_schema.pop("title", None)
            json_schema = _inline_refs(json_schema)   # flatten $defs
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name":   schema.__name__,
                    "strict": False,
                    "schema": json_schema,
                },
            }

        for attempt in range(4):
            try:
                resp = self._client.chat.completions.create(
'''

NEW_TAIL = '''                stats.api_calls  += 1
                stats.tokens_in  += resp.usage.prompt_tokens
                stats.tokens_out += resp.usage.completion_tokens
                log.debug(
                    f"system_fingerprint="
                    f"{getattr(resp, \'system_fingerprint\', None)}"
                )
                raw = resp.choices[0].message.content or ""
                # _strip_fences only needed in text-mode (schema=None)
                return raw if schema is not None else self._strip_fences(raw)
'''

# The messages=[...] block between these two anchors is left untouched, so
# whatever indentation you used for model/max_tokens/temperature/seed survives.
# Only the argument list needs **extra appended, done separately below.
EDITS = [
    ("run_pipeline.py", OLD_SIGNATURE, NEW_SIGNATURE),
    ("run_pipeline.py", OLD_TAIL, NEW_TAIL),
]


def add_extra_kwarg(src: str) -> tuple[str, str]:
    """
    Append **extra to the OpenAI create(...) call.

    Matched structurally, not by exact text: find the create( inside
    OpenAIBackend and insert **extra before its closing parenthesis, so any
    argument alignment works.
    """
    i = src.find("class OpenAIBackend")
    if i < 0:
        return src, "NOT FOUND (class OpenAIBackend)"
    j = src.find("\nclass ", i + 1)
    block = src[i: j if j > 0 else len(src)]
    k = block.find("self._client.chat.completions.create(")
    if k < 0:
        return src, "NOT FOUND (create call)"
    if "**extra" in block:
        return src, "ALREADY"

    depth, end = 0, None
    for pos in range(block.index("(", k), len(block)):
        if block[pos] == "(":
            depth += 1
        elif block[pos] == ")":
            depth -= 1
            if depth == 0:
                end = pos
                break
    if end is None:
        return src, "NOT FOUND (unbalanced parentheses)"

    line_start = block.rfind("\n", 0, end) + 1
    indent = block[line_start:end]
    insert = f"{indent}    **extra,\n{indent}"
    patched_block = block[:line_start] + insert + block[end:]
    return src[:i] + patched_block + src[i + len(block):], "OK"


def main() -> int:
    write = "--write" in sys.argv
    if not Path("run_pipeline.py").exists():
        print("Run this from the repo root.")
        return 1

    texts: dict[str, str] = {}
    bad = 0
    for path, old, new in EDITS:
        s = texts.setdefault(path, Path(path).read_text(encoding="utf-8"))
        label = old.strip().splitlines()[0][:52]
        if new in s:
            status = "ALREADY"
        elif s.count(old) == 1:
            status = "OK"
            texts[path] = s.replace(old, new)
        else:
            status = f"NOT FOUND (matches={s.count(old)})"
            bad += 1
        print(f"  {status:22s} {path}: {label!r}")

    if bad:
        print("\nNot found — nothing written. Most likely your temperature/seed "
              "lines are formatted differently from the block this script "
              "expects. Paste lines 528-552 of run_pipeline.py and I'll adjust.")
        return 1
    if not write:
        print("\nCheck passed. Re-run with --write to apply.")
        return 0

    src, status = add_extra_kwarg(texts["run_pipeline.py"])
    print(f"  {status:22s} run_pipeline.py: '**extra in create(...)'")
    if status.startswith("NOT FOUND"):
        print("\nNothing written.")
        return 1
    texts["run_pipeline.py"] = src

    for path, s in texts.items():
        Path(path).write_text(s, encoding="utf-8")

    print("\nVerifying ...")
    check = subprocess.run(
        [sys.executable, "-c",
         "import ast,inspect,sys;"
         "src=open('run_pipeline.py').read();ast.parse(src);"
         "i=src.index('class OpenAIBackend');"
         "blk=src[i:i+4000];"
         "print('  response_format present :', 'response_format' in blk);"
         "print('  schema is a parameter   :', 'schema:     Optional[type]' in blk);"
         "print('  fence-skip on schema    :', 'raw if schema is not None' in blk)"],
        capture_output=True, text=True)
    print((check.stdout or check.stderr).rstrip())
    print("\nDone. Next: re-run the variance harness and compare the noise floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())