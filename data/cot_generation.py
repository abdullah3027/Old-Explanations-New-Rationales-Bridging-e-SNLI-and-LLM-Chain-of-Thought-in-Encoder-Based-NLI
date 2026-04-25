"""CoT trace generation pipeline for Variant D using HuggingFace Inference API.

Generates ~2,000 chain-of-thought reasoning traces from e-SNLI training examples
using DeepSeek-R1-Distill-Qwen-7B — the best open-source reasoning model available
on the HF free tier.

Target: 667 per label × 3 labels = 2,001 traces total (balanced).

Rate limit strategy: HF free tier allows ~500-1000 requests/day. The script
auto-pauses on 429 responses and resumes. Run it daily until complete (~2 days).
Checkpoint/resume means no work is ever lost on interruption.

Usage:
    python data/cot_generation.py --hf-token hf_xxxx
    python data/cot_generation.py --hf-token hf_xxxx --validate-only
    python data/cot_generation.py --hf-token hf_xxxx --output path/to/out.csv

Or set HF_TOKEN env variable to avoid passing it every time:
    export HF_TOKEN=hf_xxxx
    python data/cot_generation.py
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Set

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_API_URL   = "https://api-inference.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_OUTPUT = "Datasets/CoT/cot_traces.csv"

N_PER_LABEL  = 667          # 667 × 3 = 2,001 total (~2K)
RANDOM_SEED  = 42

VALID_LABELS        = {"entailment", "neutral", "contradiction"}
LABEL_LEAKAGE_WORDS = {"entailment", "neutral", "contradiction"}

# Describes the relationship without naming the label — prevents MLM leakage.
LABEL_DESCRIPTIONS = {
    "entailment":    "the hypothesis logically follows from the premise",
    "neutral":       "the hypothesis is neither confirmed nor contradicted by the premise",
    "contradiction": "the hypothesis contradicts what the premise states",
}

PROMPT_TEMPLATE = """\
You are a logical reasoning assistant for Natural Language Inference.

Premise: {premise}
Hypothesis: {hypothesis}
Relationship: {label_description}

Write a chain-of-thought reasoning trace explaining WHY this relationship holds.

Rules:
- Exactly 3 to 5 sentences. Label each "Step 1:", "Step 2:", etc.
- Do NOT use the words "entailment", "neutral", or "contradiction".
- Do NOT copy the premise or hypothesis verbatim.
- Each step must make a distinct logical point.
- The final step states the conclusion.

Reasoning:\
"""

# ---------------------------------------------------------------------------
# HuggingFace Inference API
# ---------------------------------------------------------------------------

def _strip_think_tags(text: str) -> str:
    """Remove DeepSeek-R1 internal <think>...</think> blocks from output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_hf_inference(
    prompt: str,
    hf_token: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 6,
    timeout: int = 60,
) -> Optional[str]:
    """Call HF Inference API (v1 chat completions). Returns generated text or None.

    Handles:
      - 429 Too Many Requests  → sleeps for Retry-After (or 60s), retries up to max_retries
      - 503 Model Loading      → sleeps 30s, retries
      - other errors           → returns None immediately (logged)
    """
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.3,
        "stream": False,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.Timeout:
            print(f"  [timeout on attempt {attempt + 1}]", flush=True)
            time.sleep(10)
            continue
        except requests.exceptions.RequestException as e:
            print(f"  [request error: {e}]", flush=True)
            return None

        if resp.status_code == 200:
            try:
                text = resp.json()["choices"][0]["message"]["content"]
                text = _strip_think_tags(text).strip()
                return text if len(text) >= 30 else None
            except (KeyError, IndexError, ValueError):
                return None

        elif resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            print(f"\n  [Rate limit hit — sleeping {retry_after}s before retry {attempt + 1}/{max_retries}]",
                  flush=True)
            time.sleep(retry_after)

        elif resp.status_code == 503:
            print(f"\n  [Model loading (503) — sleeping 30s, attempt {attempt + 1}/{max_retries}]",
                  flush=True)
            time.sleep(30)

        else:
            print(f"\n  [HTTP {resp.status_code}: {resp.text[:120]}]", flush=True)
            return None

    print(f"\n  [Max retries ({max_retries}) reached — skipping example]", flush=True)
    return None


def _build_prompt(premise: str, hypothesis: str, label: str) -> str:
    return PROMPT_TEMPLATE.format(
        premise=premise,
        hypothesis=hypothesis,
        label_description=LABEL_DESCRIPTIONS[label],
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_esnli_train(project_root: Path) -> pd.DataFrame:
    """Load and concatenate both e-SNLI training CSVs."""
    dfs = []
    for fname in ("esnli_train_1.csv", "esnli_train_2.csv"):
        path = project_root / "Datasets" / "E-SNLI" / fname
        df = pd.read_csv(path, usecols=["gold_label", "Sentence1", "Sentence2", "Explanation_1"])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["gold_label"].isin(VALID_LABELS)]
    df = df.dropna(subset=["Sentence1", "Sentence2", "Explanation_1"])
    return df.reset_index(drop=True)


def select_subset(
    esnli_df: pd.DataFrame,
    n_per_label: int = N_PER_LABEL,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Stratified sample: n_per_label rows per label, perfectly balanced."""
    subset = (
        esnli_df
        .groupby("gold_label", group_keys=False)
        .apply(lambda g: g.sample(min(n_per_label, len(g)), random_state=seed))
        .reset_index(drop=True)
    )
    subset.insert(0, "pair_id", range(len(subset)))
    return subset


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def generate_cot_traces(
    source_df: pd.DataFrame,
    output_path: str,
    hf_token: str,
    model: str = DEFAULT_MODEL,
    sleep_between: float = 1.5,
) -> None:
    """Generate CoT traces, appending one row per call for safe checkpoint/resume.

    Checkpoint/resume: re-running the script automatically skips completed rows.
    Rate-limited days: run the script daily — it picks up where it left off.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine already-completed pair_ids
    done_ids: Set[int] = set()
    if output_path.exists():
        existing = pd.read_csv(
            output_path,
            usecols=["pair_id", "cot_rationale"],
            dtype={"pair_id": int},
        )
        done_ids = set(
            existing.loc[
                existing["cot_rationale"].notna() &
                (existing["cot_rationale"].str.strip() != ""),
                "pair_id",
            ]
        )
        print(f"Resuming — {len(done_ids):,} already done, "
              f"{len(source_df) - len(done_ids):,} remaining.")

    remaining = source_df[~source_df["pair_id"].isin(done_ids)].reset_index(drop=True)
    total      = len(source_df)
    done_count = len(done_ids)

    if remaining.empty:
        print("All traces already generated. Run --validate-only to inspect.")
        return

    file_exists = output_path.exists()
    start_time  = time.time()

    for i, row in enumerate(remaining.itertuples(index=False)):
        prompt = _build_prompt(row.Sentence1, row.Sentence2, row.gold_label)
        cot    = call_hf_inference(prompt, hf_token=hf_token, model=model)

        record = {
            "pair_id":       row.pair_id,
            "gold_label":    row.gold_label,
            "Sentence1":     row.Sentence1,
            "Sentence2":     row.Sentence2,
            "Explanation_1": row.Explanation_1,
            "cot_rationale": cot if cot is not None else "",
            "cot_model":     model,
        }

        # Append immediately — one row = one durable checkpoint
        pd.DataFrame([record]).to_csv(
            output_path, mode="a", header=not file_exists, index=False
        )
        file_exists = True
        done_count += 1

        if (i + 1) % 50 == 0:
            elapsed = (time.time() - start_time) / 60
            rate    = (i + 1) / elapsed if elapsed > 0 else 0
            eta     = (total - done_count) / rate if rate > 0 else float("inf")
            print(
                f"[{done_count:,}/{total:,}] "
                f"elapsed={elapsed:.1f}min | "
                f"rate={rate:.1f} ex/min | "
                f"ETA=~{eta:.0f}min",
                flush=True,
            )

        time.sleep(sleep_between)

    elapsed_total = (time.time() - start_time) / 60
    print(f"\nDone. {done_count:,} traces in {elapsed_total:.1f} min.")
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_traces(path: str) -> None:
    """Print quality report for an existing cot_traces.csv."""
    path = Path(path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    df        = pd.read_csv(path, dtype={"pair_id": int})
    total     = len(df)
    non_empty = df["cot_rationale"].notna() & (df["cot_rationale"].str.strip() != "")
    filled    = non_empty.sum()

    leakage = df.loc[non_empty, "cot_rationale"].str.lower().apply(
        lambda t: any(w in t for w in LABEL_LEAKAGE_WORDS)
    ).sum()

    word_counts = df.loc[non_empty, "cot_rationale"].str.split().apply(len)
    label_dist  = df["gold_label"].value_counts().sort_index()

    print("=" * 47)
    print("  CoT Trace Validation Report")
    print("=" * 47)
    print(f"  Total rows:            {total:>6,}")
    print(f"  Non-empty rationale:   {filled:>6,}  ({filled / total * 100:.1f}%)")
    print(f"  Label word leakage:    {leakage:>6,}  ({leakage / max(filled, 1) * 100:.1f}%)  ← target <5%")
    print(f"  Avg word count:        {word_counts.mean():>6.1f}")
    print(f"  Min / Max words:       {word_counts.min():>3} / {word_counts.max()}")
    print("  Label distribution:")
    for label, count in label_dist.items():
        print(f"    {label:<15} {count:>5,}  ({count / total * 100:.1f}%)")
    print("=" * 47)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2K CoT traces for Variant D via HF Inference API."
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument("--model",         default=DEFAULT_MODEL,  help="HF model ID")
    parser.add_argument("--output",        default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--n-per-label",   type=int, default=N_PER_LABEL,
                        help="Examples per label (default 667 → ~2K total)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing file, skip generation")
    args = parser.parse_args()

    if args.validate_only:
        validate_traces(args.output)
        return

    if not args.hf_token:
        print("ERROR: HuggingFace token required.")
        print("  Pass --hf-token hf_xxxx  OR  set HF_TOKEN environment variable.")
        print("  Get your token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent

    print(f"Model:   {args.model}")
    print(f"Output:  {args.output}")
    print(f"Target:  {args.n_per_label * 3:,} traces ({args.n_per_label} per label)")
    print(f"Note:    HF free tier = ~500-1000 req/day → may take 2-3 days total.")
    print(f"         Re-run this script daily — checkpoint/resume handles the rest.")
    print()

    print("Loading e-SNLI training data...")
    esnli_df = _load_esnli_train(project_root)
    print(f"  Loaded {len(esnli_df):,} examples")

    print("Selecting stratified subset...")
    subset = select_subset(esnli_df, n_per_label=args.n_per_label)
    print(f"  Subset: {len(subset):,} examples")
    print(f"  Labels:\n{subset['gold_label'].value_counts().to_string()}")
    print()

    generate_cot_traces(
        source_df=subset,
        output_path=args.output,
        hf_token=args.hf_token,
        model=args.model,
    )

    print()
    validate_traces(args.output)


if __name__ == "__main__":
    main()
