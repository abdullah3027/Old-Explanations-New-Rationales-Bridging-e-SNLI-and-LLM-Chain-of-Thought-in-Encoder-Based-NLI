"""CoT trace generation pipeline for Variant D using Ollama (local inference).

Generates 10,000 chain-of-thought reasoning traces from e-SNLI training examples,
stratified across 3 labels (3,334 entailment + 3,333 neutral + 3,333 contradiction).

Usage:
    python data/cot_generation.py                          # generate / resume
    python data/cot_generation.py --validate-only          # check existing file
    python data/cot_generation.py --model llama3:latest    # use a different model
    python data/cot_generation.py --output path/to/out.csv # custom output path

Requires Ollama running locally: https://ollama.com
Recommended model: gemma3:latest (fastest — 4.3B Q4_K_M)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Set

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"

DEFAULT_MODEL  = "gemma3:latest"
DEFAULT_OUTPUT = "Datasets/CoT/cot_traces.csv"
N_PER_LABEL    = 3334          # 3334 + 3333 + 3333 = 10,000 total
RANDOM_SEED    = 42

VALID_LABELS = {"entailment", "neutral", "contradiction"}

LABEL_LEAKAGE_WORDS = {"entailment", "neutral", "contradiction"}

# Describes the relationship without using the label word itself.
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
# Ollama API
# ---------------------------------------------------------------------------

def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 60,
) -> Optional[str]:
    """Call Ollama chat endpoint. Returns generated text or None on failure."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 200,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to Ollama. Is it running? Start with: ollama serve")
        sys.exit(1)
    except requests.exceptions.Timeout:
        return None

    if resp.status_code != 200:
        return None

    try:
        text = resp.json()["message"]["content"].strip()
    except (KeyError, ValueError):
        return None

    return text if len(text) >= 30 else None


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


def select_subset(esnli_df: pd.DataFrame, n_per_label: int = N_PER_LABEL, seed: int = RANDOM_SEED) -> pd.DataFrame:
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
    model: str = DEFAULT_MODEL,
    sleep_between: float = 0.1,
) -> None:
    """Generate CoT traces for source_df, writing results to output_path.

    Checkpoint/resume: appends one row per successful generation.
    Re-running picks up exactly where it left off.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine which pair_ids are already done
    done_ids: Set[int] = set()
    if output_path.exists():
        existing = pd.read_csv(output_path, usecols=["pair_id", "cot_rationale"], dtype={"pair_id": int})
        done_ids = set(
            existing.loc[existing["cot_rationale"].notna() & (existing["cot_rationale"].str.strip() != ""), "pair_id"]
        )
        print(f"Resuming — {len(done_ids):,} already done, {len(source_df) - len(done_ids):,} remaining.")

    remaining = source_df[~source_df["pair_id"].isin(done_ids)].reset_index(drop=True)
    total = len(source_df)
    done_count = len(done_ids)

    if remaining.empty:
        print("All traces already generated. Run --validate-only to inspect.")
        return

    file_exists = output_path.exists()
    start_time = time.time()

    for i, row in enumerate(remaining.itertuples(index=False)):
        prompt = _build_prompt(row.Sentence1, row.Sentence2, row.gold_label)
        cot = call_ollama(prompt, model=model)

        record = {
            "pair_id":       row.pair_id,
            "gold_label":    row.gold_label,
            "Sentence1":     row.Sentence1,
            "Sentence2":     row.Sentence2,
            "Explanation_1": row.Explanation_1,
            "cot_rationale": cot if cot is not None else "",
            "cot_model":     model,
        }

        pd.DataFrame([record]).to_csv(
            output_path, mode="a", header=not file_exists, index=False
        )
        file_exists = True
        done_count += 1

        if (i + 1) % 100 == 0:
            elapsed = (time.time() - start_time) / 60
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining_count = total - done_count
            eta = remaining_count / rate if rate > 0 else float("inf")
            print(f"[{done_count:,}/{total:,}] elapsed={elapsed:.1f}min | rate={rate:.1f} ex/min | ETA=~{eta:.0f}min")

        if sleep_between > 0:
            time.sleep(sleep_between)

    elapsed_total = (time.time() - start_time) / 60
    print(f"\nDone. Generated {done_count:,} traces in {elapsed_total:.1f} min.")
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

    df = pd.read_csv(path, dtype={"pair_id": int})
    total = len(df)
    non_empty = df["cot_rationale"].notna() & (df["cot_rationale"].str.strip() != "")
    filled = non_empty.sum()

    leakage = df.loc[non_empty, "cot_rationale"].str.lower().apply(
        lambda t: any(w in t for w in LABEL_LEAKAGE_WORDS)
    ).sum()

    word_counts = df.loc[non_empty, "cot_rationale"].str.split().apply(len)

    label_dist = df["gold_label"].value_counts().sort_index()

    print("=" * 45)
    print("CoT Trace Validation")
    print("=" * 45)
    print(f"Total rows:            {total:>7,}")
    print(f"Non-empty rationale:   {filled:>7,}  ({filled/total*100:.1f}%)")
    print(f"Label word leakage:    {leakage:>7,}  ({leakage/filled*100:.1f}%)  ← target: <5%")
    print(f"Avg word count:        {word_counts.mean():>7.1f}")
    print(f"Min / Max word count:  {word_counts.min():>7} / {word_counts.max()}")
    print("Label distribution:")
    for label, count in label_dist.items():
        print(f"  {label:<15} {count:>6,}  ({count/total*100:.1f}%)")
    print("=" * 45)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CoT traces for Variant D using Ollama.")
    parser.add_argument("--model",         default=DEFAULT_MODEL,  help="Ollama model name")
    parser.add_argument("--output",        default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--n-per-label",   type=int, default=N_PER_LABEL, help="Examples per label (default 3334 → 10K total)")
    parser.add_argument("--validate-only", action="store_true",    help="Only validate existing file, no generation")
    args = parser.parse_args()

    if args.validate_only:
        validate_traces(args.output)
        return

    # Locate project root (two levels up from this file)
    project_root = Path(__file__).resolve().parent.parent

    print(f"Model:        {args.model}")
    print(f"Output:       {args.output}")
    print(f"Target:       {args.n_per_label * 3:,} traces ({args.n_per_label} per label)")
    print()

    print("Loading e-SNLI training data...")
    esnli_df = _load_esnli_train(project_root)
    print(f"  Loaded {len(esnli_df):,} examples")

    print("Selecting stratified subset...")
    subset = select_subset(esnli_df, n_per_label=args.n_per_label)
    print(f"  Subset size: {len(subset):,}")
    print(f"  Label counts:\n{subset['gold_label'].value_counts().to_string()}")
    print()

    generate_cot_traces(
        source_df=subset,
        output_path=args.output,
        model=args.model,
    )

    print()
    validate_traces(args.output)


if __name__ == "__main__":
    main()
