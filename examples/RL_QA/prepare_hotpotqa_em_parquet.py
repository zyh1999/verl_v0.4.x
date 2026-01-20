#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a QA-EM verifiable dataset for verl GRPO without any code execution.

This script downloads HotpotQA (distractor) via HuggingFace `datasets` and converts it into
verl-compatible parquet files:
  - data/RL_QA/hotpotqa_em/train.parquet     (from hotpotqa train)
  - data/RL_QA/hotpotqa_em/test.parquet      (from hotpotqa validation; used as test)

Important for verl built-in reward (QA-EM):
  - data_source must be "searchR1_hotpotqa" to trigger `search_r1_like_qa_em.compute_score`
  - reward_model.ground_truth must be a dict containing {"target": <answer or list-of-answers>}
  - model output should include <answer>...</answer> (handled by prompt instruction)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List


INSTRUCTION = (
    "Answer the question using the provided context.\n"
    "Put ONLY the final answer inside <answer> and </answer> tags.\n"
    "Example: <answer>Paris</answer>\n"
)


def build_context(ctx: Dict[str, Any]) -> str:
    """HotpotQA context: {'title': [..], 'sentences': [[..], ...]}."""
    titles = ctx.get("title", []) or []
    sents = ctx.get("sentences", []) or []
    chunks: List[str] = []
    for t, ss in zip(titles, sents):
        text = " ".join(ss) if isinstance(ss, list) else str(ss)
        chunks.append(f"[{t}] {text}")
    return "\n".join(chunks)


def to_rows(dataset, split_name: str, data_source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ex in dataset:
        question = ex.get("question", "")
        answer = ex.get("answer", "")
        context = build_context(ex.get("context", {}) or {})
        prompt = f"{INSTRUCTION}\nContext:\n{context}\n\nQuestion:\n{question}\n"

        rows.append(
            {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "qa",
                "reward_model": {"style": "rule", "ground_truth": {"target": answer}},
                "extra_info": {
                    "split": split_name,
                    "id": ex.get("id"),
                    "level": ex.get("level"),
                    "type": ex.get("type"),
                    "supporting_facts": ex.get("supporting_facts"),
                },
            }
        )
    return rows


def main() -> None:
    try:
        from datasets import Dataset, load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: datasets. Please run inside your verl python env (e.g. source .verl/bin/activate)."
        ) from e

    out_dir = os.environ.get("OUT_DIR", "data/RL_QA/hotpotqa_em")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("hotpot_qa", "distractor")
    train_rows = to_rows(ds["train"], "train", "searchR1_hotpotqa")
    test_rows = to_rows(ds["validation"], "test", "searchR1_hotpotqa")

    Dataset.from_list(train_rows).to_parquet(os.path.join(out_dir, "train.parquet"))
    Dataset.from_list(test_rows).to_parquet(os.path.join(out_dir, "test.parquet"))

    print(f"[OK] wrote {os.path.join(out_dir, 'train.parquet')} (rows={len(train_rows)})")
    print(f"[OK] wrote {os.path.join(out_dir, 'test.parquet')} (rows={len(test_rows)})")


if __name__ == "__main__":
    main()


