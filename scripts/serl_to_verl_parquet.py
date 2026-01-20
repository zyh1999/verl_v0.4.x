#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert SeRL (jsonl) datasets into verl-compatible parquet datasets.

Output schema follows verl examples (e.g. examples/data_preprocess/gsm8k.py):
  - data_source: str
  - prompt: list[{role, content}]
  - ability: str
  - reward_model: {style, ground_truth}
  - extra_info: dict
"""

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


_BOXED_RE = re.compile(r"^\\boxed\{(.*)\}$", re.DOTALL)


def _clean_ground_truth(x: Any) -> str:
    """Heuristic cleanup: unwrap \\boxed{...} and trim."""
    if x is None:
        return ""
    s = x if isinstance(x, str) else str(x)
    s = s.strip()
    m = _BOXED_RE.match(s)
    if m:
        s = m.group(1).strip()
    return s


_INT_RE = re.compile(r"(-?\d+)")


def _as_int_or_none(x: Any) -> Optional[int]:
    """Normalize numeric-like values to int to keep parquet schemas alignable across datasets.

    This is important when we concatenate multiple parquet files via HuggingFace Datasets:
    - Some sources use `level: 5` (int), others use `level: "Level 5"` (string).
    - Some sources use `idx: "123"` (string), others use `idx: 123` (int).
    We normalize them to int (or None if not parseable).
    """
    if x is None:
        return None
    if isinstance(x, bool):
        # Avoid treating bool as int here; schema-wise it's safer to drop it.
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if x.is_integer():
            return int(x)
        return None
    if isinstance(x, str):
        m = _INT_RE.search(x)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None
    # Unknown type
    return None


def _last_boxed_substring(string: str) -> Optional[str]:
    """Extract the last \\boxed{...} (or \\boxed ... form) substring from a solution/answer text.

    Copied in spirit from `verl.utils.reward_score.math_reward.last_boxed_only_string`
    to keep this script self-contained.
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def _remove_boxed_wrapper(s: str) -> str:
    s = s.strip()
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :].strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{") : -1].strip()
    return s


def to_verl_rows(
    jsonl_path: str,
    split: str,
    data_source: str,
    instruction: str,
    ability: str = "math",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ex in _iter_jsonl(jsonl_path):
        problem = ex.get("problem") or ex.get("question") or ""
        answer_raw = ex.get("answer") or ex.get("ground_truth") or ""
        # Ground truth normalization:
        # - For math_500, answer is usually exactly "\\boxed{...}" -> _clean_ground_truth works.
        # - For math_hard, "answer" can be a full explanation containing the final "\\boxed{...}".
        #   We extract the last boxed expression so reward matches `math_reward.compute_score`.
        ground_truth = _clean_ground_truth(answer_raw)
        if isinstance(answer_raw, str):
            boxed = _last_boxed_substring(answer_raw)
            if boxed:
                ground_truth = _remove_boxed_wrapper(boxed)

        prompt_text = (str(problem).strip() + " " + instruction).strip()

        rows.append(
            {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": ability,
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split,
                    # Keep schema stable across different sources (math_500 / math_hard / train_with_idx)
                    "idx": _as_int_or_none(ex.get("idx")),
                    "unique_id": ex.get("unique_id"),
                    "subject": ex.get("subject"),
                    "level": _as_int_or_none(ex.get("level")),
                    "answer_raw": answer_raw,
                    "solution": ex.get("solution"),
                    "source_file": os.path.basename(jsonl_path),
                },
            }
        )
    return rows


def _write_parquet(rows: List[Dict[str, Any]], out_path: str) -> None:
    # Prefer datasets (same dependency as verl preprocess scripts)
    try:
        from datasets import Dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: datasets. Please run this script inside your verl/apptainer python env."
        ) from e

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Dataset.from_list(rows).to_parquet(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serl-dir", default="/scratch/u6g/zhouyihe.u6g/SeRL")
    ap.add_argument(
        "--out-dir",
        default="/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/data/serl_math",
        help="Output dir where train.parquet/test.parquet will be written.",
    )
    ap.add_argument(
        "--train-jsonl",
        default=None,
        help="Optional override: path to SeRL train jsonl. Default: <serl-dir>/openrlhf/dataset/math/train_with_idx.jsonl",
    )
    ap.add_argument(
        "--test-jsonl",
        default=None,
        help="Optional override: path to eval jsonl. Default: <serl-dir>/evaluation/Math-Benchmarks/data/math_500/test_with_idx.jsonl",
    )
    ap.add_argument(
        "--instruction",
        default='Let\'s think step by step and output the final answer after "####".',
    )
    ap.add_argument("--ability", default="math")
    ap.add_argument("--train-data-source", default="serl_math_train_full")
    ap.add_argument("--test-data-source", default="serl_math_500")

    args = ap.parse_args()

    train_jsonl = args.train_jsonl or os.path.join(
        args.serl_dir, "openrlhf", "dataset", "math", "train_with_idx.jsonl"
    )
    test_jsonl = args.test_jsonl or os.path.join(
        args.serl_dir, "evaluation", "Math-Benchmarks", "data", "math_500", "test_with_idx.jsonl"
    )

    if not os.path.isfile(train_jsonl):
        raise FileNotFoundError(train_jsonl)
    if not os.path.isfile(test_jsonl):
        raise FileNotFoundError(test_jsonl)

    out_train = os.path.join(args.out_dir, "train.parquet")
    out_test = os.path.join(args.out_dir, "test.parquet")

    train_rows = to_verl_rows(
        train_jsonl,
        split="train",
        data_source=args.train_data_source,
        instruction=args.instruction,
        ability=args.ability,
    )
    test_rows = to_verl_rows(
        test_jsonl,
        split="test",
        data_source=args.test_data_source,
        instruction=args.instruction,
        ability=args.ability,
    )

    _write_parquet(train_rows, out_train)
    _write_parquet(test_rows, out_test)

    print(f"[OK] wrote {out_train} (rows={len(train_rows)})")
    print(f"[OK] wrote {out_test} (rows={len(test_rows)})")


if __name__ == "__main__":
    main()


