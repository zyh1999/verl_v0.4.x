#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download PRIME-RL/Eurus-2-RL-Data from HuggingFace and export a CODE-only
subset into parquet files that are directly consumable by verl RLHFDataset.

The produced parquet is expected to contain (at least):
  - prompt: chat messages (list[{"role": ..., "content": ...}, ...])
  - data_source: one of {"codecontests","apps","codeforces","taco"} (or your custom tag)
  - reward_model: {"ground_truth": <test_cases_json_or_dict_or_str>}

Example:
  python examples/data_preprocess/eurus2_code.py --local_dir ~/data/code

If the dataset doesn't provide a test/validation split, this script will
create one from train via --test_size / --test_ratio.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, Optional


CODE_SOURCES = {"codecontests", "apps", "codeforces", "taco"}


def _first_existing(d: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            return k
    return None


def _ensure_reward_model(example: Dict[str, Any], gt_key: Optional[str]) -> Dict[str, Any]:
    if "reward_model" in example and isinstance(example["reward_model"], dict) and "ground_truth" in example["reward_model"]:
        return example

    ground_truth = None
    if gt_key and gt_key in example:
        ground_truth = example.get(gt_key)
    else:
        # Common fallbacks across code datasets.
        gt_key_auto = _first_existing(
            example,
            keys=[
                "ground_truth",
                "test_cases",
                "tests",
                "input_output",
                "reference",
                "answer",
            ],
        )
        if gt_key_auto:
            ground_truth = example.get(gt_key_auto)

    # Normalize json string for dict-like cases if possible (optional).
    if isinstance(ground_truth, str):
        s = ground_truth.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                ground_truth = json.loads(s)
            except Exception:
                pass

    example["reward_model"] = {"ground_truth": ground_truth}
    return example


def _ensure_prompt(example: Dict[str, Any], prompt_key: Optional[str]) -> Dict[str, Any]:
    if "prompt" in example:
        return example

    if prompt_key and prompt_key in example:
        example["prompt"] = example[prompt_key]
        return example

    # Try common chat field names.
    k = _first_existing(example, keys=["messages", "conversation", "chat", "conversations"])
    if k:
        example["prompt"] = example[k]
        return example

    raise KeyError(
        "Cannot find prompt/messages field. Please pass --prompt_key to map the correct column to 'prompt'. "
        f"Available keys: {sorted(example.keys())}"
    )


def _ensure_data_source(example: Dict[str, Any], data_source_key: Optional[str]) -> Dict[str, Any]:
    if "data_source" in example:
        return example

    if data_source_key and data_source_key in example:
        example["data_source"] = example[data_source_key]
        return example

    # Try common field names.
    k = _first_existing(example, keys=["source", "task", "dataset", "tag"])
    if k:
        example["data_source"] = example[k]
        return example

    # If completely missing, set a generic code tag. You can change it later.
    example["data_source"] = "codecontests"
    return example


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PRIME-RL/Eurus-2-RL-Data")
    parser.add_argument("--local_dir", required=True, help="Output dir, e.g. ~/data/code")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--test_split", default=None, help="e.g. test/validation. If absent, will split train.")
    parser.add_argument("--test_size", type=int, default=2048, help="Used when test_split is missing.")
    parser.add_argument("--test_ratio", type=float, default=None, help="Alternative to --test_size.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prompt_key", default=None, help="Column name to map to 'prompt' if 'prompt' is absent.")
    parser.add_argument("--data_source_key", default=None, help="Column name to map to 'data_source' if absent.")
    parser.add_argument("--ground_truth_key", default=None, help="Column name used as ground_truth if reward_model is absent.")
    parser.add_argument("--code_only", action="store_true", default=True)
    parser.add_argument("--no_code_only", dest="code_only", action="store_false")
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    try:
        from datasets import DatasetDict, load_dataset
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'datasets'. Please install it in your env: `pip install datasets`"
        ) from e

    print(f"[eurus2_code] loading dataset={args.dataset} ...", flush=True)
    ds_any = load_dataset(args.dataset)
    if isinstance(ds_any, DatasetDict):
        ds_dict = ds_any
    else:
        # Some datasets return a single Dataset for load_dataset(name, split=...).
        ds_dict = DatasetDict({args.train_split: ds_any})

    if args.train_split not in ds_dict:
        raise KeyError(f"train_split='{args.train_split}' not found. Available: {list(ds_dict.keys())}")

    train_ds = ds_dict[args.train_split]
    test_ds = None
    if args.test_split and args.test_split in ds_dict:
        test_ds = ds_dict[args.test_split]
    else:
        # Create a holdout set from train.
        if args.test_ratio is not None:
            split = train_ds.train_test_split(test_size=args.test_ratio, seed=args.seed)
        else:
            test_size = min(max(args.test_size, 1), len(train_ds) // 10 if len(train_ds) > 10 else 1)
            split = train_ds.train_test_split(test_size=test_size, seed=args.seed)
        train_ds, test_ds = split["train"], split["test"]

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        ex = _ensure_prompt(ex, prompt_key=args.prompt_key)
        ex = _ensure_data_source(ex, data_source_key=args.data_source_key)
        ex = _ensure_reward_model(ex, gt_key=args.ground_truth_key)
        return ex

    print("[eurus2_code] normalizing columns -> prompt/data_source/reward_model.ground_truth ...", flush=True)
    train_ds = train_ds.map(_map_fn)
    test_ds = test_ds.map(_map_fn)

    if args.code_only:
        def _is_code(ex: Dict[str, Any]) -> bool:
            ds = ex.get("data_source")
            if isinstance(ds, str) and ds in CODE_SOURCES:
                return True
            # Some eurus-like datasets may use generic tags like "code".
            if isinstance(ds, str) and ds.lower() in {"code", "coding"}:
                return True
            return False

        before_train, before_test = len(train_ds), len(test_ds)
        train_ds = train_ds.filter(_is_code)
        test_ds = test_ds.filter(_is_code)
        print(f"[eurus2_code] code_only filter: train {before_train} -> {len(train_ds)}, test {before_test} -> {len(test_ds)}", flush=True)

    train_path = os.path.join(local_dir, "train.parquet")
    test_path = os.path.join(local_dir, "test.parquet")
    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)

    print(f"[eurus2_code] wrote: {train_path}", flush=True)
    print(f"[eurus2_code] wrote: {test_path}", flush=True)
    print("[eurus2_code] tip: run training with CODE_TRAIN_PATH/CODE_TEST_PATH if you store them elsewhere.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


