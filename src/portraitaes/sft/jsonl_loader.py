"""Optional in-memory JSONL loader for ms-swift SFT datasets.

The patch bypasses the Arrow cache for local, non-streaming JSONL files. It is
useful on storage systems where cache construction is slow, at the cost of
holding the full source dataset in memory.
"""

import json
import os
import time
from typing import Any, Dict, List

from datasets import Dataset as HfDataset


THEME_SCORE_KEYS = ("fit", "visual_consistency", "originality")


def _coerce_theme_score_labels(row: Dict[str, Any]) -> List[float]:
    labels = row.get("theme_score_labels")
    if not isinstance(labels, dict):
        return [float("nan"), float("nan"), float("nan")]
    values = []
    for key in THEME_SCORE_KEYS:
        try:
            score = float(labels[key])
        except Exception:
            score = float("nan")
        if score == score:
            score = min(5.0, max(1.0, score))
        values.append(score)
    return values


def _bulk_load_jsonl(path: str) -> tuple[HfDataset, List[List[float]]]:
    t0 = time.time()
    with open(path, 'rb') as f:
        data = [json.loads(line) for line in f if line.strip()]
    theme_score_labels = [_coerce_theme_score_labels(row) for row in data]
    ds = HfDataset.from_list(data)
    print(f'[jsonl_inmemory_loader] loaded {len(ds)} rows from {path} '
          f'in {time.time() - t0:.1f}s (in-memory)')
    return ds, theme_score_labels


def patch_swift_jsonl_loader() -> None:
    try:
        from swift.dataset import loader as _loader_mod
        from swift.dataset.preprocessor import RowPreprocessor
    except ImportError as e:
        print(f'[jsonl_inmemory_loader] swift not importable ({e}), skip patch')
        return

    _original = _loader_mod.DatasetLoader._load_dataset_path

    def _patched(self, dataset_path, dataset_meta):
        ext = os.path.splitext(dataset_path)[1].lstrip('.').lower()
        is_local_jsonl = (
            ext == 'jsonl'
            and isinstance(dataset_path, str)
            and os.path.isfile(dataset_path)
            and not getattr(self, 'streaming', False)
        )
        if not is_local_jsonl:
            return _original(self, dataset_path, dataset_meta)

        dataset, theme_score_labels = _bulk_load_jsonl(dataset_path)
        if self.columns:
            dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
        dataset = dataset_meta.preprocess_func(
            dataset,
            num_proc=self.num_proc,
            load_from_cache_file=self.load_from_cache_file,
            strict=self.strict,
        )
        if self.remove_unused_columns:
            dataset = RowPreprocessor.remove_useless_columns(dataset)
        if len(dataset) == len(theme_score_labels):
            dataset = dataset.map(
                lambda _, idx: {"theme_score_labels": theme_score_labels[idx]},
                with_indices=True,
                load_from_cache_file=False,
            )
        else:
            print('[jsonl_inmemory_loader] skip theme_score_labels attach: '
                  f'preprocessed rows={len(dataset)} raw rows={len(theme_score_labels)}')
        return dataset

    _loader_mod.DatasetLoader._load_dataset_path = _patched
    print('[jsonl_inmemory_loader] Patched swift.dataset.loader.DatasetLoader._load_dataset_path '
          'to bulk-load local .jsonl into memory')
