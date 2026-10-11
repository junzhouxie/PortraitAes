"""Optional in-memory JSONL loader for ms-swift GRPO datasets.

Besides bypassing the Arrow cache, the adapter serializes nested reward
metadata before ms-swift preprocessing and restores it afterward. This keeps
custom reward columns available without relying on nested Arrow schemas.
"""

import json
import os
import time
from typing import Any, Dict, List

from datasets import Dataset as HfDataset


THEME_SCORE_KEYS = ("fit", "visual_consistency", "originality")
REWARD_JSON_COLUMNS = ("dimension_score_labels", "gt_full_json", "weights_used", "emotion_label")
REWARD_STRING_COLUMNS = (
    "aesthetics_dimension",
    "gt_score_key",
    "weight_category",
    "weight_category_name",
    "gt_subject_code",
    "composition_form",
    "completeness_layer_used",
    "tone_eval_mode",
    "emotion_category",
)


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


def _json_dumps_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return ""


def _coerce_float_or_nan(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if value == value else float("nan")


def _coerce_reward_columns(row: Dict[str, Any]) -> Dict[str, Any]:
    """Keep reward-side columns after ms-swift preprocessing.

    ms-swift may drop or reshape non-standard columns during preprocessing.
    Re-attaching a small set of simple scalar/string columns here keeps reward
    functions independent from Arrow nested-dict compatibility.
    """
    out: Dict[str, Any] = {"theme_score_labels": _coerce_theme_score_labels(row)}
    out["gt_score"] = _coerce_float_or_nan(row.get("gt_score"))
    for key in REWARD_STRING_COLUMNS:
        value = row.get(key)
        out[key] = "" if value is None else str(value)
    for key in REWARD_JSON_COLUMNS:
        out[key] = _json_dumps_or_empty(row.get(key))
    return out


def _bulk_load_jsonl(path: str) -> tuple[HfDataset, List[Dict[str, Any]]]:
    t0 = time.time()
    with open(path, 'rb') as f:
        data = [json.loads(line) for line in f if line.strip()]
    reward_columns = [_coerce_reward_columns(row) for row in data]
    # Keep the dataset schema simple before ms-swift preprocessing.  The same
    # columns are re-attached after preprocessing from `reward_columns`.
    safe_data: List[Dict[str, Any]] = []
    for row, reward_col in zip(data, reward_columns):
        safe_row = dict(row)
        for key in REWARD_JSON_COLUMNS:
            if key in safe_row:
                safe_row[key] = reward_col[key]
        safe_row["theme_score_labels"] = reward_col["theme_score_labels"]
        safe_row["gt_score"] = reward_col["gt_score"]
        for key in REWARD_STRING_COLUMNS:
            if key in safe_row:
                safe_row[key] = reward_col[key]
        safe_data.append(safe_row)
    ds = HfDataset.from_list(safe_data)
    print(f'[jsonl_inmemory_loader] loaded {len(ds)} rows from {path} '
          f'in {time.time() - t0:.1f}s (in-memory)')
    return ds, reward_columns


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

        dataset, reward_columns = _bulk_load_jsonl(dataset_path)
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
        if len(dataset) == len(reward_columns):
            dataset = dataset.map(
                lambda _, idx: reward_columns[idx],
                with_indices=True,
                load_from_cache_file=False,
            )
        else:
            print('[jsonl_inmemory_loader] skip reward-column attach: '
                  f'preprocessed rows={len(dataset)} raw rows={len(reward_columns)}')
        return dataset

    _loader_mod.DatasetLoader._load_dataset_path = _patched
    print('[jsonl_inmemory_loader] Patched swift.dataset.loader.DatasetLoader._load_dataset_path '
          'to bulk-load local .jsonl into memory')
