"""Built-in seven-task PortraitAes inference and score fusion."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from typing import Any, Callable, Iterable, Mapping


TYPE_TASK = "type_agent"
DIMENSION_TASKS = (
    "structure",
    "composition",
    "image_quality",
    "tone",
    "theme_style",
    "emotion",
)
TASKS = (TYPE_TASK, *DIMENSION_TASKS)

SCORE_KEYS = {
    "structure": "structure_score",
    "composition": "composition_score",
    "image_quality": "image_quality_score",
    "tone": "tone_score",
    "theme_style": "theme_style_score",
    "emotion": "emotion_score",
}

TYPE_NAMES = {
    "I": "功能类",
    "II": "商业类",
    "III": "叙事类",
    "IV": "审美类",
    "V": "社交类",
    "VI": "艺术类",
    "VII": "综合类",
    "VIII": "随手拍",
    "IX": "符号类",
}

TYPE_WEIGHTS = {
    "I": {"structure": 0.20, "composition": 0.20, "image_quality": 0.30,
          "tone": 0.15, "theme_style": 0.10, "emotion": 0.05},
    "II": {"structure": 0.10, "composition": 0.15, "image_quality": 0.35,
           "tone": 0.25, "theme_style": 0.10, "emotion": 0.05},
    "III": {"structure": 0.15, "composition": 0.15, "image_quality": 0.15,
            "tone": 0.20, "theme_style": 0.10, "emotion": 0.25},
    "IV": {"structure": 0.10, "composition": 0.15, "image_quality": 0.15,
           "tone": 0.30, "theme_style": 0.20, "emotion": 0.10},
    "V": {"structure": 0.05, "composition": 0.15, "image_quality": 0.15,
          "tone": 0.20, "theme_style": 0.30, "emotion": 0.15},
    "VI": {"structure": 0.10, "composition": 0.15, "image_quality": 0.05,
           "tone": 0.20, "theme_style": 0.20, "emotion": 0.30},
    "VII": {"structure": 0.10, "composition": 0.15, "image_quality": 0.15,
            "tone": 0.20, "theme_style": 0.20, "emotion": 0.20},
    "VIII": {"structure": 0.05, "composition": 0.10, "image_quality": 0.05,
             "tone": 0.15, "theme_style": 0.15, "emotion": 0.50},
    "IX": {"structure": 0.00, "composition": 0.05, "image_quality": 0.00,
           "tone": 0.05, "theme_style": 0.30, "emotion": 0.60},
}

_TYPE_NAME_TO_CODE = {name: code for code, name in TYPE_NAMES.items()}
_PROMPT_FILENAMES = {task: f"{task}.txt" for task in TASKS}


@lru_cache(maxsize=None)
def get_system_prompt(task: str) -> str:
    """Return the released system prompt for one of the seven tasks."""
    if task not in _PROMPT_FILENAMES:
        raise ValueError(f"Unknown PortraitAes task: {task}")
    prompt = (
        resources.files("portraitaes.inference")
        .joinpath("prompts", _PROMPT_FILENAMES[task])
        .read_text(encoding="utf-8")
        .strip()
    )
    if not prompt:
        raise RuntimeError(f"Empty built-in prompt for task: {task}")
    return prompt


def build_task_messages(
    task: str,
    image_parts: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the same system-and-image messages used by the evaluation set."""
    content = [dict(part) for part in image_parts]
    if not content:
        raise ValueError("At least one image is required")

    return [
        {"role": "system", "content": get_system_prompt(task)},
        {"role": "user", "content": content},
    ]


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first complete JSON object from a generated response."""
    if not text:
        return None
    stripped = text.strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            value, _ = decoder.raw_decode(stripped[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _as_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str):
        try:
            score = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not 1.0 <= score <= 5.0:
        return None
    return round(score, 4)


def extract_dimension_score(
    task: str,
    assessment: Mapping[str, Any] | None,
    raw_text: str = "",
) -> float | None:
    """Extract a validated 1--5 aggregate score for one specialist."""
    key = SCORE_KEYS[task]
    if isinstance(assessment, Mapping):
        score = _as_score(assessment.get(key))
        if score is not None:
            return score
    match = re.search(
        rf'["\']{re.escape(key)}["\']\s*:\s*(-?\d+(?:\.\d+)?)',
        raw_text,
    )
    return _as_score(match.group(1)) if match else None


def get_type_code(router_result: Mapping[str, Any] | None) -> str | None:
    """Read and normalize the I--IX category from a router response."""
    if not isinstance(router_result, Mapping):
        return None
    photo_type = router_result.get("photo_type")
    if isinstance(photo_type, Mapping):
        candidate = photo_type.get("weight_category")
        name = photo_type.get("weight_category_name")
    else:
        candidate = router_result.get("weight_category")
        name = router_result.get("weight_category_name")
    if isinstance(candidate, str):
        code = candidate.strip().upper()
        if code in TYPE_WEIGHTS:
            return code
    if isinstance(name, str):
        return _TYPE_NAME_TO_CODE.get(name.strip())
    return None


def _router_weights(router_result: Mapping[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(router_result, Mapping):
        return None
    values = router_result.get("weights_used")
    if not isinstance(values, Mapping):
        return None
    weights: dict[str, float] = {}
    for task in DIMENSION_TASKS:
        value = values.get(task)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if float(value) < 0:
            return None
        weights[task] = float(value)
    total = sum(weights.values())
    if total <= 0:
        return None
    return {task: round(value / total, 6) for task, value in weights.items()}


def resolve_type_weights(
    router_result: Mapping[str, Any] | None,
) -> tuple[str | None, dict[str, float] | None, str | None]:
    """Resolve weights from the predicted type, with router values as fallback."""
    code = get_type_code(router_result)
    if code is not None:
        return code, dict(TYPE_WEIGHTS[code]), "type_table"
    weights = _router_weights(router_result)
    if weights is not None:
        return None, weights, "router_output"
    return None, None, None


def format_pipeline_result(
    task_outputs: Mapping[str, Mapping[str, Any]],
    excluded_dimensions: Iterable[str] = (),
    include_raw_predictions: bool = False,
) -> dict[str, Any]:
    """Format seven task outputs and calculate the type-weighted total score."""
    router_entry = task_outputs.get(TYPE_TASK, {})
    router = router_entry.get("parsed")
    if not isinstance(router, Mapping):
        router = None
    type_code, weights, weight_source = resolve_type_weights(router)

    excluded = {task for task in excluded_dimensions if task in DIMENSION_TASKS}
    dimensions: dict[str, Any] = {}
    dimension_scores: dict[str, float | None] = {}
    missing: list[str] = []
    errors: dict[str, str] = {}

    router_error = router_entry.get("error")
    if router_error:
        errors[TYPE_TASK] = str(router_error)

    for task in DIMENSION_TASKS:
        entry = task_outputs.get(task, {})
        parsed = entry.get("parsed")
        assessment = dict(parsed) if isinstance(parsed, Mapping) else None
        score = extract_dimension_score(task, assessment, str(entry.get("raw") or ""))
        dimension_scores[task] = score
        dimensions[task] = assessment
        if task in excluded:
            continue
        if score is None:
            missing.append(task)
        if entry.get("error"):
            errors[task] = str(entry["error"])

    overall_score = None
    weight_coverage = 0.0
    if weights:
        weighted_sum = 0.0
        for task, weight in weights.items():
            score = dimension_scores.get(task)
            if task in excluded or score is None or weight <= 0:
                continue
            weighted_sum += weight * score
            weight_coverage += weight
        if weight_coverage > 0:
            overall_score = round(weighted_sum / weight_coverage, 4)
            weight_coverage = round(weight_coverage, 6)

    complete = (
        type_code is not None
        and not missing
        and not errors
        and not excluded
        and overall_score is not None
    )
    photo_type = router.get("photo_type") if router else None
    if isinstance(photo_type, Mapping):
        photo_type = dict(photo_type)
    result: dict[str, Any] = {
        "status": "complete" if complete else ("partial" if overall_score is not None else "failed"),
        "photo_type": photo_type,
        "router": dict(router) if router else None,
        "weight_category": type_code,
        "weight_category_name": TYPE_NAMES.get(type_code),
        "weights_used": weights,
        "weight_source": weight_source,
        "dimension_scores": dimension_scores,
        "overall_score": overall_score,
        "score_range": [1.0, 5.0],
        "weight_coverage": weight_coverage,
        "missing_dimensions": missing,
        "excluded_dimensions": sorted(excluded),
        "dimensions": dimensions,
    }
    if errors:
        result["task_errors"] = errors
    if include_raw_predictions:
        result["raw_predictions"] = {
            task: task_outputs.get(task, {}).get("raw") for task in TASKS
        }
    return result


def run_full_pipeline(
    generate: Callable[[list[dict[str, Any]]], str],
    image_parts: Iterable[dict[str, Any]],
    excluded_dimensions: Iterable[str] = (),
    include_raw_predictions: bool = False,
) -> dict[str, Any]:
    """Run router then specialists and return one formatted result object."""
    image_parts = tuple(dict(part) for part in image_parts)
    excluded = {task for task in excluded_dimensions if task in DIMENSION_TASKS}
    task_outputs: dict[str, dict[str, Any]] = {}

    def run_task(task: str) -> None:
        entry: dict[str, Any] = {"raw": None, "parsed": None}
        try:
            messages = build_task_messages(task, image_parts)
            raw = generate(messages)
            entry["raw"] = raw
            entry["parsed"] = parse_json_object(raw)
            if entry["parsed"] is None:
                entry["error"] = "Model response is not a JSON object"
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        task_outputs[task] = entry

    run_task(TYPE_TASK)
    for task in DIMENSION_TASKS:
        if task not in excluded:
            run_task(task)

    return format_pipeline_result(
        task_outputs,
        excluded_dimensions=excluded,
        include_raw_predictions=include_raw_predictions,
    )
