"""GRPO reward plugin for PortraitAes seven-dimension aesthetic training.

The public registry names remain compatible with theme-only datasets:

  - peer_relative_dir   : R1 absolute score/category reward
  - bpr_rank_cross_dir  : R2 cross-rank BPR ranking reward for numeric agents
  - subject_code_hit    : R3 semantic-label hit reward, now generalized
  - format_len          : R4 JSON format and length reward

Supported data formats:

1) Theme-only JSONL:
   - theme_score_labels: {fit, visual_consistency, originality}
   - gt_subject_code

2) Mixed seven-agent JSONL:
   - aesthetics_dimension: type_agent/theme_style/composition/structure/
     image_quality/tone/emotion
   - gt_score, gt_score_key, dimension_score_labels, gt_full_json
   - optional semantic columns: weight_category, gt_subject_code,
     composition_form, completeness_layer_used, tone_eval_mode,
     emotion_category, weights_used.

For type_agent there is no scalar aesthetic score, so R1 optimizes category
and weight-vector accuracy, R2 returns neutral, R3 optimizes category hit.
For the six numeric agents, R1 optimizes the primary dimension score plus
sub-score consistency, R2 ranks only samples from the same agent.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from swift.rewards import ORM, orms


# Environment parsing.

def _envf(name: str, default: str) -> float:
    return float((os.getenv(name) or "").strip() or default)


def _envb(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default).strip().lower() == "true")


# Reward configuration.

RANK_MIN_GAP = _envf("GRPO_RANK_MIN_GAP", "1.0")
RANK_MARGIN = _envf("GRPO_RANK_MARGIN", "0.5")
RANK_TEMP = _envf("GRPO_RANK_TEMP", "0.3")
BPR_DIRECT = _envb("GRPO_BPR_DIRECT", "false")
BPR_CENTERED = _envb("GRPO_BPR_CENTERED", "false")
BPR_NEUTRAL_REWARD = _envf(
    "GRPO_BPR_NEUTRAL",
    "0.0" if BPR_DIRECT and BPR_CENTERED else "0.5",
)

# Theme-style sub-score weights also support the compact dataset schema.
W_FIT = _envf("GRPO_W_FIT", "0.4")
W_VC = _envf("GRPO_W_VC", "0.4")
W_ORIG = _envf("GRPO_W_ORIG", "0.2")

ALPHA = {
    "fit": _envf("GRPO_ALPHA_FIT", "1.5"),
    "vc": _envf("GRPO_ALPHA_VC", "1.0"),
    "orig": _envf("GRPO_ALPHA_ORIG", "0.8"),
}
BETA = {
    "fit": _envf("GRPO_BETA_FIT", "1.0"),
    "vc": _envf("GRPO_BETA_VC", "1.0"),
    "orig": _envf("GRPO_BETA_ORIG", "1.5"),
}

SIGMA = _envf("GRPO_SIGMA", "1.0")
R1_W_GAUSS = _envf("GRPO_R1_W_GAUSS", "1.0")

# Numeric-agent R1 weights.
GENERIC_W_TOTAL = _envf("GRPO_GENERIC_W_TOTAL", "0.8")
GENERIC_W_SUB = _envf("GRPO_GENERIC_W_SUB", "0.2")

# Type-agent R1 weights.
TYPE_W_CATEGORY = _envf("GRPO_TYPE_W_CATEGORY", "0.7")
TYPE_W_WEIGHTS = _envf("GRPO_TYPE_W_WEIGHTS", "0.3")

PARSE_FAIL_PENALTY = _envf("GRPO_PARSE_FAIL_PENALTY", "-1.0")
NEUTRAL_REWARD = 0.5
EPS = 1e-6

FORMAT_SOFT_CHARS = _envf("GRPO_FORMAT_SOFT_CHARS", "1200")
FORMAT_MAX_CHARS = _envf("GRPO_FORMAT_MAX_CHARS", "1800")

R3_HIT_FULL = _envf("GRPO_R3_HIT_FULL", "1.0")
R3_HIT_MAJOR = _envf("GRPO_R3_HIT_MAJOR", "0.4")
R3_MISS = _envf("GRPO_R3_MISS", "-0.3")

# These weights are used only to reconstruct per-agent total reward metrics.
# The four primary ORM outputs remain the sole optimization rewards.
DIAGNOSTIC_REWARD_WEIGHTS = {
    "r1": _envf("GRPO_W_R1", "1.0"),
    "r2": _envf("GRPO_W_R2", "1.0"),
    "r3": _envf("GRPO_W_R3", "1.0"),
    "r4": _envf("GRPO_W_R4", "1.0"),
}
_REWARD_DIAGNOSTIC_CACHE: Dict[str, Any] = {}


# Agent schemas.

TYPE_AGENT = "type_agent"
SCORE_AGENTS = ("theme_style", "composition", "structure", "image_quality", "tone", "emotion")
ALL_AGENTS = (TYPE_AGENT,) + SCORE_AGENTS
AGENT_ID = {agent: idx for idx, agent in enumerate(ALL_AGENTS)}
ID_AGENT = {idx: agent for agent, idx in AGENT_ID.items()}

DEFAULT_SCORE_KEY = {
    "theme_style": "theme_style_score",
    "composition": "composition_score",
    "structure": "structure_score",
    "image_quality": "image_quality_score",
    "tone": "tone_score",
    "emotion": "emotion_score",
}

SUB_SCORE_KEYS = {
    "theme_style": ("fit_score", "visual_consistency", "originality"),
    "composition": ("framing_space_score", "perspective_score", "leading_line_score", "dynamism_score"),
    "structure": ("subject_prominence", "completeness", "background_harmony"),
    "image_quality": ("sharpness_score", "texture_score", "noise_score"),
    "tone": ("light_dark_score", "purity_score", "brightness_score", "harmony_score"),
    "emotion": (
        "intent_consistency_score",
        "physiological_score",
        "context_fusion_score",
        "emotion_intensity_score",
        "body_language_score",
    ),
}

SUB_SCORE_WEIGHTS = {
    "theme_style": {"fit_score": W_FIT, "visual_consistency": W_VC, "originality": W_ORIG},
    # Other dimensions use equal sub-score weights unless the data row provides
    # explicit weights, e.g. composition.weights_used.
}

COMPOSITION_WEIGHT_TO_SCORE = {
    "framing_space": "framing_space_score",
    "perspective": "perspective_score",
    "leading_line": "leading_line_score",
    "dynamism": "dynamism_score",
}

WEIGHT_VECTOR_KEYS = ("structure", "composition", "image_quality", "tone", "theme_style", "emotion")

PRIMARY_ALPHA = {agent: _envf(f"GRPO_ALPHA_{agent.upper()}", "1.0") for agent in SCORE_AGENTS}
PRIMARY_BETA = {agent: _envf(f"GRPO_BETA_{agent.upper()}", "1.0") for agent in SCORE_AGENTS}


# Completion and supervision parsing.

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_NUMBER_RE_CACHE: Dict[str, re.Pattern] = {}
_SUBJECT_RE = re.compile(r'"subject_code"\s*:\s*"(S\d\.\d)"')
_FIT_RE = re.compile(r'"fit_score"\s*:\s*(-?\d+(?:\.\d+)?)')
_VC_RE = re.compile(r'"visual_consistency"\s*:\s*(-?\d+(?:\.\d+)?)')
_ORIG_RE = re.compile(r'"originality"\s*:\s*(-?\d+(?:\.\d+)?)')


def _finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    try:
        value = float(str(value).strip())
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _score_clip(value: Any) -> Optional[float]:
    value = _finite_float(value)
    if value is None:
        return None
    return float(max(1.0, min(5.0, value)))


def _clean_completion(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    body = _clean_completion(text)
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        obj = json.loads(body[start : end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _json_load_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "null", "nan"}:
        return None
    return value


def _number_regex(key: str) -> re.Pattern:
    pat = _NUMBER_RE_CACHE.get(key)
    if pat is None:
        pat = re.compile(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)')
        _NUMBER_RE_CACHE[key] = pat
    return pat


def _number_from_json_or_text(obj: Optional[dict], text: str, key: str) -> Optional[float]:
    if isinstance(obj, dict):
        value = _score_clip(obj.get(key))
        if value is not None:
            return value
    mm = _number_regex(key).search(_clean_completion(text))
    if mm:
        return _score_clip(mm.group(1))
    return None


def _theme_score_from_labels(labels: Optional[dict]) -> Optional[float]:
    if not isinstance(labels, dict):
        return None
    fit = _finite_float(labels.get("fit_score"))
    vc = _finite_float(labels.get("visual_consistency"))
    orig = _finite_float(labels.get("originality"))
    if fit is None or vc is None or orig is None:
        return None
    return float(W_FIT * fit + W_VC * vc + W_ORIG * orig)


def _weighted_sub_score(agent: str, sub_scores: Dict[str, float], weights_used: Any = None) -> Optional[float]:
    keys = SUB_SCORE_KEYS.get(agent)
    if not keys:
        return None
    if not all(k in sub_scores and _finite_float(sub_scores[k]) is not None for k in keys):
        return None

    if agent == "theme_style":
        weights = SUB_SCORE_WEIGHTS["theme_style"]
        return float(sum(float(sub_scores[k]) * float(weights[k]) for k in keys))

    if agent == "composition":
        weights = _json_load_maybe(weights_used)
        if isinstance(weights, dict):
            total = 0.0
            acc = 0.0
            for weight_key, score_key in COMPOSITION_WEIGHT_TO_SCORE.items():
                w = _finite_float(weights.get(weight_key))
                v = _finite_float(sub_scores.get(score_key))
                if w is None or v is None:
                    return None
                acc += w * v
                total += w
            return float(acc / total) if total > 0 else None

    # Conservative default for dimensions without explicit sub-score weights.
    return float(sum(float(sub_scores[k]) for k in keys) / len(keys))


def _extract_theme_gt(gt_item: Any) -> Optional[Dict[str, float]]:
    gt_item = _json_load_maybe(gt_item)
    if gt_item is None:
        return None
    if isinstance(gt_item, (list, tuple)):
        if len(gt_item) < 3:
            return None
        try:
            return {
                "fit_score": float(gt_item[0]),
                "visual_consistency": float(gt_item[1]),
                "originality": float(gt_item[2]),
            }
        except Exception:
            return None
    if isinstance(gt_item, dict):
        try:
            return {
                "fit_score": float(gt_item.get("fit", gt_item.get("fit_score"))),
                "visual_consistency": float(gt_item.get("visual_consistency", gt_item.get("vc"))),
                "originality": float(gt_item.get("originality", gt_item.get("orig"))),
            }
        except Exception:
            return None
    return None


def parse_pred(text: str) -> Optional[Dict[str, Any]]:
    """Parse the compact theme-style response schema."""
    obj = _extract_json_obj(text)
    body = _clean_completion(text)
    fit = _number_from_json_or_text(obj, body, "fit_score")
    vc = _number_from_json_or_text(obj, body, "visual_consistency")
    orig = _number_from_json_or_text(obj, body, "originality")
    subj = obj.get("subject_code") if isinstance(obj, dict) else None
    if subj is None:
        mm = _SUBJECT_RE.search(body)
        subj = mm.group(1) if mm else None
    if fit is None or vc is None or orig is None:
        return None
    return {
        "fit": fit,
        "vc": vc,
        "orig": orig,
        "subject_code": subj if isinstance(subj, str) else None,
    }


def _is_single_theme_label(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    return all(_finite_float(v) is not None for v in value)


def _batchify(name: str, value: Any, n: int, default: Any = None) -> List[Any]:
    if value is None:
        return [default] * n
    if name == "theme_score_labels" and _is_single_theme_label(value):
        return [value] * n
    if isinstance(value, (list, tuple)) and len(value) == n:
        return list(value)
    return [value] * n


def _normalize_agent(value: Any, has_theme_labels: bool) -> str:
    value = _safe_str(value)
    if value == "type":
        return TYPE_AGENT
    if value in ALL_AGENTS:
        return value
    # Theme-only rows omit the explicit agent column.
    if has_theme_labels:
        return "theme_style"
    return "theme_style"


def _normalize_weight_vector(weights: Any) -> Optional[Dict[str, float]]:
    weights = _json_load_maybe(weights)
    if not isinstance(weights, dict):
        return None
    out: Dict[str, float] = {}
    for key in WEIGHT_VECTOR_KEYS:
        value = _finite_float(weights.get(key))
        if value is None or value < 0:
            return None
        out[key] = value
    total = sum(out.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in out.items()}


def _type_category_from_json(obj: Optional[dict]) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    photo_type = obj.get("photo_type")
    if isinstance(photo_type, dict):
        value = _safe_str(photo_type.get("weight_category"))
        if value:
            return value
    return _safe_str(obj.get("weight_category"))


def _weights_from_json(obj: Optional[dict]) -> Optional[Dict[str, float]]:
    if not isinstance(obj, dict):
        return None
    return _normalize_weight_vector(obj.get("weights_used"))


def _semantic_from_json(obj: Optional[dict], agent: str) -> Any:
    if not isinstance(obj, dict):
        return None
    if agent == TYPE_AGENT:
        return _type_category_from_json(obj)
    if agent == "theme_style":
        return _safe_str(obj.get("subject_code"))
    if agent == "composition":
        return _safe_str(obj.get("composition_form"))
    if agent == "structure":
        return _safe_str(obj.get("completeness_layer_used"))
    if agent == "tone":
        return _safe_str(obj.get("tone_eval_mode"))
    if agent == "emotion":
        return obj.get("emotion_category")
    return None


def _make_bundles(n: int, theme_score_labels: Any = None, **kwargs) -> List[Dict[str, Any]]:
    theme_batch = _batchify("theme_score_labels", theme_score_labels, n, None)
    col_names = (
        "aesthetics_dimension",
        "gt_score",
        "gt_score_key",
        "dimension_score_labels",
        "gt_full_json",
        "weight_category",
        "weights_used",
        "gt_subject_code",
        "composition_form",
        "completeness_layer_used",
        "tone_eval_mode",
        "emotion_category",
        "emotion_label",
    )
    cols = {name: _batchify(name, kwargs.get(name), n, None) for name in col_names}

    bundles: List[Dict[str, Any]] = []
    for i in range(n):
        theme_gt = _extract_theme_gt(theme_batch[i])
        agent = _normalize_agent(cols["aesthetics_dimension"][i], theme_gt is not None)

        gt_full = _json_load_maybe(cols["gt_full_json"][i])
        if not isinstance(gt_full, dict):
            gt_full = {}

        dim_labels = _json_load_maybe(cols["dimension_score_labels"][i])
        if not isinstance(dim_labels, dict):
            dim_labels = {}
        if agent == "theme_style" and theme_gt is not None and not dim_labels:
            dim_labels = theme_gt

        gt_score = _finite_float(cols["gt_score"][i])
        score_key = _safe_str(cols["gt_score_key"][i]) or DEFAULT_SCORE_KEY.get(agent)
        if gt_score is None and score_key:
            gt_score = _finite_float(gt_full.get(score_key))
        if gt_score is None and agent == "theme_style":
            gt_score = _theme_score_from_labels(dim_labels)
        if gt_score is None and agent in SCORE_AGENTS:
            gt_score = _weighted_sub_score(agent, dim_labels, cols["weights_used"][i] or gt_full.get("weights_used"))

        weights_used = _normalize_weight_vector(cols["weights_used"][i])
        if weights_used is None:
            weights_used = _normalize_weight_vector(gt_full.get("weights_used"))

        weight_category = _safe_str(cols["weight_category"][i])
        if weight_category is None:
            weight_category = _type_category_from_json(gt_full)

        semantic_gt = None
        if agent == TYPE_AGENT:
            semantic_gt = weight_category
        elif agent == "theme_style":
            semantic_gt = _safe_str(cols["gt_subject_code"][i]) or _safe_str(gt_full.get("subject_code"))
        elif agent == "composition":
            semantic_gt = _safe_str(cols["composition_form"][i]) or _safe_str(gt_full.get("composition_form"))
        elif agent == "structure":
            semantic_gt = _safe_str(cols["completeness_layer_used"][i]) or _safe_str(gt_full.get("completeness_layer_used"))
        elif agent == "tone":
            semantic_gt = _safe_str(cols["tone_eval_mode"][i]) or _safe_str(gt_full.get("tone_eval_mode"))
        elif agent == "emotion":
            semantic_gt = _safe_str(cols["emotion_category"][i]) or _safe_str(gt_full.get("emotion_category"))

        bundles.append({
            "agent": agent,
            "gt_score": gt_score,
            "score_key": score_key,
            "dim_labels": dim_labels,
            "gt_full": gt_full,
            "weights_used": weights_used,
            "weight_category": weight_category,
            "semantic_gt": semantic_gt,
            "emotion_label": _json_load_maybe(cols["emotion_label"][i]) or gt_full.get("emotion_label"),
        })
    return bundles


def _parse_score_completion(text: str, bundle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    agent = bundle["agent"]
    obj = _extract_json_obj(text)
    body = _clean_completion(text)
    if not isinstance(obj, dict):
        return None

    score_key = bundle.get("score_key") or DEFAULT_SCORE_KEY.get(agent)
    score = _number_from_json_or_text(obj, body, score_key) if score_key else None

    sub_scores: Dict[str, float] = {}
    for key in SUB_SCORE_KEYS.get(agent, ()):
        value = _number_from_json_or_text(obj, body, key)
        if value is not None:
            sub_scores[key] = value

    if score is None:
        score = _weighted_sub_score(agent, sub_scores, obj.get("weights_used") or bundle.get("weights_used"))

    return {
        "json": obj,
        "score": score,
        "sub_scores": sub_scores,
        "semantic": _semantic_from_json(obj, agent),
    }


# Scalar reward primitives.

def _gaussian_reward(pred: float, gt: float, alpha: float, beta: float) -> float:
    diff = pred - gt
    scale = alpha if diff > 0 else beta
    weighted_err = abs(diff) * scale
    return float(math.exp(-(weighted_err ** 2) / (2 * max(SIGMA, EPS) ** 2)) * R1_W_GAUSS)


def _score_r1_reward(pred: Dict[str, Any], bundle: Dict[str, Any]) -> float:
    agent = bundle["agent"]
    gt_score = _finite_float(bundle.get("gt_score"))
    pred_score = _finite_float(pred.get("score"))
    if gt_score is None or pred_score is None:
        return PARSE_FAIL_PENALTY

    total_r = _gaussian_reward(pred_score, gt_score, PRIMARY_ALPHA[agent], PRIMARY_BETA[agent])

    sub_r = _subscore_r1_reward(agent, pred.get("sub_scores") or {}, bundle)
    if sub_r is None:
        return float(total_r)
    total_w = max(GENERIC_W_TOTAL + GENERIC_W_SUB, EPS)
    return float((GENERIC_W_TOTAL * total_r + GENERIC_W_SUB * sub_r) / total_w)


def _subscore_weights(agent: str, bundle: Dict[str, Any]) -> Dict[str, float]:
    keys = SUB_SCORE_KEYS.get(agent, ())
    if not keys:
        return {}
    if agent == "theme_style":
        weights = dict(SUB_SCORE_WEIGHTS["theme_style"])
    elif agent == "composition" and isinstance(bundle.get("gt_full"), dict):
        raw_weights = bundle["gt_full"].get("weights_used") or bundle.get("weights_used")
        raw_weights = _json_load_maybe(raw_weights)
        weights = {}
        if isinstance(raw_weights, dict):
            for weight_key, score_key in COMPOSITION_WEIGHT_TO_SCORE.items():
                w = _finite_float(raw_weights.get(weight_key))
                if w is not None:
                    weights[score_key] = w
        if not weights:
            weights = {key: 1.0 for key in keys}
    else:
        weights = {key: 1.0 for key in keys}
    total = sum(max(0.0, float(w)) for w in weights.values())
    if total <= 0:
        return {key: 1.0 / len(keys) for key in keys}
    return {key: max(0.0, float(weights.get(key, 0.0))) / total for key in keys}


def _theme_alpha_beta_for_key(key: str) -> Tuple[float, float]:
    if key == "fit_score":
        return ALPHA["fit"], BETA["fit"]
    if key == "visual_consistency":
        return ALPHA["vc"], BETA["vc"]
    if key == "originality":
        return ALPHA["orig"], BETA["orig"]
    return 1.0, 1.0


def _subscore_r1_reward(agent: str, pred_sub: Dict[str, float], bundle: Dict[str, Any]) -> Optional[float]:
    gt_sub = bundle.get("dim_labels")
    if not isinstance(gt_sub, dict):
        return None
    keys = SUB_SCORE_KEYS.get(agent, ())
    if not keys:
        return None
    weights = _subscore_weights(agent, bundle)
    acc = 0.0
    used = False
    for key in keys:
        w = float(weights.get(key, 0.0))
        if w <= 0:
            continue
        gt = _finite_float(gt_sub.get(key))
        pred = _finite_float(pred_sub.get(key))
        if gt is None:
            continue
        used = True
        if pred is None:
            r = 0.0
        else:
            if agent == "theme_style":
                alpha, beta = _theme_alpha_beta_for_key(key)
            else:
                alpha, beta = 1.0, 1.0
            r = _gaussian_reward(pred, gt, alpha, beta)
        acc += w * r
    return float(acc) if used else None


def _type_r1_reward(obj: Optional[dict], bundle: Dict[str, Any]) -> float:
    if not isinstance(obj, dict):
        return PARSE_FAIL_PENALTY
    gt_cat = _safe_str(bundle.get("weight_category"))
    pred_cat = _type_category_from_json(obj)
    if not gt_cat:
        cat_r = NEUTRAL_REWARD
    else:
        cat_r = 1.0 if pred_cat == gt_cat else 0.0

    gt_w = bundle.get("weights_used")
    pred_w = _weights_from_json(obj)
    if isinstance(gt_w, dict) and isinstance(pred_w, dict):
        l1 = sum(abs(float(pred_w.get(k, 0.0)) - float(gt_w.get(k, 0.0))) for k in WEIGHT_VECTOR_KEYS)
        weight_r = max(0.0, 1.0 - l1 / 2.0)
    else:
        weight_r = NEUTRAL_REWARD

    total_w = max(TYPE_W_CATEGORY + TYPE_W_WEIGHTS, EPS)
    return float((TYPE_W_CATEGORY * cat_r + TYPE_W_WEIGHTS * weight_r) / total_w)


# R1: absolute score and category agreement.

class PeerRelativeDirORM(ORM):
    """R1: generic absolute reward.

    - type_agent: weight_category hit + weights_used vector closeness.
    - score agents: primary score Gaussian + sub-score Gaussian.
    """

    def __call__(self, completions: List[str], theme_score_labels: List[Any] = None, **kwargs) -> List[float]:
        n = len(completions)
        bundles = _make_bundles(n, theme_score_labels, **kwargs)
        rewards: List[float] = []
        for completion, bundle in zip(completions, bundles):
            agent = bundle["agent"]
            if agent == TYPE_AGENT:
                rewards.append(_type_r1_reward(_extract_json_obj(completion), bundle))
                continue
            pred = _parse_score_completion(completion, bundle)
            if pred is None:
                rewards.append(PARSE_FAIL_PENALTY)
            else:
                rewards.append(_score_r1_reward(pred, bundle))
        _cache_reward_component("r1", bundles, rewards, reset=True)
        return rewards


orms["peer_relative_dir"] = PeerRelativeDirORM


# R2: within-agent pairwise ranking.

def _bpr_rank_pool_1d(
    pool_pred: torch.Tensor,
    pool_gt: torch.Tensor,
    pool_agent: torch.Tensor,
    pool_valid: torch.Tensor,
    local_slice: slice,
) -> torch.Tensor:
    n_pool = pool_pred.shape[0]
    # Centered BPR lives in [-1, 1], so its no-information value must be 0.
    # A 0.5 neutral would incorrectly record no valid pair as a
    # positive ranking reward and inflates the aggregate R2 metric.
    r_all = torch.full((n_pool,), BPR_NEUTRAL_REWARD, dtype=torch.float32)
    if n_pool < 2:
        return r_all[local_slice]

    for agent in SCORE_AGENTS:
        aid = AGENT_ID[agent]
        idxs = torch.nonzero((pool_valid > 0.5) & (pool_agent == aid), as_tuple=False).flatten()
        if idxs.numel() < 2:
            continue

        p = pool_pred[idxs]
        g = pool_gt[idxs]
        gi = g.unsqueeze(1)
        gj = g.unsqueeze(0)
        pi = p.unsqueeze(1)
        pj = p.unsqueeze(0)
        gt_diff = gi - gj
        pred_diff = pi - pj
        valid_pair = (gt_diff.abs() >= RANK_MIN_GAP).float()

        signed_pred = torch.sign(gt_diff) * pred_diff
        pair_correct = torch.sigmoid((signed_pred - RANK_MARGIN) / max(RANK_TEMP, EPS))

        alpha = PRIMARY_ALPHA[agent]
        beta = PRIMARY_BETA[agent]
        d_signed_i = pi - gi
        d_signed_j = pj - gj
        scale_i = torch.where(d_signed_i > 0, torch.tensor(alpha), torch.tensor(beta)).to(p.dtype)
        scale_j = torch.where(d_signed_j > 0, torch.tensor(alpha), torch.tensor(beta)).to(p.dtype)
        werr_i = d_signed_i.abs() * scale_i
        werr_j = d_signed_j.abs() * scale_j
        blame_i = werr_i / (werr_i + werr_j + EPS)

        if BPR_DIRECT:
            r_pair = pair_correct
            if BPR_CENTERED:
                r_pair = 2.0 * r_pair - 1.0
        else:
            r_pair = pair_correct * blame_i + torch.clamp(pair_correct, min=0.5) * (1 - blame_i)

        weight = gt_diff.abs() * valid_pair
        denom = weight.sum(dim=1).clamp(min=EPS)
        r_dim = (weight * r_pair).sum(dim=1) / denom
        no_pair = valid_pair.sum(dim=1) == 0
        r_dim = torch.where(no_pair, torch.full_like(r_dim, BPR_NEUTRAL_REWARD), r_dim)
        r_all[idxs] = r_dim

    return r_all[local_slice]


class BPRRankCrossDirORM(ORM):
    """R2: cross-rank BPR ranking reward.

    Ranking is computed only within the same numeric agent. type_agent rows are
    not comparable by scalar score and therefore receive neutral R2.
    """

    def __call__(self, completions: List[str], theme_score_labels: List[Any] = None, **kwargs) -> List[float]:
        n = len(completions)
        bundles = _make_bundles(n, theme_score_labels, **kwargs)

        pred_values: List[float] = []
        gt_values: List[float] = []
        agent_ids: List[int] = []
        valid_flags: List[float] = []
        parse_fail_flags: List[bool] = []

        for completion, bundle in zip(completions, bundles):
            agent = bundle["agent"]
            agent_ids.append(AGENT_ID.get(agent, AGENT_ID["theme_style"]))
            if agent == TYPE_AGENT:
                pred_values.append(0.0)
                gt_values.append(0.0)
                valid_flags.append(0.0)
                parse_fail_flags.append(False)
                continue
            pred = _parse_score_completion(completion, bundle)
            gt = _finite_float(bundle.get("gt_score"))
            pv = _finite_float(pred.get("score")) if pred is not None else None
            if pv is None or gt is None:
                pred_values.append(0.0)
                gt_values.append(0.0)
                valid_flags.append(0.0)
                parse_fail_flags.append(True)
            else:
                pred_values.append(float(pv))
                gt_values.append(float(gt))
                valid_flags.append(1.0)
                parse_fail_flags.append(False)

        pred_arr = torch.tensor(pred_values, dtype=torch.float32)
        gt_arr = torch.tensor(gt_values, dtype=torch.float32)
        agent_arr = torch.tensor(agent_ids, dtype=torch.long)
        valid_arr = torch.tensor(valid_flags, dtype=torch.float32)

        try:
            import torch.distributed as dist
            do_gather = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        except Exception:
            do_gather = False

        if do_gather:
            world = dist.get_world_size()
            my_rank = dist.get_rank()
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else pred_arr.device

            def gather_tensor(t: torch.Tensor) -> torch.Tensor:
                tg = t.to(device).contiguous()
                out = [torch.zeros_like(tg) for _ in range(world)]
                dist.all_gather(out, tg)
                return torch.cat(out, dim=0).cpu()

            pool_pred = gather_tensor(pred_arr)
            pool_gt = gather_tensor(gt_arr)
            pool_agent = gather_tensor(agent_arr)
            pool_valid = gather_tensor(valid_arr)
            local_slice = slice(my_rank * n, (my_rank + 1) * n)
        else:
            pool_pred, pool_gt, pool_agent, pool_valid = pred_arr, gt_arr, agent_arr, valid_arr
            local_slice = slice(0, n)

        r_local = _bpr_rank_pool_1d(pool_pred, pool_gt, pool_agent, pool_valid, local_slice)

        out: List[float] = []
        for i, r in enumerate(r_local):
            out.append(PARSE_FAIL_PENALTY if parse_fail_flags[i] else float(r.item()))
        _cache_reward_component("r2", bundles, out)
        return out


orms["bpr_rank_cross_dir"] = BPRRankCrossDirORM


# R3: semantic-label agreement.

def _major_subject(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    return code.split(".")[0] if "." in code else code


def _token_set(value: Any) -> set[str]:
    loaded = _json_load_maybe(value)
    if loaded is not None:
        value = loaded
    if value is None:
        return set()
    if isinstance(value, list):
        raw = " ".join(str(x) for x in value)
    else:
        raw = str(value)
    toks = re.split(r"[/,，、\s]+", raw)
    return {t.strip() for t in toks if t.strip()}


def _semantic_hit_reward(bundle: Dict[str, Any], obj: Optional[dict]) -> float:
    agent = bundle["agent"]
    if agent == "image_quality":
        return NEUTRAL_REWARD
    if not isinstance(obj, dict):
        return PARSE_FAIL_PENALTY

    gt = bundle.get("semantic_gt")
    pred = _semantic_from_json(obj, agent)
    gt_s = _safe_str(gt)
    pred_s = _safe_str(pred)

    if agent == "theme_style":
        if not gt_s or not pred_s:
            return NEUTRAL_REWARD
        if gt_s == pred_s:
            return R3_HIT_FULL
        return R3_HIT_MAJOR if _major_subject(gt_s) == _major_subject(pred_s) else R3_MISS

    if agent == "emotion":
        gt_set = _token_set(gt)
        pred_set = _token_set(pred)
        if not gt_set or not pred_set:
            return NEUTRAL_REWARD
        if gt_set == pred_set or gt_s == pred_s:
            return R3_HIT_FULL
        return R3_HIT_MAJOR if gt_set & pred_set else R3_MISS

    if not gt_s or not pred_s:
        return NEUTRAL_REWARD
    return R3_HIT_FULL if gt_s == pred_s else R3_MISS


class SubjectCodeORM(ORM):
    """R3: semantic hit reward.

    Name kept as `subject_code_hit` for CLI compatibility, but the reward now
    maps by agent:
      - type_agent: weight_category
      - theme_style: subject_code with major-code partial credit
      - composition: composition_form
      - structure: completeness_layer_used
      - tone: tone_eval_mode
      - emotion: emotion_category with overlap partial credit
      - image_quality: neutral
    """

    def __call__(self, completions: List[str], theme_score_labels: List[Any] = None, **kwargs) -> List[float]:
        bundles = _make_bundles(len(completions), theme_score_labels, **kwargs)
        rewards = [_semantic_hit_reward(bundle, _extract_json_obj(completion))
                   for completion, bundle in zip(completions, bundles)]
        _cache_reward_component("r3", bundles, rewards)
        return rewards


orms["subject_code_hit"] = SubjectCodeORM


# R4: structured-output compliance.

def _format_base_reward(text: str, bundle: Dict[str, Any]) -> float:
    obj = _extract_json_obj(text)
    if not isinstance(obj, dict):
        return -0.6

    agent = bundle["agent"]
    if agent == TYPE_AGENT:
        has_cat = _type_category_from_json(obj) is not None
        has_weights = _weights_from_json(obj) is not None
        if has_cat and has_weights:
            return 1.0
        if has_cat or has_weights:
            return 0.2
        return -0.2

    score_key = bundle.get("score_key") or DEFAULT_SCORE_KEY.get(agent)
    has_primary = _number_from_json_or_text(obj, text, score_key) is not None if score_key else False
    sub_keys = SUB_SCORE_KEYS.get(agent, ())
    n_sub = sum(1 for key in sub_keys if _number_from_json_or_text(obj, text, key) is not None)
    if has_primary and n_sub == len(sub_keys):
        return 1.0
    if has_primary:
        return 0.6
    if n_sub >= max(1, len(sub_keys) // 2):
        return 0.2
    return -0.2


def _format_len_reward(text: str, bundle: Dict[str, Any]) -> float:
    if not text:
        return -1.0
    body = _clean_completion(text)
    char_len = len(body)
    soft = max(1, int(FORMAT_SOFT_CHARS))
    hard = max(soft + 1, int(FORMAT_MAX_CHARS))
    if char_len <= soft:
        length_penalty = 0.0
    elif char_len <= hard:
        length_penalty = 0.4 * (char_len - soft) / max(hard - soft, 1)
    else:
        length_penalty = 0.4 + 0.6 * min((char_len - hard) / max(hard, 1), 1.0)
    return float(max(-1.0, min(1.0, _format_base_reward(text, bundle) - length_penalty)))


class FormatLenORM(ORM):
    """R4: JSON schema completeness + length guardrail."""

    def __call__(self, completions: List[str], theme_score_labels: List[Any] = None, **kwargs) -> List[float]:
        bundles = _make_bundles(len(completions), theme_score_labels, **kwargs)
        rewards = [_format_len_reward(completion, bundle)
                   for completion, bundle in zip(completions, bundles)]
        _cache_reward_component("r4", bundles, rewards)
        return rewards


orms["format_len"] = FormatLenORM


# Zero-weight, per-agent diagnostic rewards.

def _cache_reward_component(
    name: str,
    bundles: List[Dict[str, Any]],
    rewards: List[float],
    reset: bool = False,
) -> None:
    if reset:
        _REWARD_DIAGNOSTIC_CACHE.clear()
        _REWARD_DIAGNOSTIC_CACHE["agents"] = [bundle["agent"] for bundle in bundles]
        _REWARD_DIAGNOSTIC_CACHE["components"] = {}
    agents = _REWARD_DIAGNOSTIC_CACHE.get("agents")
    if agents is None or len(agents) != len(rewards):
        raise RuntimeError("Per-agent reward diagnostics cache is out of sync with the current batch.")
    _REWARD_DIAGNOSTIC_CACHE["components"][name] = list(rewards)


def _dimension_diagnostic_rewards(
    target_agent: str,
    component_name: str = "total",
) -> List[Optional[float]]:
    agents = _REWARD_DIAGNOSTIC_CACHE.get("agents")
    components = _REWARD_DIAGNOSTIC_CACHE.get("components") or {}
    missing = [name for name in DIAGNOSTIC_REWARD_WEIGHTS if name not in components]
    if agents is None or missing:
        raise RuntimeError(
            "Per-agent diagnostics must run after the four primary rewards; "
            f"missing components: {missing}"
        )
    rewards: List[Optional[float]] = []
    for index, agent in enumerate(agents):
        if agent != target_agent:
            rewards.append(None)
            continue
        if component_name == "total":
            value = sum(
                DIAGNOSTIC_REWARD_WEIGHTS[name] * components[name][index]
                for name in DIAGNOSTIC_REWARD_WEIGHTS
            )
        else:
            value = components[component_name][index]
        rewards.append(float(value))
    return rewards


class _DimensionRewardORM(ORM):
    target_agent = ""
    component_name = "total"

    def __call__(self, completions: List[str], **kwargs) -> List[Optional[float]]:
        return _dimension_diagnostic_rewards(self.target_agent, self.component_name)


_DIAGNOSTIC_CLASS_PREFIX = {
    "type_agent": "TypeAgent",
    "theme_style": "ThemeStyle",
    "composition": "Composition",
    "structure": "Structure",
    "image_quality": "ImageQuality",
    "tone": "Tone",
    "emotion": "Emotion",
}
for _agent, _class_prefix in _DIAGNOSTIC_CLASS_PREFIX.items():
    for _component in ("r1", "r2", "r3", "r4", "total"):
        _class_name = f"{_class_prefix}{_component.upper() if _component != 'total' else 'Total'}RewardORM"
        _diagnostic_class = type(
            _class_name,
            (_DimensionRewardORM,),
            {"target_agent": _agent, "component_name": _component},
        )
        orms[f"dim_{_agent}_{_component}"] = _diagnostic_class


if int(os.getenv("LOCAL_RANK", "0")) == 0:
    print("[grpo-reward-plugin] registered 7-agent compatible rewards: "
          "peer_relative_dir, bpr_rank_cross_dir, subject_code_hit, format_len", flush=True)
    print("[grpo-reward-plugin] registered 35 zero-weight per-agent diagnostics: "
          "R1/R2/R3/R4/Total for type_agent, theme_style, composition, "
          "structure, image_quality, tone, emotion", flush=True)
    print(f"[grpo-reward-plugin] R2: RANK_MIN_GAP={RANK_MIN_GAP} MARGIN={RANK_MARGIN} "
          f"TEMP={RANK_TEMP} BPR_DIRECT/CENTERED={BPR_DIRECT}/{BPR_CENTERED} "
          f"NEUTRAL={BPR_NEUTRAL_REWARD}", flush=True)
    print(f"[grpo-reward-plugin] R1: SIGMA={SIGMA} R1_W_GAUSS={R1_W_GAUSS} "
          f"GENERIC_W_TOTAL/SUB={GENERIC_W_TOTAL}/{GENERIC_W_SUB} "
          f"TYPE_W_CATEGORY/WEIGHTS={TYPE_W_CATEGORY}/{TYPE_W_WEIGHTS}", flush=True)
    print(f"[grpo-reward-plugin] theme sub ALPHA={ALPHA} BETA={BETA} "
          f"R3_FULL/MAJOR/MISS={R3_HIT_FULL}/{R3_HIT_MAJOR}/{R3_MISS} "
          f"PARSE_FAIL={PARSE_FAIL_PENALTY} FORMAT_SOFT/MAX={FORMAT_SOFT_CHARS}/{FORMAT_MAX_CHARS}", flush=True)
    print(f"[grpo-reward-plugin] diagnostic total weights={DIAGNOSTIC_REWARD_WEIGHTS}", flush=True)
