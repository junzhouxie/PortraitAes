"""SFT trainer with portrait-aesthetic regression metrics.

Evaluation uses teacher-forced logits at the three score-token positions. The
expected score is computed over the tokens for integers 1--5, then aggregated
across distributed workers to report PLCC, SRCC, and MAE for each score and
their weighted total. Training loss and optimization are unchanged.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F

from swift.trainers import Seq2SeqTrainer


# Qwen3.5-VL encodes the standalone integers 1--5 as consecutive token IDs.
SCORE_TOKEN_IDS = [16, 17, 18, 19, 20]

# Match field names without a leading quote so tokenization remains stable
# when JSON fields follow different whitespace or punctuation.
DIM_PATTERNS = {
    "fit": [6078, 10095, 763],  # fit_score":
    "vc": [28263, 30919, 45522, 763],  # visual_consistency":
    "orig": [9596, 477, 763],  # originality":
}

WEIGHTS = {"fit": 0.4, "vc": 0.4, "orig": 0.2}


def _find_subseq(seq: List[int], pat: List[int]) -> int | None:
    """Return the start of the first matching subsequence, if present."""
    n, m = len(seq), len(pat)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if seq[i:i + m] == pat:
            return i
    return None


class BaselineSeq2SeqTrainer(Seq2SeqTrainer):
    """Extend ms-swift SFT with score-correlation evaluation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Buffer score pairs until the complete evaluation pass is available.
        self._eval_buffer: Dict[str, Dict[str, list]] = defaultdict(lambda: {"pred": [], "gt": []})
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            print(f"[baseline-trainer] SCORE_TOKEN_IDS={SCORE_TOKEN_IDS} (should encode as '1'..'5')",
                  flush=True)
            for dim, pat in DIM_PATTERNS.items():
                print(f"[baseline-trainer] DIM_PATTERN[{dim}] = {pat}", flush=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch,
        )
        if not self.model.training:
            try:
                self._collect_eval_pred(inputs, outputs)
            except Exception as e:
                if int(os.getenv("LOCAL_RANK", "0")) == 0:
                    print(f"[baseline-trainer] collect eval pred failed: {e}", flush=True)
        return (loss, outputs) if return_outputs else loss

    def _collect_eval_pred(self, inputs, outputs):
        input_ids = inputs.get("input_ids")
        logits = getattr(outputs, "logits", None)
        if input_ids is None or logits is None:
            return
        # Causal logits at position t predict the token at position t + 1.
        B, T = input_ids.shape
        device = logits.device
        score_id_tensor = torch.tensor(SCORE_TOKEN_IDS, dtype=torch.long, device=device)
        score_val_tensor = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float32, device=device)

        for b in range(B):
            ids_list = input_ids[b].tolist()
            for dim, pat in DIM_PATTERNS.items():
                pos = _find_subseq(ids_list, pat)
                if pos is None:
                    continue
                # Serialized targets place one whitespace token before the score.
                t_num = pos + len(pat) + 1
                if t_num >= T:
                    continue
                gt_token = ids_list[t_num]
                if gt_token not in SCORE_TOKEN_IDS:
                    continue
                gt_val = gt_token - 15
                if t_num - 1 < 0:
                    continue
                logit_5 = logits[b, t_num - 1, :].index_select(-1, score_id_tensor).float()  # (5,)
                prob = F.softmax(logit_5, dim=-1)
                pred_val = float((prob * score_val_tensor).sum().detach().cpu().item())
                self._eval_buffer[dim]["pred"].append(pred_val)
                self._eval_buffer[dim]["gt"].append(float(gt_val))

    def evaluate(self, *args, **kwargs):
        self._eval_buffer.clear()
        result = super().evaluate(*args, **kwargs)
        try:
            self._compute_and_print_metrics()
        except Exception as e:
            if int(os.getenv("LOCAL_RANK", "0")) == 0:
                print(f"[baseline-eval] compute/print metrics failed: {e}", flush=True)
        return result

    def _compute_and_print_metrics(self):
        import numpy as np

        rank = int(os.getenv("LOCAL_RANK", "0"))
        # Object gathering preserves the variable number of valid samples per rank.
        buffer_copy = {k: {"pred": list(v["pred"]), "gt": list(v["gt"])}
                       for k, v in self._eval_buffer.items()}
        merged = buffer_copy
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_world_size() > 1:
                world = dist.get_world_size()
                gathered = [None] * world
                dist.all_gather_object(gathered, buffer_copy)
                if rank != 0:
                    return
                merged = {}
                for buf in gathered:
                    if buf is None:
                        continue
                    for k, v in buf.items():
                        if k not in merged:
                            merged[k] = {"pred": [], "gt": []}
                        merged[k]["pred"].extend(v["pred"])
                        merged[k]["gt"].extend(v["gt"])
            elif rank != 0:
                return
        except Exception:
            if rank != 0:
                return

        print(f"[baseline-eval] step={self.state.global_step}:", flush=True)
        per_dim_pred: Dict[str, "np.ndarray"] = {}
        per_dim_gt: Dict[str, "np.ndarray"] = {}
        for dim in ("fit", "vc", "orig"):
            buf = merged.get(dim, {"pred": [], "gt": []})
            preds = np.array(buf["pred"], dtype=np.float64)
            gts = np.array(buf["gt"], dtype=np.float64)
            n = len(preds)
            if n < 2:
                print(f"  {dim:22s} n={n}  (skip: too few samples)", flush=True)
                continue
            if float(np.std(preds)) == 0 or float(np.std(gts)) == 0:
                print(f"  {dim:22s} n={n}  (skip: zero variance)", flush=True)
                continue
            plcc = float(np.corrcoef(preds, gts)[0, 1])
            rk_p = np.argsort(np.argsort(preds))
            rk_g = np.argsort(np.argsort(gts))
            srcc = float(np.corrcoef(rk_p, rk_g)[0, 1])
            mae = float(np.mean(np.abs(preds - gts)))
            print(f"  {dim:22s} n={n:4d}  PLCC={plcc:.4f}  SRCC={srcc:.4f}  MAE={mae:.4f}",
                  flush=True)
            per_dim_pred[dim] = preds
            per_dim_gt[dim] = gts

        # All three fields originate from the same ordered examples. Truncate to
        # the shortest buffer defensively if a malformed target omitted a field.
        if all(d in per_dim_pred for d in ("fit", "vc", "orig")):
            n_min = min(len(per_dim_pred[d]) for d in ("fit", "vc", "orig"))
            if n_min >= 2:
                pred_t = np.zeros(n_min, dtype=np.float64)
                gt_t = np.zeros(n_min, dtype=np.float64)
                for d, w in WEIGHTS.items():
                    pred_t += w * per_dim_pred[d][:n_min]
                    gt_t += w * per_dim_gt[d][:n_min]
                if float(np.std(pred_t)) > 0 and float(np.std(gt_t)) > 0:
                    plcc = float(np.corrcoef(pred_t, gt_t)[0, 1])
                    srcc = float(np.corrcoef(np.argsort(np.argsort(pred_t)),
                                              np.argsort(np.argsort(gt_t)))[0, 1])
                    mae = float(np.mean(np.abs(pred_t - gt_t)))
                    print(f"  {'total_weighted':22s} n={n_min:4d}  "
                          f"PLCC={plcc:.4f}  SRCC={srcc:.4f}  MAE={mae:.4f}", flush=True)
