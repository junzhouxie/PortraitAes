#!/usr/bin/env python3
"""PortraitAes GRPO training launcher.

It installs the optional JSONL adapter before delegating to the ms-swift RLHF
entry point. The reward module is supplied through ms-swift's
``--external_plugins`` argument so that ms-swift owns reward registration.
"""
import os


def maybe_disable_hf_datasets_cache() -> None:
    if os.environ.get("DISABLE_HF_DATASETS_CACHE", "0") != "1":
        return
    try:
        from datasets import disable_caching
        disable_caching()
        print("[datasets_cache] disabled by DISABLE_HF_DATASETS_CACHE=1")
    except Exception as e:
        print(f"[datasets_cache] failed to disable cache: {e}")


maybe_disable_hf_datasets_cache()

from portraitaes.grpo.jsonl_loader import patch_swift_jsonl_loader


def patch_qwen35_causal_conv1d_tuple_return():
    """Keep Qwen3.5 linear attention compatible with causal-conv1d tuple outputs.

    QWEN35_FORCE_TORCH_CAUSAL_CONV1D=1 forces the multi-token conv path to use
    the model's PyTorch fallback instead of FLA/Triton causal_conv1d. This is
    slower, but avoids CUDA illegal memory access observed during GRPO
    PyTorch rollouts.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, torch_causal_conv1d_update
    except Exception as exc:
        print(f"[qwen35_patch] Skip causal_conv1d tuple patch: {exc}")
        return

    if getattr(Qwen3_5GatedDeltaNet, "_portraitaes_causal_conv1d_tuple_patched", False):
        return

    origin_forward = Qwen3_5GatedDeltaNet.forward

    def _forward_with_tensor_causal_conv1d(self, *args, **kwargs):
        original_fn = getattr(self, "causal_conv1d_fn", None)
        original_update = getattr(self, "causal_conv1d_update", None)
        force_torch_conv = os.environ.get("QWEN35_FORCE_TORCH_CAUSAL_CONV1D", "0") == "1"

        if original_fn is None and not force_torch_conv:
            return origin_forward(self, *args, **kwargs)

        def _tensor_only_causal_conv1d(*fn_args, **fn_kwargs):
            output = original_fn(*fn_args, **fn_kwargs)
            return output[0] if isinstance(output, tuple) else output

        if force_torch_conv:
            self.causal_conv1d_fn = None
            self.causal_conv1d_update = torch_causal_conv1d_update
        else:
            self.causal_conv1d_fn = _tensor_only_causal_conv1d
        try:
            return origin_forward(self, *args, **kwargs)
        finally:
            self.causal_conv1d_fn = original_fn
            self.causal_conv1d_update = original_update

    Qwen3_5GatedDeltaNet.forward = _forward_with_tensor_causal_conv1d
    Qwen3_5GatedDeltaNet._portraitaes_causal_conv1d_tuple_patched = True
    if os.environ.get("QWEN35_FORCE_TORCH_CAUSAL_CONV1D", "0") == "1":
        print("[qwen35_patch] Patched Qwen3.5 causal_conv1d to force torch fallback")
    else:
        print("[qwen35_patch] Patched Qwen3.5 causal_conv1d tuple output compatibility")


if os.environ.get("USE_JSONL_INMEMORY_LOADER", "1") != "0":
    patch_swift_jsonl_loader()
else:
    print("[jsonl_inmemory_loader] disabled by USE_JSONL_INMEMORY_LOADER=0")
patch_qwen35_causal_conv1d_tuple_return()

from swift.pipelines import rlhf_main

if __name__ == "__main__":
    rlhf_main()
