#!/usr/bin/env python3
"""PortraitAes SFT launcher with PLCC/SRCC/MAE evaluation metrics.

The launcher registers :class:`BaselineSeq2SeqTrainer` for causal language
modeling and otherwise delegates argument parsing and training to ms-swift.
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

from portraitaes.sft.jsonl_loader import patch_swift_jsonl_loader

if os.environ.get("USE_JSONL_INMEMORY_LOADER", "1") != "0":
    patch_swift_jsonl_loader()
else:
    print("[jsonl_inmemory_loader] disabled by USE_JSONL_INMEMORY_LOADER=0")

from swift.trainers.trainer_factory import TrainerFactory

TrainerFactory.TRAINER_MAPPING["causal_lm"] = "portraitaes.sft.trainer.BaselineSeq2SeqTrainer"

from swift.pipelines import sft_main


def main() -> None:
    sft_main()


if __name__ == "__main__":
    main()
