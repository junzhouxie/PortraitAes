#!/usr/bin/env python3
"""Sharded local inference for the merged PortraitAes model.

Each image is evaluated by the built-in router and six specialists, then
formatted with the type-specific weighted score.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from portraitaes.inference.pipeline import run_full_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PortraitAes merged-model inference.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model", required=True, help="Merged model directory or model ID.")
    parser.add_argument("--result-key", default="portraitaes_result")
    parser.add_argument("--error-key", default="inference_error")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--skip-existing-jsonl",
        action="append",
        default=[],
        help="Skip records whose first-image basename appears in these JSONL files.",
    )
    parser.add_argument(
        "--exclude-dimensions",
        default="",
        help="Comma-separated specialists to omit; remaining weights are renormalized.",
    )
    parser.add_argument("--include-raw-predictions", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _normalize_one_image(item: Any) -> str | None:
    if isinstance(item, dict):
        return item.get("path") or None
    if isinstance(item, str):
        return item or None
    return None


def get_images(record: dict[str, Any]) -> list[str]:
    images = record.get("images", record.get("image"))
    if images is None:
        return []
    if isinstance(images, (str, dict)):
        image = _normalize_one_image(images)
        return [image] if image else []
    if isinstance(images, list):
        return [image for image in map(_normalize_one_image, images) if image]
    raise TypeError(f"Unsupported image field type: {type(images)}")


def resolve_image_input(path: str) -> str:
    """Return a public URL or a validated absolute local path."""
    path = str(path)
    if path.startswith(("http://", "https://", "data:")):
        return path
    if path.startswith("file://"):
        path = path[7:]
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    return str(resolved)


def image_key(path: str) -> str:
    return os.path.basename(str(path).split("?", 1)[0]) if path else ""


def record_image_key(record: dict[str, Any]) -> str:
    images = get_images(record)
    return image_key(images[0]) if images else ""


def load_existing_image_keys(paths: list[str]) -> set[str]:
    keys: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not raw_path or not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record_image_key(record)
                if key:
                    keys.add(key)
    return keys


def load_model(model_dir: str, args: argparse.Namespace):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model_kwargs = {
        "dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForImageTextToText.from_pretrained(model_dir, **model_kwargs)
    model.eval()
    return model, processor


def generate(model, processor, messages, args) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    generation_kwargs = {
        key: value for key, value in generation_kwargs.items() if value is not None
    }
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    generated_ids = outputs[0, inputs.input_ids.shape[1]:]
    text = processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    for token in ("<|im_end|>", processor.tokenizer.eos_token or ""):
        if token and text.endswith(token):
            text = text[:-len(token)]
    return text.strip()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, num-shards)")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1")

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    print(f"[load] model={args.model}", flush=True)
    model, processor = load_model(args.model, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skip_rows = count_jsonl(output_path) if args.resume else 0
    existing_keys = load_existing_image_keys(args.skip_existing_jsonl)
    excluded = {
        value.strip()
        for value in args.exclude_dimensions.split(",")
        if value.strip()
    }
    mode = "a" if args.resume and output_path.exists() else "w"
    processed = seen = shard_rows = skipped_existing = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        mode,
        encoding="utf-8",
    ) as destination:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            seen += 1
            if (seen - 1) % args.num_shards != args.shard_id:
                continue
            shard_rows += 1
            if shard_rows <= skip_rows:
                continue
            if args.limit is not None and processed >= args.limit:
                break

            record = json.loads(line)
            key = record_image_key(record)
            if key and key in existing_keys:
                skipped_existing += 1
                continue
            try:
                images = [resolve_image_input(path) for path in get_images(record)]
                if not images:
                    raise ValueError("record has no image/images field")
                image_parts = [{"type": "image", "image": image} for image in images]
                record[args.result_key] = run_full_pipeline(
                    lambda messages: generate(model, processor, messages, args),
                    image_parts,
                    excluded_dimensions=excluded,
                    include_raw_predictions=args.include_raw_predictions,
                )
                record.pop(args.error_key, None)
            except Exception as exc:
                record[args.error_key] = f"{type(exc).__name__}: {exc}"
                print(
                    f"[error] line={line_no} {record[args.error_key]}",
                    file=sys.stderr,
                    flush=True,
                )

            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            destination.flush()
            processed += 1
            if processed % args.log_every == 0:
                print(
                    f"[progress] shard={args.shard_id}/{args.num_shards} "
                    f"processed={processed} input_line={line_no}",
                    flush=True,
                )

    print(
        f"[done] shard={args.shard_id}/{args.num_shards} processed={processed} "
        f"skipped_existing={skipped_existing} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
