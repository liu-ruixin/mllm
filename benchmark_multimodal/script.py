#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import base64
import binascii
import io
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "benchmark_multimodal/data"
DEFAULT_MMLU_OUTPUT = DATA_DIR / "mmlu.jsonl"
DEFAULT_MMMU_OUTPUT = DATA_DIR / "mmmu.jsonl"
DEFAULT_MMMU_IMAGE_DIR = DATA_DIR / "mmmu_images"
DEFAULT_EMBSPATIAL_OUTPUT = DATA_DIR / "embspatial.jsonl"
DEFAULT_EMBSPATIAL_IMAGE_DIR = DATA_DIR / "embspatial_images"
DEFAULT_ERQA_OUTPUT = DATA_DIR / "erqa.jsonl"
DEFAULT_ERQA_IMAGE_DIR = DATA_DIR / "erqa_images"
DEFAULT_REFSPATIAL_OUTPUT = DATA_DIR / "refspatial.jsonl"
DEFAULT_REFSPATIAL_IMAGE_DIR = DATA_DIR / "refspatial_images"
DEFAULT_REFSPATIAL_MASK_DIR = DATA_DIR / "refspatial_masks"
REFSPATIAL_SPLITS = ["location", "placement", "unseen"]
REFSPATIAL_MIXED_SPLITS = {
    "locplace": ["location", "placement"],
    "all": REFSPATIAL_SPLITS,
}
MMMU_SUBJECTS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]
OPTION_LETTERS = [chr(ord("A") + i) for i in range(26)]


def option_letters(count: int) -> List[str]:
    if count < 0:
        raise ValueError(f"Option count must be non-negative, got: {count}")
    if count > len(OPTION_LETTERS):
        raise ValueError(f"Too many options: {count}; max supported is {len(OPTION_LETTERS)}")
    return OPTION_LETTERS[:count]


def sanitize_id_part(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def normalize_answer(answer: Any, valid_letters: List[str] | None = None) -> str:
    """Convert answer formats into an option letter."""
    letters = valid_letters or OPTION_LETTERS
    if isinstance(answer, int):
        if answer < 0 or answer >= len(letters):
            raise ValueError(f"Answer index out of range: {answer}")
        return letters[answer]

    text = str(answer).strip()
    if text.isdigit():
        idx = int(text)
        if idx < 0 or idx >= len(letters):
            raise ValueError(f"Answer index out of range: {answer}")
        return letters[idx]

    letter_pattern = "".join(re.escape(letter) for letter in letters)
    match = re.search(rf"\b([{letter_pattern}])\b", text.upper())
    letter = match.group(1) if match else text[:1].upper()
    if letter not in letters:
        raise ValueError(f"Unsupported answer value: {answer!r}")
    return letter


def resolve_output_path(path: str | Path) -> Path:
    output_path = Path(path).expanduser()
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    return output_path


def default_output_for_benchmark(benchmark: str, split: str) -> Path:
    defaults = {
        "mmlu": DEFAULT_MMLU_OUTPUT,
        "mmmu": DEFAULT_MMMU_OUTPUT,
        "embspatial": DEFAULT_EMBSPATIAL_OUTPUT,
        "erqa": DEFAULT_ERQA_OUTPUT,
    }
    if benchmark == "refspatial":
        split_name = "locplace" if split in {"location+placement", "placement+location"} else split
        return DATA_DIR / f"refspatial_{sanitize_id_part(split_name)}.jsonl"
    return defaults[benchmark]


def parse_choice_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(choice) for choice in value]
    if isinstance(value, tuple):
        return [str(choice) for choice in value]

    text = str(value).strip()
    if not text or text == "[]":
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, (list, tuple)):
        return [str(choice) for choice in parsed]

    # Fallback for simple newline-separated or semicolon-separated option blobs.
    parts = re.split(r"\n+|;\s*", text)
    return [part.strip() for part in parts if part.strip()]


def extract_choices(record: Dict[str, Any]) -> List[str]:
    """Support both `choices: [...]` and separate A/B/C/D columns."""
    choices = record.get("choices")
    if choices is not None:
        choices = parse_choice_list(choices)
        if len(choices) < 4:
            raise ValueError(f"MMLU sample needs at least 4 choices, got: {choices!r}")
        return [str(choice) for choice in choices[:4]]

    values = []
    for key in OPTION_LETTERS:
        if key not in record:
            break
        values.append(str(record[key]))
    if len(values) < 4:
        raise ValueError("Sample missing `choices` and at least A/B/C/D choice columns")
    return values


def build_mmlu_prompt(question: str, choices: List[str]) -> str:
    letters = option_letters(len(choices))
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(letters, choices)
    )
    return (
        "Answer the following multiple-choice question. "
        "Only output the letter of the correct option.\n\n"
        f"Question: {question}\n\n"
        f"{options}\n\n"
        "Answer:"
    )


def convert_mmlu_record(
    record: Dict[str, Any],
    *,
    dataset_name: str,
    subject: str,
    split: str,
    index: int,
) -> Dict[str, Any]:
    question = str(record.get("question", "")).strip()
    if not question:
        raise ValueError("Sample missing non-empty `question`")

    choices = extract_choices(record)
    letters = option_letters(len(choices))
    answer = normalize_answer(record.get("answer"), letters)
    resolved_subject = record.get("subject") or subject
    sample_id = (
        f"mmlu_{sanitize_id_part(resolved_subject)}_"
        f"{sanitize_id_part(split)}_{index:06d}"
    )

    return {
        "id": sample_id,
        "task": "mmlu",
        "subject": resolved_subject,
        "answer_type": "option",
        "answer": answer,
        "messages": [
            {
                "role": "user",
                "content": build_mmlu_prompt(question, choices),
            }
        ],
        "meta": {
            "dataset": dataset_name,
            "split": split,
            "input_modality": "text",
            "source_index": index,
            "question": question,
            "choices": {
                letter: choice for letter, choice in zip(letters, choices)
            },
        },
    }


def convert_record(
    record: Dict[str, Any],
    *,
    dataset_name: str,
    subject: str,
    split: str,
    index: int,
) -> Dict[str, Any]:
    """Backward-compatible alias for MMLU conversion."""
    return convert_mmlu_record(
        record,
        dataset_name=dataset_name,
        subject=subject,
        split=split,
        index=index,
    )


def iter_hf_mmlu(
    *,
    dataset_name: str,
    subjects: List[str],
    split: str,
) -> Iterable[tuple[str, int, Dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install it with: pip install datasets"
        ) from exc

    for subject in subjects:
        dataset = load_dataset(dataset_name, subject, split=split)
        for index, record in enumerate(dataset):
            yield subject, index, dict(record)


def extract_mmmu_choices(record: Dict[str, Any]) -> List[str]:
    for key in ("options", "choices"):
        choices = parse_choice_list(record.get(key))
        if choices:
            return choices
    return []


def is_mmmu_multiple_choice(record: Dict[str, Any], choices: List[str]) -> bool:
    question_type = str(record.get("question_type", "")).lower()
    if "multiple" in question_type or "choice" in question_type:
        return True
    letters = option_letters(len(choices))
    return bool(choices) and str(record.get("answer", "")).strip()[:1].upper() in letters


def normalize_mmmu_answer(
    answer: Any,
    *,
    is_multiple_choice: bool,
    choices: List[str],
) -> str:
    if is_multiple_choice:
        letters = option_letters(len(choices))
        try:
            return normalize_answer(answer, letters)
        except ValueError:
            answer_text = str(answer).strip().lower()
            for letter, choice in zip(letters, choices):
                if answer_text == str(choice).strip().lower():
                    return letter
            raise
    return "" if answer is None else str(answer).strip()


def build_mmmu_prompt(
    question: str,
    choices: List[str],
    *,
    is_multiple_choice: bool,
) -> str:
    if is_multiple_choice:
        letters = option_letters(len(choices))
        letters_text = ", ".join(letters)
        options = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(letters, choices)
        )
        return (
            "Answer the following multimodal multiple-choice question. "
            "Solve the problem using the image when needed. "
            f"The final line must be exactly `Final answer: X`, where X is one of {letters_text}.\n\n"
            f"Question: {question}\n\n"
            f"{options}\n\n"
            "Final answer:"
        )

    return (
        "Answer the following multimodal question briefly and directly.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def iter_image_values(record: Dict[str, Any]) -> Iterable[tuple[str, Any]]:
    yielded_image_list = False
    images = record.get("images")
    if isinstance(images, list):
        for idx, image in enumerate(images, start=1):
            if image is not None:
                yielded_image_list = True
                yield f"image_{idx}", image

    images_base64 = record.get("images_base64") if not yielded_image_list else None
    if isinstance(images_base64, list):
        for idx, image in enumerate(images_base64, start=1):
            if image:
                yield f"image_base64_{idx}", image

    if record.get("image") is not None:
        yield "image", record["image"]
    elif record.get("image_base64"):
        yield "image_base64", record["image_base64"]

    for key in sorted(record):
        if key in {"image", "images", "image_base64", "images_base64"}:
            continue
        if not key.startswith("image"):
            continue
        value = record.get(key)
        if value is not None:
            yield key, value


def save_pil_image(image: Any, output_path: Path) -> Path:
    """Materialize and save a PIL image without relying on lazy dataset handles."""
    image.load()
    image = image.copy()

    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(output_path, format="JPEG", quality=95)
        return output_path

    if image.mode not in {"RGB", "RGBA", "L"}:
        image = image.convert("RGB")
    image.save(output_path, format="PNG", compress_level=1)
    return output_path


def save_image_value(image: Any, output_path: Path) -> Path | None:
    """Save one benchmark image value to disk and return the path if successful."""
    if image is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image export: pip install pillow") from exc

    if isinstance(image, Image.Image):
        try:
            return save_pil_image(image, output_path)
        except (OSError, ValueError):
            fallback_path = output_path.with_suffix(".jpg")
            return save_pil_image(image, fallback_path)

    if isinstance(image, (str, Path)):
        src = Path(image).expanduser()
        if src.exists():
            shutil.copy2(src, output_path)
            return output_path
        try:
            return save_pil_image(
                Image.open(io.BytesIO(base64.b64decode(str(image), validate=True))),
                output_path,
            )
        except (binascii.Error, OSError, ValueError):
            return None

    if isinstance(image, bytes):
        return save_pil_image(Image.open(io.BytesIO(image)), output_path)

    if isinstance(image, dict):
        if image.get("path"):
            src = Path(image["path"]).expanduser()
            if src.exists():
                shutil.copy2(src, output_path)
                return output_path
        if image.get("bytes"):
            return save_pil_image(Image.open(io.BytesIO(image["bytes"])), output_path)

    return None


def save_record_images(
    record: Dict[str, Any],
    *,
    image_dir: str | Path,
    subject: str,
    sample_id: str,
    extension: str = "png",
) -> List[str]:
    root = resolve_output_path(image_dir)
    subject_dir = root / sanitize_id_part(subject)
    paths: List[str] = []
    seen_keys = set()
    suffix = extension.lstrip(".") or "png"

    for idx, (key, image) in enumerate(iter_image_values(record), start=1):
        if key in seen_keys:
            continue
        seen_keys.add(key)
        output_path = subject_dir / f"{sample_id}_{idx:02d}.{suffix}"
        saved = save_image_value(image, output_path)
        if saved is not None:
            paths.append(str(saved.resolve()))

    return paths


def save_record_mask(
    record: Dict[str, Any],
    *,
    mask_dir: str | Path,
    split: str,
    sample_id: str,
) -> str | None:
    mask = record.get("mask")
    if mask is None:
        return None

    root = resolve_output_path(mask_dir)
    output_path = root / sanitize_id_part(split) / f"{sample_id}_mask.png"
    saved = save_image_value(mask, output_path)
    return str(saved.resolve()) if saved is not None else None


def get_image_size(path: str | Path) -> List[int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to inspect image size: pip install pillow") from exc

    with Image.open(path) as image:
        width, height = image.size
    return [int(width), int(height)]


def save_mmmu_images(
    record: Dict[str, Any],
    *,
    image_dir: str | Path,
    subject: str,
    sample_id: str,
) -> List[str]:
    return save_record_images(
        record,
        image_dir=image_dir,
        subject=subject,
        sample_id=sample_id,
    )


def build_multimodal_content(prompt: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_path}})
    return content


def build_placeholder_multimodal_content(
    prompt: str,
    image_paths: List[str],
) -> List[Dict[str, Any]]:
    """Insert image objects at <image 1>, <image 2>, ... placeholders."""
    if not image_paths:
        return [{"type": "text", "text": prompt}]

    content: List[Dict[str, Any]] = []
    used_indices: set[int] = set()
    cursor = 0
    pattern = re.compile(r"<image\s+(\d+)>", flags=re.IGNORECASE)

    for match in pattern.finditer(prompt):
        before = prompt[cursor : match.start()]
        if before:
            content.append({"type": "text", "text": before})

        image_index = int(match.group(1)) - 1
        if 0 <= image_index < len(image_paths):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_paths[image_index]},
                }
            )
            used_indices.add(image_index)
        else:
            content.append({"type": "text", "text": match.group(0)})
        cursor = match.end()

    tail = prompt[cursor:]
    if tail:
        content.append({"type": "text", "text": tail})

    if not used_indices:
        return build_multimodal_content(prompt, image_paths)

    for index, image_path in enumerate(image_paths):
        if index not in used_indices:
            content.append({"type": "image_url", "image_url": {"url": image_path}})

    return content


def convert_mmmu_record(
    record: Dict[str, Any],
    *,
    dataset_name: str,
    subject: str,
    split: str,
    index: int,
    image_dir: str | Path = DEFAULT_MMMU_IMAGE_DIR,
) -> Dict[str, Any]:
    question = str(record.get("question", "")).strip()
    if not question:
        raise ValueError("MMMU sample missing non-empty `question`")

    choices = extract_mmmu_choices(record)
    is_multiple_choice = is_mmmu_multiple_choice(record, choices)
    answer = normalize_mmmu_answer(
        record.get("answer"),
        is_multiple_choice=is_multiple_choice,
        choices=choices,
    )
    resolved_subject = (
        record.get("subject")
        or record.get("subfield")
        or record.get("discipline")
        or subject
    )
    source_id = record.get("id")
    sample_id = (
        f"mmmu_{sanitize_id_part(resolved_subject)}_"
        f"{sanitize_id_part(split)}_{sanitize_id_part(source_id or index)}"
    )

    image_paths = save_mmmu_images(
        record,
        image_dir=image_dir,
        subject=str(resolved_subject),
        sample_id=sample_id,
    )

    content = build_placeholder_multimodal_content(
        build_mmmu_prompt(
            question,
            choices,
            is_multiple_choice=is_multiple_choice,
        ),
        image_paths,
    )

    return {
        "id": sample_id,
        "task": "mmmu",
        "subject": resolved_subject,
        "answer_type": "option" if is_multiple_choice else "short_text",
        "answer": answer,
        "messages": [{"role": "user", "content": content}],
        "meta": {
            "dataset": dataset_name,
            "split": split,
            "input_modality": "image_text" if image_paths else "text",
            "source_index": index,
            "source_id": source_id,
            "question": question,
            "choices": {
                letter: choice
                for letter, choice in zip(option_letters(len(choices)), choices)
            },
            "question_type": record.get("question_type"),
            "subfield": record.get("subfield"),
            "topic_difficulty": record.get("topic_difficulty"),
            "image_count": len(image_paths),
            "images": image_paths,
        },
    }


def iter_hf_mmmu(
    *,
    dataset_name: str,
    subjects: List[str],
    split: str,
) -> Iterable[tuple[str, int, Dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install it with: pip install datasets"
        ) from exc

    for subject in subjects:
        dataset = load_dataset(dataset_name, subject, split=split)
        for index, record in enumerate(dataset):
            yield subject, index, dict(record)


def extract_embspatial_choices(record: Dict[str, Any]) -> List[str]:
    for key in ("answer_options", "options", "choices"):
        choices = parse_choice_list(record.get(key))
        if choices:
            return choices
    return []


def build_embspatial_prompt(question: str, choices: List[str]) -> str:
    letters = option_letters(len(choices))
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(letters, choices)
    )
    return (
        "Answer the embodied spatial multiple-choice question. "
        "Only output the letter of the correct option.\n\n"
        f"Question: {question}\n\n"
        f"{options}\n\n"
        "Answer:"
    )


def convert_embspatial_record(
    record: Dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    index: int,
    image_dir: str | Path = DEFAULT_EMBSPATIAL_IMAGE_DIR,
) -> Dict[str, Any]:
    question = str(record.get("question", "")).strip()
    if not question:
        raise ValueError("EmbSpatial sample missing non-empty `question`")

    choices = extract_embspatial_choices(record)
    if len(choices) < 2:
        raise ValueError(f"EmbSpatial sample needs at least 2 choices, got: {choices!r}")

    letters = option_letters(len(choices))
    answer = normalize_answer(record.get("answer"), letters)
    relation = record.get("relation") or "unknown"
    source_id = record.get("question_id") or record.get("id") or index
    sample_id = (
        f"embspatial_{sanitize_id_part(relation)}_"
        f"{sanitize_id_part(split)}_{sanitize_id_part(source_id)}"
    )

    image_paths = save_record_images(
        record,
        image_dir=image_dir,
        subject=str(relation),
        sample_id=sample_id,
        extension="jpg",
    )

    return {
        "id": sample_id,
        "task": "embspatial",
        "subject": relation,
        "answer_type": "option",
        "answer": answer,
        "messages": [
            {
                "role": "user",
                "content": build_multimodal_content(
                    build_embspatial_prompt(question, choices),
                    image_paths,
                ),
            }
        ],
        "meta": {
            "dataset": dataset_name,
            "split": split,
            "input_modality": "image_text" if image_paths else "text",
            "source_index": index,
            "source_id": source_id,
            "data_source": record.get("data_source"),
            "scene_id": record.get("scene_id"),
            "question_id": record.get("question_id"),
            "question": question,
            "relation": relation,
            "choices": {
                letter: choice for letter, choice in zip(letters, choices)
            },
            "objects": record.get("objects"),
            "bbox": record.get("bbox"),
            "name": record.get("name"),
            "image_count": len(image_paths),
            "images": image_paths,
        },
    }


def build_erqa_prompt(question: str) -> str:
    return (
        "You are given one or more images in order. "
        "Answer the robotics multiple-choice question. "
        "Only output the letter of the correct option.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def convert_erqa_record(
    record: Dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    index: int,
    image_dir: str | Path = DEFAULT_ERQA_IMAGE_DIR,
) -> Dict[str, Any]:
    question = str(record.get("question", "")).strip()
    if not question:
        raise ValueError("ERQA sample missing non-empty `question`")

    answer = normalize_answer(record.get("answer"))
    question_type = record.get("question_type") or "unknown"
    source_id = record.get("question_id") or record.get("id") or index
    sample_id = (
        f"erqa_{sanitize_id_part(question_type)}_"
        f"{sanitize_id_part(split)}_{sanitize_id_part(source_id)}"
    )

    image_paths = save_record_images(
        record,
        image_dir=image_dir,
        subject=str(question_type),
        sample_id=sample_id,
        extension="jpg",
    )

    return {
        "id": sample_id,
        "task": "erqa",
        "subject": question_type,
        "answer_type": "option",
        "answer": answer,
        "messages": [
            {
                "role": "user",
                "content": build_multimodal_content(
                    build_erqa_prompt(question),
                    image_paths,
                ),
            }
        ],
        "meta": {
            "dataset": dataset_name,
            "split": split,
            "input_modality": "image_text" if image_paths else "text",
            "source_index": index,
            "source_id": source_id,
            "question_id": record.get("question_id"),
            "question": question,
            "question_type": question_type,
            "visual_indices": record.get("visual_indices"),
            "image_count": len(image_paths),
            "images": image_paths,
        },
    }


def build_refspatial_prompt(prompt: str, suffix: str) -> str:
    prompt = prompt.strip()
    suffix = suffix.strip()
    if not suffix:
        return prompt
    return f"{prompt} {suffix}"


def convert_refspatial_record(
    record: Dict[str, Any],
    *,
    dataset_name: str,
    split: str,
    index: int,
    image_dir: str | Path = DEFAULT_REFSPATIAL_IMAGE_DIR,
    mask_dir: str | Path = DEFAULT_REFSPATIAL_MASK_DIR,
) -> Dict[str, Any]:
    prompt = str(record.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("RefSpatial sample missing non-empty `prompt`")

    suffix = str(record.get("suffix", "")).strip()
    source_id = record.get("id") if record.get("id") is not None else index
    sample_id = (
        f"refspatial_{sanitize_id_part(split)}_"
        f"{sanitize_id_part(source_id)}"
    )

    image_paths = save_record_images(
        record,
        image_dir=image_dir,
        subject=split,
        sample_id=sample_id,
        extension="jpg",
    )
    if not image_paths:
        raise ValueError(f"RefSpatial sample missing exportable image: {sample_id}")

    mask_path = save_record_mask(
        record,
        mask_dir=mask_dir,
        split=split,
        sample_id=sample_id,
    )
    if mask_path is None:
        raise ValueError(f"RefSpatial sample missing exportable mask: {sample_id}")

    image_size = get_image_size(image_paths[0])
    answer = {
        "mask_path": mask_path,
        "image_size": image_size,
        "coordinate_system": "normalized_xy",
    }

    return {
        "id": sample_id,
        "task": "refspatial",
        "subject": split,
        "answer_type": "point",
        "answer": answer,
        "messages": [
            {
                "role": "user",
                "content": build_multimodal_content(
                    build_refspatial_prompt(prompt, suffix),
                    image_paths,
                ),
            }
        ],
        "meta": {
            "dataset": dataset_name,
            "split": split,
            "input_modality": "image_text",
            "source_index": index,
            "source_id": source_id,
            "object": record.get("object"),
            "prompt": prompt,
            "suffix": suffix,
            "step": record.get("step"),
            "image_count": len(image_paths),
            "images": image_paths,
            "mask": mask_path,
            "image_size": image_size,
        },
    }


def iter_hf_refspatial(
    *,
    dataset_name: str,
    split: str,
) -> Iterable[tuple[str, int, Dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install it with: pip install datasets"
        ) from exc

    split_key = "locplace" if split in {"location+placement", "placement+location"} else split
    splits = REFSPATIAL_MIXED_SPLITS.get(split_key, [split_key])
    for split_name in splits:
        dataset = load_dataset(dataset_name, split=split_name)
        for index, record in enumerate(dataset):
            yield split_name, index, dict(record)


def iter_hf_single_split(
    *,
    dataset_name: str,
    split: str,
) -> Iterable[tuple[str, int, Dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install it with: pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    for index, record in enumerate(dataset):
        yield split, index, dict(record)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    output_path = resolve_output_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_subjects(value: str) -> List[str]:
    subjects = [item.strip() for item in value.split(",") if item.strip()]
    if not subjects:
        raise ValueError("At least one subject must be provided")
    return subjects


def resolve_subjects(benchmark: str, value: str) -> List[str]:
    subjects = parse_subjects(value)
    if len(subjects) == 1 and subjects[0].lower() == "all":
        if benchmark == "mmmu":
            return MMMU_SUBJECTS
    return subjects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace benchmark data to benchmark JSONL."
    )
    parser.add_argument(
        "--benchmark",
        choices=["mmlu", "mmmu", "embspatial", "erqa", "refspatial"],
        default="mmlu",
        help="Benchmark to convert. Default: mmlu",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="HuggingFace dataset name. Uses the benchmark-specific default when omitted.",
    )
    parser.add_argument(
        "--subject",
        default="all",
        help="Dataset config/subject, or comma-separated subjects. Examples: all, Art, college_computer_science.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split. Uses the benchmark-specific default when omitted.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to benchmark_multimodal/data/<benchmark>.jsonl.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory for exported benchmark images.",
    )
    parser.add_argument(
        "--mask-dir",
        default=None,
        help="Directory for exported benchmark masks, used by RefSpatial.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of converted samples, useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_datasets = {
        "mmlu": "cais/mmlu",
        "mmmu": "MMMU/MMMU",
        "embspatial": "FlagEval/EmbSpatial-Bench",
        "erqa": "FlagEval/ERQA",
        "refspatial": "BAAI/RefSpatial-Bench",
    }
    default_splits = {
        "mmlu": "test",
        "mmmu": "validation",
        "embspatial": "test",
        "erqa": "test",
        "refspatial": "location",
    }
    default_image_dirs = {
        "mmmu": DEFAULT_MMMU_IMAGE_DIR,
        "embspatial": DEFAULT_EMBSPATIAL_IMAGE_DIR,
        "erqa": DEFAULT_ERQA_IMAGE_DIR,
        "refspatial": DEFAULT_REFSPATIAL_IMAGE_DIR,
    }

    dataset_name = args.dataset or default_datasets[args.benchmark]
    split = args.split or default_splits[args.benchmark]
    output = args.output or str(default_output_for_benchmark(args.benchmark, split))
    image_dir = args.image_dir or str(
        default_image_dirs.get(args.benchmark, DEFAULT_MMMU_IMAGE_DIR)
    )
    mask_dir = args.mask_dir or str(DEFAULT_REFSPATIAL_MASK_DIR)
    subjects = (
        resolve_subjects(args.benchmark, args.subject)
        if args.benchmark in {"mmlu", "mmmu"}
        else []
    )

    def converted_rows() -> Iterable[Dict[str, Any]]:
        emitted = 0
        if args.benchmark == "mmlu":
            iterator = iter_hf_mmlu(
                dataset_name=dataset_name,
                subjects=subjects,
                split=split,
            )
            for subject, index, record in iterator:
                if args.limit is not None and emitted >= args.limit:
                    break
                yield convert_mmlu_record(
                    record,
                    dataset_name=dataset_name,
                    subject=subject,
                    split=split,
                    index=index,
                )
                emitted += 1
            return

        if args.benchmark == "mmmu":
            iterator = iter_hf_mmmu(
                dataset_name=dataset_name,
                subjects=subjects,
                split=split,
            )
            for subject, index, record in iterator:
                if args.limit is not None and emitted >= args.limit:
                    break
                yield convert_mmmu_record(
                    record,
                    dataset_name=dataset_name,
                    subject=subject,
                    split=split,
                    index=index,
                    image_dir=image_dir,
                )
                emitted += 1
            return

        if args.benchmark == "refspatial":
            iterator = iter_hf_refspatial(dataset_name=dataset_name, split=split)
            for split_name, index, record in iterator:
                if args.limit is not None and emitted >= args.limit:
                    break
                yield convert_refspatial_record(
                    record,
                    dataset_name=dataset_name,
                    split=split_name,
                    index=index,
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                )
                emitted += 1
            return

        iterator = iter_hf_single_split(dataset_name=dataset_name, split=split)
        for _, index, record in iterator:
            if args.limit is not None and emitted >= args.limit:
                break

            if args.benchmark == "embspatial":
                yield convert_embspatial_record(
                    record,
                    dataset_name=dataset_name,
                    split=split,
                    index=index,
                    image_dir=image_dir,
                )
            elif args.benchmark == "erqa":
                yield convert_erqa_record(
                    record,
                    dataset_name=dataset_name,
                    split=split,
                    index=index,
                    image_dir=image_dir,
                )
            else:
                raise ValueError(f"Unsupported benchmark: {args.benchmark}")
            emitted += 1

    count = write_jsonl(output, converted_rows())
    print(
        f"Wrote {count} {args.benchmark.upper()} samples to "
        f"{resolve_output_path(output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
