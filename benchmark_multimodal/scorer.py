from __future__ import annotations

import ast
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List


ScoreResult = Dict[str, Any]
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^|]+?\|>|</?s>|<pad>")


def _as_answer_list(answer: Any) -> List[str]:
    if answer is None:
        return []
    if isinstance(answer, list):
        return [str(item) for item in answer if item is not None]
    return [str(answer)]


def normalize_text(text: Any, options: Dict[str, Any] | None = None) -> str:
    """Normalize text for short-answer matching."""
    opts = options or {}
    value = "" if text is None else str(text)

    if opts.get("lowercase", True):
        value = value.lower()

    if opts.get("strip_punctuation", True):
        table = str.maketrans("", "", string.punctuation)
        value = value.translate(table)

    if opts.get("strip_articles", False):
        value = re.sub(r"\b(a|an|the)\b", " ", value)

    if opts.get("collapse_whitespace", True):
        value = " ".join(value.split())
    else:
        value = value.strip()

    return value


def strip_special_tokens(text: Any) -> str:
    value = "" if text is None else str(text)
    return SPECIAL_TOKEN_PATTERN.sub("", value).strip()


def exact_match(
    prediction: Any,
    answers: Any,
    normalization: Dict[str, Any] | None = None,
) -> ScoreResult:
    """Strict exact match after stripping leading/trailing whitespace."""
    pred = "" if prediction is None else str(prediction).strip()
    answer_list = [answer.strip() for answer in _as_answer_list(answers)]
    correct = pred in answer_list
    return {
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "scorer": "exact_match",
        "normalized_prediction": pred,
        "normalized_answer": answer_list,
        "reason": "exact string match" if correct else "prediction did not exactly match any answer",
    }


def normalized_match(
    prediction: Any,
    answers: Any,
    normalization: Dict[str, Any] | None = None,
) -> ScoreResult:
    """Exact match after normalization."""
    pred = normalize_text(prediction, normalization)
    answer_list = [normalize_text(answer, normalization) for answer in _as_answer_list(answers)]
    correct = pred in answer_list
    return {
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "scorer": "normalized_match",
        "normalized_prediction": pred,
        "normalized_answer": answer_list,
        "reason": "normalized text matched" if correct else "normalized text did not match",
    }

# 选择题
def extract_explicit_option(prediction: Any) -> str | None:
    pred = strip_special_tokens(prediction)
    normalized_pred = normalize_text(
        pred,
        {"lowercase": True, "strip_punctuation": False, "collapse_whitespace": True},
    )

    priority_patterns = [
        r"\bfinal\s+answer\b\s*(?::|：|=|\bis\b)\s*\(?\s*([a-j])\s*\)?",
        r"答案\s*(?:=|:|：)\s*\(?\s*([a-j])\s*\)?",
    ]
    for pattern in priority_patterns:
        matches = re.findall(pattern, normalized_pred, flags=re.IGNORECASE)
        if matches:
            return matches[-1].lower()

    return None


def extract_single_option(prediction: Any) -> str | None:
    pred = strip_special_tokens(prediction)
    match = re.fullmatch(r"\s*\(?\s*([a-jA-J])\s*\)?\.?\s*", pred)
    return match.group(1).lower() if match else None


def extract_option_candidates(prediction: Any) -> List[str]:
    pred = strip_special_tokens(prediction)
    explicit = extract_explicit_option(pred)
    if explicit:
        return [explicit]
    single = extract_single_option(pred)
    if single:
        return [single]

    return []


def postprocess_prediction(sample: Dict[str, Any], prediction: Any) -> str:
    """Clean raw model output for scoring/reporting while preserving raw text elsewhere."""
    cleaned = strip_special_tokens(prediction)
    answer_type = sample.get("answer_type")

    if answer_type == "option":
        explicit = extract_explicit_option(cleaned)
        if explicit:
            return explicit.upper()
        single = extract_single_option(cleaned)
        return single.upper() if single else ""

    final_patterns = [
        r"\bfinal\s+answer\b\s*(?:=|:|：)\s*(.+)$",
        r"\bfinal\s+answer\b\s+is\s+(.+)$",
        r"答案\s*(?:=|:|：)\s*(.+)$",
    ]
    final_matches: List[str] = []
    for pattern in final_patterns:
        final_matches.extend(re.findall(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL))
    if final_matches:
        return strip_special_tokens(final_matches[-1])
    return cleaned


def option_match(
    prediction: Any,
    answers: Any,
    normalization: Dict[str, Any] | None = None,
) -> ScoreResult:
    answer_list = [
        normalize_text(answer, {"lowercase": True, "strip_punctuation": True})
        for answer in _as_answer_list(answers)
    ]
    answer_list = [answer for answer in answer_list if answer]
    opts = normalization or {}
    if opts.get("require_final_or_single", False):
        candidate = extract_explicit_option(prediction) or extract_single_option(prediction)
        candidates = [candidate] if candidate else []
    else:
        candidates = extract_option_candidates(prediction)

    correct = any(candidate in answer_list for candidate in candidates)
    return {
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "scorer": "option_match",
        "normalized_prediction": candidates,
        "normalized_answer": answer_list,
        "reason": "option matched" if correct else "no predicted option matched",
    }


def regex_match(
    prediction: Any,
    answers: Any,
    normalization: Dict[str, Any] | None = None,
) -> ScoreResult:
    """Treat answers as regular expressions and search in prediction."""
    pred = "" if prediction is None else str(prediction)
    patterns = _as_answer_list(answers)
    for pattern in patterns:
        if re.search(pattern, pred, flags=re.IGNORECASE):
            return {
                "correct": True,
                "score": 1.0,
                "scorer": "regex_match",
                "normalized_prediction": pred,
                "normalized_answer": patterns,
                "reason": f"regex matched: {pattern}",
            }

    return {
        "correct": False,
        "score": 0.0,
        "scorer": "regex_match",
        "normalized_prediction": pred,
        "normalized_answer": patterns,
        "reason": "no regex matched",
    }


def _answer_dict(answer: Any) -> Dict[str, Any]:
    if isinstance(answer, dict):
        return answer
    if isinstance(answer, str):
        try:
            parsed = json.loads(answer)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _strip_code_fence(text: str) -> str:
    value = text.strip()
    fence = re.match(r"^```(?:json|python)?\s*(.*?)\s*```$", value, flags=re.DOTALL)
    return fence.group(1).strip() if fence else value


def _collect_points_from_obj(obj: Any, points: List[List[float]]) -> None:
    if isinstance(obj, dict):
        if isinstance(obj.get("point"), (list, tuple)) and len(obj["point"]) == 2:
            # Gemini-style output often uses [y, x]. The RefSpatial prompt asks
            # for [(x, y)], so only accept explicit `point` as y/x when present.
            y, x = obj["point"]
            points.append([float(x), float(y)])
            return
        if "x" in obj and "y" in obj:
            points.append([float(obj["x"]), float(obj["y"])])
            return
        for value in obj.values():
            _collect_points_from_obj(value, points)
        return

    if isinstance(obj, (list, tuple)):
        if len(obj) == 2 and all(isinstance(item, (int, float)) for item in obj):
            points.append([float(obj[0]), float(obj[1])])
            return
        for value in obj:
            _collect_points_from_obj(value, points)


def extract_points(prediction: Any) -> List[List[float]]:
    text = _strip_code_fence("" if prediction is None else str(prediction))
    points: List[List[float]] = []

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
        _collect_points_from_obj(parsed, points)
        if points:
            return points

    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    for x, y in re.findall(rf"\(\s*({number})\s*,\s*({number})\s*\)", text):
        points.append([float(x), float(y)])
    for x, y in re.findall(rf"\[\s*({number})\s*,\s*({number})\s*\]", text):
        points.append([float(x), float(y)])
    for x, y in re.findall(
        rf"\bx\s*[=:]\s*({number})\D+?\by\s*[=:]\s*({number})",
        text,
        flags=re.IGNORECASE,
    ):
        points.append([float(x), float(y)])

    return points


def _scale_point(
    point: List[float],
    *,
    image_size: List[int] | None,
    mask_size: tuple[int, int],
) -> tuple[int, int]:
    x, y = point
    image_width, image_height = image_size or [mask_size[0], mask_size[1]]

    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        px = x * (image_width - 1)
        py = y * (image_height - 1)
    elif 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0 and (
        x > image_width or y > image_height
    ):
        px = x / 1000.0 * (image_width - 1)
        py = y / 1000.0 * (image_height - 1)
    else:
        px = x
        py = y

    if image_width != mask_size[0] and image_width > 0:
        px = px * mask_size[0] / image_width
    if image_height != mask_size[1] and image_height > 0:
        py = py * mask_size[1] / image_height

    return int(round(px)), int(round(py))


def _mask_hit(mask: Any, x: int, y: int, radius_px: int) -> bool:
    width, height = mask.size
    if x < 0 or y < 0 or x >= width or y >= height:
        return False

    pixels = mask.load()
    radius = max(0, int(radius_px))
    for yy in range(max(0, y - radius), min(height, y + radius + 1)):
        for xx in range(max(0, x - radius), min(width, x + radius + 1)):
            if pixels[xx, yy] > 0:
                return True
    return False


def point_in_mask_match(
    prediction: Any,
    answers: Any,
    normalization: Dict[str, Any] | None = None,
) -> ScoreResult:
    """Score point predictions by checking whether any predicted point hits the mask."""
    opts = normalization or {}
    answer = _answer_dict(answers)
    mask_path = answer.get("mask_path")
    if not mask_path:
        return {
            "correct": False,
            "score": 0.0,
            "scorer": "point_in_mask_match",
            "normalized_prediction": None,
            "normalized_answer": answer,
            "reason": "answer missing mask_path",
        }

    points = extract_points(prediction)
    if not points:
        return {
            "correct": False,
            "score": 0.0,
            "scorer": "point_in_mask_match",
            "normalized_prediction": [],
            "normalized_answer": answer,
            "reason": "no point parsed from prediction",
        }

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for point-in-mask scoring") from exc

    path = Path(mask_path).expanduser()
    if not path.exists():
        return {
            "correct": False,
            "score": 0.0,
            "scorer": "point_in_mask_match",
            "normalized_prediction": points,
            "normalized_answer": answer,
            "reason": f"mask file not found: {path}",
        }

    radius_px = int(opts.get("radius_px", 0) or 0)
    with Image.open(path) as image:
        mask = image.convert("L")
        scaled_points = [
            _scale_point(
                point,
                image_size=answer.get("image_size"),
                mask_size=mask.size,
            )
            for point in points
        ]
        correct = any(_mask_hit(mask, x, y, radius_px) for x, y in scaled_points)

    return {
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "scorer": "point_in_mask_match",
        "normalized_prediction": {
            "raw_points": points,
            "pixel_points": scaled_points,
        },
        "normalized_answer": answer,
        "reason": "point inside target mask" if correct else "no predicted point hit target mask",
    }


SCORERS = {
    "exact_match": exact_match,
    "normalized_match": normalized_match,
    "option_match": option_match,
    "regex_match": regex_match,
    "point_in_mask_match": point_in_mask_match,
}


def resolve_scorer_name(sample: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    scoring_cfg = cfg.get("scoring", {})
    task = sample.get("task")
    answer_type = sample.get("answer_type")

    task_scorers = scoring_cfg.get("task_scorers") or {}
    if task in task_scorers:
        return task_scorers[task]

    type_scorers = scoring_cfg.get("scorers") or {}
    if answer_type in type_scorers:
        return type_scorers[answer_type]

    return scoring_cfg.get("default_scorer", "normalized_match")


def score_prediction(
    sample: Dict[str, Any],
    prediction: str,
    cfg: Dict[str, Any],
) -> ScoreResult:
    """Score one prediction according to the benchmark config."""
    scoring_cfg = cfg.get("scoring", {})
    if not scoring_cfg.get("enabled", True):
        return {
            "correct": None,
            "score": None,
            "scorer": "disabled",
            "normalized_prediction": None,
            "normalized_answer": None,
            "reason": "scoring disabled",
        }

    scorer_name = resolve_scorer_name(sample, cfg)
    if scorer_name == "disabled":
        return {
            "correct": None,
            "score": None,
            "scorer": "disabled",
            "normalized_prediction": None,
            "normalized_answer": None,
            "reason": "scorer disabled for sample",
        }

    scorer = SCORERS.get(scorer_name)
    if scorer is None:
        raise ValueError(f"Unknown scorer: {scorer_name}")

    scorer_options = dict(scoring_cfg.get("normalization", {}) or {})
    for key in (sample.get("answer_type"), scorer_name):
        value = scoring_cfg.get(key)
        if isinstance(value, dict):
            scorer_options.update(value)

    return scorer(
        prediction,
        sample.get("answer"),
        scorer_options,
    )


def attach_score(
    row: Dict[str, Any],
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a copy of a prediction row with scoring fields attached."""
    scored = dict(row)
    result = score_prediction(sample, row.get("prediction", ""), cfg)
    scored.update(result)
    return scored
