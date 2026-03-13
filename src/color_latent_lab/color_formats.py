from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass

from .color_palette import COLOR_WORD_SYNONYMS

FORMAT_PROMPTS: dict[str, str] = {
    "word": "What color do you associate with the word {word}? Reply with one color word.",
    "hex": "What color do you associate with the word {word}? Reply with a hex code.",
    "rgb": "What color do you associate with the word {word}? Reply with an RGB triplet like 255,0,0.",
}
DEFAULT_FORMATS = ("word", "hex", "rgb")

FAMILY_PALETTE: dict[str, str] = {
    "red": "#c93a3a",
    "orange": "#d9741b",
    "yellow": "#c7a400",
    "green": "#31844a",
    "cyan": "#1a8f9f",
    "blue": "#2b63c9",
    "purple": "#7a46af",
    "magenta": "#ba4b93",
    "brown": "#7c5b3d",
    "black": "#222222",
    "white": "#f4f1e8",
    "gray": "#8f8f8f",
}
FORMAT_STROKES: dict[str, str] = {
    "word": "#1a1a1a",
    "hex": "#4a4a4a",
    "rgb": "#7a7a7a",
}

HEX_CODE_RE = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})")
RGB_TRIPLET_RE = re.compile(r"(?<!\d)(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?!\d)")
WORD_TOKEN_RE = re.compile(r"[a-zA-Z]+")


@dataclass(frozen=True)
class ParsedCompletion:
    normalized_output: str | None
    color_family: str | None
    temperature: str | None


def _normalize_hex_code(raw_completion: str) -> str | None:
    match = HEX_CODE_RE.search(raw_completion)
    if match is None:
        return None
    value = match.group(0).lower()
    if len(value) == 4:
        return "#" + "".join(character * 2 for character in value[1:])
    return value


def _hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    return (
        int(hex_code[1:3], 16),
        int(hex_code[3:5], 16),
        int(hex_code[5:7], 16),
    )


def _color_family_from_rgb(red: int, green: int, blue: int) -> str:
    red_f = red / 255.0
    green_f = green / 255.0
    blue_f = blue / 255.0
    hue, saturation, value = colorsys.rgb_to_hsv(red_f, green_f, blue_f)
    hue_degrees = hue * 360.0

    if value <= 0.15:
        return "black"
    if saturation <= 0.12 and value >= 0.9:
        return "white"
    if saturation <= 0.15:
        return "gray"
    if 15.0 <= hue_degrees < 45.0 and value < 0.65:
        return "brown"
    if hue_degrees < 15.0 or hue_degrees >= 345.0:
        return "red"
    if hue_degrees < 45.0:
        return "orange"
    if hue_degrees < 75.0:
        return "yellow"
    if hue_degrees < 165.0:
        return "green"
    if hue_degrees < 195.0:
        return "cyan"
    if hue_degrees < 255.0:
        return "blue"
    if hue_degrees < 300.0:
        return "purple"
    return "magenta"


def _color_family_from_hex(hex_code: str) -> str:
    return _color_family_from_rgb(*_hex_to_rgb(hex_code))


def _temperature_from_family(family: str | None) -> str | None:
    if family is None:
        return None
    if family in {"red", "orange", "yellow", "brown", "magenta"}:
        return "warm"
    if family in {"green", "cyan", "blue", "purple"}:
        return "cool"
    return "neutral"


def parse_format_completion(format_name: str, raw_completion: str) -> ParsedCompletion:
    stripped = raw_completion.strip()
    if format_name == "word":
        for token in WORD_TOKEN_RE.findall(stripped.lower()):
            family = COLOR_WORD_SYNONYMS.get(token)
            if family is not None:
                return ParsedCompletion(
                    normalized_output=token,
                    color_family=family,
                    temperature=_temperature_from_family(family),
                )
        return ParsedCompletion(None, None, None)
    if format_name == "hex":
        hex_code = _normalize_hex_code(stripped)
        if hex_code is None:
            return ParsedCompletion(None, None, None)
        family = _color_family_from_hex(hex_code)
        return ParsedCompletion(
            normalized_output=hex_code,
            color_family=family,
            temperature=_temperature_from_family(family),
        )
    if format_name == "rgb":
        match = RGB_TRIPLET_RE.search(stripped)
        if match is None:
            return ParsedCompletion(None, None, None)
        values = tuple(int(group) for group in match.groups())
        if any(value < 0 or value > 255 for value in values):
            return ParsedCompletion(None, None, None)
        normalized = ",".join(str(value) for value in values)
        family = _color_family_from_rgb(*values)
        return ParsedCompletion(
            normalized_output=normalized,
            color_family=family,
            temperature=_temperature_from_family(family),
        )
    raise ValueError(f"Unsupported format {format_name!r}")


__all__ = [
    "COLOR_WORD_SYNONYMS",
    "DEFAULT_FORMATS",
    "FAMILY_PALETTE",
    "FORMAT_PROMPTS",
    "FORMAT_STROKES",
    "ParsedCompletion",
    "parse_format_completion",
]
