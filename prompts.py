"""Versioned prompt loading.

Layout:
    prompts/
    ├── v1/
    │   ├── system.md
    │   ├── writer_nudge.md
    │   ├── force_synthesis.md
    │   ├── json_schema.md
    │   └── judge.md
    └── v2/  (future)
        ...

Add new versions by creating a new vN directory with the same file names.
The CLI exposes `--prompts vN` and the env var DEEP_RESEARCH_PROMPTS.

The "system" prompt is the only one localised; pass `lang` to `load()` and
the language directive is appended for non-English outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

_PROMPTS_DIR = Path(__file__).parent / "prompts"

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh-CN": "Simplified Chinese (简体中文)",
    "zh-TW": "Traditional Chinese (繁體中文)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "es": "Spanish (Español)",
    "de": "German (Deutsch)",
    "fr": "French (Français)",
    "pt": "Portuguese (Português)",
    "ru": "Russian (Русский)",
}


_REQUIRED = ("system", "writer_nudge", "force_synthesis", "json_schema", "judge")


def list_versions() -> list[str]:
    if not _PROMPTS_DIR.exists():
        return []
    return sorted(p.name for p in _PROMPTS_DIR.iterdir() if p.is_dir())


def load(version: str = "v1", *, lang: str = "en") -> dict[str, str]:
    """Load all prompts for the given version. Append a language directive
    to the system prompt when `lang` is not English.
    """
    base = _PROMPTS_DIR / version
    if not base.exists():
        available = list_versions()
        raise ValueError(
            f"Prompt version {version!r} not found at {base}. "
            f"Available: {available}"
        )

    out: dict[str, str] = {}
    for name in _REQUIRED:
        p = base / f"{name}.md"
        if not p.exists():
            raise ValueError(f"Missing prompt file: {p}")
        out[name] = p.read_text(encoding="utf-8").strip() + "\n"

    if lang and lang != "en":
        out["system"] = with_language(out["system"], lang)

    return out


def with_language(system_prompt: str, lang: str) -> str:
    """Append a language directive to the system prompt."""
    name = LANGUAGE_NAMES.get(lang, lang)
    addendum = (
        "\n\n## Output language\n\n"
        f"Write the final markdown report in **{name}**. Internal reasoning, "
        "tool-call arguments, and search queries may stay in English (often "
        "more effective for retrieval), but the final report — including "
        "headings, body, and source notes — MUST be in {name}. Source titles "
        "may stay in their original language; add an inline translation in "
        "parentheses if helpful."
    ).format(name=name)
    return system_prompt + addendum


def resolve_lang(raw: Optional[str]) -> str:
    """Normalise a user-provided language code to a known key."""
    if not raw:
        return "en"
    raw = raw.strip()
    if raw in LANGUAGE_NAMES:
        return raw
    # Common aliases
    aliases = {
        "zh": "zh-CN", "cn": "zh-CN", "chinese": "zh-CN",
        "tw": "zh-TW",
        "english": "en",
        "spanish": "es", "german": "de", "french": "fr",
        "japanese": "ja", "korean": "ko",
    }
    return aliases.get(raw.lower(), raw)
