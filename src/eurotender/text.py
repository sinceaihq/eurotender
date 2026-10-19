"""Text preparation shared by training and inference.

Two concerns live here:

* ``scrub_pii`` removes direct contact identifiers (e-mail addresses, phone
  numbers, URLs) before notice text is stored or redistributed.
* ``normalize`` turns notice text into model input. It masks anything that
  looks like a CPV code, because buyers often paste the code into the
  description ("CPV 45000000-7") and the model would otherwise learn to read
  the label instead of the text.
"""

from __future__ import annotations

import re
import unicodedata

# Also catches the common obfuscations "name(at)domain.fi", "name[at]domain.fi"
# and a stray space around the "@".
EMAIL_RE = re.compile(
    r"\b[\w.+-]+ ?(?:@|\(at\)|\[at\]|\{at\}) ?[\w-]+(?:\.[\w-]+)+\b", re.IGNORECASE
)
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"')]+", re.IGNORECASE)
# International numbers (+358 40 123 4567, 0049 30 1234567) and numbers that
# follow a telephone keyword in any EU language (tel., puh., tél., τηλ. ...).
PHONE_INTL_RE = re.compile(
    r"(?<![\w+])(?:\+|00)[1-9]\d{0,2}[\s./-]?(?:\(0\)[\s./-]?)?\d{1,4}(?:[\s./-]?\d{2,4}){1,4}\b"
)
PHONE_KEYWORD_RE = re.compile(
    r"(?i)\b(?:tel|tél|telefon|telefono|teléfono|telefone|telephone|phone|puh|puhelin"
    r"|tlf|τηλ|тел|gsm|mobile?|mob)\b\.?\s*(?:nr\.?|no\.?|:)?\s*"
    r"(\+?\(?\d[\d\s()./-]{5,}\d)"
)
# Contact persons introduced by an honorific ("Frau Müller", "Mme Dupont").
# Deliberately narrow: bare names without a title are not detected.
_HONORIFIC = (
    r"(?:Herrn?|Frau|Mr|Mrs|Ms|Mme|Mlle|Madame|Monsieur|Mevrouw|Mevr|Dhr|Sig\.ra|Sig"
    r"|Dott\.ssa|Dott|Dr|Prof|Ing)\.?"
)
_NAME_WORD = r"[A-ZÀ-ÖØ-ÞĀ-Ž][\w'-]+"
NAME_RE = re.compile(
    rf"\b{_HONORIFIC}(?:\s+{_HONORIFIC})*\s+(?:[A-Z]\.\s*)?{_NAME_WORD}(?:\s+{_NAME_WORD})?"
)
# 8 digits, optionally followed by the CPV check digit ("45000000-7").
CPV_CODE_RE = re.compile(r"(?<!\d)\d{8}(?:\s?-\s?\d)?(?!\d)")
WS_RE = re.compile(r"\s+")

EMAIL_TOKEN = "[EMAIL]"
URL_TOKEN = "[URL]"
PHONE_TOKEN = "[PHONE]"
NAME_TOKEN = "[NAME]"
CODE_TOKEN = " cpvcode "


def scrub_pii(text: str) -> str:
    """Replace e-mail addresses, URLs, phone numbers and titled names with placeholders."""
    if not text:
        return ""
    text = EMAIL_RE.sub(EMAIL_TOKEN, text)
    text = URL_RE.sub(URL_TOKEN, text)
    text = PHONE_KEYWORD_RE.sub(lambda m: m.group(0)[: m.start(1) - m.start(0)] + PHONE_TOKEN, text)
    text = PHONE_INTL_RE.sub(_phone_or_keep, text)
    text = NAME_RE.sub(NAME_TOKEN, text)
    return text


def _phone_or_keep(match: re.Match[str]) -> str:
    # Country code + subscriber number is at least 9 digits in the EU numbering
    # plans; shorter runs after "00" are usually reference numbers.
    number = match.group(0)
    number = number[2:] if number.startswith("00") else number.lstrip("+")
    digits = sum(ch.isdigit() for ch in number)
    return PHONE_TOKEN if digits >= 9 else match.group(0)


def mask_cpv_codes(text: str) -> str:
    return CPV_CODE_RE.sub(CODE_TOKEN, text)


def normalize(text: str) -> str:
    """Model-side normalisation: NFKC, PII scrubbing, CPV masking, casefold."""
    text = unicodedata.normalize("NFKC", text or "")
    text = mask_cpv_codes(scrub_pii(text))
    text = text.casefold()
    return WS_RE.sub(" ", text).strip()


def compose(title: str, description: str = "", max_description_chars: int = 2000) -> str:
    """Join the notice title and (truncated) description into one input string."""
    title = (title or "").strip()
    description = (description or "").strip()[:max_description_chars]
    if description and description != title:
        return f"{title}\n{description}"
    return title
