"""Data governance: nothing personal leaves the pipeline, by construction.

The CFPB already masks names, dates and account numbers in published narratives
(as `XXXX`, `XX/XX/XXXX`). That is the producer's promise; the pipeline does not
rely on it. Before any text is sent to an external model it is scanned for the
personal data types that matter here and redacted:

- e-mail addresses, URLs
- phone numbers (US formats)
- payment card numbers (13-19 digits that pass the Luhn checksum)
- US social security numbers
- long digit runs that look like account or reference numbers

The scan's findings are counted per complaint and reported, so leakage in the
source shows up as a governance metric instead of passing through unnoticed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")
SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
LONG_DIGITS = re.compile(r"(?<!\d)\d{9,}(?!\d)")
# The producer's own masks, normalised to a single token the models can learn to ignore.
CFPB_MASK = re.compile(r"\b(?:X{2,}/X{2,}/X{2,4}|X{2,}/X{2,}|X{4,})\b|\{\$[\d.]+\}")


def luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits][::-1]
    total = sum(
        n if i % 2 == 0 else (n * 2 - 9 if n * 2 > 9 else n * 2) for i, n in enumerate(nums)
    )
    return total % 10 == 0


@dataclass
class Redaction:
    text: str
    found: dict[str, int] = field(default_factory=dict)

    @property
    def leaked(self) -> bool:
        return bool(self.found)


def redact(text: str) -> Redaction:
    found: dict[str, int] = {}

    def sub(pattern: re.Pattern, label: str, s: str, check=None) -> str:
        def repl(m: re.Match) -> str:
            if check and not check(m.group(0)):
                return m.group(0)
            found[label] = found.get(label, 0) + 1
            return f"[{label.upper()}]"

        return pattern.sub(repl, s)

    out = text or ""
    out = sub(EMAIL, "email", out)
    out = sub(URL, "url", out)
    out = sub(SSN, "ssn", out)
    out = sub(CARD, "card", out, check=lambda s: luhn_ok(re.sub(r"\D", "", s)))
    out = sub(PHONE, "phone", out)
    out = sub(LONG_DIGITS, "number", out)
    out = CFPB_MASK.sub("[REDACTED]", out)
    out = re.sub(r"\s+", " ", out).strip()
    return Redaction(out, found)
