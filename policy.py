"""The only code that reads policy. SOPs are data in sops/*.yaml; every SOP is validated against a Pydantic schema,
and anything malformed raises PolicyError naming the file and SOP (the app then refuses to start).
Re-read on every call, so a new SOP applies on the next message without a restart.

Validate after editing:  python policy.py
"""
import operator
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from nodes.weather import METRICS

SOPS_DIR = Path(__file__).parent / "sops"
VOCAB_FILE = "vocabulary.yaml"
SEVERITY = ["info", "low", "moderate", "high", "critical"]
OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq, "!=": operator.ne}
COND = re.compile(r"^\s*(\w+)\s*(>=|<=|==|!=|>|<)\s*(\S+)\s*$")
SAMPLE = {**{k: 1 for k in METRICS}, "place": "", "hours": "", "activity": ""}
Severity = Literal["info", "low", "moderate", "high", "critical"]


class PolicyError(ValueError):
    pass


def parse(line: str):
    """'gust_max_kmh >= 40 and thunderstorm == true' -> [(metric, op, value), ...]. Raises on anything else."""
    out = []
    for part in line.split(" and "):
        m = COND.match(part)
        if not m or m[1] not in METRICS:
            raise ValueError(f"bad condition {part!r} (metrics: {', '.join(METRICS)})")
        raw = m[3].lower()
        out.append((m[1], OPS[m[2]], {"true": True, "false": False}[raw] if raw in ("true", "false") else float(raw)))
    return out


def holds(line, metrics):
    return all(op(metrics[k], v) for k, op, v in parse(line))


def describe(line, metrics):
    """'gust_max_kmh >= 40' -> 'gust_max_kmh was 48 (rule: >= 40)'. Used by `explain` to show what triggered a SOP."""
    return " and ".join(f"{m[1]} was {metrics[m[1]]} (rule: {m[2]} {m[3]})"
                        for m in (COND.match(part) for part in line.split(" and ")))


def _placeholders_ok(text):
    try:
        text.format_map(SAMPLE)
    except (KeyError, ValueError) as e:
        raise ValueError(f"unknown placeholder {e} in guidance") from e
    return text


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Severity
    guidance: str

    @field_validator("guidance")
    @classmethod
    def placeholders(cls, v):
        return _placeholders_ok(v)


class Judgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uses: list[str]
    criteria: str
    verdicts: dict[str, Verdict]

    @field_validator("uses")
    @classmethod
    def known_metrics(cls, v):
        if unknown := set(v) - set(METRICS):
            raise ValueError(f"unknown metrics {sorted(unknown)}")
        return v


class AppliesTo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    activities: Literal["any"] | list[str]
    audiences: list[str] | None = None


class Sop(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo'd field name is an error, not silently ignored
    id: str
    title: str
    category: str
    severity: Severity | None = None
    situational: bool = False
    applies_to: AppliesTo
    when: list[str] = []
    judgement: Judgement | None = None
    default: bool = False
    must_quote: list[str] = []
    guidance: str | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.judgement:
            if self.when or self.default or self.guidance or self.severity:
                raise ValueError("a judgement SOP has verdicts instead of when/default/guidance/severity")
            return self
        if not (self.severity and self.guidance):
            raise ValueError("needs severity and guidance")
        if not (self.when or self.default):
            raise ValueError("needs `when`, `judgement` or `default: true`")
        for line in self.when:
            parse(line)
        _placeholders_ok(self.guidance)
        for k in self.must_quote:
            if k not in METRICS or k == "thunderstorm":
                raise ValueError(f"must_quote: {k} is not a numeric metric")
            if f"{{{k}}}" not in self.guidance:
                raise ValueError(f"must_quote: {{{k}}} isn't in the guidance text")
        return self


def _read(path):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise PolicyError(f"{path.name}: not valid YAML ({e})") from e


def load():
    """-> {activities, audiences, sops: [validated SOP dicts, each with its `file`]}. Raises PolicyError."""
    vocab = _read(SOPS_DIR / VOCAB_FILE)
    if not (isinstance(vocab.get("activities"), dict) and isinstance(vocab.get("audiences"), dict)):
        raise PolicyError(f"{VOCAB_FILE}: needs `activities` and `audiences` maps")
    sops, seen = [], {}
    for path in sorted(SOPS_DIR.glob("*.yaml")):
        if path.name == VOCAB_FILE:
            continue
        for raw in _read(path).get("sops") or []:
            where = f"{path.name}, SOP {raw.get('id', '?') if isinstance(raw, dict) else '?'}"
            try:
                s = Sop.model_validate(raw)
            except ValidationError as e:
                why = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'sop'}: {err['msg']}" for err in e.errors())
                raise PolicyError(f"{where}: {why}") from e
            if s.id in seen:
                raise PolicyError(f"{where}: duplicate id (also in {seen[s.id]})")
            acts = s.applies_to.activities
            if acts != "any" and (bad := set(acts) - set(vocab["activities"])):
                raise PolicyError(f"{where}: unknown activities {sorted(bad)} (see {VOCAB_FILE})")
            if bad := set(s.applies_to.audiences or []) - set(vocab["audiences"]):
                raise PolicyError(f"{where}: unknown audiences {sorted(bad)} (see {VOCAB_FILE})")
            seen[s.id] = path.name
            sops.append({**s.model_dump(exclude_none=True), "file": path.name})
    if not sops:
        raise PolicyError(f"no SOPs found in {SOPS_DIR}")
    return {**vocab, "sops": sops}


if __name__ == "__main__":
    doc = load()
    files = sorted({s["file"] for s in doc["sops"]})
    print(f"policy valid: {len(doc['sops'])} SOPs in {len(files)} files ({', '.join(files)})")
    # schema rejects the mistakes a policy author is likely to make
    good = {"id": "X-01", "title": "t", "category": "c", "severity": "low", "applies_to": {"activities": "any"},
            "when": ["gust_max_kmh >= 40"], "guidance": "gusts {gust_max_kmh}", "must_quote": ["gust_max_kmh"]}
    Sop.model_validate(good)
    for bad in ({"severity": "danger"}, {"when": ["gust_max_kph >= 40"]}, {"guidance": "gusts {gusts}"},
                {"must_quote": ["uv_max"]}, {"wehn": ["uv_max >= 8"]}, {"when": []}):
        try:
            Sop.model_validate({**good, **bad})
            raise AssertionError(f"accepted {bad}")
        except ValidationError:
            pass
    print("schema ok")
