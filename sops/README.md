# SOPs: field reference

Policy lives here as data. `vocabulary.yaml` holds the activities and audiences the bot understands; every other
`*.yaml` file holds SOPs for one category. The loader (`policy.py`) validates every file against a Pydantic schema:
on the first load the app refuses to start and names the bad file and SOP; edits are re-read on every message, so a
new SOP takes effect without a restart.

**Add an SOP:** append a block to the right file (or create a new `.yaml` file), run `python policy.py`, ask the bot.

| Field | Meaning |
|---|---|
| `id`, `title`, `category` | `id` is what the bot cites; `title` is shown with it |
| `severity` | `info` < `low` < `moderate` < `high` < `critical` |
| `situational` | `true` for a weather **system** (rain system, thunderstorm): it outranks every other SOP and the graph's `override` branch names it before any activity advice |
| `applies_to.activities` | list of activity keys, or `any` (any outdoor question, including activities with no key) |
| `applies_to.audiences` | optional list of audience keys |
| `when` | list of alternatives; the SOP applies if ANY line holds. A line is comparisons joined by ` and `: `gust_max_kmh >= 40 and precip_total_mm > 2`. Operators: `>= <= > < == !=` |
| `guidance` | the advice. `{placeholders}` are filled by code with live numbers, never by the model |
| `must_quote` | optional metrics the reply MUST state (e.g. the gust value); if the wording leaves one out, the fixed template is used. Each must appear in `guidance` |
| `judgement` | instead of `when`, for fuzzy SOPs: the model picks one of `verdicts` (each with its own `severity` + `guidance`) from the numbers in `uses` only |
| `default: true` | applies only if NO other SOP matched, and (with `when`) only while `when` holds. Outside that envelope the bot says it has no guidance instead of giving an all-clear |

**Metrics** (window = the hours the user asked about): `temp_max_c`, `temp_min_c`, `feels_like_max_c`,
`feels_like_min_c`, `precip_total_mm`, `precip_prob_max_pct`, `wind_max_kmh`, `gust_max_kmh`, `uv_max`,
`thunderstorm` (true/false), and window-independent rain context around now: `rain_past_24h_mm`, `rain_next_24h_mm`,
`rain_48h_mm` (past + next 24 h), `rain_hours_48h` (hours with ≥ 0.1 mm of those 48).
Extra placeholders: `{place}`, `{hours}`, `{activity}`.

**When several SOPs apply:** all are shown. Situational SOPs first, then highest severity first (file order breaks
ties). If anything high/critical applies, info-level SOPs are dropped (no "lovely day" next to a storm warning).
