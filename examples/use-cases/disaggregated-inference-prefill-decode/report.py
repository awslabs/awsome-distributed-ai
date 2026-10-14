"""Render scored comparisons as one self-contained HTML page (standard library only). Values come
from scoring.py; this module only draws them. Page contents: README section "5.collect.py's output".
"""
import html as _html
import math
from scoring import TPOT_METRICS, UNAVAILABLE, verdict

RULE_TERM = {"mean": "mean", "itl-p99": "itl-p99", "worst-itl": "worst-itl"}
RULE_TITLE = {"mean": "mean TPOT", "itl-p99": "itl-p99", "worst-itl": "worst-itl"}
ARCH_LETTER = {"unified": "U", "disaggregated": "D"}
WINNER_TEXT = {"unified": "Unified has the higher qualifying rate", "disaggregated": "Disaggregated has the higher qualifying rate",
               "tie": "Tie: same qualifying rate", "neither": "Neither stack has a qualifying rate",
               "incomparable": "Incomparable: not every stack was judged under this rule"}


def _esc(x):
    return _html.escape("" if x is None else str(x))


def _unavailable(x):
    # Compared by value: scoring.UNAVAILABLE read back from JSON is an equal but distinct str.
    return isinstance(x, str) and x == UNAVAILABLE


def _rate_of(value):
    return value["offered"] if isinstance(value, dict) else value


def _goodput_of(value):
    """Goodput from highest_qualifying's result, else None or UNAVAILABLE unchanged."""
    return value.get("useful_per_gpu") if isinstance(value, dict) else value


def fmt_rate(x):
    if x is None:
        return "none"
    if _unavailable(x):
        return "unavailable"
    return f"{x:g}"


def fmt_goodput(x):
    if x is None:
        return "none"
    if _unavailable(x):
        return "unavailable"
    return f"{x:.4g}"


def fmt_ms(x):
    if x is None:
        return "-"
    if _unavailable(x):
        return "unavailable"
    return f"{x:,.0f}" if x >= 100 else f"{x:.1f}"


def fmt_share(x):
    if x is None:
        return "-"
    if _unavailable(x):
        return "unavailable"
    v = x * 100
    return f"{v:.0f}%" if abs(v - round(v)) < 0.05 else f"{v:.1f}%"


def term(key, text=None):
    """A clickable term: opens GLOSSARY[key] in the shared popover."""
    entry = GLOSSARY[key]
    return (f'<button type="button" class="term" data-term="{_esc(key)}" aria-haspopup="dialog" '
            f'aria-expanded="false">{_esc(text or entry["term"])}</button>')


# ---------------------------------------------------------------------------------------------
# Figures for the glossary popovers. One drawn from this report's values says so; others are
# labelled "Schematic".

# Cloudscape tokens: text-body-default, -body-secondary, link-default, status-error, charts-line-grid, status-success.
INK, MUTED, ACCENT, BAD, LINE, OK = "#0f141a", "#424650", "#006ce0", "#db0000", "#dedee3", "#00802f"


def _svg(width, height, body, label, caption):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" class="fig" role="img" viewBox="0 0 {width} {height}" '
            f'aria-label="{_esc(label)}">{body}</svg><p class="cap">{_esc(caption)}</p>')


def _text(x, y, s, size=11, anchor="start", fill=MUTED, weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
            f'font-weight="{weight}">{_esc(s)}</text>')


def fig_ttft(ctx):
    bound = ctx["objective"][0]
    body = (f'<line x1="20" y1="40" x2="300" y2="40" stroke="{LINE}" stroke-width="2"/>'
            f'<circle cx="20" cy="40" r="5" fill="{INK}"/>'
            f'<line x1="20" y1="40" x2="190" y2="40" stroke="{ACCENT}" stroke-width="4"/>'
            + "".join(f'<rect x="{190 + i * 14}" y="33" width="8" height="14" rx="2" fill="{ACCENT}" opacity="{1 if i == 0 else 0.5}"/>' for i in range(8))
            + _text(20, 66, "request sent") + _text(194, 66, "first token", anchor="middle", fill=INK)
            + _text(105, 26, "TTFT", anchor="middle", fill=ACCENT, weight="600"))
    return _svg(320, 74, body, "time to first token", f"Schematic. A call passes when its TTFT is at most {bound:g} ms.")


def fig_mean_tpot(ctx):
    bound = ctx["objective"][1]
    xs = [30, 55, 80, 105, 130, 215, 240, 265]
    body = ("".join(f'<circle cx="{x}" cy="34" r="5" fill="{ACCENT}"/>' for x in xs)
            + f'<line x1="30" y1="54" x2="265" y2="54" stroke="{INK}" stroke-width="1.5"/>'
            f'<line x1="30" y1="49" x2="30" y2="59" stroke="{INK}"/><line x1="265" y1="49" x2="265" y2="59" stroke="{INK}"/>'
            + _text(148, 72, "time from first to last chunk", anchor="middle", fill=INK)
            + _text(30, 18, "first chunk", anchor="middle") + _text(265, 18, "last chunk", anchor="middle"))
    return _svg(300, 80, body, "mean time per output token",
                f"Schematic. mean TPOT is the time from the first chunk to the last chunk, divided by the {len(xs) - 1} tokens that arrived after the first chunk. "
                f"A call passes when the result is at most {bound:g} ms.")


def fig_itl_p99(ctx):
    bound = ctx["objective"][1]
    heights = [70, 64, 58, 50, 40, 30, 22, 15, 10, 7, 5, 4, 3, 3, 2, 2]
    body = ("".join(f'<rect x="{20 + i * 17}" y="{90 - h}" width="13" height="{h}" fill="{ACCENT}" opacity="0.7"/>' for i, h in enumerate(heights))
            + f'<line x1="15" y1="90" x2="300" y2="90" stroke="{INK}"/>'
            f'<line x1="207" y1="12" x2="207" y2="92" stroke="{INK}" stroke-dasharray="4 3"/>'
            f'<line x1="250" y1="12" x2="250" y2="92" stroke="{BAD}" stroke-width="2"/>'
            + _text(203, 22, "p99", anchor="end", fill=INK, weight="600") + _text(254, 22, f"{bound:g} ms", fill=BAD, weight="600")
            + _text(15, 106, "per-token gap (ms), short to long"))
    return _svg(320, 112, body, "pooled inter-token latency p99",
                "Schematic. Each bar counts per-token gap values of one length, pooled from every completed call of one seed at one rate. "
                "99% of the values are at or left of p99 (dashed). p99 must be at or left of the TPOT bound (red).")


def fig_worst_itl(ctx):
    bound = ctx["objective"][1]
    xs = [20, 38, 56, 74, 92, 205, 223, 241, 259, 277]
    body = ("".join(f'<circle cx="{x}" cy="40" r="5" fill="{ACCENT}"/>' for x in xs)
            + f'<line x1="97" y1="40" x2="200" y2="40" stroke="{BAD}" stroke-width="3"/>'
            + _text(148, 26, "longest gap", anchor="middle", fill=BAD, weight="600")
            + _text(148, 62, f"compared with {bound:g} ms as it is", anchor="middle", fill=INK))
    return _svg(300, 70, body, "longest gap in one response",
                "Schematic. One gap longer than the TPOT bound fails the call, even when every other gap is short. "
                "The gap is not divided by the number of tokens it delivered.")


def fig_ttft_percentile(ctx):
    n = 10
    k = math.ceil(n * 0.9)
    body = ("".join(f'<rect x="{20 + i * 26}" y="{80 - (12 + i * 6)}" width="20" height="{12 + i * 6}" '
                    f'fill="{INK if i + 1 == k else ACCENT}" opacity="{1 if i + 1 == k else 0.55}"/>' for i in range(n))
            + f'<line x1="15" y1="80" x2="285" y2="80" stroke="{INK}"/>'
            + _text(20 + (k - 1) * 26 + 10, 96, f"#{k}", anchor="middle", fill=INK, weight="600"))
    return _svg(300, 102, body, "TTFT percentile by nearest rank",
                f"Schematic. Ten calls sorted by TTFT, smallest first. 10 multiplied by 0.9 is 9, so p90 is the value at position {k}.")


def _numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def fig_attainment(ctx):
    target = ctx.get("attainment") or 0.9
    rows = ctx.get("evidence", [])
    rule = next((d for d in ("worst-itl", "mean") if any(_numeric(r.get(f"{d}_attainment")) for r in rows)), None)
    if rule is None:
        return None
    row = min((r for r in rows if _numeric(r.get(f"{rule}_attainment"))), key=lambda r: r[f"{rule}_attainment"])
    share = row[f"{rule}_attainment"]
    good = round(share * row["planned_calls"])
    w = 260
    body = (f'<rect x="20" y="30" width="{w}" height="18" rx="4" fill="{LINE}"/>'
            f'<rect x="20" y="30" width="{w * share:.1f}" height="18" rx="4" fill="{OK if share >= target else BAD}"/>'
            f'<line x1="{20 + w * target:.1f}" y1="22" x2="{20 + w * target:.1f}" y2="56" stroke="{INK}" stroke-width="2"/>'
            + _text(20 + w * target, 16, f"target {fmt_share(target)}", anchor="middle", fill=INK, weight="600")
            + _text(20, 70, "0%") + _text(280, 70, "100%", anchor="end"))
    return _svg(300, 76, body, "attainment against the target",
                f"From this report's data, the row with the lowest share: {row['architecture']}, {_per_second(row['offered'])}, "
                f"seed {row['seed']}, {rule} rule. {good} of {row['planned_calls']} planned calls were good, which is {fmt_share(share)}.")


def fig_qualifying_rate(ctx):
    rows = ctx.get("evidence", [])
    archs = sorted({r["architecture"] for r in rows})
    if not archs:
        return None
    arch = archs[0]
    mine = [r for r in rows if r["architecture"] == arch]
    rates = sorted({r["offered"] for r in mine})
    seeds = sorted({r["seed"] for r in mine}, key=lambda v: (not _numeric(v), v if _numeric(v) else str(v)))
    if not rates or len(rates) > 8 or len(seeds) > 4:
        return None
    qualified = _rate_of(ctx["qualified"].get((arch, "mean")))
    x0, dx, y0, dy = 92, 32, 34, 20
    parts = [_text(x0 + j * dx, 18, fmt_rate(rate), anchor="middle", fill=INK) for j, rate in enumerate(rates)]
    parts.append(_text(x0 - 26, 18, "tasks/s", anchor="end"))
    for i, seed in enumerate(seeds):
        parts.append(_text(x0 - 14, y0 + i * dy + 4, f"seed {seed}", anchor="end"))
        for j, rate in enumerate(rates):
            hit = [r for r in mine if r["offered"] == rate and r["seed"] == seed]
            passed = bool(hit) and hit[0].get("mean_pass") is True
            parts.append(f'<circle cx="{x0 + j * dx}" cy="{y0 + i * dy}" r="6" fill="{OK}"/>' if passed else
                         f'<circle cx="{x0 + j * dx}" cy="{y0 + i * dy}" r="5" fill="#fff" stroke="{BAD}" stroke-width="2"/>')
    if _numeric(qualified) and qualified in rates:
        j = rates.index(qualified)
        parts.append(f'<rect x="{x0 + j * dx - 12}" y="{y0 - 12}" width="24" height="{dy * (len(seeds) - 1) + 24}" rx="6" fill="none" stroke="{INK}" stroke-width="2"/>')
    return _svg(x0 + dx * len(rates), y0 + dy * len(seeds), "".join(parts), "qualifying rate walk",
                f"From this report's data: {arch}, mean rule. A filled green dot means the seed passed at that rate; an open red ring means it failed. "
                f"The qualifying rate, {_per_second(qualified)}, is boxed.")


def fig_goodput(ctx):
    rows = ctx.get("evidence", [])
    for arch in sorted(ctx["archs"]):
        rate = _rate_of(ctx["qualified"].get((arch, "mean")))
        at = [r for r in rows if r["architecture"] == arch and r["offered"] == rate and _numeric(r.get("mean_goodput"))]
        if at:
            r = min(at, key=lambda x: x["mean_goodput"])
            break
    else:
        return None
    good = round(r["mean_attainment"] * r["planned_calls"])
    window, gpus = r["measurement_including_drain_s"], r["gpu_count"]
    body = _text(10, 26, f"{good} good calls / {window:.1f} s / {gpus} GPUs = {fmt_goodput(r['mean_goodput'])}", size=14, fill=ACCENT, weight="600")
    return _svg(320, 36, body, "goodput calculation",
                f"From this report's data: {r['architecture']} at its mean-rule qualifying rate, {_per_second(rate)}, "
                f"seed {r['seed']} (the smallest goodput across seeds). Good calls are divided by the measurement window, "
                f"including the wait for the last calls, and by the GPUs. The unit is useful calls per second per GPU.")


# Keep in step with README "Three TPOT definitions" and "How a qualifying rate and its goodput are computed".
GLOSSARY = {
    "ttft": {"term": "TTFT", "figure": fig_ttft,
             "text": "TTFT (time to first token) is the time from sending a request until the first part of the answer arrives. Each call's TTFT is compared with the TTFT bound."},
    "tpot": {"term": "TPOT", "figure": fig_mean_tpot,
             "text": "TPOT (time per output token) is the time each token of the answer takes to arrive after the first part of the answer. A larger value means a slower answer. This page judges it in three ways, called rules: mean, itl-p99 and worst-itl."},
    "mean": {"term": "mean", "figure": fig_mean_tpot,
             "text": "The mean rule takes the time from a call's first chunk to its last chunk and divides it by the number of tokens that arrived after the first chunk. A chunk is one piece of the streamed answer, and it can hold several tokens. The call is good when this value is at most the TPOT bound."},
    "itl-p99": {"term": "itl-p99", "figure": fig_itl_p99,
                "text": "The itl-p99 rule (99th percentile of the inter-token latency) looks at one seed at one rate. It collects every gap between two consecutive chunks from every completed call. Each gap is divided by the number of tokens in the later chunk, and that value is counted once per token. The 99th percentile (p99) of all these values must be at most the TPOT bound. Under this rule, a call only needs to pass the TTFT bound to be good."},
    "worst-itl": {"term": "worst-itl", "figure": fig_worst_itl,
                  "text": "The worst-itl rule takes the single longest gap between two consecutive chunks of one call. The gap is not divided by the number of tokens in the chunk. The call is good when that gap is at most the TPOT bound."},
    "ttft-percentile": {"term": "TTFT p90", "figure": fig_ttft_percentile,
                        "text": "TTFT p90 is the 90th percentile of TTFT over the completed calls. To find it, sort the values from smallest to largest, multiply the number of calls by 0.9, round up to a whole number, and take the value at that position (the nearest-rank method)."},
    "attainment": {"term": "attainment", "figure": fig_attainment,
                   "text": "Attainment is the share of planned calls that were good under a rule. Failed, truncated and skipped calls stay in the count of planned calls. A seed passes at a rate when its attainment reaches the target."},
    "qualifying-rate": {"term": "qualifying rate", "figure": fig_qualifying_rate,
                        "text": "The qualifying rate is the highest offered rate at which every seed that ran it passed, and every lower planned rate also passed. The rates are checked from the lowest up, and the check stops at the first rate that fails. A planned rate with no result counts as failing."},
    "goodput": {"term": "goodput", "figure": fig_goodput,
                "text": "Goodput is the number of good calls at the qualifying rate, divided by the measurement time and by the number of GPUs. The measurement time includes the wait for the last calls to finish. The unit is useful calls per second per GPU. With several seeds, the smallest value is reported."},
    "offered-rate": {"term": "offered rate", "figure": None,
                     "text": "The offered rate is how many new tasks per second the load generator starts, whether or not the server keeps up. In this traffic shape, one task is one call."},
    "seed": {"term": "seed", "figure": None,
             "text": "A seed is a number that fixes the random order and timing of the generated traffic. Repeating a rate with different seeds shows whether a result holds for more than one arrival pattern."},
    "unified": {"term": "unified", "figure": None,
                "text": "In the unified stack, each copy of the server processes the whole prompt (prefill) and then generates the answer token by token (decode), on the same GPUs. A router, a program that forwards requests, spreads the requests across the copies."},
    "disaggregated": {"term": "disaggregated", "figure": None,
                      "text": "In the disaggregated stack, one set of GPUs processes prompts (prefill) and another set generates answers (decode). Between the two steps, the prompt's KV cache (the key and value tensors the model computed for the prompt) is copied from the prefill GPUs to the decode GPUs."},
    "promise-map": {"term": "promise map", "figure": None,
                    "text": "A promise map is a grid that repeats the verdict for many pairs of TTFT and TPOT bounds. Each cell shows which stack has the higher qualifying rate for that pair. It shows how the answer changes when the objective changes."},
    "tie": {"term": "tie", "figure": None,
            "text": "Tie means both stacks have the same qualifying rate at that objective."},
    "neither": {"term": "neither", "figure": None,
                "text": "Neither means both stacks were measured and neither met the objective at any offered rate."},
    "incomparable": {"term": "incomparable", "figure": None,
                     "text": "Incomparable means at least one stack was never judged under this rule, usually because its files have no per-chunk timing. It is a gap in the measurement, not a loss."},
    "unavailable": {"term": "unavailable", "figure": None,
                    "text": "Unavailable means the file has no per-chunk timing, so the itl-p99 and worst-itl rules cannot judge it. The mean rule still can."},
}


# ---------------------------------------------------------------------------------------------
# Sections.

def _objective_source(ctx):
    given = ctx.get("objective_given") or []
    return "recorded with the runs" + (f", except the {' and '.join(given)} given to 5.collect.py on the command line" if given else "")


def _objective_text(ctx):
    ttft_ms, tpot_ms = ctx["objective"]
    attainment = ctx.get("attainment")
    metric = ctx.get("ttft_metric") or "per-call"
    who = "each call's time to first token (TTFT)" if metric == "per-call" else f"the {metric} of the time to first token (TTFT) at each rate"
    text = (f"{who} is at most {ttft_ms:g} ms, "
            f"the time per output token (TPOT) is at most {tpot_ms:g} ms under the chosen rule")
    if attainment is not None:
        text += f", and at least {fmt_share(attainment)} of the planned calls are good"
    return text


def _winner(ctx, d):
    by_arch = {arch: ctx["qualified"].get((arch, d)) for arch in ctx["archs"]}
    return verdict(by_arch)[1], by_arch


def _per_second(rate):
    return f"{fmt_rate(rate)} task per second" if rate == 1 else f"{fmt_rate(rate)} tasks per second"


RULE_PLAIN = {"mean": "the mean TPOT rule (each call's average time per token)",
              "itl-p99": "the itl-p99 rule (the 99th percentile of the time between tokens, over all calls)",
              "worst-itl": "the worst-itl rule (each call's single longest pause)"}


def _rule_sentence(ctx, d):
    winner, by_arch = _winner(ctx, d)
    rates = {a: _rate_of(v) for a, v in by_arch.items()}
    goodputs = {a: _goodput_of(v) for a, v in by_arch.items()}
    lead = f"Under {RULE_PLAIN[d]},"
    if winner in ARCH_LETTER:
        other = [a for a in rates if a != winner][0]
        if rates[other] is None:
            return (f"{lead} only {winner} met the objective. Its qualifying rate is {_per_second(rates[winner])}, "
                    f"with a goodput of {fmt_goodput(goodputs[winner])} useful calls per second per GPU. "
                    f"{other.capitalize()} met it at no offered rate.")
        return (f"{lead} {winner} has the higher qualifying rate: {_per_second(rates[winner])}, against {_per_second(rates[other])} for {other}. "
                f"Its goodput there is {fmt_goodput(goodputs[winner])} useful calls per second per GPU, against {fmt_goodput(goodputs[other])} for {other}.")
    if winner == "tie":
        rate = next(iter(rates.values()))
        names = sorted(goodputs)
        values = [fmt_goodput(goodputs[a]) for a in names]
        if len(set(values)) == 1:
            gp = f"Both have a goodput of {values[0]} useful calls per second per GPU there."
        else:
            gp = (f"Their goodput there is {values[0]} useful calls per second per GPU for {names[0]} "
                  f"and {' and '.join(f'{v} for {n}' for v, n in zip(values[1:], names[1:]))}.")
        return f"{lead} both stacks have the same qualifying rate, {_per_second(rate)}. {gp}"
    if winner == "neither":
        return f"{lead} neither stack met the objective at any measured rate."
    return f"{lead} the stacks cannot be compared, because at least one was not judged under this rule."


def answer_section(ctx):
    cards = []
    for d in ctx.get("definitions", TPOT_METRICS):
        winner, by_arch = _winner(ctx, d)
        rows = "".join(
            f'<div class="stackline"><div class="sname"><span class="dot {arch}"></span>{_esc(arch)}</div>'
            + (f'<div class="metric"><span class="big">{_esc(fmt_rate(_rate_of(by_arch[arch])))}</span><span class="unit">tasks/s</span></div>'
               f'<div class="metric"><span class="mid">{_esc(fmt_goodput(_goodput_of(by_arch[arch])))}</span><span class="unit">useful calls/s/GPU</span></div>'
               if isinstance(_rate_of(by_arch[arch]), (int, float)) else
               f'<div class="metric wide"><span class="mid">{"not judged" if _unavailable(_rate_of(by_arch[arch])) else "no qualifying rate"}</span>'
               f'<span class="unit">{"no per-chunk timing" if _unavailable(_rate_of(by_arch[arch])) else "failed at every offered rate"}</span></div>')
            + '</div>'
            for arch in sorted(ctx["archs"]))
        cards.append(f'<article class="rule-card w-{_esc(winner)}"><header><h3>{term(RULE_TERM[d], RULE_TITLE[d])} rule</h3>'
                     f'<span class="badge w-{_esc(winner)}">{_esc(WINNER_TEXT.get(winner, winner))}</span></header>'
                     f'{rows}</article>')
    return (f'<section id="answer"><h2>The answer</h2>'
            f'<p class="lead">The objective ({_esc(_objective_source(ctx))}): {_esc(_objective_text(ctx))}. There are three rules for judging {term("tpot")}. '
            f'For each rule, the card shows each stack\'s {term("qualifying-rate")} and its {term("goodput")} at that rate.</p>'
            f'<div class="summary">{"".join(f"<p>{_esc(_rule_sentence(ctx, d))}</p>" for d in ctx.get("definitions", TPOT_METRICS))}</div>'
            f'<div class="cards">{"".join(cards)}</div></section>')


def measured_section(ctx):
    rows = ctx.get("evidence", [])
    rates = sorted({r["offered"] for r in rows})
    seeds = [str(x) for x in sorted({r["seed"] for r in rows}, key=lambda v: (not isinstance(v, (int, float)), v if isinstance(v, (int, float)) else str(v)))]
    instances = sorted({str(r.get("instance_type")) for r in rows if r.get("instance_type")})
    gpus = {r["architecture"]: r.get("gpu_count") for r in rows}
    windows = [r["measurement_including_drain_s"] for r in rows if isinstance(r.get("measurement_including_drain_s"), (int, float))]
    items = [("Stacks", " and ".join(term(a) for a in sorted(ctx["archs"]))),
             ("Traffic shape", _esc(ctx.get("shape", "-")) + ' <small class="muted">(the traffic file the load generator replayed; see the README for each shape)</small>'),
             (term("offered-rate", "Offered rates"), _esc(", ".join(fmt_rate(r) for r in rates) + " tasks/s") if rates else "-"),
             (term("seed", "Seeds"), _esc(", ".join(seeds)) if seeds else "-"),
             ("GPUs per stack", _esc(", ".join(f"{a} {g}" for a, g in sorted(gpus.items()))) if gpus else "-"),
             ("Amazon EC2 instance type", _esc(", ".join(instances)) if instances else "-"),
             ("Time per rate", _esc((f"{min(windows):.0f} s" if round(min(windows)) == round(max(windows)) else f"{min(windows):.0f} to {max(windows):.0f} s") + ", including the wait for the last calls") if windows else "-"),
             ("Objective", f'{term("ttft")} at most {ctx["objective"][0]:g} ms (judged {_esc(ctx.get("ttft_metric") or "per-call")}); '
                           f'{term("tpot")} at most {ctx["objective"][1]:g} ms, judged by each rule; '
                           f'{term("attainment")} at least {_esc(fmt_share(ctx.get("attainment") or 0.9))} of planned calls '
                           f'<small class="muted">({_esc(_objective_source(ctx))})</small>')]
    body = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in items)
    return f'<section id="measured"><h2>What was measured</h2><dl class="facts">{body}</dl></section>'


def verdict_table(ctx):
    archs = sorted(ctx["archs"])
    head = "".join(f'<th scope="col">{_esc(a)}<br><small>{term("qualifying-rate")} (tasks/s)</small></th>'
                   f'<th scope="col">{_esc(a)}<br><small>{term("goodput")} (useful calls/s/GPU)</small></th>' for a in archs)
    rows = []
    for d in ctx.get("definitions", TPOT_METRICS):
        winner, by_arch = _winner(ctx, d)
        cells = "".join(f'<td class="num">{_esc(fmt_rate(_rate_of(by_arch[a])))}</td><td class="num">{_esc(fmt_goodput(_goodput_of(by_arch[a])))}</td>' for a in archs)
        rows.append(f'<tr><th scope="row">{term(RULE_TERM[d], RULE_TITLE[d])}</th>{cells}<td><span class="badge w-{_esc(winner)}">{_esc(WINNER_TEXT.get(winner, winner))}</span></td></tr>')
    return (f'<section id="verdict"><h2>Verdict by TPOT rule</h2><div class="scroll"><table class="verdict">'
            f'<caption>At the objective {_esc(_objective_source(ctx))}: {_esc(_objective_text(ctx))}.</caption>'
            f'<thead><tr><th scope="col">Rule</th>{head}<th scope="col">Result</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>')


def _cell_label(cell, archs):
    winner, by_arch = cell.get("winner"), cell.get("by_arch", {})
    if winner == "neither":
        return "-"
    if winner == "incomparable":
        return "?"
    if winner == "tie":
        return f"= {fmt_rate(_rate_of(by_arch[sorted(archs)[0]]))}"
    return f"{ARCH_LETTER.get(winner, winner[:1].upper())} {fmt_rate(_rate_of(by_arch.get(winner)))}"


def _cell_detail(definition, ttft, tpot, cell, archs):
    by_arch = cell.get("by_arch", {})
    parts = [f"TTFT at most {ttft:g} ms, TPOT at most {tpot:g} ms, {RULE_TITLE.get(definition, definition)} rule."]
    for arch in sorted(archs):
        value = by_arch.get(arch)
        if _unavailable(_rate_of(value)):
            parts.append(f"{arch}: not judged under this rule.")
        elif _rate_of(value) is None:
            parts.append(f"{arch}: does not meet this objective at any measured rate.")
        else:
            parts.append(f"{arch}: qualifying rate {_per_second(_rate_of(value))}, goodput {fmt_goodput(_goodput_of(value))} useful calls per second per GPU.")
    parts.append(WINNER_TEXT.get(cell.get("winner"), str(cell.get("winner"))) + ".")
    return " ".join(parts)


def promise_map(definition, cells, ttft_axis, tpot_axis, archs, objective=None, map_id="m"):
    """One grid as HTML: columns are TTFT bounds, rows TPOT bounds, one button per cell."""
    cols = len(ttft_axis)
    out = [f'<div class="pmap" id="{_esc(map_id)}" data-rule="{_esc(definition)}" style="--cols:{cols}">',
           '<div class="corner">rows: TPOT bound (ms)<br>columns: TTFT bound (ms)</div>']
    out += [f'<div class="colh">{ttft:g}</div>' for ttft in ttft_axis]
    for tpot in tpot_axis:
        out.append(f'<div class="rowh">{tpot:g}</div>')
        for ttft in ttft_axis:
            cell = cells.get((ttft, tpot), {})
            winner = cell.get("winner") or "neither"
            is_objective = objective is not None and (ttft, tpot) == tuple(objective)
            detail = _cell_detail(definition, ttft, tpot, cell, archs)
            out.append(f'<button type="button" class="cell w-{_esc(winner)}{" objective" if is_objective else ""}" '
                       f'data-detail="{_esc(detail)}" data-objective="{"1" if is_objective else "0"}" '
                       f'aria-label="{_esc(detail)}">{_esc(_cell_label(cell, archs))}</button>')
    out.append("</div>")
    return "".join(out)


def _legend():
    items = [("w-unified", "U", "unified reaches the higher rate"), ("w-disaggregated", "D", "disaggregated reaches the higher rate"),
             ("w-tie", "=", "tie"), ("w-neither", "-", "neither meets the objective"), ("w-incomparable", "?", "incomparable")]
    chips = "".join(f'<span class="chip"><span class="sw {c}">{_esc(sym)}</span>{_esc(t)}</span>' for c, sym, t in items)
    return (f'<div class="legend">{chips}<span class="chip"><span class="sw objective-sw"></span>objective used for the verdict</span></div>'
            f'<p class="hint">The number in a cell is the higher of the two qualifying rates, in tasks per second; for a tie, it is the rate both stacks share. '
            f'Definitions: {term("tie")}, {term("neither")}, {term("incomparable")}.</p>')


def maps_section(ctx, prefix="r"):
    definitions = ctx.get("definitions", TPOT_METRICS)
    tabs, panes = [], []
    for i, d in enumerate(definitions):
        tid = f"{prefix}-tab-{i}"
        tabs.append(f'<input type="radio" name="{prefix}-tabs" id="{tid}"{" checked" if i == 0 else ""}><label for="{tid}">{_esc(RULE_TITLE[d])}</label>')
        panes.append(f'<div class="pane">{promise_map(d, ctx["grid"][d], ctx["ttft_axis"], ctx["tpot_axis"], ctx["archs"], ctx.get("objective"), f"{prefix}-map-{i}")}</div>')
    return (f'<section id="maps"><h2>Promise maps</h2>'
            f'<p class="lead">A {term("promise-map")} repeats the verdict for many pairs of TTFT and TPOT bounds. '
            f'Columns are TTFT bounds; the bound gets looser to the right. Rows are TPOT bounds; the bound gets looser further down. '
            f'Select a cell to read both stacks\' values.</p>{_legend()}'
            f'<div class="tabs">{"".join(tabs)}<div class="panes">{"".join(panes)}</div></div>'
            f'<div class="detail" aria-live="polite">Select a cell.</div></section>')


def _flag(value, bound):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or bound is None:
        return ""
    return " over" if value > bound else ""


def _pass_cell(row, d, target, tpot_bound):
    passed, share = row.get(f"{d}_pass"), row.get(f"{d}_attainment")
    if _unavailable(passed):
        return '<td class="rule"><span class="chip-na">unavailable</span></td>'
    extra = ""
    if d == "itl-p99":
        p99 = row.get("itl_ms_p99")
        extra = f'<small class="{_flag(p99, tpot_bound).strip()}">p99 {_esc(fmt_ms(p99))} ms</small>'
    low = isinstance(share, (int, float)) and share < target
    return (f'<td class="rule"><span class="pf {"pass" if passed is True else "fail"}">{"pass" if passed is True else "fail"}</span> '
            f'<span class="share{" over" if low else ""}">{_esc(fmt_share(share))}</span>{extra}</td>')


def evidence_section(ctx):
    rows = ctx.get("evidence", [])
    if not rows:
        return '<section id="evidence"><h2>Evidence for each rate and seed</h2><p class="empty">No rows.</p></section>'
    ttft_bound, tpot_bound = ctx["objective"]
    target = ctx.get("attainment") or 0.9
    metric = ctx.get("ttft_metric") or "per-call"
    ttft_key = {"p95": "p95_ttft_ms_success_only", "p99": "p99_ttft_ms_success_only"}.get(metric, "p90_ttft_ms_success_only")
    ttft_label = {"p95": "TTFT p95", "p99": "TTFT p99"}.get(metric, "TTFT p90")
    definitions = ctx.get("definitions", TPOT_METRICS)
    head = ("<th>Stack</th><th>Rate<br><small>tasks/s</small></th><th>" + term("seed", "Seed") + "</th>"
            "<th>Calls<br><small>completed / planned</small></th>"
            f'<th>{term("ttft-percentile", ttft_label)}<br><small>ms, completed calls</small></th>'
            "<th>Mean TPOT p90<br><small>ms, over completed calls</small></th>"
            "<th>Longest gap p90<br><small>ms, longest gap of each completed call</small></th>"
            + "".join(f'<th>{term(RULE_TERM[d], RULE_TITLE[d])}<br><small>{"pass or fail, share passing TTFT, pooled p99" if d == "itl-p99" else "pass or fail, " + term("attainment")}</small></th>' for d in definitions))
    body = []
    for r in sorted(rows, key=lambda x: (x["architecture"], x["offered"], x["seed"] if isinstance(x["seed"], (int, float)) else 0, str(x["seed"]))):
        body.append(f'<tr class="{_esc(r["architecture"])}"><td class="stack"><span class="dot {_esc(r["architecture"])}"></span>{_esc(r["architecture"])}</td>'
                    f'<td class="num">{_esc(fmt_rate(r["offered"]))}</td><td class="num">{_esc(r["seed"])}</td>'
                    f'<td class="num">{_esc(r["completed_calls"])} / {_esc(r["planned_calls"])}</td>'
                    f'<td class="num{_flag(r.get(ttft_key), ttft_bound)}">{_esc(fmt_ms(r.get(ttft_key)))}</td>'
                    f'<td class="num{_flag(r.get("p90_tpot_ms_success_only"), tpot_bound)}">{_esc(fmt_ms(r.get("p90_tpot_ms_success_only")))}</td>'
                    f'<td class="num{_flag(r.get("longest_gap_ms_p90"), tpot_bound)}">{_esc(fmt_ms(r.get("longest_gap_ms_p90")))}</td>'
                    + "".join(_pass_cell(r, d, target, tpot_bound) for d in definitions) + "</tr>")
    return (f'<section id="evidence"><h2>Evidence for each rate and seed</h2>'
            f'<p class="lead">Each row is one stack at one offered rate with one seed. A red value is over its bound, or under the {_esc(fmt_share(target))} target. '
            f'Red values show why that seed failed at that rate. A rate qualifies only when every seed passed there and at every lower rate.</p>'
            f'<div class="scroll tall"><table class="evidence"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></section>')


# ---------------------------------------------------------------------------------------------
# Page.

def _glossary_templates(blocks):
    # A figure drawn from a shape's own data gets one template per shape; the page script picks the one for the shape the term sits in.
    out = []
    for key, entry in GLOSSARY.items():
        for prefix, ctx in (blocks if entry["figure"] is not None else [(None, None)]):
            figure = ""
            if ctx is not None:
                try:
                    figure = entry["figure"](ctx) or ""
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    figure = ""
            tid = f"g-{prefix}-{key}" if prefix else f"g-{key}"
            out.append(f'<template id="{_esc(tid)}"><h4>{_esc(entry["term"])}</h4><p>{_esc(entry["text"])}</p>{figure}</template>')
    return "".join(out)


_STYLE = """
/* Cloudscape design tokens (light mode, Visual Refresh), copied from
   @cloudscape-design/design-tokens 3.0.114; https://cloudscape.design/foundation/visual-foundation/design-tokens/
   Unified and disaggregated use chart palette categorical-1 and -5, distinct under red-green
   colour vision deficiency; every cell also carries a letter or symbol. */
:root{
--text:#0f141a;--text-2:#424650;--text-inactive:#656871;--divider:#c6c6cd;--grid:#dedee3;--popover-border:#b4b4bb;
--link:#006ce0;--bg-page:#f6f6f9;--bg-card:#ffffff;--bg-info:#f0fbff;
--ok:#00802f;--ok-bg:#effff1;--err:#db0000;--err-bg:#fff5f5;
--hdr-bg:#0f141a;--hdr-text:#ebebf0;--hdr-text-2:#c6c6cd;
--u:#688ae8;--u-ink:#015b9d;--d:#e07941;--d-ink:#983c02;--tie:#8456ce;--tie-ink:#6b40b2;--na:#8c8c94;
--font:'Open Sans','Helvetica Neue',Roboto,Arial,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
--fs-body:14px;--lh-body:20px;--fs-s:12px;--lh-s:16px;--fs-xl:24px;--lh-xl:30px;--fs-l:20px;--lh-l:24px;--fs-hs:16px;--lh-hs:20px;
--r-container:16px;--r-button:20px;--r-badge:4px;--r-item:8px;
--sp-xs:8px;--sp-s:12px;--sp-m:16px;--sp-l:20px;--sp-xl:24px;
--shadow-pop:0px 1px 1px 1px #e9ebed,0px 6px 36px #0007161a}
*{box-sizing:border-box}
body{margin:0;font-family:var(--font);font-size:var(--fs-body);line-height:var(--lh-body);color:var(--text);background:var(--bg-page)}
.top{background:var(--hdr-bg);color:var(--hdr-text);padding:var(--sp-xl) 0 var(--sp-l)}
.wrap{max-width:1180px;margin:0 auto;padding:0 var(--sp-xl)}
.top h1{margin:0 0 4px;font-size:var(--fs-xl);line-height:var(--lh-xl);font-weight:700}
.top p{margin:0;color:var(--hdr-text-2)}
nav.toc{display:flex;gap:var(--sp-l);flex-wrap:wrap;margin-top:var(--sp-s)}
nav.toc a{color:var(--hdr-text);text-decoration:none;border-bottom:1px solid var(--hdr-text-2)}
nav.toc a:hover{color:#fff}
section{background:var(--bg-card);border:1px solid var(--divider);border-radius:var(--r-container);padding:var(--sp-l) var(--sp-l);margin:var(--sp-l) 0}
h2{margin:0 0 var(--sp-xs);font-size:var(--fs-l);line-height:var(--lh-l);font-weight:700}
h3{margin:0;font-size:var(--fs-hs);line-height:var(--lh-hs);font-weight:700}
.muted{color:var(--text-2)}
.lead{margin:0 0 var(--sp-m);color:var(--text-2);max-width:900px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:var(--sp-m)}
.rule-card{border:1px solid var(--divider);border-radius:var(--r-item);padding:var(--sp-m);border-top:4px solid var(--na)}
.rule-card.w-unified{border-top-color:var(--u)}.rule-card.w-disaggregated{border-top-color:var(--d)}.rule-card.w-tie{border-top-color:var(--tie)}
.rule-card header{display:flex;flex-direction:column;gap:var(--sp-xs);margin-bottom:var(--sp-xs)}
.stackline{display:grid;grid-template-columns:1fr auto auto;column-gap:var(--sp-l);align-items:end;padding:var(--sp-xs) 0;border-top:1px solid var(--grid)}
.metric{text-align:right;font-variant-numeric:tabular-nums}
.metric.wide{grid-column:2/4}
.metric .unit{display:block;font-size:var(--fs-s);line-height:var(--lh-s);color:var(--text-2)}
.big{font-size:var(--fs-xl);line-height:var(--lh-xl);font-weight:700}.mid{font-size:var(--fs-hs);font-weight:700;color:var(--text-2)}
.summary{margin:0 0 var(--sp-m);padding:var(--sp-s) var(--sp-m);border-left:4px solid var(--link);background:var(--bg-info);border-radius:var(--r-item)}
.summary p{margin:4px 0}
.sname{font-weight:700;white-space:nowrap}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--na);margin-right:6px}
.dot.unified{background:var(--u)}.dot.disaggregated{background:var(--d);border-radius:2px}
.badge{display:inline-block;max-width:100%;align-self:flex-start;font-size:var(--fs-s);line-height:var(--lh-s);font-weight:700;padding:2px 8px;border-radius:var(--r-badge);background:var(--bg-page);color:var(--text-2);border:1px solid var(--grid)}
.badge.w-unified{color:var(--u-ink);border-color:var(--u)}.badge.w-disaggregated{color:var(--d-ink);border-color:var(--d)}.badge.w-tie{color:var(--tie-ink);border-color:var(--tie)}
dl.facts{display:grid;grid-template-columns:max-content 1fr;gap:var(--sp-xs) var(--sp-l);margin:0}
dl.facts dt{color:var(--text-2);font-weight:700}dl.facts dd{margin:0}
.scroll{overflow-x:auto}.scroll.tall{max-height:70vh;overflow:auto;border:1px solid var(--divider);border-radius:var(--r-item)}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:var(--fs-body)}
caption{text-align:left;color:var(--text-2);padding-bottom:var(--sp-xs)}
th,td{padding:var(--sp-xs) var(--sp-s);text-align:left;border-bottom:1px solid var(--grid);vertical-align:top}
tbody th{white-space:nowrap}
thead th{background:var(--bg-card);color:var(--text-2);font-weight:700;border-bottom:1px solid var(--divider);position:sticky;top:0;z-index:1}
th small,td small{display:block;font-weight:400;color:var(--text-inactive);font-size:var(--fs-s);line-height:var(--lh-s)}
tbody tr:nth-child(even) td,tbody tr:nth-child(even) th{background:var(--bg-page)}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.over{color:var(--err);font-weight:700}
td.rule,td.stack{white-space:nowrap}
.pf{display:inline-block;min-width:38px;text-align:center;font-size:var(--fs-s);line-height:var(--lh-s);font-weight:700;border-radius:var(--r-badge);padding:1px 6px;margin-right:6px}
.pf.pass{background:var(--ok-bg);color:var(--ok);border:1px solid var(--ok)}.pf.fail{background:var(--err-bg);color:var(--err);border:1px solid var(--err)}
.share{font-variant-numeric:tabular-nums}
.chip-na{font-size:var(--fs-s);color:var(--text-inactive)}
.legend{display:flex;flex-wrap:wrap;gap:var(--sp-xs) var(--sp-m);margin:4px 0 6px}
.chip{display:inline-flex;align-items:center;gap:6px}
.sw{min-width:24px;height:20px;border-radius:var(--r-badge);display:inline-flex;align-items:center;justify-content:center;font-size:var(--fs-s);font-weight:700}
.hint{margin:0 0 var(--sp-s);font-size:var(--fs-s);line-height:var(--lh-s);color:var(--text-2)}
.tabs{display:flex;flex-wrap:wrap;gap:var(--sp-xs)}
.tabs>input{position:absolute;opacity:0;pointer-events:none}
.tabs>label{padding:4px var(--sp-m);border-radius:var(--r-button);border:1px solid var(--text-2);cursor:pointer;font-weight:700;color:var(--text-2);background:var(--bg-card)}
.tabs>input:checked+label{background:var(--link);border-color:var(--link);color:#fff}
.tabs>input:focus-visible+label{outline:2px solid var(--link);outline-offset:2px}
.panes{flex-basis:100%;margin-top:var(--sp-s);overflow-x:auto}
.pane{display:none}
.tabs>input:nth-of-type(1):checked~.panes>.pane:nth-child(1),.tabs>input:nth-of-type(2):checked~.panes>.pane:nth-child(2),.tabs>input:nth-of-type(3):checked~.panes>.pane:nth-child(3){display:block}
.pmap{display:grid;grid-template-columns:132px repeat(var(--cols),minmax(54px,1fr));gap:3px;min-width:max-content;font-variant-numeric:tabular-nums}
.pmap .corner{font-size:var(--fs-s);line-height:var(--lh-s);color:var(--text-2);align-self:end;padding:0 4px 4px 0}
.pmap .colh{font-size:var(--fs-s);color:var(--text-2);text-align:center;align-self:end;padding-bottom:4px}
.pmap .rowh{font-size:var(--fs-s);color:var(--text-2);text-align:right;padding-right:var(--sp-xs);align-self:center}
.cell{height:34px;border:1px solid transparent;border-radius:var(--r-badge);font:inherit;font-size:var(--fs-s);font-weight:700;cursor:pointer;white-space:nowrap;padding:0 4px}
.w-unified.cell,.sw.w-unified{background:color-mix(in srgb,var(--u) 22%,#fff);color:var(--u-ink);border-color:color-mix(in srgb,var(--u) 45%,#fff)}
.w-disaggregated.cell,.sw.w-disaggregated{background:color-mix(in srgb,var(--d) 22%,#fff);color:var(--d-ink);border-color:color-mix(in srgb,var(--d) 45%,#fff)}
.w-tie.cell,.sw.w-tie{background:color-mix(in srgb,var(--tie) 14%,#fff);color:var(--tie-ink);border-color:color-mix(in srgb,var(--tie) 35%,#fff)}
.w-neither.cell,.sw.w-neither{background:var(--bg-page);color:var(--text-inactive);border-color:var(--grid)}
.w-incomparable.cell,.sw.w-incomparable{background:repeating-linear-gradient(45deg,var(--bg-page),var(--bg-page) 4px,var(--grid) 4px,var(--grid) 8px);color:var(--text-2);border-color:var(--grid)}
.cell.objective,.objective-sw{box-shadow:inset 0 0 0 2px var(--text)}
.cell:hover{filter:brightness(.96)}.cell:focus-visible{outline:2px solid var(--link);outline-offset:1px}
.cell.selected{box-shadow:inset 0 0 0 2px var(--link)}
.detail{margin-top:var(--sp-s);padding:var(--sp-xs) var(--sp-s);border-radius:var(--r-item);background:var(--bg-info);border:1px solid color-mix(in srgb,var(--link) 30%,#fff)}
.detail a,footer a{color:var(--link)}
.term{font:inherit;color:inherit;background:none;border:0;padding:0;cursor:pointer;border-bottom:1px dashed var(--link);line-height:inherit}
.term:hover{color:var(--link)}
.term:focus-visible{outline:2px solid var(--link);outline-offset:2px;border-radius:2px}
#pop{position:absolute;z-index:10;max-width:360px;width:calc(100vw - 32px);background:var(--bg-card);border:1px solid var(--popover-border);border-radius:var(--r-item);
box-shadow:var(--shadow-pop);padding:var(--sp-s) var(--sp-m);color:var(--text);font-weight:400;text-align:left}
#pop[hidden]{display:none}
#pop h4{margin:0 0 6px;font-size:var(--fs-hs);line-height:var(--lh-hs);font-weight:700}#pop p{margin:0 0 var(--sp-xs)}
#pop .fig{display:block;width:100%;height:auto;margin-top:6px;overflow:visible}
#pop .cap{font-size:var(--fs-s);line-height:var(--lh-s);color:var(--text-2);margin:4px 0 0}
#pop .close{position:absolute;top:6px;right:8px;border:0;background:none;font-size:18px;cursor:pointer;color:var(--text-2)}
footer{color:var(--text-2);font-size:var(--fs-s);padding:var(--sp-xs) 0 var(--sp-xl)}
"""

_SCRIPT = """
(function(){
var pop=document.getElementById('pop'),opener=null;
function close(){if(pop.hidden)return;pop.hidden=true;if(opener){opener.setAttribute('aria-expanded','false');opener.focus();}opener=null;}
function open(btn){var b=btn.closest('.shape-block'),t=(b&&document.getElementById('g-'+b.dataset.shape+'-'+btn.dataset.term))||document.getElementById('g-'+btn.dataset.term);if(!t)return;
pop.innerHTML='<button type="button" class="close" aria-label="Close">\\u00d7</button>';pop.appendChild(t.content.cloneNode(true));
pop.hidden=false;opener=btn;btn.setAttribute('aria-expanded','true');
var r=btn.getBoundingClientRect(),w=pop.offsetWidth,h=pop.offsetHeight,sx=window.scrollX,sy=window.scrollY;
var left=Math.min(Math.max(8,r.left),document.documentElement.clientWidth-w-8);
var top=r.bottom+8;if(top+h>window.innerHeight&&r.top-h-8>0)top=r.top-h-8;
pop.style.left=(left+sx)+'px';pop.style.top=(top+sy)+'px';pop.querySelector('.close').focus();}
document.addEventListener('click',function(e){var b=e.target.closest('.term');
if(b){e.preventDefault();if(opener===b){close();}else{close();open(b);}return;}
if(e.target.closest('#pop .close')){close();return;}
if(!e.target.closest('#pop'))close();
var c=e.target.closest('.cell');if(c){var sec=c.closest('section');sec.querySelectorAll('.cell.selected').forEach(function(x){x.classList.remove('selected');});
c.classList.add('selected');var d=sec.querySelector('.detail');if(d){d.textContent=c.dataset.detail;
var ev=c.closest('.shape-block').querySelector('section[id$="-evidence"]');if(c.dataset.objective==='1'&&ev){var a=document.createElement('a');a.href='#'+ev.id;a.textContent=' See the evidence for this objective.';d.appendChild(a);}}}});
document.addEventListener('keydown',function(e){if(e.key==='Escape')close();});
})();
"""


def _page(title, subtitle, body, blocks, toc=True):
    nav = ('<nav class="toc"><a href="#r-answer">Answer</a><a href="#r-measured">What was measured</a><a href="#r-verdict">Verdict</a>'
           '<a href="#r-maps">Promise maps</a><a href="#r-evidence">Evidence</a></nav>') if toc else ""
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_esc(title)}</title><style>{_STYLE}</style></head><body>'
            f'<div class="top"><div class="wrap"><h1>{_esc(title)}</h1><p>{_esc(subtitle)}</p>{nav}</div></div>'
            f'<main class="wrap">{body}<footer>Generated by 5.collect.py from the stored ramp files. Definitions: README sections '
            f'"Three TPOT definitions" and "How a qualifying rate and its goodput are computed".</footer></main>'
            f'<div id="pop" role="dialog" aria-modal="false" hidden></div>{_glossary_templates(blocks)}<script>{_SCRIPT}</script></body></html>')


def _report_body(ctx, prefix="r"):
    # Section ids carry the shape's prefix so that two shapes on one page never share an anchor.
    body = answer_section(ctx) + measured_section(ctx) + verdict_table(ctx) + maps_section(ctx, prefix) + evidence_section(ctx)
    for sid in ("answer", "measured", "verdict", "maps", "evidence"):
        body = body.replace(f'<section id="{sid}">', f'<section id="{prefix}-{sid}">')
    return f'<div class="shape-block" data-shape="{_esc(prefix)}">{body}</div>'


def render_report(context):
    """`context` keys: shape, archs, objective, attainment, ttft_metric, objective_given (names overridden on the command line), qualified
    {(arch, definition): result}, grid, ttft_axis, tpot_axis, evidence (5.collect.wide_evidence rows);
    optional definitions."""
    try:
        return _page("Disaggregated vs. unified: results", f"Traffic shape {context.get('shape', '')}", _report_body(context), [(None, context)])
    except Exception as exc:  # pragma: no cover - 5.collect.py keeps its result files when this fails
        raise RuntimeError(f"report.py failed to render report.html: {exc}") from exc


def render_sections(contexts):
    """render_report for several traffic shapes on one page."""
    if len(contexts) == 1:
        return render_report(contexts[0])
    blocks = [(f"s{i}", ctx) for i, ctx in enumerate(contexts)]
    body = "".join(f'<h2 class="shape">Traffic shape {_esc(ctx["shape"])}</h2>{_report_body(ctx, prefix)}' for prefix, ctx in blocks)
    return _page("Disaggregated vs. unified: results", ", ".join(c["shape"] for c in contexts), body, blocks, toc=False)

