"""Redraw images/drift_other_share.png from the reference run's 05_other_share_per_minute output (README step 6). Values are hard-coded."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Step 8 query output: clock minute (UTC), tickets, tickets routed to Other
rows = [("15:52", 20, 0), ("15:53", 60, 3), ("15:54", 60, 4), ("15:55", 60, 4), ("15:56", 60, 2),
        ("15:57", 60, 1), ("15:58", 60, 3), ("15:59", 60, 9), ("16:00", 60, 17), ("16:01", 60, 14),
        ("16:02", 60, 19), ("16:03", 60, 21), ("16:04", 60, 20), ("16:05", 60, 32), ("16:06", 60, 28),
        ("16:07", 40, 26)]
assert sum(r[1] for r in rows) == 900 and sum(r[2] for r in rows) == 203

START = 52 + 40 / 60          # RUN_START 15:52:40, in minutes past 15:00
LAUNCH = 59 + 20 / 60 - START # LAUNCH_TS 15:59:20
END = 67 + 40 / 60 - START    # RUN_END 16:07:40

def m(hhmm):                  # minutes past 15:00
    h, mm = map(int, hhmm.split(":")); return (h - 15) * 60 + mm

# per-minute bars span each clock minute, clipped to the run window
bars = []
for t, n, o in rows:
    lo, hi = max(m(t) - START, 0), min(m(t) + 1 - START, END)
    bars.append((lo, hi, 100 * o / n))
# cumulative share at the end of each clock minute
cx, cy, tot, oth = [0], [0], 0, 0
for (t, n, o), (lo, hi, _) in zip(rows, bars):
    tot += n; oth += o; cx.append(hi); cy.append(100 * oth / tot)

BLUE, GRAY = "#2E7EBB", "#7F7F7F"
fig, ax = plt.subplots(figsize=(13.66, 6.77), dpi=100)
fig.patch.set_facecolor("#FDFDFB"); ax.set_facecolor("#FDFDFB")
ax.bar([lo for lo, _, _ in bars], [p for _, _, p in bars], width=[hi - lo for lo, hi, _ in bars],
       align="edge", color="#C9DDF0", edgecolor="#FDFDFB", linewidth=1.5, label="Per-minute share", zorder=2)
ax.plot(cx, cy, color=BLUE, linewidth=3.5, label="Cumulative share", zorder=3)
ax.scatter([cx[-1]], [cy[-1]], color=BLUE, s=90, zorder=4)
ax.annotate(f"{cy[-1]:.1f}%", (cx[-1], cy[-1]), xytext=(-10, 12), textcoords="offset points",
            ha="right", fontsize=19, fontweight="bold", color="#1A1A1A")

ax.axvline(LAUNCH, color=GRAY, linestyle=(0, (4, 3)), linewidth=2, zorder=3)
ax.text(LAUNCH + 0.15, 61, "Arctis air-conditioner launch", fontsize=15, color="#404040", va="center")
ax.text(LAUNCH / 2, 12, "before launch\n4.8% Other", ha="center", fontsize=13.5, color="#404040")
ax.text(LAUNCH + (END - LAUNCH) / 2, 72, "after launch: 36.8% Other", ha="center", fontsize=13.5, color="#404040")

ax.set_title("Share of tickets routed to \u201cOther\u201d climbs after launch", loc="left",
             fontsize=23, fontweight="bold", color="#111111", pad=16)
ax.set_xlabel("Minutes into the capture window", fontsize=15, color="#333333")
ax.set_ylabel("Tickets routed to \u201cOther\u201d", fontsize=15, color="#333333")
ax.set_xlim(0, END + 0.3); ax.set_ylim(0, 80)
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.tick_params(labelsize=13, colors="#555555")
ax.grid(axis="y", color="#D9D9D9", linewidth=1); ax.set_axisbelow(True)
for side in ("top", "right"): ax.spines[side].set_visible(False)
for side in ("left", "bottom"): ax.spines[side].set_color("#999999")
ax.legend(loc="upper left", frameon=False, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "..", "images", "drift_other_share.png"), facecolor=fig.get_facecolor())
print("cumulative end:", round(cy[-1], 2), "launch at min", round(LAUNCH, 2), "end", round(END, 2))
