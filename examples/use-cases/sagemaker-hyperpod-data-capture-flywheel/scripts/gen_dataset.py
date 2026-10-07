import json, random
rng = random.Random(7)

# product -> (variants, symptoms seen in production traffic, symptoms reserved for eval)
P = {
 "ProductFridge": (["French-door fridge", "side-by-side refrigerator", "mini fridge", "top-freezer fridge"],
   ["is not cooling", "ice maker stopped making ice", "leaks water onto the floor", "hums loudly all night",
    "freezer keeps frosting over", "keeps food warm on the top shelf"],
   ["temperature display keeps flashing", "interior light never turns on", "clicks and restarts every few minutes"]),
 "ProductWasher": (["front-load washer", "top-load washing machine", "washer-dryer combo", "stackable washer"],
   ["won't drain", "shakes violently on spin", "leaves clothes soaking wet", "door won't unlock after the cycle",
    "smells musty", "leaks from underneath"],
   ["shows error code E21 halfway through", "overflows the detergent drawer with foam", "takes four hours to finish a cycle"]),
 "ProductOven": (["wall oven", "gas range", "convection oven", "electric range"],
   ["won't heat past 200 degrees", "broiler stopped glowing", "burners won't ignite", "bakes unevenly on one side",
    "locked the door during self-clean", "resets its clock when opened"],
   ["keeps its fan running after switching off", "smells of gas when idle", "reads 100 degrees too high"]),
 "ProductDishwasher": (["built-in dishwasher", "countertop dishwasher", "drawer dishwasher", "compact dishwasher"],
   ["leaves dishes dirty", "won't drain the water at the bottom", "leaks from the door", "never dissolves the pod",
    "leaves a white film on glasses", "won't start the wash cycle"],
   ["makes a grinding noise on the rinse", "leaves dishes wet and cold", "beeps and the panel goes dark"]),
 "ProductMicrowave": (["over-the-range microwave", "countertop microwave", "built-in microwave", "microwave drawer"],
   ["runs but doesn't heat", "turntable won't spin", "sparks inside when running", "keypad is unresponsive",
    "door won't latch shut", "hums then shuts off"],
   ["has its vent fan stuck on high", "shows random characters on the display", "gets hot on the outside casing"]),
 # The new launch: the retailer's own "Arctis" air-conditioner line. Customers mostly use the product name.
 "ProductAC": (["Arctis 9K", "Arctis 12K", "Arctis Mini", "Arctis 12K unit", "window air conditioner"],
   ["blows warm air", "drips water onto the carpet", "compressor won't kick on", "lost its Wi-Fi link to the app",
    "ices over after an hour", "rattles loudly when cooling"],
   ["reads the room 10 degrees wrong", "short-cycles every two minutes", "smells like mildew when it starts"]),
}
EVAL_VARIANTS = {"ProductAC": ["Arctis 18K", "Arctis Duo", "Arctis Pro", "portable air conditioner"]}  # unseen models
OTHER_TRAFFIC = ["I was charged twice on my last invoice, can you refund one?", "Where is my delivery? It was due yesterday.",
  "How do I update the billing address on my account?", "I'd like to return an item I bought last week.",
  "Can I reschedule my installation appointment to Friday?", "Do you offer financing on large purchases?",
  "My promo code isn't working at checkout.", "The technician never showed up for my appointment."]
OTHER_EVAL = ["Can you recommend which model suits a household of six?", "My password reset link has expired.",
  "Do you deliver to rural addresses?", "Why was my gift card declined?", "Do you haul away old appliances?",
  "Is there a student discount?", "Can I change the name on my order before it ships?", "Do you have job openings?"]
T_TRAFFIC = ["My {v} {s}.", "{V} {s}, can someone take a look?", "Hi, our {v} {s} since last week. Order #{n}.",
             "Need help: the {v} {s}", "Our {v} {s} again. Second time this month."]
T_EVAL = ["Hello team, quick question - my {v} {s}. Any advice?", "Reporting a fault: the {v} {s}.",
          "Not happy. Brand new {v} and it already {s}.", "Could you book a repair? The {v} {s}."]
KNOWN = [c for c in P if c != "ProductAC"]

def ticket(cat, pool, templates):
    v = rng.choice(EVAL_VARIANTS.get(cat, P[cat][0]) if pool == 2 else P[cat][0])
    s = rng.choice(P[cat][pool])
    t = rng.choice(templates).format(v=v, V=v[0].upper() + v[1:], s=s, n=rng.randint(10000, 99999))
    return {"ticket": t, "label": cat}

def traffic_row(p_ac):
    r = rng.random()
    if r < p_ac: return ticket("ProductAC", 1, T_TRAFFIC)
    if r < p_ac + 0.04: return {"ticket": rng.choice(OTHER_TRAFFIC), "label": "Other"}
    return ticket(rng.choice(KNOWN), 1, T_TRAFFIC)

# 400 baseline tickets (no Arctis yet), then 500 after launch with the Arctis share ramping 15% -> 55%
traffic = [traffic_row(0.0) for _ in range(400)] + [traffic_row(0.15 + 0.40 * i / 499) for i in range(500)]

# Eval: 40 per class, from symptoms and templates that never appear in traffic (held out by content);
# AC eval tickets also use Arctis models that never appear in traffic.
evalset, seen = [], set()
for cat in P:
    n = 0
    while n < 40:
        row = ticket(cat, 2, T_EVAL)
        if row["ticket"] not in seen: seen.add(row["ticket"]); evalset.append(row); n += 1
while sum(r["label"] == "Other" for r in evalset) < 40:
    t = rng.choice(["", "Hi, ", "Quick one: ", "Hello - "]) + rng.choice(OTHER_EVAL) + rng.choice(["", " Thanks!", " Order #%d." % rng.randint(10000, 99999)])
    if t not in seen: seen.add(t); evalset.append({"ticket": t, "label": "Other"})

for name, rows in [("tickets_traffic.jsonl", traffic), ("tickets_eval.jsonl", evalset)]:
    with open(name, "w") as f: f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{name}: {len(rows)} tickets")
