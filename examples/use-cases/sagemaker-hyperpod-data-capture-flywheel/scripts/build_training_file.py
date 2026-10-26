import csv, json, sys
from router_client import V2 as CATS, system_prompt
PROMPT = system_prompt(CATS)
n = 0
with open(sys.argv[1], newline="") as fin, open(sys.argv[2], "w") as fout:
    for row in csv.DictReader(fin):
        fout.write(json.dumps({"messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": row["ticket_text"]},
            {"role": "assistant", "content": json.dumps({"category": row["corrected_category"], "confidence": 1.0})},
        ]}) + "\n"); n += 1
print(f"wrote {n} training examples -> {sys.argv[2]}")
