import csv, json, sys
CATS = ["ProductFridge", "ProductWasher", "ProductOven", "ProductDishwasher", "ProductMicrowave", "ProductAC", "Other"]
PROMPT = ("You are a support-ticket router for a home-appliance retailer. Route each ticket to the product it "
          'concerns. Respond with ONLY a JSON object {"category": "...", "confidence": <0-1>} where category is '
          "one of: " + ", ".join(CATS) + '. Use "Other" if the ticket does not clearly concern one of these products.')
n = 0
with open(sys.argv[1], newline="") as fin, open(sys.argv[2], "w") as fout:
    for row in csv.DictReader(fin):
        fout.write(json.dumps({"messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": row["ticket_text"]},
            {"role": "assistant", "content": json.dumps({"category": row["corrected_category"], "confidence": 1.0})},
        ]}) + "\n"); n += 1
print(f"wrote {n} training examples -> {sys.argv[2]}")
