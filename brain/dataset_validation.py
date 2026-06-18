import os
import json
import hashlib
from collections import defaultdict

# =========================
# PATH
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CLEAN_FILE = os.path.join(DATA_DIR, "clean_no_errors.jsonl")

stats = defaultdict(int)
seen = set()

# =========================
# SCORE
# =========================

def score(conv):
    text = " ".join(m.get("value", "") for m in conv).lower()

    s = 0
    s += min(len(text) / 2000, 1)

    if "?" in text:
        s += 0.8

    twitch_tokens = ["lol", "lmao", "wow", "bruh", "wtf"]
    s += min(sum(text.count(t) for t in twitch_tokens) * 0.05, 0.3)

    if len(text.strip()) < 10:
        s -= 0.5

    return s

# =========================
# CHECK
# =========================

def check(conv):
    if not isinstance(conv, list) or len(conv) < 2:
        return "bad"

    h = hashlib.md5(str(conv).encode()).hexdigest()
    if h in seen:
        return "dup"
    seen.add(h)

    for i, m in enumerate(conv):
        if "from" not in m or "value" not in m:
            return "malformed"

        if not isinstance(m["value"], str) or len(m["value"].strip()) < 2:
            return "empty"

        if m["from"] not in ["human", "gpt"]:
            return "role"

        if i % 2 == 0 and m["from"] != "human":
            return "order"
        if i % 2 == 1 and m["from"] != "gpt":
            return "order"

    if score(conv) < 0.5:
        return "low"

    return "ok"

# =========================
# RUN
# =========================

with open(CLEAN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
            result = check(obj.get("conversations", []))
            stats[result] += 1
            stats["total"] += 1
        except:
            stats["json_error"] += 1

# =========================
# REPORT
# =========================

print("\n===== CLEAN DATASET REPORT =====\n")

for k, v in stats.items():
    print(f"{k}: {v}")

print("\nvalid_ratio:", stats["ok"] / max(stats["total"], 1))