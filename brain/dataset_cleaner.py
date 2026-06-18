import os
import json

# =========================
# PATH
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

OUTPUT_FILE = os.path.join(DATA_DIR, "clean_no_errors.jsonl")

files = [
    "ai_identity_01.jsonl",
    "creator_relationship_01.jsonl",
    "donations_and_support_01.jsonl",
    "emotions_and_reactions_01.jsonl",
    "fran_dataset.jsonl",
    "gaming_01.jsonl",
    "general_banter_01.jsonl",
    "moderation_and_trolls_01.jsonl",
    "multiturn_banter_01.jsonl",
    "multiturn_banter_02.jsonl",
    "multiturn_banter_03.jsonl",
    "multiturn_disagreements_01.jsonl",
    "philosophy_and_values_01.jsonl",
    "programming_ai_01.jsonl",
    "stream_chat_community_01.jsonl",
    "whisper_noise_01.jsonl",
]

# =========================
# VALIDATION
# =========================

def is_valid(conv):
    if not isinstance(conv, list) or len(conv) < 2:
        return False

    for i, m in enumerate(conv):
        if not isinstance(m, dict):
            return False

        if "from" not in m or "value" not in m:
            return False

        if not isinstance(m["value"], str):
            return False

        if len(m["value"].strip()) < 2:
            return False

        if m["from"] not in ["human", "gpt"]:
            return False

        # enforce turn order
        if i % 2 == 0 and m["from"] != "human":
            return False
        if i % 2 == 1 and m["from"] != "gpt":
            return False

    return True

# =========================
# CLEANING
# =========================

kept = 0
dropped = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for f in files:
        path = os.path.join(DATA_DIR, f)

        if not os.path.exists(path):
            print(f"[MISSING] {path}")
            continue

        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    obj = json.loads(line)

                    conv = obj.get("conversations", [])

                    # STRICT FILTER (removes json_error + bad)
                    if not is_valid(conv):
                        dropped += 1
                        continue

                    out.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
                    kept += 1

                except:
                    dropped += 1  # removes json_error safely

# =========================
# REPORT
# =========================

print("\n===== CLEAN COMPLETE =====\n")
print("kept:", kept)
print("dropped:", dropped)
print("output:", OUTPUT_FILE)