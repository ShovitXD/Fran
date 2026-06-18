"""
Fran Interactive Chat Test
Run: python test_fran.py
Exit: Ctrl+C
"""

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ADAPTER_PATH       = "D:/Fran/brain/model"
SYSTEM_PROMPT_PATH = "D:/Fran/brain/system_prompt.txt"

MAX_SEQ_LENGTH     = 2048
MAX_MEMORY_PAIRS   = 15

MAX_NEW_TOKENS     = 200
TEMPERATURE        = 0.8
TOP_P              = 0.9
TOP_K              = 50
REPETITION_PENALTY = 1.3

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("Loading Fran...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
)

tokenizer.pad_token = tokenizer.eos_token

with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

print("Fran is ready. Type a message. Ctrl+C to exit.\n")
print("─" * 50)

# ─────────────────────────────────────────────
# CHAT LOOP
# ─────────────────────────────────────────────
history = []

try:
    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        history.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        # Keep only last 15 conversation pairs
        if len(history) > MAX_MEMORY_PAIRS * 2:
            history = history[-(MAX_MEMORY_PAIRS * 2):]

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ] + history

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

        inputs = {
            k: v.to("cuda")
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_tokens = output_ids[0][
            inputs["input_ids"].shape[-1]:
        ]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        # Keep only last 15 conversation pairs again
        if len(history) > MAX_MEMORY_PAIRS * 2:
            history = history[-(MAX_MEMORY_PAIRS * 2):]

        print(f"\nFran: {response}")

except KeyboardInterrupt:
    print("\n\nFran: bye i guess.")