"""
Fran Fine-tuning Script
Model  : unsloth/Meta-Llama-3.1-8B-Instruct
Dataset: D:/Fran/brain/data/FinalDataset.jsonl (ShareGPT JSONL, from/value format)
Output : D:/Fran/brain/model/
"""

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig

# ────────────────────────────────
# 1. CONFIG
# ────────────────────────────────
BASE_MODEL          = "D:/Fran/brain/base_model"   # local unsloth/Meta-Llama-3.1-8B-Instruct
DATASET_PATH        = "D:/Fran/brain/data/FinalDataset.jsonl"  # updated to directly reference the JSONL file
OUTPUT_DIR          = "D:/Fran/brain/model"
SYSTEM_PROMPT_PATH  = "D:/Fran/brain/system_prompt.txt"

MAX_SEQ_LENGTH  = 2048   # longer context = less degeneration on multi-turn
LORA_RANK       = 32     # sweet spot for persona coherence
LOAD_IN_4BIT    = True

# ────────────────────────────────
# 2. LOAD BASE MODEL
# ────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = BASE_MODEL,
    max_seq_length  = MAX_SEQ_LENGTH,
    dtype           = None,   # auto-detect; uses bf16 on RTX 4080
    load_in_4bit    = LOAD_IN_4BIT,
)

# Apply Llama-3 chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

# ────────────────────────────────
# 3. LORA ADAPTER
# ────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r                   = LORA_RANK,
    target_modules      = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha          = LORA_RANK * 2,   # 2x rank is standard best practice
    lora_dropout        = 0.05,            # small dropout reduces overfitting on 600 examples
    bias                = "none",
    use_gradient_checkpointing = "unsloth",
    random_state        = 42,
)

# ────────────────────────────────
# 4. DATASET
# ────────────────────────────────
from datasets import load_dataset

print(f"Loading dataset from: {DATASET_PATH}")

raw_dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

print(f"Loaded system prompt ({len(SYSTEM_PROMPT)} chars):\n{SYSTEM_PROMPT[:300]}\n...")

def format_conversations(example):
    """Convert ShareGPT from/value → ChatML messages with system prompt."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in example["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = raw_dataset.map(format_conversations, remove_columns=raw_dataset.column_names)

print(f"Dataset size: {len(dataset)} examples")
print(f"Sample:\n{dataset[0]['text'][:500]}\n")

# ────────────────────────────────
# 5. TRAINER
# ────────────────────────────────
trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = dataset,
    args            = SFTConfig(
        dataset_text_field      = "text",
        max_seq_length          = MAX_SEQ_LENGTH,
        per_device_train_batch_size     = 2,
        gradient_accumulation_steps     = 8,   # effective batch = 16
        warmup_steps            = 20,
        num_train_epochs        = 3,           # 3 epochs on 600 examples hits ~1800 steps
        learning_rate           = 2e-4,
        fp16                    = not torch.cuda.is_bf16_supported(),
        bf16                    = torch.cuda.is_bf16_supported(),
        logging_steps           = 10,
        optim                   = "adamw_8bit",
        weight_decay            = 0.01,        # helps prevent repetition loops
        lr_scheduler_type       = "cosine",    # cosine decay = better convergence than linear
        seed                    = 42,
        output_dir              = OUTPUT_DIR,
        save_strategy           = "no",        # avoids Windows TRL checkpoint bug
        report_to               = "none",
    ),
)

# ────────────────────────────────
# 6. TRAIN
# ────────────────────────────────
print("Starting training...")
trainer_stats = trainer.train()

print(f"\nTraining complete.")
print(f"  Runtime : {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"  Loss    : {trainer_stats.metrics['train_loss']:.4f}")

# ────────────────────────────────
# 7. SAVE LORA ADAPTER
# ────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nAdapter saved to {OUTPUT_DIR}")