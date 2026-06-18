import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

FRAN_MODEL = None
TOKENIZER = None


def load_llm(base_model_path):
    global FRAN_MODEL, TOKENIZER

    print("[Fran] Loading LLM (4-bit)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    TOKENIZER = AutoTokenizer.from_pretrained(base_model_path)

    FRAN_MODEL = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        quantization_config=bnb_config,
    )

    FRAN_MODEL.generation_config.max_length = 4096
    FRAN_MODEL.eval()
    FRAN_MODEL.config.use_cache = True

    print("[Fran] LLM Ready")
    return FRAN_MODEL, TOKENIZER


def generate(system_prompt, history, text, max_new_tokens=200):
    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": text})

    prompt = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(FRAN_MODEL.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = FRAN_MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.95,
            repetition_penalty=1.15,
            use_cache=True,
        )

    return TOKENIZER.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()