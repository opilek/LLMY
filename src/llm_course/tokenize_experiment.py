from transformers import AutoTokenizer

MODEL_NAME_V1 = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer_v1 = AutoTokenizer.from_pretrained(MODEL_NAME_V1)
MODEL_NAME_V2 = "bert-base-cased"
tokenizer_v2 = AutoTokenizer.from_pretrained(MODEL_NAME_V2)

samples = [
    "Opiła",
    "for(int i=0;i<10;i++)",
    "127000.999990",
    "😍❤️🤣😘👌",
    "Mądry żółw przechadza się wzdłuż kępy trawy",
    "Duży model językowy",
    "Large language model",
]

for name, tokenizer in [(MODEL_NAME_V1, tokenizer_v1), (MODEL_NAME_V2, tokenizer_v2)]:
    print("\n", name)
    for text in samples:
        ids = tokenizer.encode(text, add_special_tokens=False)
        pieces = [tokenizer.decode([token_id]) for token_id in ids]
        print(f"\nTEXT:   {text!r}")
        print(f"IDS:    {ids}")
        print(f"PIECES: {pieces}")
        print(f"COUNT:  {len(ids)}")
        print(f"DECODE==TEXT: {tokenizer.decode(ids) == text}")