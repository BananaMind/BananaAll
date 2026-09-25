import pathlib


def load_model(reference, trust_remote_code=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from modeling_bananaall import register
    register()
    reference = str(reference or "").strip()
    if reference.startswith("https://huggingface.co/"):
        reference = reference[len("https://huggingface.co/"):].strip("/")
    if not reference:
        raise ValueError("Enter a Hugging Face model ID, URL, or local model folder")
    tokenizer = AutoTokenizer.from_pretrained(reference, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    is_adapter = pathlib.Path(reference, "adapter_config.json").exists()
    if is_adapter:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(reference, trust_remote_code=trust_remote_code, device_map="auto", torch_dtype="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(reference, trust_remote_code=trust_remote_code, device_map="auto", torch_dtype="auto")
    model.eval()
    return model, tokenizer, reference
