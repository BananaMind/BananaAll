"""Single generation worker. The UI keeps the transcript; each request is reproducible."""
import json
import pathlib
import sys
from common import emit, fail
from model_loader import load_model


def render_prompt(tokenizer, config):
    mode = config.get("inferenceMode", "instruct")
    messages = config.get("messages") or []
    if mode == "base":
        return config.get("prompt") or "\n".join(item.get("content", "") for item in messages)
    system = config.get("systemPrompt", "")
    if system:
        messages = [{"role": "system", "content": system}] + messages
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "".join(f"<|{item['role']}|>\n{item['content']}\n" for item in messages) + "<|assistant|>\n"


def main(config):
    import torch
    emit("status", message=f"Loading {config['model']}")
    model, tokenizer, reference = load_model(config["model"], bool(config.get("trustRemoteCode")))
    prompt = render_prompt(tokenizer, config)
    if not prompt.strip():
        raise ValueError("Enter a prompt")
    max_context = getattr(model.config, "max_position_embeddings", 2048)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=max(16, max_context - int(config.get("maxNewTokens", 128))))
    encoded = {key: value.to(next(model.parameters()).device) for key, value in encoded.items()}
    temperature = float(config.get("temperature", 0.7))
    kwargs = dict(max_new_tokens=int(config.get("maxNewTokens", 128)),
                  repetition_penalty=float(config.get("repetitionPenalty", 1.0)),
                  pad_token_id=tokenizer.pad_token_id,
                  eos_token_id=tokenizer.eos_token_id,
                  use_cache=not bool(getattr(model.config, "lft", False)))
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature,
                      top_p=float(config.get("topP", 0.95)), top_k=int(config.get("topK", 50)))
    else:
        kwargs["do_sample"] = False
    emit("status", message="Generating")
    with torch.inference_mode():
        generated = model.generate(**encoded, **kwargs)
    text = tokenizer.decode(generated[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True)
    output = pathlib.Path(config["outputPath"]) / "generation.json"
    output.write_text(json.dumps({"model": reference, "prompt": prompt, "response": text, "settings": kwargs}, indent=2, default=str))
    emit("complete", message="Generation complete", text=text, outputPath=str(output))


if __name__ == "__main__":
    try:
        with open(sys.argv[1], encoding="utf-8") as handle:
            main(json.load(handle))
    except Exception as exc:
        fail(exc)
