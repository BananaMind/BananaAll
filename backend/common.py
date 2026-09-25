"""Shared dataset and event helpers for BananaAll workers."""
import json
import os
import pathlib
import sys


def emit(kind, **fields):
    print(json.dumps({"type": kind, **fields}, default=str), flush=True)


def fail(exc):
    import traceback
    emit("error", message=str(exc), detail=traceback.format_exc())
    sys.exit(1)


def normalize_repo(value):
    value = str(value or "").strip()
    prefix = "https://huggingface.co/datasets/"
    if value.startswith(prefix):
        return value[len(prefix):].strip("/")
    return value


def load_source(source, streaming=True):
    from datasets import load_dataset
    repo = normalize_repo(source.get("id"))
    if not repo:
        raise ValueError("Dataset source is empty")
    split = source.get("split") or "train"
    config = source.get("config") or None
    path = pathlib.Path(os.path.expanduser(repo))
    if path.exists():
        extension = path.suffix.lower()
        format_name = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet"}.get(extension)
        if not format_name:
            raise ValueError(f"Unsupported local dataset format: {extension}")
        return load_dataset(format_name, data_files={split: str(path)}, split=split, streaming=streaming)
    kwargs = {"split": split, "streaming": streaming}
    if config:
        kwargs["name"] = config
    return load_dataset(repo, **kwargs)


def auto_mapping(sample):
    keys = set(sample)
    pick = lambda names: next((key for key in names if key in keys), "")
    messages = pick(["messages", "conversations", "conversation", "chat"])
    system = pick(["system", "system_prompt"])
    user = pick(["user", "prompt", "question", "instruction", "input", "problem"])
    assistant = pick(["assistant", "completion", "response", "answer", "output", "generated_solution", "solution"])
    text = pick(["text", "content", "document", "body", "raw_content", "article"])
    if messages:
        return {"messages": messages, "text": "", "system": "", "user": "", "assistant": ""}
    if not text and not (user and assistant):
        text = next((key for key, value in sample.items() if isinstance(value, str) and len(value) > 30), "")
    return {"messages": "", "text": text, "system": system, "user": user, "assistant": assistant}


def stringify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(stringify(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_example(row, mapping, mode):
    mapping = mapping or auto_mapping(row)
    if mode == "pretraining":
        text_field = mapping.get("text")
        if text_field and row.get(text_field) is not None:
            return {"text": stringify(row[text_field])}
        parts = [stringify(row.get(mapping.get(name))) for name in ("system", "user", "assistant") if mapping.get(name)]
        return {"text": "\n\n".join(part for part in parts if part)}
    messages_field = mapping.get("messages")
    if messages_field and isinstance(row.get(messages_field), list):
        messages = []
        for item in row[messages_field]:
            if isinstance(item, dict):
                role = item.get("role") or item.get("from") or item.get("speaker") or "user"
                role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
                content = item.get("content") or item.get("value") or item.get("text") or ""
                if role in ("system", "user", "assistant"):
                    messages.append({"role": role, "content": stringify(content)})
        return {"messages": messages}
    messages = []
    for role in ("system", "user", "assistant"):
        field = mapping.get(role)
        if field and row.get(field) is not None:
            content = stringify(row[field])
            if content:
                messages.append({"role": role, "content": content})
    if not messages and mapping.get("text"):
        return {"text": stringify(row.get(mapping["text"]))}
    return {"messages": messages}
