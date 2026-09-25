"""BananaAll training worker.

Logging contract for editable custom training code:
Write one JSON object per stdout line with a `type` field. Keep `status`,
`metric`, `complete`, and `error` events so the app can show progress and
finished artifacts. Other diagnostics should go to stderr.
"""
import json
import os
import pathlib
import random
import shutil
import sys
import time

from common import emit, extract_example, fail, load_source
from architecture import architecture_for_size


def pretraining_sources(sources):
    result = []
    for source in sources:
        if not source.get("enabled", True) or float(source.get("weight", 1)) <= 0:
            continue
        limit = source.get("tokenLimit")
        if limit is None:
            raise ValueError(f"Set a token target for {source['id']}")
        limit = int(limit)
        if limit <= 0:
            raise ValueError(f"Token target must be positive: {source['id']}")
        result.append((source, limit))
    if not result:
        raise ValueError("Select at least one dataset with a positive token target")
    return result


def source_rows(sources, seed):
    enabled = [s for s in sources if s.get("enabled", True) and float(s.get("weight", 1)) > 0]
    if not enabled:
        raise ValueError("Select at least one dataset with a positive mix weight")
    rng = random.Random(seed)
    iterators = [iter(load_source(source, streaming=True)) for source in enabled]
    weights = [float(source.get("weight", 1)) for source in enabled]
    while True:
        index = rng.choices(range(len(enabled)), weights=weights, k=1)[0]
        try:
            row = next(iterators[index])
        except StopIteration:
            iterators[index] = iter(load_source(enabled[index], streaming=True))
            try:
                row = next(iterators[index])
            except StopIteration:
                raise ValueError(f"Dataset has no rows: {enabled[index]['id']}")
        yield row, enabled[index]


def make_tokenizer(config):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    if config["mode"] != "pretraining":
        reference = config.get("model")
        if not reference:
            raise ValueError("Enter a base model ID or local model folder")
        tokenizer = AutoTokenizer.from_pretrained(reference, trust_remote_code=bool(config.get("trustRemoteCode")))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    vocab_size = architecture_for_size(config.get("parametersM", 25))["vocab_size"]
    emit("status", message=f"Training {vocab_size:,}-token byte-level BPE tokenizer from the dataset mix")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                 special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"])
    limit = max(100, int(config.get("tokenizerSamples", 2000)))
    def corpus():
        selected_sources = [source for source, _ in pretraining_sources(config["datasets"])]
        for index, (row, source) in enumerate(source_rows(selected_sources, int(config.get("seed", 1337)))):
            if index >= limit:
                break
            example = extract_example(row, source.get("mapping"), "pretraining")
            if example.get("text"):
                yield example["text"]
    tokenizer.train_from_iterator(corpus(), trainer=trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]", pad_token="[PAD]")
    if len(fast) < vocab_size:
        fast.add_tokens([f"[unused_{index}]" for index in range(vocab_size - len(fast))])
    fast.model_max_length = int(config.get("sequenceLength", 1024))
    return fast


def make_model(config, tokenizer):
    from transformers import AutoConfig, AutoModelForCausalLM
    from modeling_bananaall import BananaAllForCausalLM, register
    import torch
    register()
    mode = config["mode"]
    if mode != "pretraining":
        emit("status", message=f"Loading {config['model']}")
        model = AutoModelForCausalLM.from_pretrained(config["model"], trust_remote_code=bool(config.get("trustRemoteCode")), torch_dtype="auto")
        if mode == "lora":
            from peft import LoraConfig, get_peft_model
            lora = LoraConfig(r=int(config.get("loraRank", 16)), lora_alpha=int(config.get("loraAlpha", 32)),
                              lora_dropout=float(config.get("loraDropout", 0.05)), target_modules="all-linear",
                              task_type="CAUSAL_LM", bias="none")
            model = get_peft_model(model, lora)
        return model
    architecture = config.get("architecture", "bananamind2")
    spec = architecture_for_size(config.get("parametersM", 25))
    spec["vocab_size"] = len(tokenizer)
    spec["max_position_embeddings"] = max(spec["max_position_embeddings"], int(config.get("sequenceLength", 1024)))
    if architecture == "custom":
        if not config.get("trustRemoteCode"):
            raise ValueError("Custom architectures require the Trust remote code setting")
        custom_folder = pathlib.Path(config["outputPath"]) / "custom_architecture"
        custom_folder.mkdir(parents=True, exist_ok=True)
        config_path = pathlib.Path(config.get("customConfigPath", ""))
        modeling_path = pathlib.Path(config.get("customModelingPath", ""))
        if not config_path.is_file() or not modeling_path.is_file():
            raise ValueError("Select both config.json and modeling Python file for the custom architecture")
        custom_json = json.loads(config_path.read_text())
        if not custom_json.get("auto_map", {}).get("AutoModelForCausalLM"):
            raise ValueError("Custom config.json needs auto_map.AutoModelForCausalLM")
        shutil.copy2(config_path, custom_folder / "config.json")
        shutil.copy2(modeling_path, custom_folder / modeling_path.name)
        for name in config_path.parent.glob("*.py"):
            shutil.copy2(name, custom_folder / name.name)
        for key, value in spec.items():
            if key in custom_json:
                custom_json[key] = value
        custom_json["vocab_size"] = len(tokenizer)
        (custom_folder / "config.json").write_text(json.dumps(custom_json, indent=2))
        emit("status", message="Loading reviewed custom architecture")
        model_config = AutoConfig.from_pretrained(str(custom_folder), trust_remote_code=True)
        return AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    from configuration_bananaall import BananaAllConfig
    model_config = BananaAllConfig(**spec,
                                   architecture_style="lft" if architecture == "lft" else "bananamind2",
                                   lft=architecture in ("lft", "bananamind2lft"),
                                   bos_token_id=tokenizer.bos_token_id,
                                   eos_token_id=tokenizer.eos_token_id,
                                   pad_token_id=tokenizer.pad_token_id)
    model = BananaAllForCausalLM(model_config)
    executions = 3 * spec["num_hidden_layers"] - 4 if model_config.lft else spec["num_hidden_layers"]
    emit("architecture", targetM=float(config.get("parametersM", 25)), parameters=sum(p.numel() for p in model.parameters()),
         layers=spec["num_hidden_layers"], executions=executions, config=model_config.to_dict())
    return model


def text_for_messages(tokenizer, messages):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "".join(f"<|{item['role']}|>\n{item['content']}\n" for item in messages)


def instruction_tokens(tokenizer, example, max_length):
    if "text" in example:
        ids = tokenizer(example["text"], truncation=True, max_length=max_length - 1, add_special_tokens=False)["input_ids"]
        ids.append(tokenizer.eos_token_id)
        return ids, ids.copy()
    messages = example.get("messages") or []
    if not messages:
        return [], []
    last_assistant = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    if last_assistant < 0:
        return [], []
    prompt = messages[:last_assistant]
    target = messages[last_assistant]["content"]
    if getattr(tokenizer, "chat_template", None):
        prefix = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    else:
        prefix = text_for_messages(tokenizer, prompt) + "<|assistant|>\n"
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    ids = (prefix_ids + target_ids)[:max_length]
    labels = ([-100] * len(prefix_ids) + target_ids)[:max_length]
    return ids, labels


def make_training_data(config, tokenizer):
    import torch
    from torch.utils.data import IterableDataset
    mode = config["mode"]
    max_length = int(config.get("sequenceLength", 1024))
    seed = int(config.get("seed", 1337))
    sources = config["datasets"]
    class MixedDataset(IterableDataset):
        def __init__(self):
            super().__init__()
            if mode == "pretraining":
                selected = pretraining_sources(sources)
                self.sources = [source for source, _ in selected]
                self.remaining = [limit for _, limit in selected]
                self.total_sequences = (sum(self.remaining) + max_length - 1) // max_length

        def __iter__(self):
            buffer = []
            if mode == "pretraining":
                rng = random.Random(seed)
                iterators = [iter(load_source(source, streaming=True)) for source in self.sources]
                weights = [float(source.get("weight", 1)) for source in self.sources]
                while True:
                    available = [index for index, remaining in enumerate(self.remaining) if remaining is None or remaining > 0]
                    if not available:
                        if len(buffer) == 1:
                            buffer.insert(0, tokenizer.bos_token_id)
                        if buffer:
                            yield {"input_ids": buffer, "labels": buffer.copy(), "attention_mask": [1] * len(buffer)}
                        return
                    index = rng.choices(available, weights=[weights[i] for i in available], k=1)[0]
                    source = self.sources[index]
                    try:
                        row = next(iterators[index])
                    except StopIteration:
                        iterators[index] = iter(load_source(source, streaming=True))
                        try:
                            row = next(iterators[index])
                        except StopIteration:
                            raise ValueError(f"Dataset has no rows: {source['id']}")
                    example = extract_example(row, source.get("mapping"), mode)
                    text = example.get("text", "")
                    if not text.strip():
                        continue
                    # Documents are packed into sequence_length blocks below; a long
                    # source row is never sent to the model as one sequence.
                    tokens = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"] + [tokenizer.eos_token_id]
                    remaining = self.remaining[index]
                    if remaining is not None:
                        tokens = tokens[:remaining]
                        self.remaining[index] -= len(tokens)
                    buffer.extend(tokens)
                    while len(buffer) >= max_length:
                        ids, buffer = buffer[:max_length], buffer[max_length:]
                        yield {"input_ids": ids, "labels": ids.copy(), "attention_mask": [1] * len(ids)}
            else:
                for row, source in source_rows(sources, seed):
                    example = extract_example(row, source.get("mapping"), mode)
                    ids, labels = instruction_tokens(tokenizer, example, max_length)
                    if len(ids) > 1 and any(label != -100 for label in labels[1:]):
                        yield {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    class SizedPretrainingDataset(MixedDataset):
        def __len__(self):
            # Trainer uses dataloader length to request only the available
            # microbatches in the final, incomplete accumulation step.
            return self.total_sequences

    def collate(rows):
        length = max(len(row["input_ids"]) for row in rows)
        length = min(max_length, ((length + 7) // 8) * 8)
        result = {}
        for key, pad in (("input_ids", tokenizer.pad_token_id), ("labels", -100), ("attention_mask", 0)):
            result[key] = torch.tensor([row[key][:length] + [pad] * (length - len(row[key][:length])) for row in rows], dtype=torch.long)
        return result
    return (SizedPretrainingDataset() if mode == "pretraining" else MixedDataset()), collate


def main(config):
    import inspect
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_callback import PrinterCallback, ProgressCallback
    output = pathlib.Path(config["outputPath"])
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = make_tokenizer(config)
    tokenizer.save_pretrained(output)
    model = make_model(config, tokenizer)
    dataset, collator = make_training_data(config, tokenizer)
    device_cuda = torch.cuda.is_available()
    precision = config.get("precision", "auto")
    bf16 = device_cuda and (precision == "bf16" or precision == "auto" and torch.cuda.is_bf16_supported())
    fp16 = device_cuda and precision == "fp16"
    if config["mode"] == "pretraining":
        selected = pretraining_sources(config["datasets"])
        total_tokens = sum(limit for _, limit in selected)
        sequence_length = int(config.get("sequenceLength", 1024))
        total_sequences = (total_tokens + sequence_length - 1) // sequence_length
        batch_size = max(1, int(config.get("batchSize", 2))) * max(1, torch.cuda.device_count())
        total_batches = (total_sequences + batch_size - 1) // batch_size
        accumulation = max(1, int(config.get("gradientAccumulation", 8)))
        max_steps = (total_batches + accumulation - 1) // accumulation
        emit("status", message=f"Training until all {total_tokens:,} selected dataset tokens are consumed ({max_steps:,} optimizer steps planned)")
    else:
        max_steps = max(1, int(config.get("maxSteps", 500)))
    args = TrainingArguments(
        output_dir=str(output), max_steps=max_steps,
        per_device_train_batch_size=max(1, int(config.get("batchSize", 2))),
        gradient_accumulation_steps=max(1, int(config.get("gradientAccumulation", 8))),
        learning_rate=float(config.get("learningRate", 0.0002)),
        warmup_steps=max(0, round(max_steps * float(config.get("warmupRatio", 0.03)))),
        weight_decay=float(config.get("weightDecay", 0.01)),
        max_grad_norm=float(config.get("maxGradNorm", 1.0)),
        lr_scheduler_type=config.get("scheduler", "cosine"),
        logging_steps=max(1, int(config.get("loggingSteps", 10))),
        save_steps=max(1, int(config.get("saveSteps", 100))),
        save_strategy="steps", save_total_limit=3,
        bf16=bf16, fp16=fp16, report_to="none", remove_unused_columns=False,
        dataloader_num_workers=0, seed=int(config.get("seed", 1337)),
        logging_first_step=True, disable_tqdm=True,
    )
    class Progress(TrainerCallback):
        def __init__(self):
            self.started = None
            self.accumulation_step = 0

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()
            self.accumulation_step = 0
            optimizer = kwargs.get("optimizer")
            fields = dict(step=state.global_step, totalSteps=state.max_steps,
                          accumulationStep=0, accumulationTotal=args.gradient_accumulation_steps,
                          learningRate=float(optimizer.param_groups[0]["lr"] if optimizer else args.learning_rate),
                          elapsedSeconds=0)
            if config["mode"] == "pretraining":
                fields.update(tokensSeen=0, totalTokens=total_tokens)
            emit("progress", **fields)

        def on_step_begin(self, args, state, control, **kwargs):
            self.accumulation_step = 0
            emit("progress", step=state.global_step, totalSteps=state.max_steps,
                 accumulationStep=0, accumulationTotal=args.gradient_accumulation_steps)

        def on_substep_end(self, args, state, control, **kwargs):
            self.accumulation_step += 1
            emit("progress", step=state.global_step, totalSteps=state.max_steps,
                 accumulationStep=self.accumulation_step,
                 accumulationTotal=args.gradient_accumulation_steps)

        def on_step_end(self, args, state, control, **kwargs):
            optimizer = kwargs.get("optimizer")
            fields = dict(step=state.global_step, totalSteps=state.max_steps,
                          accumulationStep=0, accumulationTotal=args.gradient_accumulation_steps,
                          learningRate=float(optimizer.param_groups[0]["lr"] if optimizer else args.learning_rate),
                          elapsedSeconds=time.monotonic() - self.started)
            if config["mode"] == "pretraining":
                fields["tokensSeen"] = total_tokens - sum(dataset.remaining)
                fields["totalTokens"] = total_tokens
            emit("progress", **fields)
            self.accumulation_step = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                emit("metric", step=state.global_step, totalSteps=state.max_steps, **logs)
        def on_save(self, args, state, control, **kwargs):
            emit("status", message=f"Checkpoint saved at step {state.global_step}")
    emit("status", message="Training started", outputPath=str(output))
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator,
                      processing_class=tokenizer, callbacks=[Progress()])
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    trainer.train(resume_from_checkpoint=config.get("resumeCheckpoint") or None)
    if config["mode"] == "pretraining":
        if any(remaining != 0 for remaining in dataset.remaining):
            raise RuntimeError("Training stopped before all selected dataset token targets were consumed")
        usage = [{"dataset": source["id"], "tokens": limit} for source, limit in pretraining_sources(config["datasets"])]
        (output / "dataset_tokens.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
        emit("status", message=f"Consumed all {sum(item['tokens'] for item in usage):,} selected dataset tokens")
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    if config["mode"] == "pretraining" and config.get("architecture") != "custom":
        model.config.auto_map = {
            "AutoConfig": "configuration_bananaall.BananaAllConfig",
            "AutoModelForCausalLM": "modeling_bananaall.BananaAllForCausalLM",
        }
        model.config.save_pretrained(output)
        for name in ("configuration_bananaall.py", "modeling_bananaall.py"):
            shutil.copy2(pathlib.Path(__file__).with_name(name), output / name)
    elif config["mode"] == "pretraining" and config.get("architecture") == "custom":
        for name in (output / "custom_architecture").glob("*.py"):
            shutil.copy2(name, output / name.name)
    elif config["mode"] == "full":
        source = pathlib.Path(str(config["model"]))
        if source.is_dir():
            for name in source.glob("*.py"):
                shutil.copy2(name, output / name.name)
        else:
            module_file = pathlib.Path(inspect.getfile(model.__class__))
            if "transformers_modules" in str(module_file):
                for name in module_file.parent.glob("*.py"):
                    shutil.copy2(name, output / name.name)
    emit("complete", message="Training complete", outputPath=str(output))


if __name__ == "__main__":
    try:
        with open(sys.argv[1], encoding="utf-8") as handle:
            main(json.load(handle))
    except Exception as exc:
        fail(exc)
