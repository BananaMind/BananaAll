"""BananaAll training worker.

Logging contract for editable custom training code:
Write one JSON object per stdout line with a `type` field. Keep `status`,
`metric`, `complete`, and `error` events so the app can show progress and
finished artifacts. Other diagnostics should go to stderr.
"""
import json
import gc
import inspect
import os
import pathlib
import random
import shutil
import sys
import time
from collections import deque

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
    pretraining = config["mode"] == "pretraining"
    use_existing = pretraining and config.get("tokenizerSource", "train") == "existing"
    if not pretraining or use_existing:
        reference = config.get("tokenizerModel") if use_existing else config.get("model")
        if not reference:
            raise ValueError("Enter a tokenizer model ID or local folder" if use_existing else "Enter a base model ID or local model folder")
        reference = str(reference).strip()
        if reference.startswith("https://huggingface.co/"):
            reference = reference.removeprefix("https://huggingface.co/").rstrip("/")
        if use_existing:
            emit("status", message=f"Loading tokenizer from {reference}")
        trust_tokenizer_code = config.get("tokenizerTrustRemoteCode", False) if use_existing else config.get("trustRemoteCode", False)
        tokenizer = AutoTokenizer.from_pretrained(reference, trust_remote_code=bool(trust_tokenizer_code))
        if use_existing:
            if tokenizer.eos_token is None:
                if tokenizer.sep_token is not None:
                    tokenizer.eos_token = tokenizer.sep_token
                else:
                    tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
            if tokenizer.bos_token is None:
                tokenizer.bos_token = tokenizer.cls_token or tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if use_existing:
            if any(getattr(tokenizer, f"{name}_token_id") is None for name in ("bos", "eos", "pad")):
                raise ValueError("The selected tokenizer needs usable BOS, EOS, and PAD token IDs")
            emit("status", message=f"Using existing tokenizer with {len(tokenizer):,} tokens; model vocabulary will match it")
        return tokenizer
    if config.get("architecture") == "custom":
        config_path = pathlib.Path(config.get("customConfigPath", ""))
        if not config_path.is_file():
            raise ValueError("Select a valid custom config.json before training the tokenizer")
        vocab_size = int(json.loads(config_path.read_text()).get("vocab_size", 0))
        if vocab_size < 256:
            raise ValueError("Custom config.json needs a vocab_size of at least 256")
    else:
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
    if architecture == "ternary":
        if not config.get("trustRemoteCode"):
            raise ValueError("Ternary architecture requires Trust model code to reload its custom layers")
        if not torch.cuda.is_available() or torch.version.hip or not torch.version.cuda:
            raise RuntimeError("Experimental Ternary training requires an NVIDIA GPU with CUDA PyTorch")
    spec = architecture_for_size(config.get("parametersM", 25))
    spec["vocab_size"] = len(tokenizer)
    spec["max_position_embeddings"] = max(spec["max_position_embeddings"], int(config.get("sequenceLength", 1024)))
    if architecture == "custom":
        if not config.get("trustRemoteCode"):
            raise ValueError("Custom architectures require the Trust remote code setting")
        custom_folder = pathlib.Path(config["outputPath"]) / "custom_architecture"
        custom_folder.mkdir(parents=True, exist_ok=True)
        config_path = pathlib.Path(config.get("customConfigPath", ""))
        configuration_path = pathlib.Path(config.get("customConfigurationPath", ""))
        modeling_path = pathlib.Path(config.get("customModelingPath", ""))
        if not all(item.is_file() and item.suffix == ".py" for item in (configuration_path, modeling_path)) or not config_path.is_file():
            raise ValueError("Select config.json, configuration Python, and modeling Python files for the custom architecture")
        custom_json = json.loads(config_path.read_text())
        for key, source in (("AutoConfig", configuration_path), ("AutoModelForCausalLM", modeling_path)):
            module = str(custom_json.get("auto_map", {}).get(key, "")).split(".")[0]
            if module != source.stem:
                raise ValueError(f"config.json auto_map.{key} must point to {source.name}")
        sources = {configuration_path, modeling_path}
        for directory in {config_path.parent, configuration_path.parent, modeling_path.parent}:
            sources.update(directory.glob("*.py"))
        modules = {}
        for source in sorted(sources):
            if source.name in modules and modules[source.name].read_bytes() != source.read_bytes():
                raise ValueError(f"Conflicting custom Python files named {source.name}")
            modules[source.name] = source
        for name, source in modules.items():
            shutil.copy2(source, custom_folder / name)
        custom_json["vocab_size"] = len(tokenizer)
        custom_json["max_position_embeddings"] = max(int(custom_json.get("max_position_embeddings", 0)), int(config.get("sequenceLength", 1024)))
        for key in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
            custom_json[key] = getattr(tokenizer, key)
        (custom_folder / "config.json").write_text(json.dumps(custom_json, indent=2))
        emit("status", message="Loading reviewed custom architecture")
        model_config = AutoConfig.from_pretrained(str(custom_folder), trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
        # Trainer treats **kwargs as proof that the model normalizes its loss
        # over the whole accumulated batch. Most custom forwards merely ignore
        # num_items_in_batch and return a mean loss for each microbatch.
        if not hasattr(model, "accepts_loss_kwargs"):
            model.accepts_loss_kwargs = "num_items_in_batch" in inspect.signature(model.forward).parameters
        return model
    from configuration_bananaall import BananaAllConfig
    model_config = BananaAllConfig(**spec,
                                   architecture_style="lft" if architecture == "lft" else "bananamind2",
                                   lft=architecture in ("lft", "bananamind2lft"),
                                   ternary=architecture == "ternary",
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
        # Built-in pretraining is causal and pads only on the right. Real tokens
        # cannot attend to later padding, and padded labels are ignored, so the
        # model can use SDPA's causal path without constructing a dense mask.
        if mode == "pretraining" and config.get("architecture") != "custom":
            del result["attention_mask"]
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
    device_xpu = not device_cuda and hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_xpu:
        from accelerate.utils import is_xpu_available
        if not is_xpu_available():
            raise RuntimeError("Intel GPU found, but Accelerate cannot use XPU. Install XPU PyTorch 2.7+ and Accelerate 1.13+.")
    compile_requested = bool(config.get("compile", True))
    precision = config.get("precision", "auto")
    bf16_supported = torch.cuda.is_bf16_supported() if device_cuda else torch.xpu.is_bf16_supported() if device_xpu else False
    bf16 = (device_cuda or device_xpu) and (precision == "bf16" or precision == "auto" and bf16_supported)
    fp16 = (device_cuda or device_xpu) and precision == "fp16"
    if config["mode"] == "pretraining":
        selected = pretraining_sources(config["datasets"])
        total_tokens = sum(limit for _, limit in selected)
        sequence_length = int(config.get("sequenceLength", 1024))
        total_sequences = (total_tokens + sequence_length - 1) // sequence_length
        # Trainer runs one XPU per process; unlike CUDA DataParallel it does
        # not spread this local worker across every visible Intel GPU.
        device_count = torch.cuda.device_count() if device_cuda else 1
        batch_size = max(1, int(config.get("batchSize", 2))) * max(1, device_count)
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
        torch_compile=compile_requested,
        torch_compile_backend="inductor" if compile_requested else None,
        torch_compile_mode="reduce-overhead" if compile_requested and device_cuda and not torch.version.hip else None,
    )
    processed_tokens = 0

    class CountingTrainer(Trainer):
        compile_fallback = False

        def training_step(self, model, inputs, *args, **kwargs):
            nonlocal processed_tokens
            mask = inputs.get("attention_mask")
            tokens = mask if mask is not None else inputs["labels"].ne(-100)
            processed_tokens += int(tokens.sum().item())
            if self.compile_fallback:
                return super().training_step(self.model, inputs, *args, **kwargs)
            try:
                return super().training_step(model, inputs, *args, **kwargs)
            except Exception as error:
                original = self.accelerator.unwrap_model(model, keep_torch_compile=False)
                if not compile_requested or original is model:
                    raise
                self.model.zero_grad(set_to_none=True)
                try:
                    result = super().training_step(original, inputs, *args, **kwargs)
                except Exception:
                    raise error from None
                self.compile_fallback = True
                emit("compile-disabled", message=f"Compile failed ({type(error).__name__}); continuing without compilation.")
                return result

    class Progress(TrainerCallback):
        def __init__(self):
            self.started = None
            self.accumulation_step = 0
            self.speed_samples = deque(maxlen=8)

        def speed(self, now):
            self.speed_samples.append((now, processed_tokens))
            first_time, first_tokens = self.speed_samples[0]
            elapsed = now - first_time
            return (processed_tokens - first_tokens) / elapsed if elapsed > 0 else 0.0

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()
            self.accumulation_step = 0
            self.speed_samples.clear()
            self.speed_samples.append((self.started, processed_tokens))
            optimizer = kwargs.get("optimizer")
            fields = dict(step=state.global_step, totalSteps=state.max_steps,
                          accumulationStep=0, accumulationTotal=args.gradient_accumulation_steps,
                          learningRate=float(optimizer.param_groups[0]["lr"] if optimizer else args.learning_rate),
                          elapsedSeconds=0, processedTokens=processed_tokens, tokensPerSecond=0)
            if config["mode"] == "pretraining":
                fields.update(tokensSeen=0, totalTokens=total_tokens)
            emit("progress", **fields)

        def on_step_begin(self, args, state, control, **kwargs):
            self.accumulation_step = 0
            emit("progress", step=state.global_step, totalSteps=state.max_steps,
                 accumulationStep=0, accumulationTotal=args.gradient_accumulation_steps)

        def on_substep_end(self, args, state, control, **kwargs):
            self.accumulation_step += 1
            now = time.monotonic()
            emit("progress", step=state.global_step, totalSteps=state.max_steps,
                 accumulationStep=self.accumulation_step,
                 accumulationTotal=args.gradient_accumulation_steps,
                 elapsedSeconds=now - self.started, processedTokens=processed_tokens,
                 tokensPerSecond=self.speed(now))

        def on_step_end(self, args, state, control, **kwargs):
            optimizer = kwargs.get("optimizer")
            now = time.monotonic()
            fields = dict(step=state.global_step, totalSteps=state.max_steps,
                          accumulationStep=0, accumulationTotal=args.gradient_accumulation_steps,
                          learningRate=float(optimizer.param_groups[0]["lr"] if optimizer else args.learning_rate),
                          elapsedSeconds=now - self.started, processedTokens=processed_tokens,
                          tokensPerSecond=self.speed(now))
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
    trainer = CountingTrainer(model=model, args=args, train_dataset=dataset, data_collator=collator,
                              processing_class=tokenizer, callbacks=[Progress()])
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    resume_checkpoint = config.get("resumeCheckpoint") or None
    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except Exception as error:
        cause = error
        compile_error = False
        while cause is not None:
            module = type(cause).__module__
            if module.startswith(("torch._dynamo", "torch._inductor", "triton")) or isinstance(cause, torch.cuda.OutOfMemoryError):
                compile_error = True
                break
            cause = cause.__cause__
        if not compile_requested or trainer.compile_fallback or trainer.state.global_step != 0 or not compile_error:
            raise
        emit("compile-disabled", message=f"Compile setup failed ({type(error).__name__}); restarting in eager mode.")
        trainer.accelerator.state.dynamo_plugin.backend = type(trainer.accelerator.state.dynamo_plugin.backend).NO
        del trainer
        gc.collect()
        if device_cuda:
            torch.cuda.empty_cache()
        elif device_xpu:
            torch.xpu.empty_cache()
        processed_tokens = 0
        model.zero_grad(set_to_none=True)
        dataset, collator = make_training_data(config, tokenizer)
        args.torch_compile = False
        args.torch_compile_backend = None
        args.torch_compile_mode = None
        trainer = CountingTrainer(model=model, args=args, train_dataset=dataset, data_collator=collator,
                                  processing_class=tokenizer, callbacks=[Progress()])
        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        trainer.train(resume_from_checkpoint=resume_checkpoint)
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
