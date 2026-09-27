"""Benchmark worker for lm-evaluation-harness and continuation benchmarks."""
import json
import pathlib
import signal
import subprocess
import sys
from collections import defaultdict

from common import emit, fail, load_source
from model_loader import load_model


STANDARD = {
    "piqa": "piqa", "lambada": "lambada_openai", "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge", "hellaswag": "hellaswag",
}
CUSTOM = {
    "arithmark": ("AxiomicLabs/Arithmark-3.0", "train"),
    "tiny_tom": ("AxiomicLabs/Tiny_Theory_of_Mind", "train"),
}
OFFICIAL = {
    "base_bench": "BananaMind/BananaMind-Base-Bench-1.1",
    "safety_bench": "BananaMind/BananaMind-Safety-Bench-1.1",
}
active_benchmark = None


def stop_benchmark(_signal, _frame):
    if active_benchmark is not None and active_benchmark.poll() is None:
        active_benchmark.terminate()
    raise SystemExit(143)


signal.signal(signal.SIGTERM, stop_benchmark)


def run_official_benchmark(config, key, limit):
    """Execute the dataset's own benchmark.py and display its report.json."""
    from huggingface_hub import hf_hub_download
    global active_benchmark
    repo = OFFICIAL[key]
    emit("status", message=f"Loading official evaluator for {repo}")
    script = hf_hub_download(repo_id=repo, filename="benchmark.py", repo_type="dataset")
    out_dir = pathlib.Path(config["outputPath"]) / key
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, script, "--model", config["model"],
               "--out-dir", str(out_dir), "--batch-size", str(max(1, int(config.get("batchSize", 1))))]
    command.append("--trust-remote-code" if config.get("trustRemoteCode") else "--no-trust-remote-code")
    if limit:
        command.extend(["--limit", str(limit)])
    emit("status", message=f"Running official {repo.split('/')[-1]} script")
    recent = []
    active_benchmark = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
    try:
        for line in active_benchmark.stdout:
            line = line.rstrip()
            if line:
                recent.append(line)
                recent = recent[-12:]
                emit("log", message=line)
                if line.startswith("[") and "/" in line[:16] or line.startswith("Scored "):
                    emit("status", message=line)
        code = active_benchmark.wait()
    finally:
        active_benchmark = None
    if code != 0:
        raise RuntimeError(f"Official {repo} evaluator exited with {code}. " + " | ".join(recent[-4:]))
    report_path = out_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Official evaluator did not write {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if key == "base_bench":
        summary = report["summary"]
        result = {
            "official": bool(summary["official_complete_run"]),
            "overall_elo": summary["overall_elo"],
            "accuracy": summary["accuracy"],
            "weighted_accuracy": summary["weighted_accuracy"],
            "correct": summary["passed"],
            "total": summary["cases"],
            "categories": summary["categories"],
            "difficulties": summary["difficulties"],
            "report_path": str(report_path),
        }
    else:
        result = {
            "official": bool(report["official"]),
            "overall": report["overall"],
            "categories": report["categories"],
            "report_path": str(report_path),
        }
    emit("result", task=key, result=result)
    return result


def score_continuation(model, tokenizer, context, continuation, mode="mean_logprob"):
    import torch
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    common = 0
    for a, b in zip(context_ids, full_ids):
        if a != b:
            break
        common += 1
    # Boundary merges can change the final context token. Score that token too.
    start = max(1, common)
    if not full_ids or start >= len(full_ids):
        return float("-inf")
    max_context = getattr(model.config, "max_position_embeddings", 2048)
    if len(full_ids) > max_context:
        cut = len(full_ids) - max_context
        full_ids = full_ids[cut:]
        start = max(1, start - cut)
    device = next(model.parameters()).device
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[0, :-1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = ids[0, 1:]
        values = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[start - 1:]
    if values.numel() == 0:
        return float("-inf")
    return values.sum().item() if mode == "sum_logprob" else values.mean().item()


def custom_eval(model, tokenizer, key, limit):
    repo, split = CUSTOM[key]
    emit("status", message=f"Loading {repo}")
    dataset = load_source({"id": repo, "split": split}, streaming=True)
    correct = total = 0
    category = defaultdict(lambda: [0, 0])
    grades = []
    examples = []
    for row in dataset:
        if limit and total >= limit:
            break
        context = row.get("context") or row.get("ctx")
        choices = row.get("continuations") or row.get("endings")
        label = row.get("label")
        if not isinstance(context, str) or not isinstance(choices, list) or label is None:
            raise ValueError(f"Unexpected schema for {repo}: {list(row)}")
        mode = row.get("score_mode") or "mean_logprob"
        scores = [score_continuation(model, tokenizer, context, str(choice), mode) for choice in choices]
        prediction = max(range(len(scores)), key=scores.__getitem__)
        label = int(label)
        category_name = row.get("category") or row.get("activity_label") or "overall"
        hit = prediction == label
        correct += int(hit)
        total += 1
        category[category_name][0] += int(hit)
        category[category_name][1] += 1
        if "choice_grades" in row:
            grades.append(row["choice_grades"][prediction])
        if len(examples) < 20:
            examples.append({"id": row.get("id", row.get("ind", total)), "category": category_name,
                             "prediction": prediction, "label": label, "correct": hit, "scores": scores})
        if total % 10 == 0:
            emit("metric", task=key, completed=total, correct=correct, accuracy=correct / total)
    if not total:
        raise ValueError(f"No examples found in {repo}")
    result = {"accuracy": correct / total, "correct": correct, "total": total,
              "categories": {name: {"accuracy": hit / count, "correct": hit, "total": count} for name, (hit, count) in category.items()},
              "examples": examples, "score_mode": "mean conditional token log probability unless dataset specifies otherwise"}
    if grades:
        result["mean_choice_grade"] = sum(grades) / len(grades)
    emit("result", task=key, result=result)
    return result


def main(config):
    import torch
    keys = config.get("tasks") or []
    if not keys:
        raise ValueError("Select at least one benchmark")
    model = str(config.get("model", "")).strip()
    if model.startswith("https://huggingface.co/"):
        config["model"] = model[len("https://huggingface.co/"):].strip("/")
    limit = int(config.get("limit", 0) or 0)
    result = {}
    standard = [STANDARD[key] for key in keys if key in STANDARD]
    if standard:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        emit("status", message="Loading lm-evaluation-harness tasks")
        lm = HFLM(pretrained=config["model"], trust_remote_code=bool(config.get("trustRemoteCode")),
                  batch_size=int(config.get("batchSize", 1)),
                  device="cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        output = lm_eval.simple_evaluate(model=lm, tasks=standard, num_fewshot=int(config.get("fewShot", 0)),
                                         limit=limit or None, log_samples=False)
        for key, task in STANDARD.items():
            if task in output["results"] and key in keys:
                result[key] = output["results"][task]
                emit("result", task=key, result=result[key])
    for key in keys:
        if key in OFFICIAL:
            result[key] = run_official_benchmark(config, key, limit)
    custom = [key for key in keys if key in CUSTOM]
    if custom:
        emit("status", message=f"Loading model for {len(custom)} continuation benchmark(s)")
        model, tokenizer, _ = load_model(config["model"], bool(config.get("trustRemoteCode")))
        for key in custom:
            result[key] = custom_eval(model, tokenizer, key, limit)
    output_path = pathlib.Path(config["outputPath"]) / "evaluation-results.json"
    output_path.write_text(json.dumps({"model": config["model"], "tasks": result}, indent=2, default=str))
    emit("complete", message="Evaluation complete", outputPath=str(output_path), results=result)


if __name__ == "__main__":
    try:
        with open(sys.argv[1], encoding="utf-8") as handle:
            main(json.load(handle))
    except Exception as exc:
        fail(exc)
