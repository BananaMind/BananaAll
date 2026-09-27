# BananaAll

BananaAll is a local, dark-mode SLM Super App. It lets you do everything you need to make SLMs, evaluate them, use them in a single app.

## Run it

To have a coding agent install BananaAll for your operating system, send it: “Install BananaAll for me. Fetch and follow https://raw.githubusercontent.com/BananaMind/BananaAll/main/agent_install.txt.”

```bash
npm install
python3 -m pip install -r requirements.txt
npm run dev
```

`npm run check` type-checks and builds the renderer. `npm run build && npm start` runs the built Electron app. You can set the Python executable in the Train sidebar if the ML packages are installed in a virtual environment.

### Windows with an AMD GPU

Use a Windows and GPU combination supported by [AMD's ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html). Install the matching AMD driver, then create a project virtual environment in PowerShell:

```powershell
npm install
py -3.12 -m venv .venv
```

Install the ROCm build of PyTorch into `.venv` using [AMD's PyTorch installer instructions](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html). Select **Windows** and the user's GPU there; the package command depends on the GPU and ROCm release. Then install BananaAll's remaining dependencies and check that PyTorch sees the GPU:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
npm run dev
```

On Windows, BananaAll selects `.venv\Scripts\python.exe` automatically when that environment exists in the project. The header shows `ROCm` and the GPU name when the selected Python can use the AMD GPU; `CPU` or `PyTorch unavailable` means that environment is not using it. A different environment can be selected through **Train → Advanced → Python executable**. Install the ROCm PyTorch build before `requirements.txt` so the general `torch>=2.5` requirement is already satisfied by the GPU build. GPU detection confirms that PyTorch can see the device; training support still depends on the specific ROCm release and GPU.

### Linux or Windows with an Intel GPU

Use a GPU and operating system supported by [PyTorch's Intel GPU guide](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html). Install the Intel GPU driver, then install a recent XPU-enabled PyTorch build in the Python environment used by BananaAll. PyTorch 2.7 or newer is needed for the current Accelerate integration. For example, after activating that environment:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
python -m pip install -r requirements.txt
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'No XPU')"
```

On Windows, a project `.venv\Scripts\python.exe` is detected automatically; on either OS you can choose another interpreter in **Train → Advanced → Python executable**. The header shows `Intel XPU` and the GPU name when the selected environment can use it. In-app training uses XPU through Transformers Trainer, standard benchmarks select XPU, and inference loads onto the available accelerator. BF16 is selected automatically when supported; compile can fall back to eager training if it fails. Experimental Ternary remains NVIDIA-only.

### macOS with Apple Silicon

Install the Python dependencies with an MPS-enabled PyTorch build. On macOS, BananaAll defaults to `python3`; set **Train → Advanced → Python executable** if the packages are in another environment. The header shows `MPS · Metal GPU` when PyTorch can use the GPU. Training uses the device selected automatically by Hugging Face Trainer, and standard benchmark evaluation selects MPS. BananaAll currently runs MPS training in FP32. Some PyTorch operations may still fall back to CPU or be unsupported on MPS; see the [Hugging Face Apple Silicon guide](https://huggingface.co/docs/transformers/perf_train_special).

For private or gated Hugging Face repositories, sign in once with `hf auth login`. The header shows the current login. BananaAll uses the token stored by Hugging Face; it does not ask you to paste the token into the app.

## Notebook training

Press **Start training** and choose **In-App-Training** (recommended) to run locally, or **Notebook** to save a standalone Python script for Google Colab or Molab. Notebook export accepts Hugging Face dataset IDs and, for fine tuning, Hugging Face base-model IDs. When pretraining with an existing tokenizer, its source must also be a Hugging Face model ID or URL. Local dataset files, tokenizer folders, and base-model folders cannot be used in a cloud notebook. Custom architecture source files and reviewed training code are bundled into the script.

Open a GPU notebook, paste the entire saved `.py` file into one Python cell, and run it. The script installs the training dependencies, trains the model, and writes `<run-name>-model.zip` with the final model and tokenizer. Colab starts a download; in Molab, download the ZIP from the Files panel. Extract the ZIP to use the model locally. LoRA runs produce an adapter that still needs its base model.

Notebook export checks the selected Hugging Face repositories online and includes a saved token only if a model or dataset is gated or private. Public-only exports contain no token. If the `.py` file contains a credential, keep it private and delete it when it is no longer needed. If a protected repository needs a token and none is saved locally, sign in to Hugging Face in the notebook before running the script.

## What it does

- **Pretraining:** streamed, weighted dataset mixes with a token target for each source; sample previews and field mapping; either a byte-level BPE tokenizer trained from the chosen mix or a tokenizer loaded from an existing Hugging Face model or local folder; a continuous 3M–200M model size control that generates the architecture configuration (with the supplied 3M, 10M, 25M, 50M, and 140M configurations preserved exactly); BananaMind 2 style, LFT, BananaMind 2 + LFT, experimental Ternary, and custom Transformers code.

Experimental **Ternary** pretraining is available only when the selected Python environment sees an NVIDIA CUDA GPU. It keeps the BananaMind 2 blocks and applies 1.58-bit ternary weight and 8-bit activation fake quantization to their linear layers with straight-through gradients. Training still uses floating-point master weights and optimizer states, and checkpoints are not packed 1.58-bit files. Keep **Trust model code** enabled to reload a saved Ternary model with `trust_remote_code=True`; the export includes the configuration and modeling Python files.

**Compile model** is on by default for training. PyTorch compiles on the first training step, which can take noticeably longer than later steps. If compilation fails, BananaAll retries in regular eager mode, turns the option off for the selected Python environment, and continues training when the eager step succeeds. Change the Python environment or restart the app to make the option available again.
- **Fine tuning:** full weight training or LoRA adapters from a Hugging Face or local causal language model. Message arrays and separate system, user, and assistant columns are supported.
- **Custom architecture:** select `config.json`, the configuration Python module named by `auto_map.AutoConfig`, and the modeling Python module named by `auto_map.AutoModelForCausalLM`. BananaAll either trains a byte-level BPE tokenizer using the custom config's `vocab_size` or loads an existing model tokenizer. It saves the resulting vocabulary size and special token IDs in the model config. Review the generated script manually or use AI Review before approving the run. The stdout JSON event contract powers the in-app loss chart and status display.
- **AI Review:** first-run setup can save an optional OpenRouter API key. The AI Review button sends the generated training script and custom architecture Python files to `openai/gpt-6-luna`, applies suggested script edits to the review editor after local syntax and logging-contract checks, and waits for your approval. You can add or remove the key later from the lock button in the toolbar. Keys use operating-system secure storage when available; otherwise they last only for the current app session.
- **Evaluation:** PIQA, LAMBADA, ARC Easy, ARC Challenge, and HellaSwag through `lm-evaluation-harness`; BananaMind Base Bench 1.1, BananaMind Safety Bench 1.1, ArithMark 3.0, and Tiny Theory of Mind through continuation likelihood scoring. Reports are saved as JSON.
- **Inference:** instruct or base completion from a Hugging Face ID, URL, or local run folder, with generation controls and a saved generation record.

The selected size is an approximate target; the generated configuration and estimated count appear before training, and the true count appears when training starts. LFT reuses adjacent blocks and performs `3N - 4` block executions for `N` unique layers, so it takes more compute than a standard model with the same parameters. Large runs require adequate GPU memory and time. Dataset previews use the Hugging Face viewer when available, while training streams the underlying dataset.

For pretraining, set each enabled dataset's **tokens** target in millions of tokenizer tokens. Mix weights choose the next active source. The run ends after all targets are consumed; BananaAll calculates the required optimizer steps from the total tokens, sequence length, batch size, and gradient accumulation. Full fine tuning and LoRA still use a max-step setting. A `dataset_tokens.json` report records the completed per-dataset amounts.

## Source layout

- `src/`: React interface
- `electron/`: desktop window and process bridge
- `backend/`: Python training, dataset preview, evaluation, and inference workers

### Worker events

Each worker writes JSON lines to stdout. The app uses `status`, `metric`, `result`, `complete`, and `error` event types. Edited custom training scripts must keep these events if the app is to show progress and the completed artifact.
