# BananaAll

BananaAll is a local, dark-mode SLM Super App. It lets you do everything you need to make SLMs, evaluate them, use them in a single app.

## Run it

```bash
npm install
python3 -m pip install -r requirements.txt
npm run dev
```

`npm run check` type-checks and builds the renderer. `npm run build && npm start` runs the built Electron app. You can set the Python executable in the Train sidebar if the ML packages are installed in a virtual environment.

For private or gated Hugging Face repositories, sign in once with `hf auth login`. The header shows the current login. BananaAll uses the token stored by Hugging Face; it does not ask you to paste the token into the app.

## What it does

- **Pretraining:** streamed, weighted dataset mixes with a token target for each source; sample previews and field mapping; a byte-level BPE tokenizer trained from the chosen mix; a continuous 3M–200M model size control that generates the architecture configuration (with the supplied 3M, 10M, 25M, 50M, and 140M configurations preserved exactly); BananaMind 2 style, LFT, BananaMind 2 + LFT, and custom Transformers code.
- **Fine tuning:** full weight training or LoRA adapters from a Hugging Face or local causal language model. Message arrays and separate system, user, and assistant columns are supported.
- **Custom architecture:** select `config.json` and a modeling Python file, review the generated training script, optionally edit it, then approve the run. Its stdout JSON event contract powers the in-app loss chart and status display.
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
