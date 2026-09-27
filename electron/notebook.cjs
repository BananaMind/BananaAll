const fs = require('node:fs');
const path = require('node:path');

const workerFiles = [
  'train.py', 'common.py', 'architecture.py',
  'configuration_bananaall.py', 'modeling_bananaall.py'
];

function remoteId(value, kind) {
  const reference = String(value || '').trim();
  const prefix = kind === 'dataset' ? 'https://huggingface.co/datasets/' : 'https://huggingface.co/';
  const id = reference.startsWith(prefix) ? reference.slice(prefix.length).replace(/\/$/, '') : reference;
  if (fs.existsSync(path.resolve(reference)) ||
      !/^[A-Za-z0-9][A-Za-z0-9._-]*(?:\/[A-Za-z0-9][A-Za-z0-9._-]*)?$/.test(id) ||
      /\.(?:csv|json|jsonl|parquet|txt|py)$/i.test(id)) {
    throw new Error(`Notebook training needs a Hugging Face ${kind} ID or URL. Local ${kind} paths cannot be used in a cloud notebook: ${reference}`);
  }
  return id;
}

function safeName(value) {
  return String(value || 'bananaall-run').replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^[-.]+|[-.]+$/g, '').slice(0, 80) || 'bananaall-run';
}

function validateNotebookConfig(config) {
  const repositories = (config.datasets || []).map(source => ({ id: remoteId(source.id, 'dataset'), type: 'dataset' }));
  if (config.mode !== 'pretraining') repositories.push({ id: remoteId(config.model, 'model'), type: 'model' });
  if (config.mode === 'pretraining' && config.tokenizerSource === 'existing') {
    repositories.push({ id: remoteId(config.tokenizerModel, 'model'), type: 'model' });
  }
  return repositories;
}

function pythonLiteral(contents) {
  for (const delimiter of ["'''", '"""']) {
    if (!contents.includes(delimiter) && !contents.endsWith('\\')) {
      return `r${delimiter}${contents}${delimiter}`;
    }
  }
  const lines = contents.match(/[^\n]*\n|[^\n]+$/g) || [''];
  return `(\n${lines.map(line => `    ${JSON.stringify(line)}`).join('\n')}\n)`;
}

function buildNotebook(config, backend, code, token) {
  validateNotebookConfig(config);
  const files = Object.fromEntries(workerFiles.map(name => [name, fs.readFileSync(path.join(backend, name), 'utf8')]));
  const portableConfig = { ...config };
  delete portableConfig.python;
  delete portableConfig.outputDir;
  delete portableConfig.customConfigPath;
  delete portableConfig.customConfigurationPath;
  delete portableConfig.customModelingPath;
  portableConfig.datasets = (config.datasets || []).map(source => ({ ...source, id: remoteId(source.id, 'dataset') }));
  if (config.mode !== 'pretraining') portableConfig.model = remoteId(config.model, 'model');
  if (config.mode === 'pretraining' && config.tokenizerSource === 'existing') {
    portableConfig.tokenizerModel = remoteId(config.tokenizerModel, 'model');
  }

  if (config.mode === 'pretraining' && config.architecture === 'custom') {
    const configPath = path.resolve(String(config.customConfigPath || ''));
    const configurationPath = path.resolve(String(config.customConfigurationPath || ''));
    const modelingPath = path.resolve(String(config.customModelingPath || ''));
    if (![configPath, configurationPath, modelingPath].every(file => fs.existsSync(file) && fs.statSync(file).isFile())) {
      throw new Error('Choose valid config.json, configuration Python, and modeling Python files before exporting.');
    }
    const customJson = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    for (const [key, file] of [['AutoConfig', configurationPath], ['AutoModelForCausalLM', modelingPath]]) {
      if (String(customJson.auto_map?.[key] || '').split('.')[0] !== path.basename(file, '.py')) {
        throw new Error(`config.json auto_map.${key} must point to ${path.basename(file)}`);
      }
    }
    files['custom_source/config.json'] = fs.readFileSync(configPath, 'utf8');
    for (const directory of new Set([path.dirname(configPath), path.dirname(configurationPath), path.dirname(modelingPath)])) {
      for (const name of fs.readdirSync(directory)) {
        if (!name.endsWith('.py')) continue;
        const key = `custom_source/${name}`;
        const source = fs.readFileSync(path.join(directory, name), 'utf8');
        if (files[key] !== undefined && files[key] !== source) throw new Error(`Conflicting custom Python files named ${name}`);
        files[key] = source;
      }
    }
    portableConfig.customConfigPath = 'custom_source/config.json';
    portableConfig.customConfigurationPath = `custom_source/${path.basename(configurationPath)}`;
    portableConfig.customModelingPath = `custom_source/${path.basename(modelingPath)}`;
    if (typeof code !== 'string' || !code.trim()) throw new Error('Review the custom training code before exporting the notebook.');
    files['train.py'] = code;
  }

  const runName = safeName(config.runName);
  portableConfig.runName = runName;
  const zipName = `${runName}-model.zip`;
  const sources = Object.entries(files).map(([name, contents]) => `    ${JSON.stringify(name)}: ${pythonLiteral(contents)},`).join('\n');
  const tokenDeclaration = token ? `# This token grants access to your Hugging Face account. Keep this file private.\nHF_TOKEN = ${JSON.stringify(token)}\n` : '';
  const tokenSetup = token ? '    os.environ["HF_TOKEN"] = HF_TOKEN\n' : '';
  const script = `# BananaAll notebook training script
# Paste this entire file into one Python cell in Google Colab or Molab and run it.
# Use a GPU runtime. Review the configuration and bundled source below before running.
# The script writes ${zipName} after training finishes.
import json
import os
import pathlib
import runpy
import subprocess
import sys
import tempfile
import zipfile

RUN_NAME = ${JSON.stringify(runName)}
ZIP_NAME = ${JSON.stringify(zipName)}
${tokenDeclaration}

# Training settings selected in BananaAll.
CONFIG = json.loads(${pythonLiteral(JSON.stringify(portableConfig, null, 2))})

# Bundled BananaAll training source. These files are written into a temporary
# workspace so the same worker runs in the notebook without a project checkout.
SOURCES = {
${sources}
}

def _bananaall_notebook_train():
    dependencies = [
        "transformers>=5.7", "datasets>=4.8", "accelerate>=1.0",
        "peft>=0.19", "tokenizers>=0.21", "huggingface_hub>=1.0"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *dependencies])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU is visible to PyTorch. Enable a GPU runtime in Colab or Molab and run this cell again.")
    if CONFIG.get("architecture") == "ternary" and (torch.version.hip or not torch.version.cuda):
        raise RuntimeError("Experimental Ternary training requires an NVIDIA GPU with CUDA PyTorch.")
    print("Training on", torch.cuda.get_device_name(0), flush=True)
${tokenSetup}    workdir = pathlib.Path(tempfile.mkdtemp(prefix="bananaall-notebook-"))
    for name, contents in SOURCES.items():
        target = workdir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")
    config = dict(CONFIG)
    config["outputPath"] = str(workdir / "model")
    if config.get("customConfigPath"):
        config["customConfigPath"] = str(workdir / config["customConfigPath"])
        config["customConfigurationPath"] = str(workdir / config["customConfigurationPath"])
        config["customModelingPath"] = str(workdir / config["customModelingPath"])
    config_path = workdir / "bananaall-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    sys.path.insert(0, str(workdir))
    old_argv = sys.argv
    try:
        sys.argv = [str(workdir / "train.py"), str(config_path)]
        runpy.run_path(str(workdir / "train.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path.remove(str(workdir))
    model_dir = pathlib.Path(config["outputPath"])
    archive = pathlib.Path.cwd() / ZIP_NAME
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for item in sorted(model_dir.rglob("*")):
            relative = item.relative_to(model_dir)
            if item.is_file() and not any(part.startswith("checkpoint-") for part in relative.parts):
                bundle.write(item, pathlib.Path(RUN_NAME) / relative)
    print("Finished model ZIP:", archive, flush=True)
    try:
        from google.colab import files
    except ImportError:
        print("Download the ZIP from the notebook's Files panel, then extract it on your computer.", flush=True)
    else:
        try:
            files.download(str(archive))
        except Exception:
            print("Automatic download failed. Download the ZIP from the notebook's Files panel.", flush=True)

_bananaall_notebook_train()
`;
  return { script, zipName, tokenEmbedded: Boolean(token) };
}

module.exports = { buildNotebook, safeName, validateNotebookConfig };
