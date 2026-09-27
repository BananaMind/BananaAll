const { app, safeStorage } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');

let sessionKey = '';

function file(name) { return path.join(app.getPath('userData'), name); }
function canRemember() {
  return safeStorage.isEncryptionAvailable() &&
    (process.platform !== 'linux' || safeStorage.getSelectedStorageBackend() !== 'basic_text');
}
function getKey() {
  if (sessionKey) return sessionKey;
  if (!canRemember()) return '';
  try { return safeStorage.decryptString(fs.readFileSync(file('openrouter-key.bin'))); }
  catch { return ''; }
}
function status() {
  return { firstRun: !fs.existsSync(file('setup-complete')), hasKey: Boolean(getKey()), canRemember: canRemember() };
}
function finishSetup() {
  fs.mkdirSync(app.getPath('userData'), { recursive: true });
  fs.writeFileSync(file('setup-complete'), '1', { mode: 0o600 });
  return status();
}
function saveKey(value) {
  const key = String(value || '').trim();
  if (!key || key.length > 4096) throw new Error('Enter a valid OpenRouter API key.');
  sessionKey = key;
  fs.mkdirSync(app.getPath('userData'), { recursive: true });
  if (canRemember()) fs.writeFileSync(file('openrouter-key.bin'), safeStorage.encryptString(key), { mode: 0o600 });
  else { try { fs.unlinkSync(file('openrouter-key.bin')); } catch (error) { if (error.code !== 'ENOENT') throw error; } }
  return finishSetup();
}
function clearKey() {
  sessionKey = '';
  try { fs.unlinkSync(file('openrouter-key.bin')); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  return status();
}

function checkedSource(value, label) {
  const source = String(value || '');
  if (!source.trim() || source.length > 250000) throw new Error(`${label} is empty or too large for AI review.`);
  return source;
}
function readSource(filePath, label) {
  const resolved = path.resolve(String(filePath || ''));
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) throw new Error(`Choose a valid ${label} file.`);
  return { name: path.basename(resolved), source: checkedSource(fs.readFileSync(resolved, 'utf8'), label) };
}
function checkPython(source, python) {
  return new Promise((resolve, reject) => {
    const program = `import ast,sys\ntree=ast.parse(sys.stdin.read())\nevents={n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='emit' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)}\nassert {'status','metric','complete'} <= events, 'Missing required status, metric, or complete events'\nassert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='fail' for n in ast.walk(tree)), 'Missing error handler'\n`;
    const child = spawn(String(python || 'python3'), ['-c', program], { windowsHide: true });
    let stderr = '';
    const timer = setTimeout(() => { child.kill(); reject(new Error('Python syntax check timed out.')); }, 10000);
    child.stderr.on('data', data => { stderr += data.toString(); });
    child.on('error', error => { clearTimeout(timer); reject(error); });
    child.on('close', code => { clearTimeout(timer); code === 0 ? resolve() : reject(new Error(`AI response failed local code validation: ${stderr.trim()}`)); });
    child.stdin.on('error', () => {});
    child.stdin.end(source);
  });
}

async function review({ code, configPath, configurationPath, modelingPath, python }) {
  const key = getKey();
  if (!key) throw new Error('Add an OpenRouter API key in AI review settings first.');
  const script = checkedSource(code, 'Training script');
  const selected = [readSource(configPath, 'config.json'), readSource(configurationPath, 'configuration Python'), readSource(modelingPath, 'modeling Python')];
  const sources = [selected[0]];
  const modules = new Map();
  for (const directory of new Set([path.dirname(path.resolve(configPath)), path.dirname(path.resolve(configurationPath)), path.dirname(path.resolve(modelingPath))])) {
    for (const name of fs.readdirSync(directory)) {
      if (!name.endsWith('.py')) continue;
      const moduleSource = readSource(path.join(directory, name), name);
      if (modules.has(name) && modules.get(name).source !== moduleSource.source) throw new Error(`Conflicting custom Python files named ${name}`);
      modules.set(name, moduleSource);
    }
  }
  for (const item of selected.slice(1)) modules.set(item.name, item);
  sources.push(...modules.values());
  if (script.length + sources.reduce((sum, item) => sum + item.source.length, 0) > 400000) throw new Error('The custom code is too large for AI review. Use manual review.');
  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'openai/gpt-6-luna', max_tokens: 18000,
      messages: [
        { role: 'system', content: 'Review this Transformers custom-architecture training worker for concrete correctness errors. Preserve its one-JSON-event-per-stdout-line contract: status, metric, complete, and error (via fail) must remain available. Preserve dataset/tokenizer behavior unless a correctness fix is required. Fix only the training script; do not claim edits to the supplied architecture files. Return ONLY JSON with keys summary (short string), revised_code (complete Python source string, or null when no fixes are needed).' },
        { role: 'user', content: `Custom architecture files:\n${sources.map(item => `=== ${item.name} ===\n${item.source}`).join('\n\n')}\n\n=== train.py ===\n${script}` }
      ] }),
    signal: AbortSignal.timeout(180000)
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(`OpenRouter review failed (${response.status}): ${payload.error?.message || response.statusText}`);
  const content = payload.choices?.[0]?.message?.content;
  const answer = Array.isArray(content) ? content.map(part => part.text || '').join('') : content;
  if (typeof answer !== 'string') throw new Error('OpenRouter returned no review.');
  let parsed;
  try { parsed = JSON.parse(answer.replace(/^\s*```(?:json)?\s*|\s*```\s*$/g, '').trim()); }
  catch { throw new Error('OpenRouter returned an unreadable review. Try again or review manually.'); }
  if (typeof parsed.summary !== 'string' || (parsed.revised_code !== null && typeof parsed.revised_code !== 'string')) throw new Error('OpenRouter returned an incomplete review.');
  const revisedCode = parsed.revised_code === null ? null : checkedSource(parsed.revised_code, 'Revised script');
  if (revisedCode) await checkPython(revisedCode, python);
  return { summary: parsed.summary.slice(0, 4000), revisedCode };
}

module.exports = { status, finishSetup, saveKey, clearKey, review };
