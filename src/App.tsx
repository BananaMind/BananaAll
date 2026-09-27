import { useEffect, useMemo, useState } from 'react';
import { Activity, ArrowDownToLine, ArrowRight, Beaker, Check, ChevronDown, Code2, Database, ExternalLink, FolderOpen, GitBranch, Globe2, LoaderCircle, LockKeyhole, Play, Plus, Settings2, SlidersHorizontal, Sparkles, Square, Terminal, Trash2, TriangleAlert, WandSparkles, X } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import type { BananaBridge, DatasetSource, DeviceStatus, JobEvent, JobState, Mapping, ReviewSettings, Tab, TrainMode } from './types';
import { architectureForSize, estimateParameters } from './architecture';

const emptyMapping: Mapping = { text: '', messages: '', system: '', user: '', assistant: '' };
const textMapping: Mapping = { ...emptyMapping, text: 'text' };
const datasetPresets = [
  { name: 'FineWeb EDU', id: 'HuggingFaceFW/fineweb-edu', config: '', split: 'train', note: 'Full educational web text', mapping: textMapping },
  { name: 'FineWeb HQ', id: 'epfml/FineWeb-HQ', config: '', split: 'train', note: 'Filtered high quality web', mapping: textMapping },
  { name: 'DCLM', id: 'mlfoundations/dclm-baseline-1.0-parquet', config: '', split: 'train', note: 'DCLM baseline parquet', mapping: textMapping },
  { name: 'FineMath', id: 'HuggingFaceTB/finemath', config: 'finemath-4plus', split: 'train', note: 'High quality math text', mapping: textMapping },
  { name: 'OpenMathInstruct 2', id: 'nvidia/OpenMathInstruct-2', config: '', split: 'train', note: 'Math instruction data', mapping: { ...emptyMapping, user: 'problem', assistant: 'generated_solution' } },
  { name: 'Cosmopedia v2', id: 'HuggingFaceTB/smollm-corpus', config: 'cosmopedia-v2', split: 'train', note: 'Synthetic textbook corpus', mapping: textMapping },
];
const benchmarkGroups = [
  { title: 'Standard benchmarks', tasks: [
    ['piqa', 'PIQA', 'Physical commonsense'], ['lambada', 'LAMBADA', 'Long-range word prediction'],
    ['arc_easy', 'ARC Easy', 'Grade-school science'], ['arc_challenge', 'ARC Challenge', 'Harder science questions'],
    ['hellaswag', 'HellaSwag', 'Sentence continuation'],
  ] },
  { title: 'Continuation suites', tasks: [
    ['base_bench', 'BananaMind Base Bench 1.1', '350 base-model questions'],
    ['safety_bench', 'BananaMind Safety Bench 1.1', 'Graded safety continuations'],
    ['arithmark', 'ArithMark 3.0', 'Arithmetic word problems'],
    ['tiny_tom', 'Tiny Theory of Mind', 'Social reasoning continuations'],
  ] },
];

const mockBridge: BananaBridge = {
  system: async () => ({ platform: 'browser', defaultOutput: '~/Documents/BananaAll/runs', python: 'python3' }),
  authStatus: async () => ({ loggedIn: false, message: 'Open the Electron app to check Hugging Face login' }),
  deviceStatus: async () => ({ backend: 'unavailable', name: 'Electron required' }),
  pickPath: async () => null, openPath: async () => '',
  inspectDataset: async () => { throw new Error('Dataset preview requires the Electron app'); },
  trainingCode: async () => '', exportNotebook: async () => { throw new Error('Run BananaAll in Electron to export a notebook'); },
  reviewSettings: async () => ({ firstRun: false, hasKey: false, canRemember: false }),
  saveReviewKey: async () => { throw new Error('OpenRouter settings require Electron'); },
  clearReviewKey: async () => { throw new Error('OpenRouter settings require Electron'); },
  skipReviewSetup: async () => ({ firstRun: false, hasKey: false, canRemember: false }),
  aiReview: async () => { throw new Error('AI review requires Electron'); },
  startJob: async () => { throw new Error('Run BananaAll in Electron to start jobs'); },
  stopJob: async () => false, onJobEvent: () => () => {},
};
const api = window.banana || mockBridge;
type TrainingTarget = 'in-app' | 'notebook';

function Field({ label, hint, children, className = '' }: { label: string; hint?: string; children: React.ReactNode; className?: string }) {
  return <label className={`field ${className}`}><span className="field-label">{label}</span>{children}{hint && <span className="field-hint">{hint}</span>}</label>;
}
function Toggle({ checked, onChange, label, hint }: { checked: boolean; onChange: (value: boolean) => void; label: string; hint?: string }) {
  return <label className="toggle-row"><span><strong>{label}</strong>{hint && <small>{hint}</small>}</span><input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} /><span className="switch" /></label>;
}
function Section({ eyebrow, title, aside, children, id }: { eyebrow: string; title: string; aside?: React.ReactNode; children: React.ReactNode; id?: string }) {
  return <section className="section" id={id}><div className="section-head"><div><div className="eyebrow">{eyebrow}</div><h2>{title}</h2></div>{aside}</div>{children}</section>;
}
function statusLabel(job?: JobState) {
  if (!job) return 'Ready';
  return { running: 'In progress', complete: 'Complete', error: 'Needs attention', stopped: 'Stopped' }[job.status];
}
function compactNumber(value: number) { return value >= 1_000_000_000 ? `${(value / 1_000_000_000).toFixed(2)}B` : value >= 1_000_000 ? `${(value / 1_000_000).toFixed(2)}M` : value >= 1_000 ? `${(value / 1_000).toFixed(1)}K` : value.toLocaleString(); }
function elapsedTime(value: number) { const seconds = Math.floor(value); return `${Math.floor(seconds / 3600).toString().padStart(2, '0')}:${Math.floor(seconds % 3600 / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`; }
function pct(value: unknown) { return typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '—'; }
function percentValue(value: unknown) { return typeof value === 'number' ? `${value.toFixed(2)}%` : '—'; }
function StandardScores({ results }: { results: Record<string, unknown> }) {
  const rows = benchmarkGroups[0].tasks.filter(([key]) => results[key]);
  if (!rows.length) return null;
  return <div className="result-group"><div className="result-group-heading"><div><div className="eyebrow">LM-EVAL</div><h3>Standard benchmarks</h3></div><p>Acc is raw accuracy. Acc norm scores choices by length-normalized likelihood.</p></div>
    <div className="score-table-wrap"><table className="score-table"><thead><tr><th>Benchmark</th><th>Acc</th><th>Acc norm</th><th>Samples</th></tr></thead><tbody>{rows.map(([key, name, description]) => {
      const result = results[key] as Record<string, unknown>;
      const acc = result['acc,none'];
      const accNorm = result['acc_norm,none'];
      const accError = result['acc_stderr,none'];
      const accNormError = result['acc_norm_stderr,none'];
      return <tr key={key}><td><strong>{name}</strong><small>{description}{typeof result['perplexity,none'] === 'number' ? ` · Perplexity ${Number(result['perplexity,none']).toFixed(2)}` : ''}</small></td><td className="score-value"><strong>{pct(acc)}</strong>{typeof accError === 'number' && <small>±{(accError * 100).toFixed(2)} pp</small>}</td><td className="score-value"><strong>{pct(accNorm)}</strong>{typeof accNormError === 'number' ? <small>±{(accNormError * 100).toFixed(2)} pp</small> : <small>{accNorm === undefined ? 'Not reported' : ''}</small>}</td><td className="score-samples">{typeof result.sample_len === 'number' ? result.sample_len.toLocaleString() : '—'}</td></tr>;
    })}</tbody></table></div>
  </div>;
}
function ContinuationScores({ results }: { results: Record<string, unknown> }) {
  const rows = benchmarkGroups[1].tasks.filter(([key]) => key !== 'base_bench' && key !== 'safety_bench' && results[key]);
  if (!rows.length) return null;
  return <div className="result-group"><div className="result-group-heading"><div><div className="eyebrow">CONTINUATION SUITES</div><h3>Choice accuracy</h3></div><p>Each choice is scored by conditional token likelihood.</p></div>
    <div className="score-table-wrap"><table className="score-table continuation-table"><thead><tr><th>Benchmark</th><th>Accuracy</th><th>Correct</th></tr></thead><tbody>{rows.map(([key, name, description]) => {
      const result = results[key] as Record<string, unknown>;
      return <tr key={key}><td><strong>{name}</strong><small>{description}</small></td><td className="score-value"><strong>{pct(result.accuracy)}</strong></td><td className="score-samples">{typeof result.correct === 'number' && typeof result.total === 'number' ? `${result.correct.toLocaleString()} / ${result.total.toLocaleString()}` : '—'}</td></tr>;
    })}</tbody></table></div>
  </div>;
}
function EvaluationResult({ task, raw }: { task: string; raw: unknown }) {
  const result = raw as Record<string, unknown>;
  const name = benchmarkGroups.flatMap(group => group.tasks).find(item => item[0] === task)?.[1] || task;
  if (task === 'base_bench') {
    const categories = (result.categories || {}) as Record<string, Record<string, number>>;
    return <div className="official-result"><div className="official-title"><div><strong>{name}</strong><small>{result.official ? 'Official complete run' : 'Limited run · non-official'}</small></div><button className="text-button" onClick={() => api.openPath(String(result.report_path))}><ExternalLink size={14}/> Open full report</button></div>
      <div className="official-summary"><div><span>Overall Elo</span><b>{String(result.overall_elo ?? '—')}</b></div><div><span>Accuracy</span><b>{pct(result.accuracy)}</b><small>{String(result.correct ?? '—')} / {String(result.total ?? '—')}</small></div><div><span>Weighted accuracy</span><b>{pct(result.weighted_accuracy)}</b></div></div>
      <div className="table-scroll"><table className="category-table"><thead><tr><th>Category</th><th>Elo</th><th>Accuracy</th><th>Weighted</th></tr></thead><tbody>{Object.entries(categories).map(([category, score]) => <tr key={category}><td>{category.replaceAll('_', ' ')}</td><td>{score.elo}</td><td>{pct(score.accuracy)}</td><td>{pct(score.weighted_accuracy)}</td></tr>)}</tbody></table></div></div>;
  }
  if (task === 'safety_bench') {
    const overall = (result.overall || {}) as Record<string, number>;
    const categories = (result.categories || {}) as Record<string, Record<string, number>>;
    return <div className="official-result"><div className="official-title"><div><strong>{name}</strong><small>{result.official ? 'Official complete run' : 'Limited run · non-official'}</small></div><button className="text-button" onClick={() => api.openPath(String(result.report_path))}><ExternalLink size={14}/> Open full report</button></div>
      <div className="official-summary"><div><span>Safety score</span><b>{overall.safety_score?.toFixed(2) ?? '—'}</b></div><div><span>Aligned accuracy</span><b>{percentValue(overall.aligned_accuracy)}</b></div><div><span>Misalignment index</span><b>{overall.misalignment_index?.toFixed(2) ?? '—'}</b></div></div>
      <div className="safety-breakdown"><span>Partial compliance <b>{percentValue(overall.partial_compliance_rate)}</b></span><span>Covert misalignment <b>{percentValue(overall.covert_misalignment_rate)}</b></span><span>Severe misalignment <b>{percentValue(overall.severe_misalignment_rate)}</b></span></div>
      <div className="table-scroll"><table className="category-table"><thead><tr><th>Category</th><th>Aligned</th><th>Misalignment</th><th>Severe</th></tr></thead><tbody>{Object.entries(categories).map(([category, score]) => <tr key={category}><td>{category.replaceAll('_', ' ')}</td><td>{percentValue(score.aligned_accuracy)}</td><td>{score.misalignment_index?.toFixed(2) ?? '—'}</td><td>{percentValue(score.severe_misalignment_rate)}</td></tr>)}</tbody></table></div></div>;
  }
  return null;
}

export default function App() {
  const [tab, setTab] = useState<Tab>('train');
  const [trainMode, setTrainMode] = useState<TrainMode>('pretraining');
  const [python, setPython] = useState('python3');
  const [outputDir, setOutputDir] = useState('');
  const [auth, setAuth] = useState<{ loggedIn: boolean; username?: string; message?: string } | null>(null);
  const [device, setDevice] = useState<DeviceStatus | null>(null);
  const [datasets, setDatasets] = useState<DatasetSource[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [datasetInput, setDatasetInput] = useState('');
  const [datasetError, setDatasetError] = useState('');
  const [model, setModel] = useState('BananaMind/BananaMind-2-Medium');
  const [parametersM, setParametersM] = useState(25);
  const [sizeInput, setSizeInput] = useState('25');
  const [architecture, setArchitecture] = useState('auto');
  const [customConfigPath, setCustomConfigPath] = useState('');
  const [customConfigurationPath, setCustomConfigurationPath] = useState('');
  const [customModelingPath, setCustomModelingPath] = useState('');
  const [trustRemoteCode, setTrustRemoteCode] = useState(true);
  const [runName, setRunName] = useState('bananaall-run');
  const [learningRate, setLearningRate] = useState(0.0002);
  const [batchSize, setBatchSize] = useState(2);
  const [gradientAccumulation, setGradientAccumulation] = useState(8);
  const [maxSteps, setMaxSteps] = useState(500);
  const [sequenceLength, setSequenceLength] = useState(1024);
  const [precision, setPrecision] = useState('auto');
  const [compileTraining, setCompileTraining] = useState(true);
  const [compileUnavailable, setCompileUnavailable] = useState(false);
  const [scheduler, setScheduler] = useState('cosine');
  const [warmupRatio, setWarmupRatio] = useState(0.03);
  const [weightDecay, setWeightDecay] = useState(0.01);
  const [maxGradNorm, setMaxGradNorm] = useState(1);
  const [loggingSteps, setLoggingSteps] = useState(10);
  const [saveSteps, setSaveSteps] = useState(100);
  const [seed, setSeed] = useState(1337);
  const [tokenizerSamples, setTokenizerSamples] = useState(2000);
  const [tokenizerSource, setTokenizerSource] = useState<'train' | 'existing'>('train');
  const [tokenizerModel, setTokenizerModel] = useState('');
  const [tokenizerTrustRemoteCode, setTokenizerTrustRemoteCode] = useState(false);
  const [loraRank, setLoraRank] = useState(16);
  const [loraAlpha, setLoraAlpha] = useState(32);
  const [loraDropout, setLoraDropout] = useState(0.05);
  const [advanced, setAdvanced] = useState(false);
  const [codeModal, setCodeModal] = useState(false);
  const [trainPicker, setTrainPicker] = useState(false);
  const [trainingTarget, setTrainingTarget] = useState<TrainingTarget>('in-app');
  const [notebookExport, setNotebookExport] = useState<{ path: string; zipName: string; tokenEmbedded: boolean; tokenNeeded: boolean } | null>(null);
  const [codeDraft, setCodeDraft] = useState('');
  const [codeEditing, setCodeEditing] = useState(false);
  const [reviewSettings, setReviewSettings] = useState<ReviewSettings | null>(null);
  const [reviewSetupOpen, setReviewSetupOpen] = useState(false);
  const [reviewKey, setReviewKey] = useState('');
  const [reviewSetupError, setReviewSetupError] = useState('');
  const [reviewing, setReviewing] = useState(false);
  const [reviewNotice, setReviewNotice] = useState('');
  const [reviewError, setReviewError] = useState('');
  const [jobs, setJobs] = useState<Partial<Record<Tab, JobState>>>({});
  const [error, setError] = useState('');
  const [evalModel, setEvalModel] = useState('BananaMind/BananaMind-2-Medium');
  const [evalTasks, setEvalTasks] = useState<string[]>(['piqa', 'arc_easy', 'hellaswag']);
  const [evalTrust, setEvalTrust] = useState(true);
  const [evalBatch, setEvalBatch] = useState(1);
  const [evalFewShot, setEvalFewShot] = useState(0);
  const [evalLimit, setEvalLimit] = useState(0);
  const [inferModel, setInferModel] = useState('BananaMind/BananaMind-2-Medium-Chat');
  const [inferMode, setInferMode] = useState<'instruct' | 'base'>('instruct');
  const [inferTrust, setInferTrust] = useState(true);
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful assistant.');
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([]);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.95);
  const [topK, setTopK] = useState(50);
  const [maxNewTokens, setMaxNewTokens] = useState(128);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1);
  const [showLogs, setShowLogs] = useState(false);

  useEffect(() => { api.system().then(info => { setOutputDir(info.defaultOutput); setPython(info.python); }).catch(() => {}); }, []);
  useEffect(() => { api.reviewSettings().then(settings => { setReviewSettings(settings); setReviewSetupOpen(settings.firstRun); }).catch(() => {}); }, []);
  useEffect(() => { api.authStatus(python).then(setAuth).catch(e => setAuth({ loggedIn: false, message: e.message })); }, [python]);
  useEffect(() => {
    let cancelled = false;
    setDevice(null);
    const timer = window.setTimeout(() => {
      api.deviceStatus(python)
        .then(status => { if (!cancelled) setDevice(status); })
        .catch(e => { if (!cancelled) setDevice({ backend: 'unavailable', name: 'Python unavailable', message: e.message }); });
    }, 350);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [python]);
  useEffect(() => {
    if (architecture === 'ternary' && device && device.backend !== 'cuda') setArchitecture('auto');
  }, [architecture, device]);
  useEffect(() => { setCompileUnavailable(false); setCompileTraining(true); }, [python]);
  useEffect(() => api.onJobEvent((event: JobEvent) => {
    if (event.type === 'compile-disabled') {
      setCompileTraining(false);
      setCompileUnavailable(true);
    }
    setJobs(current => {
      const kind = (Object.keys(current) as Tab[]).find(key => current[key]?.id === event.id);
      if (!kind) return current;
      const job = current[kind]!;
      if (event.type === 'progress') return { ...current, [kind]: { ...job, progress: { ...job.progress, ...event } } };
      const next: JobState = { ...job, events: [...job.events, event].slice(-500) };
      if (event.type === 'compile-disabled') next.compileDisabled = true;
      if (event.type === 'architecture' && typeof event.parameters === 'number') next.modelParameters = event.parameters;
      if (event.type === 'metric' && typeof event.loss === 'number') next.metrics = [...job.metrics, { step: Number(event.step || 0), loss: event.loss }];
      if (event.type === 'result' && event.task) next.results = { ...job.results, [event.task]: event.result };
      if (event.type === 'complete') { next.status = 'complete'; if (event.results) next.results = event.results; }
      if (event.type === 'error') { next.status = 'error'; next.error = event.message; }
      if (event.type === 'exit' && job.status === 'running') next.status = event.code === 0 ? 'complete' : 'stopped';
      return { ...current, [kind]: next };
    });
    if (event.type === 'complete' && typeof event.text === 'string') {
      setMessages(current => [...current, { role: 'assistant', content: event.text! }]);
    }
  }), []);

  const selected = datasets.find(source => source.key === selectedDataset);
  const activeDatasets = datasets.filter(source => source.enabled && source.weight > 0);
  const allTokenTargetsSet = activeDatasets.length > 0 && activeDatasets.every(source => source.tokenLimitM !== null && source.tokenLimitM > 0 && Math.round(source.tokenLimitM * 1_000_000) > 0);
  const totalTargetM = activeDatasets.reduce((sum, source) => sum + (source.tokenLimitM || 0), 0);
  const spec = useMemo(() => architectureForSize(parametersM), [parametersM]);
  const estimatedM = estimateParameters(spec, architecture !== 'lft') / 1_000_000;
  const trainJob = jobs.train, evalJob = jobs.evaluate, inferJob = jobs.infer;

  async function inspect(source: DatasetSource) {
    setDatasets(current => current.map(item => item.key === source.key ? { ...item, status: 'loading', error: '' } : item));
    try {
      const result = await api.inspectDataset(source, python);
      setDatasets(current => current.map(item => item.key === source.key ? {
        ...item, columns: result.columns, samples: result.samples,
        mapping: Object.values(item.mapping).some(Boolean) ? item.mapping : result.mapping,
        status: 'ready',
      } : item));
    } catch (e) {
      setDatasets(current => current.map(item => item.key === source.key ? { ...item, status: 'error', error: String((e as Error).message) } : item));
    }
  }
  function addDataset(preset: { name: string; id: string; config?: string; split?: string; mapping?: Mapping }) {
    const existing = datasets.find(item => item.id === preset.id && item.config === (preset.config || ''));
    if (existing) { setSelectedDataset(existing.key); return; }
    const source: DatasetSource = { key: crypto.randomUUID(), id: preset.id.trim(), name: preset.name,
      config: preset.config || '', split: preset.split || 'train', weight: 1, tokenLimitM: null, enabled: true,
      mapping: preset.mapping ? { ...preset.mapping } : { ...emptyMapping }, samples: [], columns: [], status: 'idle' };
    setDatasets(current => [...current, source]);
    setSelectedDataset(source.key);
    inspect(source);
  }
  function updateDataset(key: string, patch: Partial<DatasetSource>) {
    setDatasets(current => current.map(item => item.key === key ? { ...item, ...patch } : item));
  }
  function addCustomDataset() {
    const id = datasetInput.trim();
    if (!id) return;
    setDatasetError('');
    addDataset({ name: id.split('/').pop() || id, id });
    setDatasetInput('');
  }
  async function pickDatasetFile() {
    const path = await api.pickPath('file');
    if (path) addDataset({ name: path.split('/').pop() || 'Local dataset', id: path });
  }
  function trainingConfig() {
    return { python, outputDir, runName, mode: trainMode, model, trustRemoteCode,
      parametersM, architecture, customConfigPath, customConfigurationPath, customModelingPath,
      datasets: datasets.filter(item => item.enabled).map(({ id, config, split, weight, tokenLimitM, enabled, mapping }) => ({ id, config, split, weight, tokenLimit: trainMode === 'pretraining' && tokenLimitM !== null ? Math.round(tokenLimitM * 1_000_000) : null, enabled, mapping })),
      learningRate, batchSize, gradientAccumulation, maxSteps: trainMode === 'pretraining' ? undefined : maxSteps, sequenceLength, precision, compile: compileTraining && !compileUnavailable, scheduler,
      warmupRatio, weightDecay, maxGradNorm, loggingSteps, saveSteps, seed, tokenizerSamples, tokenizerSource, tokenizerModel, tokenizerTrustRemoteCode,
      loraRank, loraAlpha, loraDropout };
  }
  async function startJob(kind: Tab, config: Record<string, unknown>, code?: string) {
    setError('');
    try {
      const result = await api.startJob(kind === 'evaluate' ? 'evaluate' : kind === 'infer' ? 'infer' : 'train', config, code);
      setJobs(current => ({ ...current, [kind]: { id: result.id, kind, status: 'running', outputPath: result.outputPath, events: [], metrics: [], results: {} } }));
      setShowLogs(false);
    } catch (e) { setError(String((e as Error).message)); }
  }
  function requestTraining() {
    if (!datasets.some(item => item.enabled && item.weight > 0)) { setError('Add and enable at least one dataset.'); return; }
    if (trainMode === 'pretraining' && !allTokenTargetsSet) { setError('Set a positive token target for every enabled dataset.'); return; }
    if (trainMode !== 'pretraining' && !model.trim()) { setError('Enter a base model.'); return; }
    if (trainMode === 'pretraining' && tokenizerSource === 'existing' && !tokenizerModel.trim()) { setError('Enter the model ID or folder containing the tokenizer.'); return; }
    if (trainMode === 'pretraining' && architecture === 'custom' && (!customConfigPath || !customConfigurationPath || !customModelingPath)) { setError('Choose config.json, configuration Python, and modeling Python files.'); return; }
    if (trainMode === 'pretraining' && architecture === 'ternary' && device?.backend !== 'cuda') { setError('Ternary training requires an NVIDIA GPU in the selected Python environment.'); return; }
    if (trainMode === 'pretraining' && architecture === 'ternary' && !trustRemoteCode) { setError('Enable Trust model code to train and reload the Ternary architecture.'); return; }
    setError('');
    setTrainPicker(true);
  }
  async function exportNotebook(code?: string) {
    try {
      const result = await api.exportNotebook(trainingConfig(), code);
      if (result) setNotebookExport(result);
    } catch (e) { setError(String((e as Error).message)); }
  }
  async function chooseTraining(target: TrainingTarget) {
    setTrainPicker(false);
    setTrainingTarget(target);
    if (trainMode === 'pretraining' && architecture === 'custom') {
      try { setCodeDraft(await api.trainingCode(customConfigPath, customConfigurationPath, customModelingPath)); setCodeEditing(false); setReviewNotice(''); setReviewError(''); setCodeModal(true); }
      catch (e) { setError(String((e as Error).message)); }
      return;
    }
    if (target === 'notebook') await exportNotebook();
    else await startJob('train', trainingConfig());
  }
  async function saveReviewKey() {
    setReviewSetupError('');
    try { setReviewSettings(await api.saveReviewKey(reviewKey)); setReviewKey(''); setReviewSetupOpen(false); }
    catch (e) { setReviewSetupError(String((e as Error).message)); }
  }
  async function skipReviewSetup() {
    try { setReviewSettings(await api.skipReviewSetup()); setReviewKey(''); setReviewSetupOpen(false); }
    catch (e) { setReviewSetupError(String((e as Error).message)); }
  }
  async function runAiReview() {
    if (!reviewSettings?.hasKey) { setReviewSetupOpen(true); return; }
    setReviewError(''); setReviewNotice(''); setReviewing(true);
    try {
      const result = await api.aiReview({ code: codeDraft, configPath: customConfigPath, configurationPath: customConfigurationPath, modelingPath: customModelingPath, python });
      if (result.revisedCode !== null && result.revisedCode !== codeDraft) { setCodeDraft(result.revisedCode); setCodeEditing(true); }
      setReviewNotice(result.summary || (result.revisedCode ? 'Review complete. Check the suggested edits before continuing.' : 'No changes suggested.'));
    } catch (e) { setReviewError(String((e as Error).message)); }
    finally { setReviewing(false); }
  }
  async function startEval() {
    if (!evalTasks.length) { setError('Choose at least one benchmark.'); return; }
    await startJob('evaluate', { python, outputDir, model: evalModel, tasks: evalTasks,
      trustRemoteCode: evalTrust, batchSize: evalBatch, fewShot: evalFewShot, limit: evalLimit });
  }
  async function startInfer() {
    if (!prompt.trim()) return;
    const nextMessages = inferMode === 'instruct' ? [...messages, { role: 'user' as const, content: prompt }] : [];
    if (inferMode === 'instruct') setMessages(nextMessages);
    await startJob('infer', { python, outputDir, model: inferModel, trustRemoteCode: inferTrust,
      inferenceMode: inferMode, systemPrompt, messages: nextMessages, prompt,
      temperature, topP, topK, maxNewTokens, repetitionPenalty });
    setPrompt('');
  }
  function useOutputFor(target: 'evaluate' | 'infer') {
    if (!trainJob?.outputPath) return;
    if (target === 'evaluate') { setEvalModel(trainJob.outputPath); setEvalTrust(true); setTab('evaluate'); }
    else { setInferModel(trainJob.outputPath); setInferTrust(true); setTab('infer'); }
  }
  function renderJob(job?: JobState) {
    if (!job) return null;
    const latest = [...job.events].reverse().find(item => item.type === 'status' && item.message);
    const latestMetric = [...job.events].reverse().find(item => item.type === 'metric' && typeof item.loss === 'number');
    const step = job.progress?.step ?? job.metrics.at(-1)?.step ?? 0;
    const totalSteps = job.progress?.totalSteps ?? latestMetric?.totalSteps;
    const learningRate = job.progress?.learningRate ?? latestMetric?.learning_rate;
    const progressPercent = totalSteps ? Math.min(100, step / totalSteps * 100) : 0;
    return <div className="job-panel">
      <div className="job-top"><span className={`status-dot ${job.status}`} /><strong>{statusLabel(job)}</strong><span className="muted">{latest?.message || job.outputPath}</span>
        {job.status === 'running' && <button className="text-button danger" onClick={() => api.stopJob(job.id)}><Square size={13} /> Stop</button>}</div>
      {job.kind === 'train' && <>
        <div className="run-progress"><div className="run-progress-head"><span>OPTIMIZER STEPS</span><strong>{step.toLocaleString()} <em>/ {totalSteps?.toLocaleString() ?? '—'}</em></strong></div><div className="run-progress-track"><div style={{ width: `${progressPercent}%` }}/></div><small>{progressPercent.toFixed(1)}% complete</small></div>
        <div className="run-stats">
          <div><span>Latest loss</span><strong>{job.metrics.at(-1)?.loss?.toFixed(4) ?? '—'}</strong></div>
          <div><span>Learning rate</span><strong>{learningRate !== undefined ? learningRate.toExponential(2) : '—'}</strong></div>
          <div><span>Grad accumulation</span><strong>{job.progress?.accumulationStep ?? 0} / {job.progress?.accumulationTotal ?? '—'}</strong></div>
          <div><span>Grad norm</span><strong>{latestMetric?.grad_norm?.toFixed(3) ?? '—'}</strong></div>
          <div><span>Speed</span><strong>{job.progress?.tokensPerSecond && job.progress.tokensPerSecond > 0 ? `${job.progress.tokensPerSecond.toLocaleString(undefined, { maximumFractionDigits: 1 })} tok/s` : '—'}</strong></div>
          {job.progress?.totalTokens !== undefined && <div><span>Dataset tokens</span><strong>{compactNumber(job.progress?.tokensSeen ?? 0)} / {compactNumber(job.progress?.totalTokens ?? 0)}</strong></div>}
          <div><span>Elapsed</span><strong>{job.progress?.elapsedSeconds !== undefined ? elapsedTime(job.progress.elapsedSeconds) : '—'}</strong></div>
          {job.modelParameters !== undefined && <div><span>Parameters</span><strong>{compactNumber(job.modelParameters)}</strong></div>}
        </div>
        {job.compileDisabled && <div className="compile-note">Compile hit an error and was turned off. This run is using normal training.</div>}
        <div className="chart-wrap">{job.metrics.length > 1 ? <ResponsiveContainer width="100%" height={210}><LineChart data={job.metrics}><CartesianGrid stroke="#373737" vertical={false}/><XAxis dataKey="step" tickLine={false} axisLine={false} tick={{ fill: '#a7a7a7', fontSize: 11 }}/><YAxis tickLine={false} axisLine={false} tick={{ fill: '#a7a7a7', fontSize: 11 }} domain={['auto', 'auto']}/><Tooltip contentStyle={{ borderRadius: 10, border: '1px solid #454545', background: '#242424', color: '#f1f1f1' }}/><Line type="monotone" dataKey="loss" stroke="#f2f2f2" strokeWidth={2} dot={false}/></LineChart></ResponsiveContainer> : <div className="chart-empty"><Activity size={18}/> Loss history appears as training logs arrive</div>}</div>
      </>}
      {job.error && <div className="inline-error"><TriangleAlert size={15}/>{job.error}</div>}
      <div className="job-actions"><button className="text-button" onClick={() => api.openPath(job.outputPath)}><FolderOpen size={15}/> Open output folder</button><button className="text-button" onClick={() => setShowLogs(value => !value)}><Terminal size={15}/>{showLogs ? 'Hide' : 'Show'} logs</button></div>
      {showLogs && <div className="log-view">{job.events.map((event, index) => <div key={index}>{event.type === 'log' ? event.message : event.message || `${event.type}${event.step ? ` · step ${event.step}` : ''}${event.loss !== undefined ? ` · loss ${event.loss}` : ''}`}</div>)}</div>}
    </div>;
  }

  return <div className="app-shell">
    <header className="topbar"><div className="brand"><div className="brand-mark" role="img" aria-label="BananaAll logo"/><span>BananaAll</span><small>SLM STUDIO</small></div>
      <nav className="top-tabs" aria-label="Main navigation"><button className={tab === 'train' ? 'active' : ''} onClick={() => setTab('train')}>Train</button><button className={tab === 'evaluate' ? 'active' : ''} onClick={() => setTab('evaluate')}>Evaluation</button><button className={tab === 'infer' ? 'active' : ''} onClick={() => setTab('infer')}>Inference</button></nav>
      <div className="top-right">{device && <span className={`auth-pill compute-pill ${['rocm', 'cuda', 'xpu', 'mps'].includes(device.backend) ? 'connected' : ''}`} title={device.message || device.name}><span className="status-dot"/><span className="compute-label">{device.backend === 'rocm' ? `ROCm · ${device.name}` : device.backend === 'cuda' ? `CUDA · ${device.name}` : device.backend === 'xpu' ? `Intel XPU · ${device.name}` : device.backend === 'mps' ? `MPS · ${device.name}` : device.name}</span></span>}<span className={`auth-pill ${auth?.loggedIn ? 'connected' : ''}`}><span className="status-dot"/>{auth?.loggedIn ? `HF · ${auth.username}` : auth?.message || 'Checking HF login'}</span><button className="icon-button" title="Refresh Hugging Face status" onClick={() => api.authStatus(python).then(setAuth)}><Globe2 size={17}/></button><button className="icon-button" title="OpenRouter AI review settings" aria-label="OpenRouter AI review settings" onClick={() => { setReviewSetupError(''); setReviewSetupOpen(true); }}><LockKeyhole size={17}/></button></div></header>
    <main className="main">
      {error && <div className="global-error"><TriangleAlert size={17}/><span>{error}</span><button onClick={() => setError('')}><X size={17}/></button></div>}
      {tab === 'train' && <>
        <div className="page-intro"><div><div className="eyebrow">WORKSPACE / TRAIN</div><h1>Make a model your own.</h1><p>From a fresh architecture to a focused adapter. Set the data, shape the run, and watch every step.</p></div><span className="intro-index">01 / 03</span></div>
        <div className="mode-tabs" role="tablist"><button className={trainMode === 'pretraining' ? 'active' : ''} onClick={() => setTrainMode('pretraining')}><Sparkles size={17}/><span>Pretraining</span><small>Start from scratch</small></button><button className={trainMode === 'full' ? 'active' : ''} onClick={() => setTrainMode('full')}><SlidersHorizontal size={17}/><span>Full fine tune</span><small>Update every weight</small></button><button className={trainMode === 'lora' ? 'active' : ''} onClick={() => setTrainMode('lora')}><GitBranch size={17}/><span>LoRA</span><small>Train a small adapter</small></button></div>
        <div className="content-grid"><div className="primary-column">
          <Section eyebrow="01 · SOURCE" title="Datasets" aside={<span className="quiet-count">{datasets.filter(item => item.enabled).length} in mix</span>} id="datasets">
            <p className="section-copy">Choose one or more sources. BananaAll previews rows and suggests a field mapping before training.{trainMode === 'pretraining' && ' Set a token target for every enabled source in millions. Training ends after those tokens are consumed.'}</p>
            {trainMode === 'pretraining' && <div className="preset-grid">{datasetPresets.map(preset => { const added = datasets.some(item => item.id === preset.id && item.config === preset.config); return <button className={`preset ${added ? 'added' : ''}`} key={preset.name} onClick={() => addDataset(preset)}><span><Database size={16}/>{preset.name}</span><small>{preset.note}</small>{added ? <Check size={16}/> : <Plus size={16}/>}</button>; })}</div>}
            <div className="add-source"><input value={datasetInput} onChange={e => setDatasetInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && addCustomDataset()} placeholder="Hugging Face dataset ID or URL"/><button className="secondary-button" onClick={addCustomDataset}><Plus size={15}/> Add source</button><button className="icon-button bordered" title="Add local file" onClick={pickDatasetFile}><FolderOpen size={17}/></button></div>
            {datasetError && <div className="inline-error">{datasetError}</div>}
            {datasets.length > 0 && <div className="dataset-list">{datasets.map(source => <div className={`dataset-row ${selectedDataset === source.key ? 'selected' : ''}`} key={source.key}>
              <input type="checkbox" aria-label={`Include ${source.name}`} checked={source.enabled} onChange={e => updateDataset(source.key, { enabled: e.target.checked })}/>
              <button className="dataset-select" onClick={() => setSelectedDataset(source.key)}><strong>{source.name}</strong><small>{source.id}{source.config ? ` · ${source.config}` : ''}</small></button>
              <span className={`source-status ${source.status}`}>{source.status === 'loading' ? <LoaderCircle size={13} className="spin"/> : source.status === 'ready' ? <Check size={13}/> : source.status === 'error' ? <TriangleAlert size={13}/> : null}{source.status}</span>
              <label className="weight-field">mix <input type="number" min="0" step="0.1" value={source.weight} onChange={e => updateDataset(source.key, { weight: Number(e.target.value) })}/></label>
              {trainMode === 'pretraining' && <label className="token-limit-field"><span>tokens</span><input type="number" min="0.000001" step="any" placeholder="set" aria-label={`${source.name} token target in millions`} value={source.tokenLimitM ?? ''} onChange={e => updateDataset(source.key, { tokenLimitM: e.target.value === '' ? null : Math.max(0, Number(e.target.value)) })}/><span>M</span></label>}
              <button className="icon-button faint" title="Remove source" onClick={() => { setDatasets(current => current.filter(item => item.key !== source.key)); if (selectedDataset === source.key) setSelectedDataset(null); }}><Trash2 size={15}/></button>
            </div>)}</div>}
            {selected && <div className="mapping-panel"><div className="panel-header"><div><div className="eyebrow">DATA PREVIEW</div><h3>{selected.name}</h3></div><button className="text-button" onClick={() => inspect(selected)}><ArrowDownToLine size={15}/> Reload samples</button></div>
              {selected.status === 'loading' && <div className="loading-line"><LoaderCircle size={15} className="spin"/> Loading samples and detecting fields…</div>}
              {selected.status === 'error' && <div className="inline-error"><TriangleAlert size={15}/>{selected.error}</div>}
              {selected.status === 'ready' && <><div className="mapping-grid">{(trainMode === 'pretraining' ? ['text', 'system', 'user', 'assistant'] : ['messages', 'system', 'user', 'assistant', 'text']).map(role => <Field label={role === 'text' ? 'Document text' : role === 'messages' ? 'Messages array' : `${role[0].toUpperCase()}${role.slice(1)} field`} key={role}><select value={selected.mapping[role as keyof Mapping]} onChange={e => updateDataset(selected.key, { mapping: { ...selected.mapping, [role]: e.target.value } })}><option value="">Ignore</option>{selected.columns.map(column => <option key={column} value={column}>{column}</option>)}</select></Field>)}</div>
                <div className="sample-title">Sample 1 of {selected.samples.length}</div><pre className="sample-json">{JSON.stringify(selected.samples[0], null, 2)}</pre></>}
              <div className="mapping-foot">Split <input value={selected.split} onChange={e => updateDataset(selected.key, { split: e.target.value })}/><span>Config <input value={selected.config} placeholder="default" onChange={e => updateDataset(selected.key, { config: e.target.value })}/></span></div>
            </div>}
          </Section>
          <Section eyebrow="02 · MODEL" title={trainMode === 'pretraining' ? 'Architecture' : 'Base model'} id="architecture">
            {trainMode === 'pretraining' ? <>
              <div className="size-setting"><div><strong>Approximate model size</strong><p>Slide anywhere from 3M to 200M. BananaAll generates the layer, attention, vocabulary, and feed-forward dimensions; the original five sizes retain their exact configurations.</p></div><div className="size-value"><input type="number" min="3" max="200" step="0.1" value={sizeInput} onChange={e => { const value = e.target.value; setSizeInput(value); const parsed = Number(value); if (value !== '' && Number.isFinite(parsed) && parsed >= 3 && parsed <= 200) setParametersM(parsed); }} onBlur={() => setSizeInput(String(parametersM))}/><span>M parameters</span></div></div>
              <input className="size-slider" type="range" min="3" max="200" step="0.1" value={parametersM} onChange={e => { const value = Number(e.target.value); setParametersM(value); setSizeInput(String(value)); }}/><div className="scale-labels"><span>3M</span><span>50M</span><span>100M</span><span>150M</span><span>200M</span></div>
              <div className="architecture-options">{[
                { key: 'auto', title: 'Auto', desc: 'BananaMind 2 style at the selected size', icon: <WandSparkles size={19}/> },
                { key: 'bananamind2', title: 'BananaMind 2 style', desc: 'GQA, QK norm, RoPE and SwiGLU', icon: <Sparkles size={19}/> },
                { key: 'lft', title: 'LFT', desc: 'Adjacent layer feedback over a Llama style block', icon: <GitBranch size={19}/> },
                { key: 'bananamind2lft', title: 'BananaMind 2 + LFT', desc: 'BananaMind block with feedback routing', icon: <Activity size={19}/> },
                { key: 'ternary', title: 'Ternary · 1.58-bit', desc: 'BananaMind 2 with ternary weight and 8-bit activation fake quantization', icon: <Beaker size={19}/>, tags: ['NVIDIA', 'Experimental'] },
                { key: 'custom', title: 'Custom', desc: 'Experimental · review the training code', icon: <Beaker size={19}/> },
              ].map(option => <button key={option.key} className={`arch-option ${architecture === option.key ? 'active' : ''}`} disabled={option.key === 'ternary' && device?.backend !== 'cuda'} title={option.key === 'ternary' && device?.backend !== 'cuda' ? 'Requires an NVIDIA GPU in the selected Python environment' : undefined} onClick={() => setArchitecture(option.key)}><span className="arch-icon">{option.icon}</span><span><strong>{option.title}</strong>{option.tags && <span className="arch-tags">{option.tags.map(tag => <em key={tag}>{tag}</em>)}</span>}<small>{option.desc}</small></span><span className="radio-indicator"/></button>)}</div>
              {architecture === 'ternary' && <div className="custom-panel ternary-panel"><div className="experimental"><Beaker size={15}/> Experimental · NVIDIA only</div><p>BananaMind 2 projections use ternary weight and 8-bit activation fake quantization during training. Master weights and checkpoints stay in normal precision; this does not create a packed 1.58-bit model or reduce training memory.</p><Toggle checked={trustRemoteCode} onChange={setTrustRemoteCode} label="Trust model code" hint="Required to load this architecture and its saved custom layers with trust_remote_code=True."/></div>}
              {architecture === 'custom' && <div className="custom-panel"><div className="experimental"><Beaker size={15}/> Experimental architecture</div><p>Provide <code>config.json</code>, its <code>PretrainedConfig</code> Python module, and the modeling Python module. The names must match <code>auto_map</code>. The saved model vocabulary will match the tokenizer chosen below.</p><div className="path-pair"><Field label="Config JSON"><div className="path-input"><input value={customConfigPath} onChange={e => setCustomConfigPath(e.target.value)} placeholder="/path/to/config.json"/><button onClick={async () => { const path = await api.pickPath('file'); if (path) setCustomConfigPath(path); }}><FolderOpen size={16}/></button></div></Field><Field label="Configuration Python"><div className="path-input"><input value={customConfigurationPath} onChange={e => setCustomConfigurationPath(e.target.value)} placeholder="/path/to/configuration_*.py"/><button onClick={async () => { const path = await api.pickPath('file'); if (path) setCustomConfigurationPath(path); }}><FolderOpen size={16}/></button></div></Field><Field label="Modeling Python"><div className="path-input"><input value={customModelingPath} onChange={e => setCustomModelingPath(e.target.value)} placeholder="/path/to/modeling_*.py"/><button onClick={async () => { const path = await api.pickPath('file'); if (path) setCustomModelingPath(path); }}><FolderOpen size={16}/></button></div></Field></div><Toggle checked={trustRemoteCode} onChange={setTrustRemoteCode} label="Trust custom model code" hint="Runs Python supplied by this architecture. Review the files before using them."/></div>}
              <div className="tokenizer-panel"><div className="eyebrow">TOKENIZER</div><div className="segmented tokenizer-choice" role="group" aria-label="Tokenizer source"><button className={tokenizerSource === 'train' ? 'active' : ''} onClick={() => setTokenizerSource('train')}>Train from datasets</button><button className={tokenizerSource === 'existing' ? 'active' : ''} onClick={() => setTokenizerSource('existing')}>Use existing model</button></div>{tokenizerSource === 'existing' ? <><Field label="Tokenizer model ID or local folder"><div className="path-input"><input value={tokenizerModel} onChange={e => setTokenizerModel(e.target.value)} placeholder="org/model-name or /path/to/model"/><button title="Choose local tokenizer folder" onClick={async () => { const path = await api.pickPath('directory'); if (path) setTokenizerModel(path); }}><FolderOpen size={16}/></button></div></Field><Toggle checked={tokenizerTrustRemoteCode} onChange={setTokenizerTrustRemoteCode} label="Trust tokenizer code" hint="Enable only if this tokenizer repository needs custom Python code."/><p>Loads only the tokenizer; model weights still start from scratch. The architecture vocabulary expands or shrinks to match it. Notebook export requires a Hugging Face model ID or URL.</p></> : <p>Train a byte-level BPE tokenizer from the selected datasets. For custom architectures, its vocabulary size comes from config.json.</p>}</div>
              <details className="spec-details" open><summary>Generated architecture · ≈{estimatedM.toFixed(2)}M parameters <ChevronDown size={15}/></summary><div className="spec-grid">{Object.entries(spec).map(([key, value]) => <div key={key}><span>{key}</span><strong>{value.toLocaleString()}</strong></div>)}</div><p>{architecture.includes('lft') || architecture === 'lft' ? `${spec.num_hidden_layers} stored blocks · ${3 * spec.num_hidden_layers - 4} block executions per pass.` : `${spec.num_hidden_layers} stored blocks · ${spec.num_hidden_layers} block executions per pass.`}</p></details>
            </> : <><p className="section-copy">Use a Hugging Face model ID, URL, or local checkpoint folder.</p><div className="model-path"><input value={model} onChange={e => setModel(e.target.value)} placeholder="org/model-name"/><button className="icon-button bordered" title="Choose local model folder" onClick={async () => { const path = await api.pickPath('directory'); if (path) setModel(path); }}><FolderOpen size={17}/></button></div><Toggle checked={trustRemoteCode} onChange={setTrustRemoteCode} label="Trust remote model code" hint="Needed for model repositories with custom Python architectures."/>{trainMode === 'lora' && <div className="settings-grid three"><Field label="LoRA rank"><input type="number" min="1" value={loraRank} onChange={e => setLoraRank(Number(e.target.value))}/></Field><Field label="LoRA alpha"><input type="number" min="1" value={loraAlpha} onChange={e => setLoraAlpha(Number(e.target.value))}/></Field><Field label="LoRA dropout"><input type="number" min="0" max="1" step="0.01" value={loraDropout} onChange={e => setLoraDropout(Number(e.target.value))}/></Field></div>}</>}
          </Section>
          <Section eyebrow="03 · OPTIMIZE" title="Training settings" id="settings">
            <div className="settings-grid"><Field label="Learning rate" hint="Peak optimizer learning rate"><div className="lr-control"><input type="range" min="-6" max="-2" step="0.01" value={Math.log10(Math.max(learningRate, 1e-6))} onChange={e => setLearningRate(Number((10 ** Number(e.target.value)).toPrecision(3)))}/><input type="number" min="0.000001" max="0.01" step="0.00001" value={learningRate} onChange={e => setLearningRate(Number(e.target.value))}/></div></Field>{trainMode === 'pretraining' ? <Field label="Training tokens" hint="The run stops when every dataset reaches its token target"><div className="token-target-readout">{allTokenTargetsSet ? `${Number(totalTargetM.toFixed(6)).toLocaleString()}M tokens across ${activeDatasets.length} dataset${activeDatasets.length === 1 ? '' : 's'}` : 'Set tokens for every enabled dataset'}</div></Field> : <Field label="Max steps"><input type="number" min="1" value={maxSteps} onChange={e => setMaxSteps(Number(e.target.value))}/></Field>}<Field label="Micro batch size"><input type="number" min="1" value={batchSize} onChange={e => setBatchSize(Number(e.target.value))}/></Field><Field label="Gradient accumulation"><input type="number" min="1" value={gradientAccumulation} onChange={e => setGradientAccumulation(Number(e.target.value))}/></Field><Field label="Sequence length"><input type="number" min="32" step="32" value={sequenceLength} onChange={e => setSequenceLength(Number(e.target.value))}/></Field><Field label="Precision"><select value={precision} onChange={e => setPrecision(e.target.value)}><option value="auto">Auto</option><option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp32">FP32</option></select></Field></div>
            <label className={`compile-setting ${compileUnavailable ? 'disabled' : ''}`}><input type="checkbox" checked={compileTraining && !compileUnavailable} disabled={compileUnavailable} onChange={e => setCompileTraining(e.target.checked)}/><span><strong>Compile model</strong><small>{compileUnavailable ? 'Disabled after a compile error. Change the Python environment or restart the app to retry.' : 'Uses torch.compile. The first step takes longer; compilation errors switch back to normal training.'}</small></span></label>
            <button className="advanced-button" onClick={() => setAdvanced(!advanced)}><Settings2 size={16}/>{advanced ? 'Hide' : 'Show'} advanced settings <ChevronDown size={15} className={advanced ? 'rotated' : ''}/></button>
            {advanced && <div className="settings-grid advanced-grid"><Field label="Scheduler"><select value={scheduler} onChange={e => setScheduler(e.target.value)}><option value="cosine">Cosine</option><option value="linear">Linear</option><option value="constant">Constant</option><option value="cosine_with_restarts">Cosine with restarts</option></select></Field><Field label="Warmup ratio"><input type="number" min="0" max="0.9" step="0.01" value={warmupRatio} onChange={e => setWarmupRatio(Number(e.target.value))}/></Field><Field label="Weight decay"><input type="number" min="0" step="0.01" value={weightDecay} onChange={e => setWeightDecay(Number(e.target.value))}/></Field><Field label="Gradient clip"><input type="number" min="0" step="0.1" value={maxGradNorm} onChange={e => setMaxGradNorm(Number(e.target.value))}/></Field><Field label="Log every steps"><input type="number" min="1" value={loggingSteps} onChange={e => setLoggingSteps(Number(e.target.value))}/></Field><Field label="Save every steps"><input type="number" min="1" value={saveSteps} onChange={e => setSaveSteps(Number(e.target.value))}/></Field><Field label="Seed"><input type="number" value={seed} onChange={e => setSeed(Number(e.target.value))}/></Field>{trainMode === 'pretraining' && tokenizerSource === 'train' && <Field label="Tokenizer sample rows"><input type="number" min="100" value={tokenizerSamples} onChange={e => setTokenizerSamples(Number(e.target.value))}/></Field>}<Field label="Python executable"><input value={python} onChange={e => setPython(e.target.value)}/></Field></div>}
          </Section>
          <Section eyebrow="04 · LAUNCH" title="Run & output" id="run"><div className="settings-grid"><Field label="Run name"><input value={runName} onChange={e => setRunName(e.target.value)}/></Field><Field label="Output directory"><div className="path-input"><input value={outputDir} onChange={e => setOutputDir(e.target.value)}/><button onClick={async () => { const path = await api.pickPath('directory'); if (path) setOutputDir(path); }}><FolderOpen size={16}/></button></div></Field></div><div className="launch-row"><span>Choose in-app training or export a cloud notebook script</span><button className="primary-button" disabled={trainJob?.status === 'running'} onClick={requestTraining}><Play size={15} fill="currentColor"/>Start training<ArrowRight size={16}/></button></div></Section>
          {trainJob && <Section eyebrow="LIVE RUN" title="Training monitor" aside={<span className="quiet-count">{statusLabel(trainJob)}</span>}>{renderJob(trainJob)}{trainJob.status === 'complete' && <div className="next-actions"><button className="secondary-button" onClick={() => useOutputFor('infer')}>Try a prompt <ArrowRight size={15}/></button><button className="secondary-button" onClick={() => useOutputFor('evaluate')}>Evaluate model <ArrowRight size={15}/></button></div>}</Section>}
        </div></div>
      </>}
      {tab === 'evaluate' && <><div className="page-intro"><div><div className="eyebrow">WORKSPACE / EVALUATION</div><h1>See what it knows.</h1><p>Measure reasoning, completion, arithmetic, safety, and more with reproducible local runs.</p></div><span className="intro-index">02 / 03</span></div><div className="eval-layout"><div className="primary-column"><Section eyebrow="01 · TARGET" title="Model to evaluate"><p className="section-copy">Hugging Face model ID, URL, or a BananaAll run folder.</p><div className="model-path"><input value={evalModel} onChange={e => setEvalModel(e.target.value)} placeholder="org/model-name or local folder"/><button className="icon-button bordered" onClick={async () => { const path = await api.pickPath('directory'); if (path) setEvalModel(path); }}><FolderOpen size={17}/></button></div><Toggle checked={evalTrust} onChange={setEvalTrust} label="Trust remote model code" hint="Allow custom model Python from the selected repository."/></Section>
        <Section eyebrow="02 · BENCHMARKS" title="Choose evaluations"><p className="section-copy">Standard tasks run through lm-evaluation-harness. Continuation suites score the conditional likelihood of each answer.</p>{benchmarkGroups.map(group => <div className="benchmark-group" key={group.title}><div className="group-title">{group.title}</div>{group.tasks.map(([key, name, desc]) => <label className="benchmark-row" key={key}><input type="checkbox" checked={evalTasks.includes(key)} onChange={e => setEvalTasks(current => e.target.checked ? [...current, key] : current.filter(item => item !== key))}/><span><strong>{name}</strong><small>{desc}</small></span><ArrowRight size={15}/></label>)}</div>)}</Section>
        <Section eyebrow="03 · RUN" title="Evaluation settings"><div className="settings-grid three"><Field label="Batch size"><input type="number" min="1" value={evalBatch} onChange={e => setEvalBatch(Number(e.target.value))}/></Field><Field label="Few-shot examples"><input type="number" min="0" value={evalFewShot} onChange={e => setEvalFewShot(Number(e.target.value))}/></Field><Field label="Limit per task" hint="0 runs the full dataset"><input type="number" min="0" value={evalLimit} onChange={e => setEvalLimit(Number(e.target.value))}/></Field></div><div className="launch-row"><span>{evalTasks.length} benchmark{evalTasks.length === 1 ? '' : 's'} selected</span><button className="primary-button" disabled={evalJob?.status === 'running'} onClick={startEval}><Play size={15} fill="currentColor"/> Run evaluation <ArrowRight size={16}/></button></div></Section>
        {evalJob && <Section eyebrow="RESULTS" title="Evaluation report">{renderJob(evalJob)}{Object.keys(evalJob.results).length > 0 && <div className="evaluation-results"><StandardScores results={evalJob.results}/>{(['base_bench', 'safety_bench'] as const).filter(key => evalJob.results[key]).map(key => <EvaluationResult key={key} task={key} raw={evalJob.results[key]}/>) }<ContinuationScores results={evalJob.results}/></div>}</Section>}</div><aside className="eval-aside"><div className="aside-block"><div className="eyebrow">EVALUATION NOTES</div><h3>Same model. Clear signals.</h3><p>Base Bench and Safety Bench use their official benchmark scripts and reports. ArithMark and Tiny Theory of Mind use mean conditional token log probability for each continuation.</p></div><div className="aside-block"><LockKeyhole size={18}/><p>{auth?.loggedIn ? `Signed in as ${auth.username}. Base Bench access is available through your saved Hugging Face token.` : 'Use hf auth login for gated benchmark access.'}</p></div></aside></div></>}
      {tab === 'infer' && <><div className="page-intro"><div><div className="eyebrow">WORKSPACE / INFERENCE</div><h1>Talk to the model.</h1><p>Load an instruct or base checkpoint from Hugging Face or a local run.</p></div><span className="intro-index">03 / 03</span></div><div className="infer-layout"><div className="conversation"><div className="conversation-head"><div><div className="eyebrow">SESSION</div><h2>{inferMode === 'instruct' ? 'Conversation' : 'Completion'}</h2></div><button className="text-button" onClick={() => { setMessages([]); setPrompt(''); }}>Clear session <X size={15}/></button></div><div className="messages">{inferMode === 'instruct' ? messages.length ? messages.map((message, index) => <div className={`message ${message.role}`} key={index}><span>{message.role === 'user' ? 'YOU' : 'MODEL'}</span><p>{message.content}</p></div>) : <div className="empty-conversation"><div className="empty-symbol">✳</div><h3>A blank page for a new idea.</h3><p>Enter a prompt below to begin.</p></div> : <div className="base-intro"><Code2 size={24}/><h3>Base model completion</h3><p>The model continues your text directly, without a chat template.</p>{inferJob?.events.find(event => event.type === 'complete')?.text && <div className="base-output">{String(inferJob.events.find(event => event.type === 'complete')?.text)}</div>}</div>}{inferJob?.status === 'running' && <div className="generating"><LoaderCircle size={16} className="spin"/>{[...inferJob.events].reverse().find(item => item.type === 'status')?.message || 'Loading model…'}</div>}{inferJob?.error && <div className="inline-error">{inferJob.error}</div>}</div><div className="composer"><textarea value={prompt} onChange={e => setPrompt(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); startInfer(); } }} placeholder={inferMode === 'base' ? 'The story begins...' : 'Ask anything...'} rows={3}/><div><span>Enter to send · Shift + Enter for a new line</span><button className="primary-button" disabled={!prompt.trim() || inferJob?.status === 'running'} onClick={startInfer}><ArrowRight size={17}/></button></div></div></div><aside className="inference-settings"><div className="eyebrow">MODEL SETTINGS</div><h2>Configure session</h2><Field label="Model"><div className="path-input"><input value={inferModel} onChange={e => setInferModel(e.target.value)} placeholder="org/model-name"/><button onClick={async () => { const path = await api.pickPath('directory'); if (path) setInferModel(path); }}><FolderOpen size={16}/></button></div></Field><div className="segmented"><button className={inferMode === 'instruct' ? 'active' : ''} onClick={() => setInferMode('instruct')}>Instruct</button><button className={inferMode === 'base' ? 'active' : ''} onClick={() => setInferMode('base')}>Base</button></div><Toggle checked={inferTrust} onChange={setInferTrust} label="Trust remote code" hint="Required for some custom HF models."/>{inferMode === 'instruct' && <Field label="System prompt"><textarea value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)} rows={3}/></Field>}<div className="inference-divider"/><div className="slider-field"><div><strong>Temperature</strong><span>{temperature.toFixed(2)}</span></div><input type="range" min="0" max="2" step="0.01" value={temperature} onChange={e => setTemperature(Number(e.target.value))}/></div><div className="slider-field"><div><strong>Top P</strong><span>{topP.toFixed(2)}</span></div><input type="range" min="0.05" max="1" step="0.01" value={topP} onChange={e => setTopP(Number(e.target.value))}/></div><div className="settings-grid two"><Field label="Top K"><input type="number" min="0" value={topK} onChange={e => setTopK(Number(e.target.value))}/></Field><Field label="Max new tokens"><input type="number" min="1" value={maxNewTokens} onChange={e => setMaxNewTokens(Number(e.target.value))}/></Field></div><Field label="Repetition penalty"><input type="number" min="0.1" step="0.05" value={repetitionPenalty} onChange={e => setRepetitionPenalty(Number(e.target.value))}/></Field>{inferJob?.outputPath && <button className="text-button output-link" onClick={() => api.openPath(inferJob.outputPath)}><FolderOpen size={15}/> Open generation folder <ExternalLink size={14}/></button>}</aside></div></>}
    </main>
    {trainPicker && <div className="modal-backdrop"><div className="training-modal" role="dialog" aria-modal="true" aria-label="Choose where to train"><div className="modal-head"><div><div className="eyebrow">TRAINING DESTINATION</div><h2>Where should this run?</h2></div><button className="icon-button" title="Close" onClick={() => setTrainPicker(false)}><X size={19}/></button></div><div className="training-choices"><button className="training-choice" onClick={() => chooseTraining('in-app')}><span className="training-choice-icon"><Play size={19}/></span><span><strong>In-App-Training <em>Recommended</em></strong><small>Train here and watch progress, checkpoints, and loss in BananaAll.</small></span><ArrowRight size={17}/></button><button className="training-choice" onClick={() => chooseTraining('notebook')}><span className="training-choice-icon"><Code2 size={19}/></span><span><strong>Notebook</strong><small>Save a Python file for Google Colab or Molab. It creates a model ZIP when training finishes.</small></span><ArrowRight size={17}/></button></div><p className="training-modal-note">Notebook export uses remote Hugging Face datasets and base models. A saved token is included only when a selected repository needs it.</p></div></div>}
    {notebookExport && <div className="modal-backdrop"><div className="training-modal notebook-ready" role="dialog" aria-modal="true" aria-label="Notebook export instructions"><div className="modal-head"><div><div className="eyebrow">NOTEBOOK SCRIPT READY</div><h2>Run it in a notebook</h2></div><button className="icon-button" title="Close" onClick={() => setNotebookExport(null)}><X size={19}/></button></div><ol className="notebook-steps"><li>Open Google Colab or Molab and choose a GPU runtime.</li><li>Open the saved Python file, copy all its contents, and paste them into one Python code cell. Run the cell.</li><li>Wait for training to finish. In Colab, the ZIP download starts automatically. In Molab, download <strong>{notebookExport.zipName}</strong> from the Files panel.</li><li>Extract the ZIP on your computer. The extracted folder contains the finished model and tokenizer.</li></ol><p className="notebook-path">Python file: <strong>{notebookExport.path}</strong></p>{notebookExport.tokenEmbedded ? <p className="notebook-token-note"><LockKeyhole size={16}/>This file contains your Hugging Face token. Keep it private and do not share the notebook.</p> : notebookExport.tokenNeeded ? <p className="training-modal-note">This job needs Hugging Face access. Sign in within the notebook before running this script.</p> : null}<div className="modal-footer"><button className="secondary-button" onClick={() => api.openPath(notebookExport.path)}><ExternalLink size={15}/>Open Python file</button><button className="primary-button" onClick={() => setNotebookExport(null)}>Done</button></div></div></div>}
    {codeModal && <div className="modal-backdrop"><div className="code-modal" role="dialog" aria-modal="true" aria-label="Review custom training code"><div className="modal-head"><div><div className="eyebrow">CUSTOM ARCHITECTURE / CODE REVIEW</div><h2>Review the training code</h2></div><button className="icon-button" disabled={reviewing} onClick={() => setCodeModal(false)}><X size={19}/></button></div><div className="red-warning"><TriangleAlert size={22}/><div><strong>Keep the logging contract exactly intact.</strong><p>The app reads one JSON event per stdout line. Your edited code must preserve status, metric, complete, and error events so loss, progress, checkpoints, and failures remain visible.</p></div></div><p className="modal-copy">Review manually, or use AI Review to check the script against your custom files. AI Review sends the script, config.json, and neighboring custom Python modules to OpenRouter; inspect any suggested edits before running.</p>{reviewNotice && <div className="review-result"><Sparkles size={16}/><span>{reviewNotice}</span></div>}{reviewError && <div className="inline-error review-result"><TriangleAlert size={16}/><span>{reviewError}</span></div>}<textarea className="code-editor" value={codeDraft} onChange={e => { setCodeDraft(e.target.value); setReviewNotice(''); }} readOnly={!codeEditing || reviewing} spellCheck={false}/><div className="modal-footer"><div className="review-actions"><button className="secondary-button" disabled={reviewing} onClick={() => setCodeEditing(true)}><Code2 size={15}/>{codeEditing ? 'Editing code' : 'Edit code'}</button><button className="secondary-button" disabled={reviewing} onClick={runAiReview}>{reviewing ? <LoaderCircle size={15} className="spin"/> : <Sparkles size={15}/>} {reviewing ? 'Reviewing…' : 'AI Review'}</button></div><button className="primary-button" disabled={reviewing} onClick={async () => { setCodeModal(false); if (trainingTarget === 'notebook') await exportNotebook(codeDraft); else await startJob('train', trainingConfig(), codeDraft); }}><Check size={16}/>{trainingTarget === 'notebook' ? 'Export notebook script' : 'Everything looks fine · start training'}</button></div></div></div>}
    {reviewSetupOpen && <div className="modal-backdrop setup-backdrop"><div className="training-modal review-setup" role="dialog" aria-modal="true" aria-label="OpenRouter AI review setup"><div className="modal-head"><div><div className="eyebrow">{reviewSettings?.firstRun ? 'FIRST TIME SETUP' : 'AI REVIEW SETTINGS'}</div><h2>OpenRouter AI review</h2></div><button className="icon-button" title="Close" onClick={reviewSettings?.firstRun ? skipReviewSetup : () => setReviewSetupOpen(false)}><X size={19}/></button></div><p className="training-modal-note">Optional. Add an OpenRouter API key to review custom training code with <strong>openai/gpt-6-luna</strong>. The script, config.json, and neighboring custom Python modules are sent only when you press AI Review.</p><div className="review-key-field"><Field label={reviewSettings?.hasKey ? 'Replace API key' : 'OpenRouter API key'}><input type="password" autoComplete="off" value={reviewKey} onChange={e => setReviewKey(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') saveReviewKey(); }} placeholder="sk-or-v1-…"/></Field><p>{reviewSettings?.hasKey ? 'A key is configured. It is never shown in the app.' : 'You can add a key later from the lock button in the toolbar.'} {reviewSettings?.canRemember ? 'The key is stored with your operating system’s secure storage.' : 'Secure storage is unavailable here, so the key is kept for this app session only.'}</p>{reviewSetupError && <div className="inline-error"><TriangleAlert size={15}/>{reviewSetupError}</div>}</div><div className="modal-footer"><div className="review-actions">{reviewSettings?.firstRun && <button className="secondary-button" onClick={skipReviewSetup}>Skip for now</button>}{reviewSettings?.hasKey && <button className="secondary-button" onClick={async () => { try { setReviewSettings(await api.clearReviewKey()); setReviewSetupError(''); } catch (e) { setReviewSetupError(String((e as Error).message)); } }}>Remove key</button>}</div><button className="primary-button" disabled={!reviewKey.trim()} onClick={saveReviewKey}><Check size={16}/>Save key</button></div></div></div>}
  </div>;
}
