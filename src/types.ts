export type Tab = 'train' | 'evaluate' | 'infer';
export type TrainMode = 'pretraining' | 'full' | 'lora';
export type Mapping = { text: string; messages: string; system: string; user: string; assistant: string };
export type DatasetSource = {
  key: string; id: string; name: string; config: string; split: string; weight: number;
  tokenLimitM: number | null;
  enabled: boolean; mapping: Mapping; samples: Record<string, unknown>[]; columns: string[];
  status: 'idle' | 'loading' | 'ready' | 'error'; error?: string;
};
export type JobEvent = { id: string; type: string; message?: string; detail?: string; outputPath?: string;
  code?: number | null; signal?: string | null; step?: number; totalSteps?: number; loss?: number; learning_rate?: number;
  learningRate?: number; accumulationStep?: number; accumulationTotal?: number;
  elapsedSeconds?: number; tokensSeen?: number; totalTokens?: number; processedTokens?: number; tokensPerSecond?: number; grad_norm?: number;
  task?: string; result?: Record<string, unknown>; results?: Record<string, unknown>;
  text?: string; [key: string]: unknown };
export type JobState = { id: string; kind: Tab; status: 'running' | 'complete' | 'error' | 'stopped';
  outputPath: string; events: JobEvent[]; metrics: { step: number; loss: number }[]; results: Record<string, unknown>;
  progress?: JobEvent; modelParameters?: number; compileDisabled?: boolean; error?: string };
export type DeviceStatus = { backend: 'rocm' | 'cuda' | 'mps' | 'cpu' | 'unavailable'; name: string; message?: string };
export type ReviewSettings = { firstRun: boolean; hasKey: boolean; canRemember: boolean };
export type BananaBridge = {
  system: () => Promise<{ platform: string; defaultOutput: string; python: string }>;
  authStatus: (python: string) => Promise<{ loggedIn: boolean; username?: string; message?: string }>;
  deviceStatus: (python: string) => Promise<DeviceStatus>;
  pickPath: (kind: 'file' | 'directory') => Promise<string | null>;
  openPath: (path: string) => Promise<string>;
  inspectDataset: (source: DatasetSource, python: string) => Promise<{ columns: string[]; samples: Record<string, unknown>[]; mapping: Mapping }>;
  trainingCode: (configPath: string, configurationPath: string, modelingPath: string) => Promise<string>;
  reviewSettings: () => Promise<ReviewSettings>;
  saveReviewKey: (key: string) => Promise<ReviewSettings>;
  clearReviewKey: () => Promise<ReviewSettings>;
  skipReviewSetup: () => Promise<ReviewSettings>;
  aiReview: (request: { code: string; configPath: string; configurationPath: string; modelingPath: string; python: string }) => Promise<{ summary: string; revisedCode: string | null }>;
  exportNotebook: (config: Record<string, unknown>, code?: string) => Promise<{ path: string; zipName: string; tokenEmbedded: boolean; tokenNeeded: boolean } | null>;
  startJob: (kind: 'train' | 'evaluate' | 'infer', config: Record<string, unknown>, code?: string) => Promise<{ id: string; outputPath: string }>;
  stopJob: (id: string) => Promise<boolean>;
  onJobEvent: (callback: (event: JobEvent) => void) => () => void;
};
declare global { interface Window { banana: BananaBridge } }
