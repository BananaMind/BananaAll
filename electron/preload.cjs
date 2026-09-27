const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('banana', {
  system: () => ipcRenderer.invoke('system'),
  authStatus: (python) => ipcRenderer.invoke('auth-status', python),
  deviceStatus: (python) => ipcRenderer.invoke('device-status', python),
  pickPath: (kind) => ipcRenderer.invoke('pick-path', kind),
  openPath: (path) => ipcRenderer.invoke('open-path', path),
  inspectDataset: (source, python) => ipcRenderer.invoke('inspect-dataset', source, python),
  trainingCode: (configPath, configurationPath, modelingPath) => ipcRenderer.invoke('training-code', configPath, configurationPath, modelingPath),
  reviewSettings: () => ipcRenderer.invoke('review-settings'),
  saveReviewKey: (key) => ipcRenderer.invoke('review-key-save', key),
  clearReviewKey: () => ipcRenderer.invoke('review-key-clear'),
  skipReviewSetup: () => ipcRenderer.invoke('review-setup-skip'),
  aiReview: (request) => ipcRenderer.invoke('ai-review', request),
  exportNotebook: (config, code) => ipcRenderer.invoke('export-notebook', config, code),
  startJob: (kind, config, code) => ipcRenderer.invoke('start-job', kind, config, code),
  stopJob: (id) => ipcRenderer.invoke('stop-job', id),
  onJobEvent: (callback) => {
    const handler = (_event, payload) => callback(payload);
    ipcRenderer.on('job-event', handler);
    return () => ipcRenderer.removeListener('job-event', handler);
  }
});
