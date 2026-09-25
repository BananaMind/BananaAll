const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('banana', {
  system: () => ipcRenderer.invoke('system'),
  authStatus: (python) => ipcRenderer.invoke('auth-status', python),
  pickPath: (kind) => ipcRenderer.invoke('pick-path', kind),
  openPath: (path) => ipcRenderer.invoke('open-path', path),
  inspectDataset: (source, python) => ipcRenderer.invoke('inspect-dataset', source, python),
  trainingCode: (configPath, modelingPath) => ipcRenderer.invoke('training-code', configPath, modelingPath),
  startJob: (kind, config, code) => ipcRenderer.invoke('start-job', kind, config, code),
  stopJob: (id) => ipcRenderer.invoke('stop-job', id),
  onJobEvent: (callback) => {
    const handler = (_event, payload) => callback(payload);
    ipcRenderer.on('job-event', handler);
    return () => ipcRenderer.removeListener('job-event', handler);
  }
});
