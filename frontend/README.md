# Frontend MVP

This directory now contains a static dashboard prototype for uploaded-audio analysis and browser microphone capture.

Files
1. `index.html`
   - Dashboard markup, workspace tabs, recording controls, direct training-upload form, training-save form, recent-session list, playback panel, suppression preview controls, waveform and spectrogram previews, and result containers.
2. `styles.css`
   - Visual layout, workspace-tab styling, dataset-upload panel styling, recording panel styling, per-class suppression controls, comparison playback controls, waveform and spectrogram styling, and timeline styling.
3. `app.js`
   - Upload flow, compatible-WAV fast-path detection, automatic browser-side conversion of unsupported audio files into compatible WAV payloads, browser recording and WAV conversion, direct training-data upload requests, saved-session loading, training-set save requests, manifest and training controls, model-artifact activation requests, per-class suppression profile requests, original-versus-processed playback comparison, waveform and spectrogram decoding, interactive playback/timeline behavior, and workspace section toggles.

Local Run
1. From the repo root, the easiest launcher is:

```powershell
.\start-frontend.cmd
```

2. Manual fallback from the `frontend/` directory:

```powershell
python -m http.server 3000
```

3. Open `http://127.0.0.1:3000`.

The page expects the backend API at `http://127.0.0.1:8000`.
Saved sessions appear in the Recent Sessions panel when the backend session store is available.
Selected audio files are converted into compatible WAV files in the browser before `/analyze` or `/process` requests are sent.
Already-compatible PCM WAV files are reused directly so large valid uploads do not pay the browser-side conversion cost again.
Recorded clips can be labeled and saved into `training/real_recordings/` through the backend `POST /recordings` route.
The Dataset Manager can also upload local training clips, build the real-recordings manifest, start a training run, show training progress, and switch the active trained model artifact while leaving older versions available as backup candidates.
The Workspace Tabs bar can open or hide major dashboard sections and remembers that layout in local storage.
