# Sound Dashboard Prototype

This repository now includes planning documents plus a working sound analysis dashboard prototype.

Current focus:
1. Support both uploaded-audio analysis and browser-recorded capture.
2. Build a backend API for detection and suppression workflows.
3. Keep improving the trained-model path, saved review workflow, training-data collection, and model version management.
4. Improve suppression quality beyond the current prototype attenuation path.

Repository Structure
1. `backend/`
   - FastAPI backend scaffold for health checks, configuration, and audio analysis requests.
2. `frontend/`
   - Static dashboard for upload, browser recording, analysis review, suppression preview, and training-clip save flows.
3. `docs/`
   - Planning and architecture notes.
4. Root text files
   - High-level and technical build plans created earlier.

Backend MVP
The current backend scaffold provides:
1. `GET /health`
   - Verifies the service is running.
2. `GET /config`
   - Returns the configured sample rate, chunk duration, overlap, and supported classes.
3. `POST /analyze`
   - Accepts a PCM WAV file, decodes it, normalizes and resamples it to the configured target sample rate, computes basic audio features, generates log-mel spectral summaries, returns classifier detections plus classifier-source metadata, and saves the analysis as a reusable session.
4. `POST /process`
   - Accepts a PCM WAV file plus suppression settings, applies prototype class-based attenuation over matching detected spans, and returns a processed WAV preview as base64.
5. `GET /sessions`
   - Returns recent saved analysis sessions for the dashboard.
6. `GET /sessions/{session_id}`
   - Returns a saved session with the original audio payload, analysis response, and any saved processed preview.
7. `POST /recordings`
   - Accepts a compatible WAV file plus label metadata, validates it against the supported classes, and saves it into `training/real_recordings/` for future manifest generation and retraining.
8. `GET /recordings/summary`
   - Returns clip counts, split counts, duration totals, and manifest status for the collected real-recordings dataset.
9. `POST /recordings/build-manifest`
   - Builds `training/real_recordings/manifest.jsonl` from the current labeled WAV collection.
10. `GET /training/status`
   - Returns the latest training-run state, including run ID, epoch progress, and the output artifact path.
11. `POST /training/run`
   - Starts a background training run from the current real-recordings manifest and archives the exported artifact into `training/artifacts/versions/`.
12. `GET /artifacts`
   - Lists the trained model artifacts that can be used for inference, including archived versions and the current active selection.
13. `POST /artifacts/activate`
   - Switches the active trained model manifest while leaving older versions available as fallback candidates.

The current analysis route is still an MVP, but it now performs real WAV decoding, preprocessing, model-ready spectral feature generation, and baseline classifier scoring over the configured sound classes.
The spectral extraction path is `numpy`-accelerated when available, with a pure-Python fallback kept in place for minimal environments.

Suggested Next Implementation Steps
1. Collect more labeled WAV clips, including browser-recorded captures, and train the next `training/` artifact.
2. Retrain and calibrate class thresholds against the real dataset.
3. Improve suppression quality beyond span attenuation.
4. Add broader tests for model inference, artifact promotion/rollback, and API integration behavior.
5. Expand saved-session management beyond the current recent-session list.

Running the Backend
1. Fastest option from the repo root:

```powershell
.\start-backend.cmd
```

2. Or run the PowerShell script directly:

```powershell
.\start-backend.ps1
```

3. Manual fallback:

```powershell
training\.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The API will be available locally at `http://127.0.0.1:8000`.

Running Tests
1. Install backend dependencies.
2. From the repository root, run:

```powershell
training\.venv\Scripts\python -m unittest discover -s backend\tests -v
```

Running the Frontend
1. Fastest option from the repo root:

```powershell
.\start-frontend.cmd
```

2. Or run the PowerShell script directly:

```powershell
.\start-frontend.ps1
```

3. Manual fallback:

```powershell
cd frontend
python -m http.server 3000
```

4. Open `http://127.0.0.1:3000`.

Run Both
1. From the repo root, start both windows at once:

```powershell
.\start-dev.cmd
```

Notes
1. This repo now contains a trained TorchScript artifact built from the current real-data manifest, but its quality still depends on the size and quality of the labeled recordings.
2. The frontend now supports both file upload and in-browser microphone capture, and selected browser audio files are converted to compatible PCM WAV payloads before backend ingestion.
3. Browser-recorded clips are also converted to WAV in-browser before analysis or training ingest, so the backend only needs to handle one compatible file format.
4. The backend is now configured to load `training/artifacts/real-v1/manifest.json` by default, but can also switch to archived manifests stored under `training/artifacts/versions/`.
5. The active trained artifact is tracked in `training/artifacts/active-model.json`.
6. Older archived manifests remain available as backup inference candidates if the active trained model returns no classes.
7. Completed training runs now archive exported artifacts into `training/artifacts/versions/<run_id>/` and automatically promote the new manifest to active status.
8. Analysis metadata now includes preprocessing details such as target sample rate, sample count after resampling, and normalization gain.
9. The analysis response also includes compact log-mel spectrogram summary statistics for model-ready feature inspection.
10. `backend/requirements.txt` now includes `numpy` so realistic audio lengths remain fast enough for local analysis.
11. Local CORS is enabled for `http://127.0.0.1:3000` and `http://localhost:3000` so the static frontend can call the backend during development.
12. `POST /process` is a prototype suppression path that currently attenuates detected class spans rather than doing true source separation.
13. `training/` now contains an offline waveform-classifier training scaffold that exports a TorchScript model and JSON manifest for backend loading.
14. `training/generate_synthetic_dataset.py` can create a local starter WAV dataset and manifest so the training/export path can be exercised before real labeled data is available.
15. The dashboard now persists recent sessions in `backend/data/sessions/` so earlier analyses and processed previews can be reloaded from the frontend.
16. Analysis responses now expose `classifier_source` and `used_fallback` so the frontend can show whether detections came from the trained model, an archived backup model, or the baseline safety net.
17. The dataset manager now exposes manifest generation, training status, artifact history, and active-model switching from the site itself.
18. The baseline heuristic was retuned so the current keyboard sample set ranks `keyboard` ahead of obvious false positives more reliably.
19. `training/build_real_manifest.py` and `training/real_recordings/README.md` now provide the handoff path for real labeled WAV collection and retraining, and `POST /recordings` can now populate that folder directly from the browser workflow.
20. The frontend recording panel can capture microphone audio, convert it to WAV in-browser, analyze it through the same backend flow, and save labeled clips into the training set.
21. Root-level launch scripts are included: `start-backend`, `start-frontend`, and `start-dev`.
