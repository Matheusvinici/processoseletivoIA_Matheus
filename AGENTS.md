# AGENTS.md — Processo Seletivo Edge AI

## Repository structure

```
projetos/
  1-classificacao-mnist/  ← chosen project (fully implemented by Matheus Vinicius Vidal de Andrade)
.github/workflows/ci.yml  ← validates artifacts on push/PR to main
.github/scripts/          ← validation scripts run by CI (do NOT modify)
```

**CI auto-detection**: checks which folder remains in `projetos/`. If 0 or >1 found → fails. Currently only Projeto 1 is present.

## Required artifacts (must be committed — CI does NOT train)

Trained model: `model.h5` | Optimized model: `model.tflite`

Also required: `train_model.py`, `optimize_model.py`, `run_inference.py`, `requirements.txt`, `README.md`

## CI validation thresholds

- **model.h5**: accuracy ≥ 85%
- **model.tflite**: accuracy ≥ 75%

CI also re-runs all 3 scripts — all must execute without error.

## Engineering constraints

- **CPU-only** (`device="cpu"`)
- No pre-trained models
- Input shape: `(28,28,1)` grayscale, normalized [0,1]
- Max 15 epochs, EarlyStopping
- Architecture: 3× Conv2D+BatchNorm+MaxPooling → Dropout → Dense(10, softmax)
- TFLite conversion with `tf.lite.Optimize.DEFAULT` (Dynamic Range Quantization)

## Report

Embedded in project `README.md` under "Relatório do Candidato" — 6 sections.

## `.gitignore` (key ignores)

`runs/`, `__pycache__/`, `*.pyc`, `.vscode/`, `.DS_Store`

## Dev environment

- `.devcontainer/devcontainer.json` (Python 3.11)
- Requirements: `tensorflow>=2.12`, `numpy`
