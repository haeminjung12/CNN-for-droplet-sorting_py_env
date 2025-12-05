# CNN for Droplet Sorting (Python)

Python training/tuning utilities, analysis scripts, and generated artifacts (figures + saved weights) for the droplet sorting CNN. Training data is excluded from version control.

## Contents
- `train_droplet.py` — base training script (SqueezeNet1_1 backbone, MATLAB-exportable ONNX).
- `tune_droplet_extended.py` — hyperparameter sweeps (LR/WD/BS/epochs/input_size/scheduler/label smoothing).
- `summarize_tuning.py` — summarizes tuning results and produces overview figures.
- `train_best_sweep.py` — retrains the best tuned config across multiple epoch counts and saves the best checkpoint/ONNX.
- `analyze_best_worst.py` — retrains/evaluates top/bottom configs, tunes rare-class thresholds, logs macro F1/balanced acc/confusion matrices.
- `summarize_final_model.py` — evaluates a trained model, tunes rare-class threshold, and produces academic-style figures + text explanations.
- Artifacts:
  - `tuning_runs/` — tuning summaries and best models from sweeps.
  - `best_runs/` — final sweep outputs and best model weights (`.pth` + `.onnx`).
  - `figures/`, `eval_figures/`, `analysis_runs/` — generated plots and evaluation outputs.

## Environments
Scripts were run with the `droplet_cuda` conda env (Python 3.10, PyTorch/torchvision installed). Training data lives outside the repo: `matlab_env/CNN-for-droplet-sorting/Training Data/` (ignored).

## Common Commands

### Summarize Tuning
```
MPLCONFIGDIR=./.cache/matplotlib miniconda3/envs/droplet_cuda/bin/python \
  summarize_tuning.py --summary tuning_runs/tune_ext_20251204_192755/tuning_summary.json --outdir figures
```

### Analyze Best/Worst (macro F1, balanced acc, threshold tuning)
```
MPLCONFIGDIR=./.cache/matplotlib miniconda3/envs/droplet_cuda/bin/python \
  analyze_best_worst.py --summary tuning_runs/tune_ext_20251204_192755/tuning_summary.json \
  --topk 5 --output-dir analysis_runs --num-workers 0
```

### Retrain Best Config Across Epoch Counts (saves ONNX)
```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_HOME=.cache/torch MPLCONFIGDIR=./.cache/matplotlib \
USE_PRETRAINED=1 \
miniconda3/envs/droplet_cuda/bin/python train_best_sweep.py \
  --summary tuning_runs/tune_ext_20251204_192755/tuning_summary.json \
  --epochs-list 8 10 12 \
  --output-dir best_runs --num-workers 0 --device cuda
```

### Final Model Evaluation + Figures
```
MPLCONFIGDIR=./.cache/matplotlib miniconda3/envs/droplet_cuda/bin/python \
  summarize_final_model.py \
  --model-dir best_runs/best_epochs_8_20251205_005311 \
  --output-dir eval_figures --num-workers 0 --device cuda
```
This produces:
- `metrics.json` (tuned/baseline macro F1, balanced accuracy, per-class stats, confusion matrix).
- `confusion_matrix.png/.txt`, `per_class_metrics.png/.txt`, `threshold_sweep.png/.txt`.

## Notes
- Training data is **not** tracked; place it under `matlab_env/CNN-for-droplet-sorting/Training Data/`.
- Pretrained SqueezeNet weights are cached at `.cache/torch/hub/checkpoints/squeezenet1_1-b8a52dc0.pth` (or set `TORCH_HOME`).
- To avoid OpenMP/MKL shared memory issues in some environments, the commands above set `LD_PRELOAD`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS`.
