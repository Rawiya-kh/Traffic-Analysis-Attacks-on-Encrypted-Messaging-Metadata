# IM-Link

Session-level traffic analysis for encrypted messaging metadata, with activity
classification and **SIMULATED PAIRS** correlation experiments.

> Correlation results in this project use simulated second-vantage views. They
> are not measurements from captures of the same real session at both ends.

## Project layout

```text
dataset/                         Input PCAPNG captures (with .pcap extensions)
processed_r1/                    Session-level train/val/test feature CSVs
models/stageA.joblib             Saved Stage A application classifier
models/fusion_clean_seed*.joblib Reconstructed clean Step 6 fusion models
results/activity/                Activity baseline metrics
results/correlation/             Step 5 correlation metrics and ROC data
results/fusion/                  Step 6 fusion, ablation, joint, and ROC data
results/step7/                   Step 7 abstention and defence results
preprocess_r1.py                 Extract R1 features and session splits
evaluate_activity.py             Activity classification baselines
step4_simulated_paired_flows.py  SIMULATED PAIRS generation and checks
step5_correlation_baselines.py   B0/B1/B4 correlation baselines
step6a_save_stageA.py            Fit/save the selected Stage A model
step6_imlink_fusion.py           Stage C fusion and evaluation
step7_abstain_defence.py         Abstention and defence evaluation
requirements-ml.txt              Python package dependencies
```

IP addresses and ports are used only internally to group packets into flows.
They are not written to the feature CSVs or used as model features.

## Setup (Windows PowerShell)

Open PowerShell in the project folder:

```powershell
cd "D:\FAST\sem 7\C-net Project"
```

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
```

If PowerShell blocks activation, run commands with the environment's Python
directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

## Run the full pipeline

Run these commands from the project folder, in order:

```powershell
python preprocess_r1.py
python evaluate_activity.py
python step4_simulated_paired_flows.py
python step5_correlation_baselines.py
python step6a_save_stageA.py
python step6_imlink_fusion.py
```

The preprocessing script defaults to `dataset/`, writes `train.csv`,
`val.csv`, and `test.csv` under `processed_r1/`, and uses seed 42. It will
overwrite those split files if rerun. Activity results are written under
`results/activity/`; Step 5 and Step 6 results go under
`results/correlation/` and `results/fusion/`.

The activity evaluator and correlation/fusion evaluations take several
minutes on CPU. Step 6A may overwrite `models/stageA.joblib`; it fits the
validation-selected Stage A model on the training split only. Step 6 loads that
checkpoint, checks its test macro-F1 against the saved Stage A baseline, then
fits fusion scalers and logistic regressors using simulated validation groups.
It does not refit Stage A.

Step 7 reconstructs each clean fusion scaler/classifier from the Step 6 recipe
and saves it as `models/fusion_clean_seed{seed}.joblib`, because Step 6 did not
save its fitted clean fusion model. Later Step 7 runs reuse these artifacts and
stop if their stored recipe does not match.

## Run only the later correlation/fusion steps

If `processed_r1/`, the activity results, and `models/stageA.joblib` already
exist, run:

```powershell
python step4_simulated_paired_flows.py
python step5_correlation_baselines.py
python step6_imlink_fusion.py
```

Run Step 6A only if the Stage A checkpoint is missing or you intentionally want
to recreate it:

```powershell
python step6a_save_stageA.py
```

## Outputs

- Activity baseline tables: `results/activity/`
- Step 5 correlation table and ROC samples: `results/correlation/`
- Step 6 fusion, ablation, joint tables, and ROC samples:
  `results/fusion/`
- Step 7 abstention and defence tables:
  `results/step7/`
- ROC CSV files contain 200 points per method and group size. The scripts also
  expose an in-memory `ROC_DATA[(N, method)]` dictionary when called from Python.

All correlation results are marked **SIMULATED PAIRS**. The true pair is
known by construction (same row index in views A and B), so these results
evaluate the simulation setup rather than establish real-world peer-linking
performance.
