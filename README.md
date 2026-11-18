# UAV Cyber-Physical Intrusion Detection

A drone is controlled over Wi-Fi and flies on sensor readings (pitch, roll, yaw, speed). An attack can hit either side. This project trains models on both, names the attack, and shows an operator what to do.

Data comes from the public IEEE T-ITS UAV Cyber-Physical Intrusion Detection benchmark dataset: normal flights plus de-authentication DoS, replay, evil twin, and false data injection (FDI).

## Problem

A firewall only sees packets. A sensor check only sees the airframe. Neither is enough:

| Attack | What it does | Where it shows up |
|--------|----------------|-------------------|
| De-authentication DoS | Kicks the drone off Wi-Fi | Cyber |
| Replay | Re-sends old valid commands | Cyber |
| Evil twin | Fake access point impersonates the controller | Cyber |
| False data injection | Feeds wrong sensor or control values | Physical |

The published IEEE T-ITS (2023) benchmark study showed that **joining cyber and physical features** detects “attack vs normal” better than either alone. It does not say **which** attack it is, it does not explain the decision, and it is an offline batch train.

## How it works

```
raw flight log
    → standardize columns (zero mean, unit variance)
    → choose a fusion: cyber (21) / physical (11) / both (32)
    → model
    → class + (for CP-Attn-Lite) cyber vs physical weight
    → dashboard: score, live replay, alert text
```

1. **One row is one moment.** Wi-Fi fields (`frame.len`, `wlan.seq`, signal, …) and telemetry (`pitch`, `roll`, `yaw`, velocities, acceleration) sit on the same row.
2. **Fusion** is which columns the model is allowed to see. Cyber-only, physical-only, and combined are trained separately so you can see what each layer contributes.
3. **Binary models** output 0 (normal) or 1 (attack): SVM, feedforward net, LSTM (10-step window), and a 1D-CNN.
4. **Multi-class 1D-CNN** outputs one of five labels: benign, DoS, replay, evil twin, FDI. Train on `central_*` splits. `drone1_*` / `drone2_*` are half the file and do not contain every attack.
5. **CP-Attn-Lite** does not stack all 32 columns into one vector. A small conv branch reads the 21 cyber features, another reads the 11 physical features. A softmax gate mixes them and records the mix (for example DoS leans on the network branch). Global average pooling keeps the net at about 3,000 parameters. The old CNN is left in place as the baseline.
6. **Live stream** walks the CSV row by row, runs the saved model, and logs latency. It is a replay of logged flights, not a radio hooked to a real aircraft.
7. **Alerts** map the predicted class to a short description, a severity, and operator actions (check the access point, validate packet sequence, land if attitude diverges). That text is a fixed playbook, not a feature-attribution method.

## What is new here

The dataset, the four binary models, and the idea of cyber–physical fusion are from the 2023 paper. This repo adds four things on top of that:

1. **Attack type, not only attack / normal.** Five-class identification, with per-class scores and a confusion matrix.
2. **A fusion the model learns.** CP-Attn-Lite’s gate says, per sample, how much the call used Wi-Fi versus sensors. The baseline CNN cannot report that split.
3. **A streaming path.** Same trained model, one row at a time, with latency on the dashboard.
4. **An operator view.** Severity and recommended actions next to the prediction, instead of a metrics table alone.

## Run

Python 3.11. This machine uses `C:\uav-venv`.

```powershell
pip install -r requirements.txt
python scripts\prepare_dataset.py --output data\dataset_updated
python scripts\run_all_experiments.py
python "1D-CNN -based IDS\cp_attn_cnn.py" --fusion central_combined --epochs 15
python -m streamlit run dashboard/app.py
```

Dashboard: http://localhost:8501

| Sidebar page | Shows |
|--------------|--------|
| All algorithms | Binary SVM, FNN, LSTM, 1D-CNN across cyber / physical / combined |
| Multi-class attack ID | Five-class 1D-CNN |
| CP-Attn-Lite | Dual-branch scores and fusion weights |
| Live streaming demo | Row-by-row prediction |
| Explainable alerts | Description and actions for each class |

## Layout

```
scripts/prepare_dataset.py          standardized CSVs
scripts/cp_attn_model.py            dual-branch CNN
1D-CNN -based IDS/                  baseline CNN, multi-class CNN, CP-Attn-Lite
SVM-based IDS/  FNN-based IDS/  LSTM-based IDS/
dashboard/app.py                    Streamlit UI
results/metrics.json                binary scores
results/metrics_multiclass.json     five-class scores, including CP-Attn-Lite
data/raw/Dataset_T-ITS.csv          source log
```

Generated CSVs are not in git. After a clone, run `prepare_dataset.py`.
