import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.regularizers import L1, L2
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.metrics_utils import history_metric  # noqa: E402
from scripts.paths import require_dataset  # noqa: E402
from scripts.save_metrics import save_metrics  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--fusion", default="drone1_cyber", help="choose fusion")
parser.add_argument("--png", help="choose png name")
parser.add_argument("--metrics_title", default="drone1_cyber", help="print title name")
parser.add_argument("--epochs", type=int, default=100, help="max training epochs")
parser.add_argument("--max_samples", type=int, default=None, help="optional row limit before windowing")
parser.add_argument("--cv_folds", type=int, default=5, help="cross-validation folds")
args = parser.parse_args()

os.makedirs("lstm_results", exist_ok=True)
os.makedirs("np_arrays", exist_ok=True)


def create_model(neurons, act_f, hiddenlayers, ki, l1_value, l2_value, dropout_rate, learning_rate):
    model = Sequential()
    # Add L1 and L2 regularizers to the LSTM layer
    lstm_regularizer = L1(l1_value) if l1_value > 0 else L2(l2_value)
    model.add(LSTM(int(neurons), input_shape=(WS, X.shape[2]), return_sequences=True, kernel_regularizer=lstm_regularizer))

    if hiddenlayers == 2:
        model.add(LSTM(neurons, return_sequences=False))

    elif hiddenlayers == 3:
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL2'))
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=False, name='HL3'))

    elif hiddenlayers == 4:
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL2'))
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL3'))
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=False, name='HL4'))

    elif hiddenlayers == 5:
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL2'))
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL3'))
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=True, name='HL4'))
        model.add(LSTM(neurons, kernel_initializer=ki, activation=act_f, return_sequences=False, name='HL5'))

    # Add dropout layer after each LSTM layer
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid', name='Output'))
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=scores)

    return model


scores = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]

"""
# Define the list of columns to be dropped from the dataset
drop_list_c1 = ['frame.time_epoch', 'wlan_radio.signal_strength (dbm)', 'wlan_radio.Noise level (dbm)',
                'wlan_radio.SNR (db)', 'wlan_radio.preamble', 'wlan.frag', 'wlan.qos', 'wlan.qos.priority',
                'wlan.qos.ack', 'wlan.fcs.status']
drop_list_c2 = ['wlan.bssid', 'wlan.ta', 'wlan.wep.key', 'radiotap.antenna_signal', 'radiotap.channel.flags.cck',
                'wlan_radio.channel', 'wlan_radio.frequency', 'frame.number', 'radiotap.channel.flags.ofdm',
                'wlan.fc.type']  # 20/36 removed
drop_list_p = ['timestamp', 'mid', 'x', 'y', 'z', 'vgx', 'vgy', 'vgz', 'templ',
               'temph', 'baro', 'bat', 'pitch', 'roll', 'h', ]   # 15/21 removed
# 20 keep (combined)
keep_list = ['frame.len', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.duration', 'wlan.seq', 'wlan.fc.subtype',
             'wlan.flags', 'wlan.fcs', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length', 'radiotap.signal_quality',
             'wlan_radio.datarate', 'flight_time', 'agx', 'agy', 'agz', 'yaw', 'tof']
keep_list_c = ['frame.len', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.duration', 'wlan.seq', 'wlan.fc.subtype',
               'wlan.flags', 'wlan.fcs', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length', 'radiotap.signal_quality',
               'wlan_radio.datarate']
keep_list_p = ['flight_time', 'agx', 'agy', 'agz', 'yaw', 'tof', 'pitch', 'vgz']
"""
keep_list = ['frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.bssid', 'wlan.duration', 'wlan.seq', 'wlan.fc.type',
             'wlan.fc.subtype', 'wlan.flags', 'wlan.fcs', 'wlan.fcs.status', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length',
             'radiotap.signal_quality', 'wlan_radio.datarate', 'wlan_radio.channel', 'wlan_radio.SNR (db)', 'wlan_radio.preamble',
             'pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof']  # 32

keep_list_dr_c = ['frame.number', 'frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da',
                  'wlan.duration', 'wlan.seq', 'wlan.fc.type', 'wlan.flags']  # 12
keep_list_dr_cmb = ['frame.number', 'frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da',
                    'wlan.duration', 'wlan.frag', 'wlan.seq', 'wlan.fc.type', 'wlan.flags', 'data.len', 'pitch', 'roll',
                    'templ', 'temph', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof']

keep_list_c = ['frame.len', 'wlan.ta', 'wlan.sa', 'wlan.ra', 'wlan.da', 'wlan.bssid', 'wlan.duration', 'wlan.seq', 'wlan.fc.type',
               'wlan.fc.subtype', 'wlan.flags', 'wlan.fcs', 'wlan.fcs.status', 'wlan.ccmp.extiv', 'data.len', 'radiotap.hdr_length',
               'radiotap.signal_quality', 'wlan_radio.datarate', 'wlan_radio.channel', 'wlan_radio.SNR (db)', 'wlan_radio.preamble']  # 21
# remove cyber, if requires: wlan.fc.type, wlan.fcs.status, wlan_radio.channel, wlan_radio.preamble

keep_list_p = ['pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'flight_time', 'agx', 'agy', 'agz', 'tof']  # 11


# path = os.getcwd()
# fusion = drone1_cyber, drone1_physical, drone1_combined, ... central_cyber, central_physical, central_combined
fusion = args.fusion
WS = 10   # window size

df = pd.read_csv(require_dataset(fusion))
df = df.set_index(['target'])
target = df['target.value']
df = df.drop(columns=['target.value'])

if fusion in ["drone1_cyber", "drone2_cyber", "central_cyber"]:
    df = df[keep_list_c]
elif fusion in ["drone1_physical", "drone2_physical", "central_physical"]:
    df = df[keep_list_p]
elif fusion in ["drone1_combined", "drone2_combined", "central_combined"]:
    df = df[keep_list]

if args.max_samples and len(df) > args.max_samples + WS:
    df = df.iloc[: args.max_samples + WS]
    target = target.iloc[: args.max_samples + WS]

cache_tag = f"_{args.max_samples}" if args.max_samples else ""
npy_x = Path(f"np_arrays/{fusion}{cache_tag}_X.npy")
npy_y = Path(f"np_arrays/{fusion}{cache_tag}_Y.npy")
if not (npy_x.exists() and npy_y.exists()):
    values = df.values.astype(np.float32)
    labels = target.values
    n = len(values) - WS
    feats = values.shape[1]
    xs = np.zeros((n, WS, feats), dtype=np.float32)
    for i in range(n):
        xs[i] = values[i:i + WS]
    ys = labels[WS:]
    np.save(npy_x, xs)
    np.save(npy_y, ys)

X = np.load(npy_x)
Y = np.load(npy_y)

kf = KFold(n_splits=args.cv_folds, shuffle=True)
fin_f1 = []
fin_val_f1 = []
fin_loss = []
fin_val_loss = []
i = 0
itera = ''

sacc = 0
spre = 0
srec = 0
sf1 = 0
sauc = 0

# lecun_uniform, lecun_normal, zeros, ones, orthogonal, identity, constant )
act_f = 'relu'    # tanh, sigmoid, softmax,
ki = 'he_uniform'  # kernel initialization (he_uniform, glorot_uniform, glorot_normal,
l1_value = 0.0001   # 0.0001
l2_value = 0.001   # 0.001
dropout_rate = 0.7  # 0.8
learning_rate = 0.0001  # default learning_rate for adam optimizer= 0.001 # 0.0001

for train, test in kf.split(X, Y):
    if i != 0:
        itera = '_' + str(i)

    model = create_model(neurons=512, act_f=act_f, hiddenlayers=5, ki=ki, l1_value=l1_value, l2_value=l2_value,
                         dropout_rate=dropout_rate, learning_rate=learning_rate)

    early_stop = EarlyStopping(monitor='val_loss'.format(itera), min_delta=1e-2, patience=5, verbose=1, mode='auto',
                                                  baseline=None, restore_best_weights=True)
    history = model.fit(
        x=X[train], y=Y[train],
        validation_data=(X[test], Y[test]),
        callbacks=[early_stop],
        epochs=args.epochs,
        batch_size=128,
        verbose=0,
    ).history

    recall = history_metric(history, "recall")
    precision = history_metric(history, "precision")
    f1 = (2 * ((precision * recall) / (precision + recall + 1e-7)))

    v_recall = history_metric(history, "val_recall")
    v_precision = history_metric(history, "val_precision")
    v_f1 = (2 * ((v_precision * v_recall) / (v_precision * v_recall + 1e-7)))

    fin_loss.append(history['loss'])
    fin_val_loss.append(history['val_loss'])

    fin_f1.append(f1.tolist())
    fin_val_f1.append(v_f1.tolist())

    sacc += history_metric(history, "val_accuracy")[-1]
    spre += history_metric(history, "val_precision")[-1]
    srec += history_metric(history, "val_recall")[-1]
    sf1 += v_f1[-1]
    sauc += history_metric(history, "val_auc")[-1]

    i += 1

length = max(map(len, fin_f1))
y = np.array([xi+[np.nan]*(length-len(xi)) for xi in fin_f1])

length = max(map(len, fin_val_f1))
y2 = np.array([xi+[np.nan]*(length-len(xi)) for xi in fin_val_f1])

length = max(map(len, fin_loss))
y3 = np.array([xi+[np.nan]*(length-len(xi)) for xi in fin_loss])

length = max(map(len, fin_val_loss))
y4 = np.array([xi+[np.nan]*(length-len(xi)) for xi in fin_val_loss])


train_scores_mean = np.nanmean(y, axis=0)
train_scores_std = np.nanstd(y, axis=0)
test_scores_mean = np.nanmean(y2, axis=0)
test_scores_std = np.nanstd(y2, axis=0)


x_ticks = np.arange(length)
title = r"Learning Curves (5 layers, 256 neurons, relu, glorot)"
_, axes = plt.subplots(1, 1, figsize=(14, 6))
axes.grid()
axes.fill_between(
    x_ticks,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.1,
    color="r",
)
axes.fill_between(
    x_ticks,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.1,
    color="g",
)

# axes.plot(x_ticks, train_scores_mean, 'o-', color='r', label='Train', linewidth=3)  # linewidth
# axes.plot(x_ticks, test_scores_mean, 'o-', color='g', label='Validation', linewidth=3)
axes.plot(x_ticks, train_scores_mean, 'o-', color='r', label='Train')
axes.plot(x_ticks, test_scores_mean, 'o-', color='g', label='Validation')
axes.set_ylabel('F1')
axes.set_xlabel('Epoch')
axes.legend(loc='upper left')
# plt.xlim(0, 17)
# plt.savefig(path + "/Figures/FINAL/lstm_{}_F1.png".format(fusion))
# plt.savefig("lstm_results/lstm_F1_reg_{}".format(args.png))
plt.savefig("lstm_results/F1_{}.png".format(fusion))
plt.close()


train_scores_mean = np.nanmean(y3, axis=0)
train_scores_std = np.nanstd(y3, axis=0)
test_scores_mean = np.nanmean(y4, axis=0)
test_scores_std = np.nanstd(y4, axis=0)

x_ticks = np.arange(length)
# title = r"Learning Curves (FNN, 5 layers, 256 neurons, relu, glorot)"
_, axes = plt.subplots(1, 1, figsize=(14, 6))
axes.grid()
axes.fill_between(
    x_ticks,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.1,
    color="r",
)
axes.fill_between(
    x_ticks,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.1,
    color="g",
)

axes.plot(x_ticks, train_scores_mean, 'o-', color="r", label='Train')
axes.plot(x_ticks, test_scores_mean, 'o-', color="g", label='Validation')
axes.set_ylabel('Loss')
axes.set_xlabel('Epoch')
axes.legend(loc='upper left')
# plt.xlim(0, 17)
# plt.savefig(path + "/Figures/FINAL/lstm_{}_F1.png".format(fusion))
plt.savefig("lstm_results/loss_{}.png".format(fusion))
plt.close()

sacc = sacc / args.cv_folds
spre = spre / args.cv_folds
srec = srec / args.cv_folds
sf1 = sf1 / args.cv_folds
sauc = sauc / args.cv_folds

list_results = [sacc, spre, srec, sf1, sauc]

"""
# You can replace 'a' with 'w' if you want to overwrite the file each time instead of appending
with open('results_5.txt', 'a') as f:
    f.write(f"{args.metrics_title}\n")  # --metrics_title= central_combined
    f.write('accuracy, precision, recall, f1, auc :' + str(list_results) + "\n")
"""
print(args.metrics_title or fusion)
print('accuracy, precision, recall, f1, auc :', list_results)
save_metrics("LSTM", fusion, sacc, spre, srec, sf1, sauc)


# powershell command: python LSTM.py --fusion=drone1_cyber --metrics_title=drone1_cyber
