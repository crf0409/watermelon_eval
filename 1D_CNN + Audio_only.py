# -----------------------------------------------------------
# 0. 依赖
# -----------------------------------------------------------
import gc
import os, tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from keras import layers, models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
AUTOTUNE = tf.data.AUTOTUNE

# -------------------- 全局超参 -------------------------------
SAMPLE_RATE = 16_000      # 先下采样，利于 GPU / 比较文献
FRAME_LEN  = 1024         # 64 ms
HOP_LEN    = 256          # 16 ms（≈ 75 % overlap）
FFT_LEN    = 1024         # (= FRAME_LEN)
AUDIO_LEN  = 16000        # 1秒原始数据
BATCH_SIZE = 8;   EPOCHS = 100; TEST_RATIO = .30
LR = 1e-4

# -----------------------------------------------------------
# 1. 扫描数据 —— 只用音频和标签
# -----------------------------------------------------------
dataset_dir = "/home/siton02/crf/watermelon_eval/dataset/19_datasets"
wav_paths, labels, groups = [], [], []

for sd in os.listdir(dataset_dir):
    full = os.path.join(dataset_dir, sd)
    if not os.path.isdir(full): continue
    try:
        data_id, lbl = sd.split("_"); lbl = float(lbl)
    except ValueError:
        continue
    chu_dir = os.path.join(full, "chu")
    for fd in os.listdir(chu_dir):
        base = os.path.join(chu_dir, fd)
        if not os.path.isdir(base): continue
        wav = next((f for f in os.listdir(base) if f.endswith(".wav")), None)
        if not wav: continue
        wav_paths.append(os.path.join(base, wav))
        labels.append(lbl); groups.append(data_id)

wav_paths = np.array(wav_paths)
labels    = np.array(labels, dtype=np.float32)
groups    = np.array(groups)
print(f"Total samples={len(labels)} | unique watermelons={len(set(groups))}")

# -----------------------------------------------------------
# 2. Group-Shuffle-Split  → train / val 索引
# -----------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=42)
train_idx, val_idx = next(gss.split(wav_paths, labels, groups))

def make_ds(idxs):
    return tf.data.Dataset.from_tensor_slices(
        (wav_paths[idxs], labels[idxs]))

train_raw = make_ds(train_idx)
val_raw   = make_ds(val_idx)
print(f"Train={len(train_idx)}  |  Val={len(val_idx)}")

# -----------------------------------------------------------
# 3. 预处理：音频
# -----------------------------------------------------------
def load_wav(path):
    audio_bin = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio_bin, desired_channels=1)
    wav = audio[:, 0]
    wav = wav[:AUDIO_LEN]  # 取固定长度，过短补零
    zero_padding = tf.zeros([AUDIO_LEN] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([wav, zero_padding], 0)
    return wav

def preprocess(wav_p, lbl):
    wav = load_wav(wav_p)
    wav = tf.reshape(wav, (AUDIO_LEN, 1))
    return wav, lbl

train_ds = (train_raw
    .shuffle(len(train_idx), reshuffle_each_iteration=True)
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE).prefetch(AUTOTUNE))
val_ds   = (val_raw
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE).prefetch(AUTOTUNE))

# -----------------------------------------------------------
# 4. 1D CNN 声音回归模型
# -----------------------------------------------------------
def build_1dcnn_audio(audio_len=AUDIO_LEN, embed_dim=128):
    inp = layers.Input((audio_len, 1), name="wav")
    x = layers.Conv1D(32, 11, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Conv1D(64, 9, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Conv1D(128, 7, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(embed_dim, activation='relu')(x)
    out = layers.Dense(1, name="regress")(x)
    return models.Model(inp, out, name="Audio1D_CNN")

model = build_1dcnn_audio()
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss='mse', metrics=['mae'])
model.summary(line_length=110)

# -----------------------------------------------------------
# 5. 训练（按 val_mae 最小保存权重）
# -----------------------------------------------------------
class InspectPred(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        wav, lbl = next(val_ds.unbatch().take(1).as_numpy_iterator())
        pred = self.model.predict(wav[None], verbose=0)[0, 0]
        print(f'\nEpoch {epoch}: sample pred={pred:.3f}, label={lbl:.3f}')

early = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',
    patience=16,
    mode='min',
    restore_best_weights=True)

ckpt_path = "best_audio1dcnn_mae.keras"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    monitor='val_mae',
    mode='min',
    save_best_only=True,
    save_weights_only=False,
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[early, checkpoint, InspectPred()],
    verbose=2)

# -----------------------------------------------------------
# 6. 载入最佳模型权重并评估
# -----------------------------------------------------------
print(f"\n>>> Loading best checkpoint from: {ckpt_path}")
best_model = tf.keras.models.load_model(ckpt_path)

# ---------- 自定义指标（与原文一致） ---------- #
def nrmse(y_true, y_pred, norm="range"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if norm == "range":
        return rmse / (np.max(y_true) - np.min(y_true))
    elif norm == "mean":
        return rmse / np.mean(y_true)
    else:
        raise ValueError("norm must be 'range' or 'mean'")

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(
        2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    ) * 100.0

def nse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return 1.0 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

def willmott_d(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.sum(
        (np.abs(y_pred - y_true.mean()) + np.abs(y_true - y_true.mean())) ** 2
    )
    return 1.0 - np.sum((y_pred - y_true) ** 2) / denom

def ccc(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(ddof=1), y_pred.var(ddof=1)
    cov_tp = np.mean((y_true - mean_t) * (y_pred - mean_p))
    return 2.0 * cov_tp / (var_t + var_p + (mean_t - mean_p) ** 2)

def bland_altman(y_true, y_pred):
    diff = np.asarray(y_pred) - np.asarray(y_true)
    md = diff.mean()
    sd = diff.std(ddof=1)
    return md, md - 1.96 * sd, md + 1.96 * sd

y_true, y_pred = [], []
for wav, lbl in val_ds:
    y_true.extend(lbl.numpy().ravel())
    y_pred.extend(best_model.predict(wav, verbose=0).ravel())

metrics = {
    "MAE":  mean_absolute_error(y_true, y_pred),
    "MSE":  mean_squared_error(y_true, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    "NRMSE_range": nrmse(y_true, y_pred, norm="range"),
    "NRMSE_mean":  nrmse(y_true, y_pred, norm="mean"),
    "MAPE": mape(y_true, y_pred),
    "SMAPE": smape(y_true, y_pred),
    "R²":   r2_score(y_true, y_pred),
    "NSE":  nse(y_true, y_pred),
    "Willmott_d": willmott_d(y_true, y_pred),
    "CCC":  ccc(y_true, y_pred),
}
ba_mean, ba_low, ba_up = bland_altman(y_true, y_pred)
for k, v in metrics.items():
    print(f"{k:12s}: {v: .4f}")
print(f"Bland–Altman mean diff : {ba_mean: .4f}")
print(f"95% LOA (lower, upper) : ({ba_low: .4f}, {ba_up: .4f})")

best_model.save("xigua_audio1dcnn_best.keras")



best_model.save("xigua_slowfast_tiny_best.keras")

import time
import gc
import tensorflow as tf
import numpy as np
from keras.api.models import load_model

ckpt_path = "xigua_audio1dcnn_best.keras"
model = load_model(ckpt_path)

BATCHES = list(val_ds)   # 注意 val_ds batch 格式: (wav, lbl)
import tqdm
N_REPEATS = 3
infer_times = []
num_samples = 0

for _ in range(N_REPEATS):
    for wav, lbl in tqdm.tqdm(BATCHES, desc='推理测试'):
        t0 = time.perf_counter()
        pred = model.predict(wav, verbose=0)   # 只传入 wav，不要用 [img, mel]
        t1 = time.perf_counter()
        infer_times.append(t1 - t0)
        num_samples += wav.shape[0]    # shape [batch, ...]

total_infer_time = sum(infer_times)
avg_time_per_batch = total_infer_time / (len(BATCHES) * N_REPEATS)
avg_time_per_sample = total_infer_time / num_samples

print(f"模型推理耗时: 总计 {total_infer_time:.4f} 秒 | 平均每批: {avg_time_per_batch*1000:.2f} ms | 平均每样本: {avg_time_per_sample*1000:.2f} ms")

del model
gc.collect()
tf.keras.backend.clear_session()
