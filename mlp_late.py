# -----------------------------------------------------------
# 0. 依赖
# -----------------------------------------------------------
import os, tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from keras import layers, models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
AUTOTUNE = tf.data.AUTOTUNE

# -------------------- 全局超参 -------------------------------
IMG_SIZE  = 128
MEL_SIZE  = 128
SAMPLE_RATE = 16_000
FRAME_LEN  = 1024
HOP_LEN    = 256
FFT_LEN    = 1024
FREQ_BINS  = 256

BATCH_SIZE = 8;   EPOCHS = 100; TEST_RATIO = .30
LR = 1e-4

# -----------------------------------------------------------
# 1. 扫描数据
# -----------------------------------------------------------
dataset_dir = "/home/siton02/crf/watermelon_eval/dataset/19_datasets"
wav_paths, img_paths, labels, groups = [], [], [], []

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
        img = next((f for f in os.listdir(base) if f.endswith(".jpg")), None)
        if not wav or not img: continue
        wav_paths.append(os.path.join(base, wav))
        img_paths.append(os.path.join(base, img))
        labels.append(lbl); groups.append(data_id)

wav_paths = np.array(wav_paths); img_paths = np.array(img_paths)
labels    = np.array(labels, dtype=np.float32); groups = np.array(groups)
print(f"Total samples={len(labels)} | unique watermelons={len(set(groups))}")

# -----------------------------------------------------------
# 2. Group-Shuffle-Split
# -----------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=42)
train_idx, val_idx = next(gss.split(wav_paths, labels, groups))

def make_ds(idxs):
    return tf.data.Dataset.from_tensor_slices(
        (wav_paths[idxs], img_paths[idxs], labels[idxs]))

train_raw = make_ds(train_idx);  val_raw = make_ds(val_idx)
print(f"Train={len(train_idx)}  |  Val={len(val_idx)}")

# -----------------------------------------------------------
# 3. 预处理：图片+Mel，分别处理，输出两个向量
# -----------------------------------------------------------
def load_image(path):
    img = tf.io.read_file(path); img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.
    img = tf.reshape(img, [-1])      # (IMG_SIZE*IMG_SIZE*3,)
    return img

@tf.function
def wav_to_mel(path):
    audio_bin = tf.io.read_file(path)
    audio, _  = tf.audio.decode_wav(audio_bin, desired_channels=2)
    wav = audio[:, 1][:SAMPLE_RATE]
    stft = tf.signal.stft(wav, frame_length=FRAME_LEN, frame_step=HOP_LEN,
                          fft_length=FFT_LEN, window_fn=tf.signal.hann_window,
                          pad_end=True)
    power = tf.abs(stft)**2
    mel_mat = tf.signal.linear_to_mel_weight_matrix(
        FREQ_BINS, power.shape[-1], SAMPLE_RATE, 0.0, SAMPLE_RATE/2)
    mel = tf.tensordot(power, mel_mat, 1)
    mel = tf.math.log(mel + 1e-6)
    mel = tf.image.resize(mel[..., None], (MEL_SIZE, MEL_SIZE))
    mel = tf.squeeze(mel, -1)
    mel = (mel - tf.reduce_min(mel)) / (tf.reduce_max(mel) - tf.reduce_min(mel) + 1e-6)
    mel = tf.reshape(mel, [-1])      # (MEL_SIZE*MEL_SIZE,)
    return mel

def preprocess(wav_p, img_p, lbl):
    img = load_image(img_p)
    mel = wav_to_mel(wav_p)
    return (img, mel), lbl

train_ds = (train_raw
    .shuffle(len(train_idx), reshuffle_each_iteration=True)
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE).prefetch(AUTOTUNE))
val_ds   = (val_raw
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE).prefetch(AUTOTUNE))

# -----------------------------------------------------------
# 4. Late Fusion（双回归器 + Stacking MLP）
# -----------------------------------------------------------
def build_late_fusion_model(img_dim, mel_dim, embed_dim=256):
    # 图像分支
    img_in = layers.Input(shape=(img_dim,), name='img_input')
    x = layers.Dense(embed_dim, activation='relu')(img_in)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(embed_dim//2, activation='relu')(x)
    img_out = layers.Dense(1, name='img_regress')(x)

    # Mel分支
    mel_in = layers.Input(shape=(mel_dim,), name='mel_input')
    y = layers.Dense(embed_dim, activation='relu')(mel_in)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(embed_dim//2, activation='relu')(y)
    mel_out = layers.Dense(1, name='mel_regress')(y)

    # Stacking
    concat = layers.Concatenate()([img_out, mel_out])
    z = layers.Dense(32, activation='relu')(concat)
    out = layers.Dense(1, name='stacking_regress')(z)

    return models.Model([img_in, mel_in], out, name="LateFusionStacking")

img_dim = IMG_SIZE * IMG_SIZE * 3
mel_dim = MEL_SIZE * MEL_SIZE
model = build_late_fusion_model(img_dim, mel_dim)
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss='mse', metrics=['mae'])
model.summary(line_length=120)

# -----------------------------------------------------------
# 5. 训练（按 val_mae 最小保存权重）
# -----------------------------------------------------------
class InspectPred(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        (img, mel), lbl = next(val_ds.unbatch().take(1).as_numpy_iterator())
        pred = self.model.predict([img[None], mel[None]], verbose=0)[0, 0]
        print(f'\nEpoch {epoch}: sample pred={pred:.3f}, label={lbl:.3f}')

early = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',
    patience=16,
    mode='min',
    restore_best_weights=True)

ckpt_path = "best_late_fusion_stacking_mae.keras"
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
for (img, mel), lbl in val_ds:
    y_true.extend(lbl.numpy().ravel())
    y_pred.extend(best_model.predict([img, mel], verbose=0).ravel())

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

best_model.save("xigua_late_fusion_stacking_best.keras")

import time
import gc
import tensorflow as tf
import numpy as np
from keras.api.models import load_model

ckpt_path = "xigua_late_fusion_stacking_best.keras"
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
