import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time
import gc

# -------------------- 全局参数 -------------------------------
IMG_SIZE  = 128
MEL_SIZE  = 128
SAMPLE_RATE = 16_000
FRAME_LEN  = 1024
HOP_LEN    = 256
FFT_LEN    = 1024
FREQ_BINS  = 256
TEST_RATIO = .30

# -------------------- 1. 扫描数据 ---------------------------
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

wav_paths = np.array(wav_paths)
img_paths = np.array(img_paths)
labels    = np.array(labels, dtype=np.float32)
groups    = np.array(groups)
print(f"Total samples={len(labels)} | unique watermelons={len(set(groups))}")

# -------------------- 2. 分组划分 ---------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=42)
train_idx, val_idx = next(gss.split(wav_paths, labels, groups))

# -------------------- 3. 特征提取 ---------------------------
def load_image_np(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.
    img = tf.reshape(img, [-1])
    return img.numpy()

def wav_to_mel_np(path):
    audio_bin = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio_bin, desired_channels=2)
    wav = audio[:, 1][:SAMPLE_RATE]
    stft = tf.signal.stft(wav, frame_length=FRAME_LEN, frame_step=HOP_LEN,
                          fft_length=FFT_LEN, window_fn=tf.signal.hann_window, pad_end=True)
    power = tf.abs(stft)**2
    mel_mat = tf.signal.linear_to_mel_weight_matrix(
        FREQ_BINS, power.shape[-1], SAMPLE_RATE, 0.0, SAMPLE_RATE/2)
    mel = tf.tensordot(power, mel_mat, 1)
    mel = tf.math.log(mel + 1e-6)
    mel = tf.image.resize(mel[..., None], (MEL_SIZE, MEL_SIZE))
    mel = tf.squeeze(mel, -1)
    mel = (mel - tf.reduce_min(mel)) / (tf.reduce_max(mel) - tf.reduce_min(mel) + 1e-6)
    mel = tf.reshape(mel, [-1])
    return mel.numpy()

def extract_features(wav_list, img_list):
    feats = []
    for wav, img in zip(wav_list, img_list):
        img_feat = load_image_np(img)
        mel_feat = wav_to_mel_np(wav)
        feat = np.concatenate([img_feat, mel_feat])
        feats.append(feat)
    return np.stack(feats)

print("提取训练集特征...")
X_train = extract_features(wav_paths[train_idx], img_paths[train_idx])
y_train = labels[train_idx]
print("提取验证集特征...")
X_val = extract_features(wav_paths[val_idx], img_paths[val_idx])
y_val = labels[val_idx]

print(f"训练集: {X_train.shape} 验证集: {X_val.shape}")

# -------------------- 4. 特征标准化 -------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# -------------------- 5. 机器学习回归器 ---------------------
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=0),
    'SVM': SVR(kernel="rbf", C=8, epsilon=0.2, gamma='auto'),
}
try:
    from xgboost import XGBRegressor
    models['XGBoost'] = XGBRegressor(tree_method="hist", random_state=0)
except ImportError:
    print("XGBoost 未安装，跳过 XGBoost 部分。")

results = {}

for name, model in models.items():
    print(f"\n训练 {name} ...")
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_val_s)
    results[name] = y_pred

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    print(f"{name} 验证集 MAE: {mae:.4f} RMSE: {rmse:.4f} R2: {r2:.4f}")

    # 保存模型
    joblib.dump(model, f"xigua_{name.lower()}_best.joblib")
    print(f"{name} 已保存为 xigua_{name.lower()}_best.joblib")

    # 推理耗时
    infer_times = []
    N_REPEATS = 3
    num_samples = X_val_s.shape[0]
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        _ = model.predict(X_val_s)
        t1 = time.perf_counter()
        infer_times.append(t1 - t0)
    total_infer_time = sum(infer_times)
    avg_time_per_batch = total_infer_time / N_REPEATS
    avg_time_per_sample = total_infer_time / (num_samples * N_REPEATS)
    print(f"推理耗时: 总计 {total_infer_time:.4f} 秒 | 平均每批: {avg_time_per_batch*1000:.2f} ms | 平均每样本: {avg_time_per_sample*1000:.2f} ms")

    gc.collect()

# -------------------- 6. 指标输出（可复用你的自定义指标） ---------------------
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

for name, y_pred in results.items():
    print(f"\n-------- {name} 各类回归指标 --------")
    metrics = {
        "MAE":  mean_absolute_error(y_val, y_pred),
        "MSE":  mean_squared_error(y_val, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_val, y_pred)),
        "NRMSE_range": nrmse(y_val, y_pred, norm="range"),
        "NRMSE_mean":  nrmse(y_val, y_pred, norm="mean"),
        "MAPE": mape(y_val, y_pred),
        "SMAPE": smape(y_val, y_pred),
        "R²":   r2_score(y_val, y_pred),
        "NSE":  nse(y_val, y_pred),
        "Willmott_d": willmott_d(y_val, y_pred),
        "CCC":  ccc(y_val, y_pred),
    }
    ba_mean, ba_low, ba_up = bland_altman(y_val, y_pred)
    for k, v in metrics.items():
        print(f"{k:12s}: {v: .4f}")
    print(f"Bland–Altman mean diff : {ba_mean: .4f}")
    print(f"95% LOA (lower, upper) : ({ba_low: .4f}, {ba_up: .4f})")
    
gc.collect()
