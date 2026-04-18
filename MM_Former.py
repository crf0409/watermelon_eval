# -----------------------------------------------------------
# 0. 依赖
# -----------------------------------------------------------
import os, tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from keras import layers, models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
AUTOTUNE = tf.data.AUTOTUNE



# -------------------- 全局超参 -------------------------------
IMG_SIZE  = 1024          # 图像仍保持
MEL_SIZE  = 256           # ← 建议改为 256
SAMPLE_RATE = 16_000      # 先下采样，利于 GPU / 比较文献
FRAME_LEN  = 1024         # 64 ms
HOP_LEN    = 256          # 16 ms（≈ 75 % overlap）
FFT_LEN    = 1024         # (= FRAME_LEN)
N_MELS     = 128          # 梅尔频带数
FREQ_BINS   = 256

BATCH_SIZE = 1;   EPOCHS = 100; TEST_RATIO = .30
LR = 1e-5

# -----------------------------------------------------------
# 1. 扫描数据 —— 返回三条路径和分组标记
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
# 2. Group-Shuffle-Split  → train / val 索引
# -----------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=42)
train_idx, val_idx = next(gss.split(wav_paths, labels, groups))

def make_ds(idxs):
    return tf.data.Dataset.from_tensor_slices(
        (wav_paths[idxs], img_paths[idxs], labels[idxs]))

train_raw = make_ds(train_idx);  val_raw = make_ds(val_idx)
print(f"Train={len(train_idx)}  |  Val={len(val_idx)}")

# -----------------------------------------------------------
# 3. 预处理：图片 + Mel
# -----------------------------------------------------------
def load_image(path):
    img = tf.io.read_file(path); img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32) / 255.

@tf.function
def wav_to_mel(path):
    audio_bin = tf.io.read_file(path)
    audio, _  = tf.audio.decode_wav(audio_bin, desired_channels=2)
    wav = audio[:, 1][:SAMPLE_RATE]  # 取右声道
    stft = tf.signal.stft(wav, frame_length=FRAME_LEN, frame_step=HOP_LEN,
                          fft_length=FFT_LEN, window_fn=tf.signal.hann_window,
                          pad_end=True)
    power = tf.abs(stft)**2
    mel_mat = tf.signal.linear_to_mel_weight_matrix(
        FREQ_BINS, power.shape[-1], SAMPLE_RATE, 0.0, SAMPLE_RATE/2)
    mel = tf.tensordot(power, mel_mat, 1)           # (T, F)
    mel = tf.math.log(mel + 1e-6)                   # dB
    mel = tf.image.resize(mel[..., None], (MEL_SIZE, MEL_SIZE))
    return tf.tile(mel, [1, 1, 3]) / tf.math.reduce_max(mel)  # 归一化

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
# 4. SlowFast-like 双分支网络
# -----------------------------------------------------------

from keras import layers, models
from keras.api.layers import MultiHeadAttention, LayerNormalization, Dropout

def mmformer_block(x, n_heads=4, d_ff=256, dropout=0.1, name=None):
    """标准Transformer encoder block（支持token输入，适配Cross Modal场景）"""
    # Multi-Head Attention
    attn_out = MultiHeadAttention(num_heads=n_heads, key_dim=x.shape[-1], dropout=dropout)(x, x)
    x = layers.Add()([x, attn_out])
    x = LayerNormalization(epsilon=1e-5)(x)

    # Feed-Forward
    ff = layers.Dense(d_ff, activation='relu')(x)
    ff = Dropout(dropout)(ff)
    ff = layers.Dense(x.shape[-1])(ff)
    x = layers.Add()([x, ff])
    x = LayerNormalization(epsilon=1e-5)(x)
    return x

def build_mmformer(img_shape=(IMG_SIZE, IMG_SIZE, 3),
                   mel_shape=(MEL_SIZE, MEL_SIZE, 3),
                   embed_dim=128, token_len=16, n_layers=2, n_heads=4):

    # -------- 图像分支 -------- #
    img_in = layers.Input(img_shape, name="img")
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(img_in)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(embed_dim, 3, strides=2, padding='same', activation='relu')(x)

    # ❶ 把 Flatten + Dense 换成 GAP + 小全连接
    x = layers.GlobalAveragePooling2D()(x)        # (B, embed_dim)
    img_tokens = layers.Dense(embed_dim)(x)       # (B, embed_dim)
    img_tokens = layers.Reshape((1, embed_dim))(img_tokens)

    # -------- Mel 分支 -------- #
    mel_in = layers.Input(mel_shape, name="mel")
    y = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(mel_in)
    y = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(y)
    y = layers.Conv2D(embed_dim, 3, strides=2, padding='same', activation='relu')(y)

    # ❷ 同理替换
    y = layers.GlobalAveragePooling2D()(y)
    mel_tokens = layers.Dense(embed_dim)(y)
    mel_tokens = layers.Reshape((1, embed_dim))(mel_tokens)

    # -------- 拼接 + Transformer -------- #
    tokens = layers.Concatenate(axis=1)([img_tokens, mel_tokens])  # (B, 2, embed_dim)
    for i in range(n_layers):
        tokens = mmformer_block(tokens, n_heads=n_heads,
                                d_ff=embed_dim*4, dropout=0.1,
                                name=f"mmformer_blk_{i}")

    pooled = layers.GlobalAveragePooling1D()(tokens)
    h = layers.Dense(embed_dim, activation='relu')(pooled)
    out = layers.Dense(1)(h)
    return models.Model([img_in, mel_in], out, name="MMFormer_tiny")



model = build_mmformer()
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss='mse', metrics=['mae'])
model.summary(line_length=110)

# -----------------------------------------------------------
# 5. 训练（新增：按 val_mae 最小保存权重）
# -----------------------------------------------------------
class InspectPred(tf.keras.callbacks.Callback):
    """每个 epoch 结束后随机查看 1 个样本的预测值，方便监控训练走势。"""
    def on_epoch_end(self, epoch, logs=None):
        (img, mel), lbl = next(val_ds.unbatch().take(1).as_numpy_iterator())
        pred = self.model.predict([img[None], mel[None]], verbose=0)[0, 0]
        print(f'\nEpoch {epoch}: sample pred={pred:.3f}, label={lbl:.3f}')

# —— 提前停止：按 val_mae 判定           ↓ ★ 关键：换成 val_mae
early = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',             # 监控指标从 val_loss → val_mae
    patience=16,
    mode='min',
    restore_best_weights=True)

# —— 模型权重保存：只保存 val_mae 最小的那一轮 ↓ ★ 关键：save_best_only + monitor
ckpt_path = "best_slowfast_mae.keras"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    monitor='val_mae',             # 同样跟踪 val_mae
    mode='min',
    save_best_only=True,           # 仅保存最优权重
    save_weights_only=False,       # 直接存完整模型，方便日后 load_model(...)
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[early, checkpoint, InspectPred()],
    verbose=2)

# -----------------------------------------------------------
# 6. 载入最佳模型权重并做 Grad-CAM 可视化
#    （确保用的就是 val_mae 最小时那一轮）
# -----------------------------------------------------------
print(f"\n>>> Loading best checkpoint from: {ckpt_path}")
best_model = tf.keras.models.load_model(ckpt_path)   # ← ★ 载入最佳权重

def grad_cam(model, img, mel, layer_name="conv2d_5"):
    # 与之前一致，只是把 default model 换成 best_model
    grad_model = models.Model(model.inputs,
                              [model.get_layer(layer_name).output,
                               model.output])
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model([img, mel], training=False)
        loss = pred[:, 0]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE)).numpy().squeeze()
    return cam

def show_cam(idx):
    (img, mel), lbl = next(val_ds.unbatch().skip(idx).take(1).as_numpy_iterator())
    cam = grad_cam(best_model, img[None], mel[None])          # ← ★ 使用 best_model
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(img);           axs[0].axis('off'); axs[0].set_title("Input")
    axs[1].imshow(img); axs[1].imshow(cam, cmap='jet', alpha=0.5)
    axs[1].axis('off');           axs[1].set_title("Grad-CAM")
    pred = best_model.predict([img[None], mel[None]], verbose=0)[0, 0]
    plt.suptitle(f"GT:{lbl:.2f} | Pred:{pred:.2f}")
    plt.show()



# -----------------------------------------------------------
# 7. 用最佳权重做正式评估 & 再次保存（可选）
# -----------------------------------------------------------
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

# ---------- 自定义指标函数 ---------- #
def nrmse(y_true, y_pred, norm="range"):
    """归一化 RMSE；norm 可选 'range' 或 'mean'"""
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
    """Nash–Sutcliffe Efficiency"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return 1.0 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

def willmott_d(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.sum(
        (np.abs(y_pred - y_true.mean()) + np.abs(y_true - y_true.mean())) ** 2
    )
    return 1.0 - np.sum((y_pred - y_true) ** 2) / denom

def ccc(y_true, y_pred):
    """Lin’s Concordance Correlation Coefficient"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(ddof=1), y_pred.var(ddof=1)
    cov_tp = np.mean((y_true - mean_t) * (y_pred - mean_p))
    return 2.0 * cov_tp / (var_t + var_p + (mean_t - mean_p) ** 2)

def bland_altman(y_true, y_pred):
    """返回平均差 & 95 % LOA（下、上限）"""
    diff = np.asarray(y_pred) - np.asarray(y_true)
    md = diff.mean()
    sd = diff.std(ddof=1)
    return md, md - 1.96 * sd, md + 1.96 * sd


# ---------- 推理并收集结果 ---------- #
y_true, y_pred = [], []
for (img, mel), lbl in val_ds:
    y_true.extend(lbl.numpy().ravel())
    y_pred.extend(best_model.predict([img, mel], verbose=0).ravel())

# ---------- 计算所有指标 ---------- #
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

# Bland‑Altman 统计另行打印（常用于画图）
ba_mean, ba_low, ba_up = bland_altman(y_true, y_pred)

# ---------- 输出 ---------- #
for k, v in metrics.items():
    print(f"{k:12s}: {v: .4f}")
print(f"Bland–Altman mean diff : {ba_mean: .4f}")
print(f"95% LOA (lower, upper) : ({ba_low: .4f}, {ba_up: .4f})")


# 若想再存一份备份，可重新命名
best_model.save("xigua_slowfast_tiny_best.keras")

# -----------------------------------------------------------
# 载入最佳模型权重 & 测试推理速度（补充）
# -----------------------------------------------------------
import time
import tensorflow as tf
import numpy as np
from keras.api.models import load_model

# 1. 载入最优模型（假定权重文件和结构均为 best_slowfast_mae.keras）
ckpt_path = "best_slowfast_mae.keras"  # 按你的路径名改
model = load_model(ckpt_path)


BATCHES = list(val_ds)   # 提前全部拉进内存，避免 IO 干扰测速

import tqdm
N_REPEATS = 3
infer_times = []
num_samples = 0

for _ in range(N_REPEATS):
    for (img, mel), lbl in tqdm.tqdm(BATCHES, desc='推理测试'):
        t0 = time.perf_counter()
        pred = model.predict([img, mel], verbose=0)
        t1 = time.perf_counter()
        infer_times.append(t1-t0)
        num_samples += img.shape[0]

# 4. 输出速度（单位：秒/样本、毫秒/样本、批次）
total_infer_time = sum(infer_times)
avg_time_per_batch = total_infer_time / (len(BATCHES) * N_REPEATS)
avg_time_per_sample = total_infer_time / num_samples

print(f"模型推理耗时: 总计 {total_infer_time:.4f} 秒 | 平均每批: {avg_time_per_batch*1000:.2f} ms | 平均每样本: {avg_time_per_sample*1000:.2f} ms")
