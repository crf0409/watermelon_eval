# -----------------------------------------------------------
# 0. 依赖
# -----------------------------------------------------------
import os, tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from keras import layers, models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
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

BATCH_SIZE = 8;   EPOCHS = 10; TEST_RATIO = .30
LR = 1e-5

# -----------------------------------------------------------
# 1. 扫描数据 —— 返回三条路径和分组标记
# -----------------------------------------------------------
dataset_dir = "/home/siton02/md0/crf/watermelon_eval/dataset/19_datasets"
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

from keras import ops                # NEW import

def cbam_block(x, ratio=8, name=None):
    ch = x.shape[-1]

    # ---------- Channel attention (unchanged) ----------
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    share = models.Sequential([
        layers.Dense(ch // ratio, activation='relu', use_bias=False),
        layers.Dense(ch, activation='sigmoid', use_bias=False)
    ])
    ch_attn = layers.Add()([share(gap), share(gmp)])
    ch_attn = layers.Reshape((1, 1, ch))(ch_attn)
    x = layers.Multiply()([x, ch_attn])

    # ---------- Spatial attention (re-written) ----------
    avg  = ops.mean(x, axis=-1, keepdims=True)   # keras.ops is Functional-aware
    max_ = ops.max (x, axis=-1, keepdims=True)
    s    = layers.Concatenate(axis=-1)([avg, max_])
    s    = layers.Conv2D(1, 7, padding="same",
                         activation="sigmoid",
                         use_bias=False)(s)
    return layers.Multiply()([x, s])




def conv_block(x, filters, pool=True, name=None):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                      name=None if name is None else f"{name}_conv")(x)
    x = layers.BatchNormalization(
        name=None if name is None else f"{name}_bn")(x)
    x = cbam_block(x, name=None if name is None else f"{name}_cbam")  # ★ 新增
    if pool:
        x = layers.MaxPooling2D(2,
                name=None if name is None else f"{name}_pool")(x)
    return x


from keras.layers import MultiHeadAttention, LayerNormalization, Add, Dense

def cross_attention(fast, slow, n_heads=4, key_dim=64, name="xattn"):
    # 投影 slow 到 fast 的维度
    if fast.shape[-1] != slow.shape[-1]:
        slow = layers.Dense(fast.shape[-1], activation='linear', name=f"{name}_proj")(slow)

    # (B, C) --> (B, 1, C)
    q = layers.Reshape((1, -1))(fast)
    k = layers.Reshape((1, -1))(slow)
    v = k
    attn = MultiHeadAttention(num_heads=n_heads, key_dim=key_dim, name=f"{name}_mha")(query=q, key=k, value=v)
    attn = Add()([q, attn])         # residual
    attn = LayerNormalization()(attn)
    attn = Dense(fast.shape[-1], activation='relu')(attn)
    attn = layers.Reshape(fast.shape[1:])(attn)  # (B,C) 复原
    return attn



def build_slowfast(img_shape=(IMG_SIZE, IMG_SIZE, 3),
                   mel_shape=(MEL_SIZE, MEL_SIZE, 3),
                   embed_dim=128):
    # ----- Fast branch：Image -----
    img_in = layers.Input(img_shape, name="img")
    x = conv_block(img_in, 32, pool=True, name="fast1")
    x = conv_block(x, 64, pool=True,  name="fast2")
    x = conv_block(x, 128, pool=True, name="fast3")
    fast_feat = layers.GlobalAveragePooling2D(name="fast_gap")(x)

    # ----- Slow branch：Mel -----
    mel_in = layers.Input(mel_shape, name="mel")
    y = conv_block(mel_in, 16, pool=True,  name="slow1")
    y = conv_block(y, 32, pool=True,  name="slow2")
    y = conv_block(y, 64, pool=True,  name="slow3")
    slow_feat = layers.GlobalAveragePooling2D(name="slow_gap")(y)

    # ----- Cross‑Attention（可关闭） -----
    fast_feat = cross_attention(fast_feat, slow_feat, name="img2mel")

    # ----- Fuse & head -----
    fused = layers.Concatenate(name="fusion")([fast_feat, slow_feat])
    h = layers.Dense(embed_dim, activation='relu', name="fc1")(fused)
    out = layers.Dense(1, name="regress")(h)
    return models.Model([img_in, mel_in], out, name="SlowFastAttn")


model = build_slowfast()
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
    patience=8,
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
    callbacks=[early, checkpoint],
    verbose=2)

# -----------------------------------------------------------
# 6. 载入最佳模型权重并做 Grad-CAM 可视化
#    （确保用的就是 val_mae 最小时那一轮）
# -----------------------------------------------------------
print(f"\n>>> Loading best checkpoint from: {ckpt_path}")
best_model = tf.keras.models.load_model(ckpt_path)   # ← ★ 载入最佳权重

def grad_cam(model, img, mel, layer_name="fast3_conv"):
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

for i in range(3):
    show_cam(i)

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
from keras.models import load_model

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
# -----------------------------------------------------------
# 8. 将最佳模型转换为 ONNX 格式
# -----------------------------------------------------------
import tf2onnx
import onnx

print("\n>>> 开始转换模型为 ONNX 格式...")

# 方法1: 直接从 Keras 模型转换
onnx_model_path = "xigua_slowfast_best.onnx"

# 指定输入规格
spec = (
    tf.TensorSpec((None, IMG_SIZE, IMG_SIZE, 3), tf.float32, name="img"),
    tf.TensorSpec((None, MEL_SIZE, MEL_SIZE, 3), tf.float32, name="mel")
)

# 转换模型
model_proto, _ = tf2onnx.convert.from_keras(
    best_model,
    input_signature=spec,
    opset=13,  # ONNX opset 版本,可根据需要调整
    output_path=onnx_model_path
)

print(f"✓ ONNX 模型已保存至: {onnx_model_path}")

# 验证 ONNX 模型
try:
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX 模型验证通过")
    
    # 打印模型信息
    print(f"\n模型输入:")
    for inp in onnx_model.graph.input:
        print(f"  - {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    print(f"\n模型输出:")
    for out in onnx_model.graph.output:
        print(f"  - {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
        
except Exception as e:
    print(f"✗ ONNX 模型验证失败: {e}")

# 可选: 测试 ONNX 模型推理
print("\n>>> 测试 ONNX 模型推理...")
try:
    import onnxruntime as ort
    
    # 创建推理会话
    ort_session = ort.InferenceSession(onnx_model_path)
    
    # 获取一个测试样本
    (test_img, test_mel), test_lbl = next(val_ds.unbatch().take(1).as_numpy_iterator())
    
    # 准备输入
    ort_inputs = {
        ort_session.get_inputs()[0].name: test_img[None].astype(np.float32),
        ort_session.get_inputs()[1].name: test_mel[None].astype(np.float32)
    }
    
    # ONNX 推理
    ort_pred = ort_session.run(None, ort_inputs)[0][0, 0]
    
    # Keras 推理
    keras_pred = best_model.predict([test_img[None], test_mel[None]], verbose=0)[0, 0]
    
    print(f"Keras 预测: {keras_pred:.4f}")
    print(f"ONNX  预测: {ort_pred:.4f}")
    print(f"差异: {abs(keras_pred - ort_pred):.6f}")
    
    if abs(keras_pred - ort_pred) < 1e-3:
        print("✓ ONNX 模型转换成功,预测结果一致!")
    else:
        print("⚠ 注意: ONNX 和 Keras 预测存在差异")
        
except ImportError:
    print("⚠ 未安装 onnxruntime,跳过推理测试")
    print("  可通过 'pip install onnxruntime' 安装")
except Exception as e:
    print(f"✗ ONNX 推理测试失败: {e}")

print("\n>>> 转换完成!")