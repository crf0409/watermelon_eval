# -----------------------------------------------------------
# 0. 依赖
# -----------------------------------------------------------
import os, tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from keras import layers, models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
AUTOTUNE = tf.data.AUTOTUNE

# -------------------- 全局超参 -------------------------------
IMG_SIZE  = 224        # ResNet50 标准输入 224
BATCH_SIZE = 8;   EPOCHS = 100; TEST_RATIO = .30
LR = 1e-5

# -----------------------------------------------------------
# 1. 扫描数据 —— 只用图片和标签
# -----------------------------------------------------------
dataset_dir = "/home/siton02/crf/watermelon_eval/dataset/19_datasets"
img_paths, labels, groups = [], [], []

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
        img = next((f for f in os.listdir(base) if f.endswith(".jpg")), None)
        if not img: continue
        img_paths.append(os.path.join(base, img))
        labels.append(lbl); groups.append(data_id)

img_paths = np.array(img_paths)
labels    = np.array(labels, dtype=np.float32)
groups    = np.array(groups)
print(f"Total samples={len(labels)} | unique watermelons={len(set(groups))}")

# -----------------------------------------------------------
# 2. Group-Shuffle-Split  → train / val 索引
# -----------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=42)
train_idx, val_idx = next(gss.split(img_paths, labels, groups))

def make_ds(idxs):
    return tf.data.Dataset.from_tensor_slices(
        (img_paths[idxs], labels[idxs]))

train_raw = make_ds(train_idx)
val_raw = make_ds(val_idx)
print(f"Train={len(train_idx)}  |  Val={len(val_idx)}")

# -----------------------------------------------------------
# 3. 预处理：图片
# -----------------------------------------------------------
def load_image(path):
    img = tf.io.read_file(path); img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.resnet50.preprocess_input(img) # 标准化
    return img

def preprocess(img_p, lbl):
    img = load_image(img_p)
    return img, lbl

train_ds = (train_raw
    .shuffle(len(train_idx), reshuffle_each_iteration=True)
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE).prefetch(AUTOTUNE))
val_ds   = (val_raw
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE).prefetch(AUTOTUNE))

# -----------------------------------------------------------
# 4. ResNet50 图像回归模型
# -----------------------------------------------------------
def build_resnet50_regressor(img_shape=(IMG_SIZE, IMG_SIZE, 3), embed_dim=128):
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=img_shape)
    base.trainable = True  # 若数据较少可以设置为 False
    inputs = layers.Input(img_shape, name="img")
    x = base(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embed_dim, activation='relu')(x)
    out = layers.Dense(1, name="regress")(x)
    return models.Model(inputs, out, name="ResNet50_Reg")

model = build_resnet50_regressor()
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss='mse', metrics=['mae'])
model.summary(line_length=110)

# -----------------------------------------------------------
# 5. 训练（按 val_mae 最小保存权重）
# -----------------------------------------------------------
class InspectPred(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        img, lbl = next(val_ds.unbatch().take(1).as_numpy_iterator())
        pred = self.model.predict(img[None], verbose=0)[0, 0]
        print(f'\nEpoch {epoch}: sample pred={pred:.3f}, label={lbl:.3f}')

early = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',
    patience=8,
    mode='min',
    restore_best_weights=True)

ckpt_path = "best_resnet50_img_mae.keras"
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
# 6. 载入最佳模型权重并做 Grad-CAM 可视化
# -----------------------------------------------------------
print(f"\n>>> Loading best checkpoint from: {ckpt_path}")
best_model = tf.keras.models.load_model(ckpt_path)

def grad_cam(model, img, layer_name="dense"):
    grad_model = models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img, training=False)
        loss = pred[:, 0]
    grads = tape.gradient(loss, conv_out)
    # grads 和 conv_out 都是 [B, H, W, C]
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE)).numpy().squeeze()
    return cam


def show_cam(idx):
    img, lbl = next(val_ds.unbatch().skip(idx).take(1).as_numpy_iterator())
    cam = grad_cam(best_model, img[None])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(tf.keras.applications.resnet50.preprocess_input(img)[..., ::-1]) # 原图
    axs[0].axis('off'); axs[0].set_title("Input")
    axs[1].imshow(tf.keras.applications.resnet50.preprocess_input(img)[..., ::-1])
    axs[1].imshow(cam, cmap='jet', alpha=0.5)
    axs[1].axis('off'); axs[1].set_title("Grad-CAM")
    pred = best_model.predict(img[None], verbose=0)[0, 0]
    plt.suptitle(f"GT:{lbl:.2f} | Pred:{pred:.2f}")
    plt.show()



# -----------------------------------------------------------
# 7. 用最佳权重做正式评估
# -----------------------------------------------------------
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
for img, lbl in val_ds:
    y_true.extend(lbl.numpy().ravel())
    y_pred.extend(best_model.predict(img, verbose=0).ravel())

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

best_model.save("xigua_resnet50_img_best.keras")

# -----------------------------------------------------------
# 载入最佳模型权重 & 测试推理速度（可选）
# -----------------------------------------------------------
import time
from keras.api.models import load_model

model = load_model(ckpt_path)
BATCHES = list(val_ds)
import tqdm
N_REPEATS = 3
infer_times = []
num_samples = 0
for _ in range(N_REPEATS):
    for img, lbl in tqdm.tqdm(BATCHES, desc='推理测试'):
        t0 = time.perf_counter()
        pred = model.predict(img, verbose=0)
        t1 = time.perf_counter()
        infer_times.append(t1-t0)
        num_samples += img.shape[0]

total_infer_time = sum(infer_times)
avg_time_per_batch = total_infer_time / (len(BATCHES) * N_REPEATS)
avg_time_per_sample = total_infer_time / num_samples

print(f"模型推理耗时: 总计 {total_infer_time:.4f} 秒 | 平均每批: {avg_time_per_batch*1000:.2f} ms | 平均每样本: {avg_time_per_sample*1000:.2f} ms")

for i in range(3):
    show_cam(i)