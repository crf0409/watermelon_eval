````markdown
# 数据加载指南

本文档基于 `data_preprocess.py` 脚本，说明 **西瓜品质评估** 数据集的目录结构、依赖环境以及如何加载并划分训练 / 验证集。

---

## 1. 依赖环境

| 库 | 版本建议 | 说明 |
| --- | --- | --- |
| Python | ≥ 3.9 | 推荐使用 conda 或 venv |
| TensorFlow | ≥ 2.15 | 提供 tf.data、tf.audio、ResNet50 等 |
| NumPy | ≥ 1.23 | 处理 wav 数据 |
| （可选）CUDA / cuDNN | 与 TensorFlow 匹配 | 若使用 GPU 加速 |

安装示例：

```bash
conda create -n wm_eval python=3.9
conda activate wm_eval
pip install tensorflow numpy
````

---

## 2. 数据目录结构

假设数据根目录为

```
/home/siton01/watermelon_eval/datasets
```

整体层级如下（`*` 表示任意名称；`{label}` 为浮点型品质分数）：

```
datasets
└── {data_id}_{label}                 # 每个批次 / 样本文件夹
    └── chu                           # 固定子文件夹
        ├── {stage1}                  # 任意阶段或分组名称
        │   ├── xxx.wav               # 单通道或双通道音频
        │   └── xxx.jpg               # 与音频对应的图像
        ├── {stage2}
        │   ├── yyy.wav
        │   └── yyy.jpg
        └── ...                       # 可能存在多个子阶段
```

要点：

1. **顶层文件夹命名**：`<编号>_<品质标签>`，脚本通过下划线分割提取 `label`（`float` 类型）。
2. **阶段文件夹** (`stage*`)：内部必须各含 **唯一** 的 `*.wav` 和 `*.jpg`。
3. **音频文件**：脚本取 **右声道前 16000 个采样点**（约 1 秒）。
4. **图像文件**：读取后统一 `1080 × 1080` 并做 ResNet-50 预处理。

---

## 3. 脚本工作流程

<div align="center"><img src="https://mermaid.ink/img/pako:eNp1kV1P2jAUx78mv5uwbG3jbVUQYEFLEW3C8JkUKWIStEznC4Kd7XRJEVHX--0oo53Keneu6zMfDOfN2ZO7MZBMQGoPKPQxv5xzTHChmTS_9LBGY2KKfwqRX4P-kLoqhJ4YC8ZG9zB2oltFbMXPa-A_GOjwedxVpGc5Khrpcp2SNzaaTo-kpFpXKUh1XU3Nm-IkwUmETbwU2hEXmjXVHWg8eEnzlf4axrdaInx0xtPENoSL4Bga9k5jOQSE7HTUGULekQKjb3bjlyW4d3rYz6p53ro4tT1X2o3anTjdadpte8J50m3-RqMV55SWKabIBqoj1gyQV0CrKE99o7q4hLspuE6mqv1zX6-zQpXU6j5HLXTrfqAfenF7f-cnrw2-5jTz83Z-SIWyOWepfTb6_iFd9GX9udfkvzh9p_kCNpDQ==" alt="Data pipeline flowchart"></div>

1. **读取所有批次**

   * 遍历 `datasets` 下的 `N` 个子目录，按规则收集音频、图像、标签。
2. **合并为 `tf.data.Dataset`**

   * 每个批次生成一个 `Dataset`，再 `concatenate` 组合。
3. **随机洗牌** (`shuffle_buffer` 默认 1000)。
4. **划分 7 : 3 训练 / 验证集**。
5. **图像与音频转换**

   * `map`：音频右声道→NumPy array，图像→Tensor & ResNet-50 预处理。
   * 再 `map`：打包为 `((wav, img), label)`。
6. **批处理 + 预取**

   * `batch_size` 默认 4。
   * `prefetch(tf.data.AUTOTUNE)` 提前准备下一批数据。

---

## 4. 快速开始

```bash
python data_preprocess.py \
  --dataset_dir=/home/siton01/watermelon_eval/datasets \
  --batch_size=8 \
  --shuffle_buffer=2048
```

脚本会打印：

```
Train and val dataset loaded.
```

你可以在后续模型代码中直接引入：

```python
from data_preprocess import get_datasets

train_ds, val_ds = get_datasets(
    dataset_dir="/home/siton01/watermelon_eval/datasets",
    batch_size=8,
    shuffle_buffer=2048
)

for (wav, img), label in train_ds.take(1):
    print(wav.shape, img.shape, label)
```

---

## 5. 常见问题 FAQ

| 问题                                                            | 可能原因 / 解决方案                                              |
| ------------------------------------------------------------- | -------------------------------------------------------- |
| `ValueError: not enough values to unpack (expected 2, got 1)` | 某个阶段文件夹缺少配对的 `.wav` 或 `.jpg`；检查并补齐文件                     |
| `InvalidArgumentError: audio file could not be decoded`       | 音频损坏或格式不符（需 16-bit PCM WAV）；尝试重新导出                       |
| GPU 内存不足                                                      | 减小 `batch_size` 或使用 `TF_GPU_ALLOCATOR=cuda_malloc_async` |

---

## 6. 自定义 & 扩展

* **新增通道处理**：若需用左声道或立体声，可修改 `wav_data = audio[:, 1][:16000]`。
* **更多数据增强**：可在 `load_image` 前后加入随机翻转、色彩抖动等。
* **多标签任务**：把 `label` 换成列表或向量，在最终 `map` 中返回对应结构。

---

> 如对目录结构或加载逻辑有疑问，可在脚本同级新建 `issues.md` 记录问题，再行讨论或提交 PR 😄

```

---  
**使用方法**：复制上方内容存为 `DATA_LOADING_GUIDE.md`，与 `data_preprocess.py` 同目录或项目根目录，供团队成员参考。
```
