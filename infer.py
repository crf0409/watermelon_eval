#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
西瓜糖度预测 - Gradio Web 应用
支持图片和音频上传，实时预测糖度
"""

import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import librosa
import soundfile as sf
import gradio as gr
import matplotlib.pyplot as plt
import io

# ==================== 配置参数 ====================
IMG_SIZE = 1024
MEL_SIZE = 256
SAMPLE_RATE = 16000
FRAME_LEN = 1024
HOP_LEN = 256
FFT_LEN = 1024
N_MELS = 128
FREQ_BINS = 256

# 默认模型路径
DEFAULT_MODEL_PATH = "xigua_slowfast_best.onnx"


# ==================== 图像预处理 ====================
def load_and_preprocess_image(img_path):
    """加载并预处理图片"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"图片加载失败: {e}")


# ==================== 音频预处理 ====================
def load_and_preprocess_audio(wav_path):
    """加载音频并转换为 Mel 频谱图"""
    try:
        # 加载音频
        try:
            audio, sr = sf.read(wav_path)
        except:
            audio, sr = librosa.load(wav_path, sr=None, mono=False)

        # 处理立体声
        if len(audio.shape) > 1:
            audio = audio[:, 1] if audio.shape[1] > 1 else audio[:, 0]

        # 重采样
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        # 截取或填充
        if len(audio) > SAMPLE_RATE:
            audio = audio[:SAMPLE_RATE]
        else:
            audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)), mode='constant')

        # STFT
        stft = librosa.stft(
            audio,
            n_fft=FFT_LEN,
            hop_length=HOP_LEN,
            win_length=FRAME_LEN,
            window='hann'
        )

        # 功率谱 -> Mel
        power = np.abs(stft) ** 2
        mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=FFT_LEN,
            n_mels=FREQ_BINS,
            fmin=0.0,
            fmax=SAMPLE_RATE / 2
        )
        mel_spec = np.dot(mel_basis, power)
        mel_spec = np.log(mel_spec + 1e-6)

        # Resize
        mel_spec = Image.fromarray(mel_spec)
        mel_spec = mel_spec.resize((MEL_SIZE, MEL_SIZE), Image.Resampling.BILINEAR)
        mel_spec = np.array(mel_spec, dtype=np.float32)

        # 归一化
        mel_max = np.max(mel_spec)
        if mel_max > 0:
            mel_spec = mel_spec / mel_max

        # 3通道
        mel_spec = np.stack([mel_spec] * 3, axis=-1)
        mel_spec = np.expand_dims(mel_spec, axis=0)

        return mel_spec, audio

    except Exception as e:
        raise ValueError(f"音频加载失败: {e}")


# ==================== 可视化函数 ====================
def create_mel_spectrogram_plot(mel_spec):
    """生成 Mel 频谱图"""
    fig, ax = plt.subplots(figsize=(8, 4))
    mel_2d = mel_spec[0, :, :, 0]  # 取第一个通道
    im = ax.imshow(mel_2d, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel('时间帧')
    ax.set_ylabel('Mel 频带')
    ax.set_title('Mel 频谱图')
    plt.colorbar(im, ax=ax, label='归一化幅度')
    plt.tight_layout()

    # 转换为图片
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def create_waveform_plot(audio, sr=SAMPLE_RATE):
    """生成音频波形图"""
    fig, ax = plt.subplots(figsize=(10, 3))
    time = np.arange(len(audio)) / sr
    ax.plot(time, audio, linewidth=0.5, alpha=0.8)
    ax.set_xlabel('时间 (秒)')
    ax.set_ylabel('振幅')
    ax.set_title('音频波形')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


# ==================== ONNX 预测器 ====================
class WatermelonPredictor:
    """西瓜糖度预测器"""

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        print(f"✓ 模型加载成功")
        print(f"  推理设备: {self.session.get_providers()[0]}")

    def predict(self, img, mel):
        """执行预测"""
        ort_inputs = {
            self.input_names[0]: img,
            self.input_names[1]: mel
        }
        ort_outputs = self.session.run(self.output_names, ort_inputs)
        return ort_outputs[0][0, 0]


# ==================== 全局预测器 ====================
global_predictor = None


def initialize_model(model_path=DEFAULT_MODEL_PATH):
    """初始化模型"""
    global global_predictor
    try:
        global_predictor = WatermelonPredictor(model_path)
        return "✅ 模型加载成功！"
    except Exception as e:
        return f"❌ 模型加载失败: {str(e)}"


# ==================== Gradio 预测函数 ====================
def predict_watermelon(image, audio, true_brix=None):
    """Gradio 预测函数"""
    # 检查输入
    if global_predictor is None:
        raise gr.Error("模型尚未加载，请检查后台日志。")
    if image is None:
        raise gr.Error("请上传西瓜图片！")
    if audio is None:
        raise gr.Error("请上传敲击音频！")

    try:
        # 预处理
        img_array = load_and_preprocess_image(image)
        mel_array, audio_waveform = load_and_preprocess_audio(audio)

        # 预测
        prediction = global_predictor.predict(img_array, mel_array)
        prediction_float = float(prediction)

        # 生成可视化
        mel_plot = create_mel_spectrogram_plot(mel_array)
        waveform_plot = create_waveform_plot(audio_waveform)

        # 糖度等级判断
        if prediction_float < 10:
            sweetness = "🔵 **偏低** - 可能还需要继续生长"
        elif 10 <= prediction_float < 11:
            sweetness = "🟡 **一般** - 刚刚达到成熟标准"
        elif 11 <= prediction_float < 12:
            sweetness = "🟢 **良好** - 口感适中，适合食用"
        elif 12 <= prediction_float < 13:
            sweetness = "🟢 **优秀** - 甜度较高，品质很好"
        else:
            sweetness = "🟣 **特优** - 非常甜，品质极佳"

        sweetness_md = f"### 🏆 甜度评级\n{sweetness}"

        # 误差分析
        error_md = ""
        error_visibility = False
        if true_brix is not None and true_brix > 0:
            error = abs(prediction_float - true_brix)
            error_percent = (error / true_brix) * 100
            error_md += f"""
### 📈 误差分析
- **真实糖度**: **{true_brix:.2f}° Brix**
- **绝对误差**: {error:.2f}° Brix
- **相对误差**: {error_percent:.2f}%
"""
            if error < 0.5:
                error_md += "\n✅ **预测非常准确！**"
            elif error < 1.0:
                error_md += "\n✔️ **预测较为准确**"
            else:
                error_md += "\n⚠️ **预测存在一定偏差**"
            error_visibility = True

        return (
            {f"{prediction_float:.2f}° Brix": 1.0},  # 主结果，用于gr.Label
            sweetness_md,
            gr.update(value=error_md, visible=error_visibility),  # 误差分析
            mel_plot,
            waveform_plot
        )

    except Exception as e:
        raise gr.Error(f"预测失败: {str(e)}")


# ==================== Gradio 界面 ====================
def create_interface():
    """创建 Gradio 界面"""

    # 自定义 CSS
    custom_css = """
    #result-label .label-name { font-size: 20px !important; font-weight: bold; }
    #result-label .label-value { font-size: 28px !important; }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="西瓜糖度AI预测系统") as demo:
        gr.Markdown("""
        # 🍉 西瓜糖度 AI 预测系统
        融合**图像**与**音频**的多模态AI，精准预测西瓜的含糖量。
        """)

        with gr.Tabs():
            with gr.TabItem("🚀 预测界面"):
                with gr.Row(variant="panel"):
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Step 1: 上传数据")
                        image_input = gr.Image(
                            type="filepath",
                            label="西瓜外观图片",
                            height=300
                        )
                        audio_input = gr.Audio(
                            type="filepath",
                            label="敲击音频"
                        )
                        true_brix_input = gr.Number(
                            label="真实糖度值 (可选，用于对比)",
                            value=None, minimum=0, maximum=20, step=0.1
                        )
                        with gr.Row():
                            clear_btn = gr.Button("🔄 清空", variant="secondary", scale=1)
                            predict_btn = gr.Button("✨ 开始预测", variant="primary", scale=2)

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Step 2: 查看结果")
                        result_label = gr.Label(label="预测糖度 (°Brix)", elem_id="result-label")
                        sweetness_output = gr.Markdown()
                        error_output = gr.Markdown(visible=False)

                        with gr.Accordion("📈 音频分析可视化", open=True):
                            mel_output = gr.Image(label="Mel 频谱图")
                            waveform_output = gr.Image(label="音频波形")

            with gr.TabItem("📚 模型与技术说明"):
                gr.Markdown("""
                ### 🎯 模型信息
                - 基于 **SlowFast 双流网络** + **跨模态注意力机制**
                - 融合图像特征和音频特征进行糖度预测
                - 预测精度：MAE < 0.5° Brix

                ---
                ### 📚 技术说明
                - **图像分支**: 提取西瓜外观特征（成熟度、纹理等）
                - **音频分支**: 分析敲击声特征（空心度、密实度等）
                - **融合预测**: 通过跨模态注意力机制综合判断糖度

                ---
                ### ⚡ 性能指标
                - **推理速度**: < 100ms/样本 (NVIDIA GPU)
                - **预测精度**: MAE ≈ 0.3-0.5° Brix
                - **适用范围**: 8-15° Brix

                ---
                💻 Powered by **ONNX Runtime** | 🔬 Based on **SlowFast Architecture**
                """)

        # 绑定事件
        predict_btn.click(
            fn=predict_watermelon,
            inputs=[image_input, audio_input, true_brix_input],
            outputs=[result_label, sweetness_output, error_output, mel_output, waveform_output]
        )

        clear_btn.click(
            fn=lambda: (None, None, None, None, None, gr.update(value="", visible=False), None, None),
            outputs=[image_input, audio_input, true_brix_input,
                     result_label, sweetness_output, error_output,
                     mel_output, waveform_output]
        )

    return demo


# ==================== 主函数 ====================
def main():
    """启动应用"""
    import argparse

    parser = argparse.ArgumentParser(description='西瓜糖度预测 Gradio 应用')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='ONNX模型路径')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='服务器地址')
    parser.add_argument('--port', type=int, default=7860,
                        help='端口号')
    parser.add_argument('--share', action='store_true',
                        help='创建公网分享链接')

    args = parser.parse_args()

    # 初始化模型
    print("正在加载模型...")
    result = initialize_model(args.model)
    print(result)

    if "失败" in result:
        print("\n❌ 请确保模型文件存在且路径正确")
        return

    # 创建并启动界面
    demo = create_interface()

    print(f"\n🚀 正在启动 Web 服务...")
    print(f"📍 本地访问: http://{args.host}:{args.port}")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()