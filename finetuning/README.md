# 🇮🇹 Qwen3-TTS Italian Fine-Tuning

Fine-tune [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for Italian voice on Vast.ai GPU cloud.

**15,000 Italian audio samples** | **~$1.20 total cost** | **2.5 hours training time**

## ⚡ One-Command Setup

```bash
curl -sSL https://raw.githubusercontent.com/OhMySimo/Qwen3-TTS-finetuning/main/setup_vast.sh | bash
```

Then start training:

```bash
cd /workspace/Qwen3-TTS-finetuning/finetuning
./train_vast.sh
```

## 📊 What You Get

- ✅ Professional Italian TTS model
- ✅ Auto-configured for 4x or 8x RTX 3090
- ✅ Pre-processed dataset included
- ✅ Optimized training scripts
- ✅ Monitoring & validation tools

## 🚀 Quick Start

### 1. Rent GPUs on Vast.ai

- Search: `4x RTX 3090` (recommended)
- Template: NVIDIA CUDA Development Environment
- Cost: ~$0.48/hour

### 2. SSH into instance and run setup

```bash
curl -sSL https://raw.githubusercontent.com/OhMySimo/Qwen3-TTS-finetuning/main/setup_vast.sh | bash
```

### 3. Start training

```bash
cd /workspace/Qwen3-TTS-finetuning/finetuning
./train_vast.sh
```

### 4. Download your model

```bash
tar -czf checkpoint.tar.gz output_italian_tts/checkpoint-best
scp -P <PORT> root@<IP>:/workspace/Qwen3-TTS-finetuning/finetuning/checkpoint.tar.gz .
```

## 📁 Repository Structure

```
finetuning/
├── train_vast.sh          # Main training script (auto-detects GPU count)
├── sft_12hz.py           # Multi-GPU training with Accelerate
├── dataset.py            # Dataset loader
├── prepare_data.py       # Audio tokenization
├── validate_dataset.py   # Dataset validation
└── monitor_training.sh   # Live GPU/training monitor
```

## 💾 Dataset

Italian dataset v2: [Download](https://github.com/OhMySimo/Qwen3-TTS-finetuning/releases/tag/it)

- 15,000 samples
- 24kHz mono audio
- Natural Italian speech
- Ready to use

## ⚙️ Training Config

### 4x RTX 3090 (Recommended)
- Batch size: 10 per GPU
- Effective batch: 80
- Time: ~2.5 hours
- Cost: ~$1.20

### 8x RTX 3090 (Faster)
- Batch size: 10 per GPU
- Effective batch: 80
- Time: ~1.25 hours
- Cost: ~$1.20

## 🧪 Use Your Model

```python
import torch
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "output_italian_tts/checkpoint-best",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

wavs, sr = tts.generate_custom_voice(
    text="Ciao! Sono il modello italiano.",
    speaker="italian_multi",
)
```

## 📚 Full Documentation

- [Complete Setup Guide](GUIDE.md)
- [Qwen3-TTS Official](https://github.com/QwenLM/Qwen3-TTS)
- [Vast.ai](https://vast.ai/)

## 🔧 Features

- Multi-GPU training with Accelerate
- Flash Attention 2 support
- 8-bit Adam optimizer (optional)
- Automatic validation & checkpointing
- Tensorboard monitoring
- Early stopping

## 🛠️ Troubleshooting

**Out of Memory?**
```bash
# Edit train_vast.sh
BATCH_SIZE_PER_GPU=6
GRAD_ACCUM=4
```

**Training disconnected?**
```bash
# Use tmux
tmux new -s training
./train_vast.sh
# Detach: Ctrl+B, D
# Reattach: tmux attach -t training
```

## 📄 License

Apache 2.0 (same as Qwen3-TTS)

## 🙏 Credits

- [Qwen Team](https://github.com/QwenLM/Qwen3-TTS) for the base model
- Italian dataset created for this project

---

**Questions?** Open an [Issue](https://github.com/OhMySimo/Qwen3-TTS-finetuning/issues)

**Total time from zero to trained model: ~3 hours** ⏱️
