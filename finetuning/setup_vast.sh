#!/usr/bin/env bash
# One-click setup script per Vast.ai
# Esegui questo script SUBITO dopo aver effettuato SSH nell'istanza
set -e

echo "=========================================="
echo "🚀 Qwen3-TTS Vast.ai Quick Setup"
echo "=========================================="
echo ""

# Controlla se siamo in un ambiente Vast.ai
if [ ! -d "/workspace" ]; then
    echo "⚠️  Warning: /workspace directory not found."
    echo "This script is optimized for Vast.ai CUDA template."
    echo "Creating /workspace..."
    mkdir -p /workspace
fi

cd /workspace

# GPU check
echo "🔍 Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "✓ Found $NUM_GPUS GPU(s)"
echo ""

# Verifica Python
echo "🐍 Checking Python environment..."
python --version || { echo "❌ Python not found"; exit 1; }
pip --version || { echo "❌ pip not found"; exit 1; }
echo ""

# Step 1: Clone repository
echo "=========================================="
echo "📦 Step 1: Cloning Qwen3-TTS repository..."
echo "=========================================="

if [ -d "Qwen3-TTS" ]; then
    echo "✓ Repository already exists, pulling latest changes..."
    cd Qwen3-TTS
    git pull
    cd ..
else
    echo "Cloning repository..."
    git clone https://github.com/QwenLM/Qwen3-TTS.git
    echo "✓ Repository cloned"
fi

cd Qwen3-TTS/finetuning
echo "Working directory: $(pwd)"
echo ""

# Step 2: Install dependencies
echo "=========================================="
echo "📚 Step 2: Installing dependencies..."
echo "=========================================="

echo "Installing core packages..."
pip install --break-system-packages -q qwen-tts
pip install --break-system-packages -q accelerate
pip install --break-system-packages -q tensorboard
pip install --break-system-packages -q soundfile

echo "Installing optional packages for performance..."
pip install --break-system-packages -q flash-attn --no-build-isolation 2>/dev/null || echo "  (flash-attn skipped, will use default attention)"
pip install --break-system-packages -q bitsandbytes 2>/dev/null || echo "  (bitsandbytes skipped, will use standard Adam)"

echo ""
echo "✓ Dependencies installed"
echo ""

# Verifica installazione
echo "🔍 Verifying installation..."
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')"
python -c "import qwen_tts; print('  ✓ qwen-tts')"
python -c "import accelerate; print('  ✓ accelerate')"
python -c "import flash_attn; print('  ✓ flash-attention')" 2>/dev/null || echo "  ⚠ flash-attention not available"
python -c "import bitsandbytes; print('  ✓ bitsandbytes (8-bit optimizer)')" 2>/dev/null || echo "  ⚠ bitsandbytes not available"
echo ""

# Step 3: Download training scripts
echo "=========================================="
echo "📝 Step 3: Downloading optimized scripts..."
echo "=========================================="

# Backup originali se esistono
for file in sft_12hz.py dataset.py prepare_data.py train_vast.sh; do
    if [ -f "$file" ]; then
        mv "$file" "${file}.original"
        echo "  Backed up: $file → ${file}.original"
    fi
done

# I file aggiornati dovranno essere copiati manualmente o via wget
echo ""
echo "⚠️  IMPORTANT: Copy your optimized training files here:"
echo "     $(pwd)"
echo ""
echo "Required files:"
echo "  • sft_12hz.py (optimized training script)"
echo "  • dataset.py (dataset loader)"
echo "  • prepare_data.py (audio tokenization)"
echo "  • train_vast.sh (training launcher)"
echo ""
echo "You can copy them using SCP:"
echo "  scp -P <PORT> *.py train_vast.sh root@<IP>:$(pwd)/"
echo ""

# Step 4: Setup tensorboard
echo "=========================================="
echo "📊 Step 4: Setting up monitoring..."
echo "=========================================="

# Create tensorboard service se supervisor è disponibile
if command -v supervisorctl &> /dev/null; then
    echo "Setting up Tensorboard as a background service..."
    
    cat > /etc/supervisor/conf.d/tensorboard.conf <<EOL
[program:tensorboard]
command=tensorboard --logdir=/workspace/Qwen3-TTS/finetuning/output_italian_tts --host=0.0.0.0 --port=6006
directory=/workspace/Qwen3-TTS/finetuning
autostart=false
autorestart=true
stderr_logfile=/var/log/tensorboard.err.log
stdout_logfile=/var/log/tensorboard.out.log
EOL
    
    supervisorctl reread
    supervisorctl update
    echo "✓ Tensorboard configured (start with: supervisorctl start tensorboard)"
else
    echo "Supervisor not available. Start tensorboard manually when needed:"
    echo "  tensorboard --logdir output_italian_tts --host 0.0.0.0 --port 6006"
fi
echo ""

# Step 5: Create helper scripts
echo "=========================================="
echo "🛠️  Step 5: Creating helper scripts..."
echo "=========================================="

# Script per scaricare i modelli
cat > download_models.sh <<'EOF'
#!/usr/bin/env bash
echo "Pre-downloading models to avoid delays during training..."
python -c "
from transformers import AutoConfig, AutoTokenizer
from qwen_tts import Qwen3TTSTokenizer

print('Downloading Qwen3-TTS-12Hz-1.7B-Base...')
AutoConfig.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base')

print('Downloading Qwen3-TTS-Tokenizer-12Hz...')
Qwen3TTSTokenizer.from_pretrained('Qwen/Qwen3-TTS-Tokenizer-12Hz', device_map='cpu')

print('✓ All models downloaded and cached')
"
EOF
chmod +x download_models.sh

# Script per testing checkpoint
cat > test_checkpoint.py <<'EOF'
#!/usr/bin/env python
import argparse
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--speaker", type=str, default="italian_multi")
    parser.add_argument("--text", type=str, default="Questo è un test del modello.")
    parser.add_argument("--output", type=str, default="test_output.wav")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {args.checkpoint}...")
    tts = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    print(f"Generating audio for: {args.text}")
    wavs, sr = tts.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
    )
    
    sf.write(args.output, wavs[0], sr)
    print(f"✓ Audio saved to: {args.output}")

if __name__ == "__main__":
    main()
EOF
chmod +x test_checkpoint.py

# Script per dataset validation
cat > validate_dataset.py <<'EOF'
#!/usr/bin/env python
import json
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    args = parser.parse_args()
    
    print(f"Validating {args.jsonl}...")
    
    if not os.path.exists(args.jsonl):
        print(f"❌ File not found: {args.jsonl}")
        return
    
    with open(args.jsonl, 'r') as f:
        lines = f.readlines()
    
    total = len(lines)
    valid = 0
    errors = []
    
    for i, line in enumerate(lines, 1):
        try:
            data = json.loads(line.strip())
            
            # Check required fields
            if 'audio' not in data:
                errors.append(f"Line {i}: missing 'audio' field")
                continue
            if 'text' not in data:
                errors.append(f"Line {i}: missing 'text' field")
                continue
            if 'ref_audio' not in data:
                errors.append(f"Line {i}: missing 'ref_audio' field")
                continue
            
            # Check if files exist
            if not os.path.exists(data['audio']):
                errors.append(f"Line {i}: audio file not found: {data['audio']}")
                continue
            if not os.path.exists(data['ref_audio']):
                errors.append(f"Line {i}: ref_audio file not found: {data['ref_audio']}")
                continue
            
            valid += 1
            
        except json.JSONDecodeError:
            errors.append(f"Line {i}: invalid JSON")
    
    print(f"\n📊 Validation Results:")
    print(f"  Total lines: {total}")
    print(f"  Valid: {valid}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  • {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("\n✅ All samples are valid!")

if __name__ == "__main__":
    main()
EOF
chmod +x validate_dataset.py

# Monitor script
cat > monitor_training.sh <<'EOF'
#!/usr/bin/env bash
# Live monitoring del training
clear
echo "🔍 Training Monitor"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "📊 GPU Status"
    echo "=========================================="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | 
        awk -F, '{printf "GPU %s: %s\n  Usage: %s%% | Mem: %sMB/%sMB | Temp: %s°C\n", $1, $2, $3, $4, $5, $6}'
    
    echo ""
    echo "=========================================="
    echo "📈 Recent Training Logs"
    echo "=========================================="
    
    LOG_FILE=$(ls -t output_italian_tts/training_*.log 2>/dev/null | head -1)
    if [ -n "$LOG_FILE" ]; then
        tail -n 15 "$LOG_FILE" | grep -E "(loss|epoch|step)" || echo "No recent logs"
    else
        echo "No training logs found yet"
    fi
    
    echo ""
    echo "=========================================="
    echo "💾 Checkpoints"
    echo "=========================================="
    ls -lht output_italian_tts/checkpoint-* 2>/dev/null | head -5 | awk '{print $9, "(" $5 ")"}'
    
    sleep 5
done
EOF
chmod +x monitor_training.sh

echo "✓ Helper scripts created:"
echo "  • download_models.sh - Pre-download models"
echo "  • test_checkpoint.py - Test a specific checkpoint"
echo "  • validate_dataset.py - Validate training JSONL"
echo "  • monitor_training.sh - Live training monitor"
echo ""

# Step 6: Final instructions
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Upload your training data:"
echo "   scp -P <PORT> train_raw.jsonl root@<IP>:$(pwd)/"
echo ""
echo "2. (Optional) Pre-download models to avoid delays:"
echo "   ./download_models.sh"
echo ""
echo "3. Validate your dataset:"
echo "   ./validate_dataset.py --jsonl train_raw.jsonl"
echo ""
echo "4. Copy optimized training scripts (if not already done)"
echo ""
echo "5. Start training:"
echo "   chmod +x train_vast.sh"
echo "   ./train_vast.sh"
echo ""
echo "6. Monitor training (in another terminal):"
echo "   ./monitor_training.sh"
echo "   # or"
echo "   supervisorctl start tensorboard"
echo "   # Then access tensorboard at http://<INSTANCE_IP>:6006"
echo ""
echo "=========================================="
echo ""
echo "💡 Useful Commands:"
echo "  • Check GPU: nvidia-smi"
echo "  • View logs: tail -f output_italian_tts/training_*.log"
echo "  • Test checkpoint: ./test_checkpoint.py --checkpoint output_italian_tts/checkpoint-best"
echo "  • Compress checkpoint: tar -czf checkpoint.tar.gz output_italian_tts/checkpoint-best"
echo ""
echo "🆘 Need Help?"
echo "  • Check setup guide: cat ../VAST_AI_SETUP_GUIDE.md"
echo "  • Qwen3-TTS docs: https://github.com/QwenLM/Qwen3-TTS"
echo ""
echo "=========================================="
echo "Happy training! 🚀"
echo "=========================================="
