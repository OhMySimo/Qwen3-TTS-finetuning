#!/usr/bin/env bash
# Ottimizzato per Vast.ai con 4x o 8x RTX 3090
# Dataset: 15k samples
set -e

echo "=========================================="
echo "Qwen3-TTS Multi-GPU Training Setup"
echo "=========================================="

# Configurazione automatica
DEVICE="cuda"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="output_italian_tts"
SPEAKER_NAME="italian_multi"

# Auto-detect GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "🎯 Detected $NUM_GPUS GPUs"

# Verifica dataset
if [ ! -f "$RAW_JSONL" ]; then
    echo "❌ Error: $RAW_JSONL not found!"
    echo "Please upload your training data to this directory."
    echo ""
    echo "You can use one of these methods:"
    echo "1. SCP: scp -P <PORT> train_raw.jsonl root@<IP>:/workspace/Qwen3-TTS/finetuning/"
    echo "2. Jupyter: Upload via Jupyter interface"
    echo "3. wget: wget https://your-url.com/train_raw.jsonl"
    exit 1
fi

# Hyperparameters ottimizzati per 15k samples
if [ "$NUM_GPUS" -ge 8 ]; then
    # 8x RTX 3090 configuration
    BATCH_SIZE_PER_GPU=10
    GRAD_ACCUM=1
    EPOCHS=3
    echo "📊 Configuration: 8-GPU setup"
elif [ "$NUM_GPUS" -ge 4 ]; then
    # 4x RTX 3090 configuration
    BATCH_SIZE_PER_GPU=10
    GRAD_ACCUM=2
    EPOCHS=5
    echo "📊 Configuration: 4-GPU setup"
else
    # Fallback for fewer GPUs
    BATCH_SIZE_PER_GPU=8
    GRAD_ACCUM=4
    EPOCHS=8
    echo "⚠️  Warning: Less than 4 GPUs detected. Training will be slower."
fi

# Learning rate e altri parametri
LR=5e-6
WARMUP=100
SAVE_STEPS=500
EVAL_STEPS=250
VAL_SPLIT=0.05

# Calcola metriche
EFFECTIVE_BATCH=$((NUM_GPUS * BATCH_SIZE_PER_GPU * GRAD_ACCUM))
STEPS_PER_EPOCH=$((15000 / EFFECTIVE_BATCH))
TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))

# Stima tempo
if [ "$NUM_GPUS" -ge 8 ]; then
    EST_TIME_PER_EPOCH="12-15 min"
    EST_TOTAL_TIME="~1.25 hours"
    EST_COST="~\$1.20"
elif [ "$NUM_GPUS" -ge 4 ]; then
    EST_TIME_PER_EPOCH="25-30 min"
    EST_TOTAL_TIME="~2.5 hours"
    EST_COST="~\$1.20"
else
    EST_TIME_PER_EPOCH="50-60 min"
    EST_TOTAL_TIME="~8 hours"
    EST_COST="~\$2.40"
fi

echo ""
echo "📋 Training Configuration:"
echo "  ├─ GPUs: $NUM_GPUS"
echo "  ├─ Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "  ├─ Gradient accumulation: $GRAD_ACCUM"
echo "  ├─ Effective batch size: $EFFECTIVE_BATCH"
echo "  ├─ Learning rate: $LR"
echo "  ├─ Epochs: $EPOCHS"
echo "  ├─ Validation split: $VAL_SPLIT"
echo "  ├─ Steps per epoch: ~$STEPS_PER_EPOCH"
echo "  └─ Total training steps: ~$TOTAL_STEPS"
echo ""
echo "⏱️  Estimated Time:"
echo "  ├─ Per epoch: $EST_TIME_PER_EPOCH"
echo "  ├─ Total: $EST_TOTAL_TIME"
echo "  └─ Cost: $EST_COST"
echo "=========================================="

# Controlla dipendenze
echo ""
echo "🔍 Checking dependencies..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || { echo "❌ PyTorch not found"; exit 1; }
python -c "import qwen_tts; print('✓ qwen-tts installed')" || { echo "❌ qwen-tts not installed. Run: pip install qwen-tts"; exit 1; }
python -c "import accelerate; print('✓ accelerate installed')" || { echo "❌ accelerate not installed. Run: pip install accelerate"; exit 1; }

# Check flash attention
python -c "import flash_attn; print('✓ flash-attention available')" 2>/dev/null || echo "⚠️  flash-attention not found (optional, but recommended for speed)"

# Check optional 8-bit Adam
USE_8BIT_ADAM=""
python -c "import bitsandbytes; print('✓ bitsandbytes available (8-bit Adam enabled)')" 2>/dev/null && USE_8BIT_ADAM="--use_8bit_adam"

echo ""
echo "=========================================="
echo "📦 Step 1/3: Preparing training data..."
echo "=========================================="

if [ ! -f "$TRAIN_JSONL" ]; then
    echo "Extracting audio codes (this may take 10-20 minutes for 15k samples)..."
    
    python prepare_data.py \
        --device ${DEVICE}:0 \
        --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
        --input_jsonl ${RAW_JSONL} \
        --output_jsonl ${TRAIN_JSONL}
    
    echo "✓ Data preparation completed!"
    
    # Verifica output
    NUM_LINES=$(wc -l < ${TRAIN_JSONL})
    echo "  Generated $NUM_LINES training samples"
else
    echo "✓ Training data already prepared (${TRAIN_JSONL})"
    NUM_LINES=$(wc -l < ${TRAIN_JSONL})
    echo "  Found $NUM_LINES training samples"
fi

echo ""
echo "=========================================="
echo "🚀 Step 2/3: Starting multi-GPU training..."
echo "=========================================="
echo ""
echo "💡 Tips:"
echo "  • Press Ctrl+C to stop training"
echo "  • Checkpoints saved every $SAVE_STEPS steps"
echo "  • Validation every $EVAL_STEPS steps"
echo "  • Monitor with: tensorboard --logdir $OUTPUT_DIR"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Launch training con timestamp logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/training_${TIMESTAMP}.log"

accelerate launch \
    --mixed_precision bf16 \
    --num_processes ${NUM_GPUS} \
    --multi_gpu \
    sft_12hz.py \
    --init_model_path ${INIT_MODEL_PATH} \
    --output_model_path ${OUTPUT_DIR} \
    --train_jsonl ${TRAIN_JSONL} \
    --batch_size ${BATCH_SIZE_PER_GPU} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --lr ${LR} \
    --num_epochs ${EPOCHS} \
    --speaker_name ${SPEAKER_NAME} \
    --warmup_steps ${WARMUP} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --val_split ${VAL_SPLIT} \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    ${USE_8BIT_ADAM} \
    2>&1 | tee ${LOG_FILE}

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code $TRAIN_EXIT_CODE"
    echo "Check logs: $LOG_FILE"
    exit $TRAIN_EXIT_CODE
fi
echo "=========================================="

# Lista checkpoints
echo ""
echo "📁 Saved checkpoints:"
ls -lh ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | awk '{print "  ", $9, "(" $5 ")"}'

echo ""
echo "=========================================="
echo "🧪 Step 3/3: Testing best checkpoint..."
echo "=========================================="

# Trova best checkpoint
BEST_CHECKPOINT="${OUTPUT_DIR}/checkpoint-best"
if [ ! -d "$BEST_CHECKPOINT" ]; then
    # Fallback all'ultimo checkpoint
    LAST_CHECKPOINT=$(ls -td ${OUTPUT_DIR}/checkpoint-epoch-* 2>/dev/null | head -1)
    if [ -n "$LAST_CHECKPOINT" ]; then
        BEST_CHECKPOINT="$LAST_CHECKPOINT"
        echo "⚠️  checkpoint-best not found, using: $(basename $BEST_CHECKPOINT)"
    else
        echo "❌ No checkpoints found!"
        exit 1
    fi
fi

echo "Testing checkpoint: $BEST_CHECKPOINT"
echo ""

# Test script
python - <<EOF
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
checkpoint_path = "${BEST_CHECKPOINT}"

print("Loading model...")
try:
    tts = Qwen3TTSModel.from_pretrained(
        checkpoint_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print("✓ Model loaded successfully!")
    
    test_texts = [
        "Buongiorno, questo è un test del modello italiano fine-tuned.",
        "La qualità della voce sintetizzata dovrebbe essere eccellente.",
        "Oggi è il sei febbraio duemila ventisei.",
    ]
    
    print("\nGenerating test audio samples...")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text[:50]}...")
        wavs, sr = tts.generate_custom_voice(
            text=text,
            speaker="${SPEAKER_NAME}",
        )
        output_file = f"${OUTPUT_DIR}/test_output_{i}.wav"
        sf.write(output_file, wavs[0], sr)
        print(f"     ✓ Saved: {output_file}")
    
    print("\n✅ Test completed successfully!")
    print(f"\nTest audio files saved in: ${OUTPUT_DIR}/")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    print("\nYou can test manually later with:")
    print(f"  python test_checkpoint.py --checkpoint {checkpoint_path}")
    exit(1)
EOF

TEST_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "🎉 TRAINING PIPELINE COMPLETED!"
echo "=========================================="
echo ""
echo "📊 Summary:"
echo "  ├─ Training log: $LOG_FILE"
echo "  ├─ Checkpoints: ${OUTPUT_DIR}/"
echo "  ├─ Best model: $BEST_CHECKPOINT"
echo "  └─ Test audio: ${OUTPUT_DIR}/test_output_*.wav"
echo ""
echo "📈 Next steps:"
echo "  1. Listen to test_output_*.wav to verify quality"
echo "  2. View training metrics:"
echo "     tensorboard --logdir ${OUTPUT_DIR} --host 0.0.0.0"
echo "  3. Download checkpoint:"
echo "     tar -czf checkpoint.tar.gz ${BEST_CHECKPOINT}"
echo "     scp -P <PORT> root@<IP>:$(pwd)/checkpoint.tar.gz ."
echo ""
echo "💾 Checkpoint size:"
du -sh ${BEST_CHECKPOINT} 2>/dev/null || echo "  (check manually)"
echo ""
echo "=========================================="

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ All done! Your model is ready to use."
else
    echo "⚠️  Training succeeded but test failed. Check the checkpoint manually."
fi

echo "=========================================="
