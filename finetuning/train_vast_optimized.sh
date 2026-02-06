#!/usr/bin/env bash
# Versione migliorata per Vast.ai con 4x o 8x RTX 3090
# Miglioramenti: validazione dataset, dimensione dinamica, resume support
set -e

echo "=========================================="
echo "Qwen3-TTS Multi-GPU Training Setup v2"
echo "=========================================="

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# File di configurazione esterno (opzionale)
CONFIG_FILE="train_config.sh"
if [ -f "$CONFIG_FILE" ]; then
    echo "📋 Loading custom config from $CONFIG_FILE"
    source "$CONFIG_FILE"
fi

# Configurazione base
DEVICE="${DEVICE:-cuda}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-Qwen/Qwen3-TTS-Tokenizer-12Hz}"
INIT_MODEL_PATH="${INIT_MODEL_PATH:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
RAW_JSONL="${RAW_JSONL:-train_raw.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-train_with_codes.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output_italian_tts}"
SPEAKER_NAME="${SPEAKER_NAME:-italian_multi}"

# Hyperparameters (possono essere sovrascritti da train_config.sh)
LR="${LR:-5e-6}"
WARMUP="${WARMUP:-100}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-250}"
VAL_SPLIT="${VAL_SPLIT:-0.05}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-3}"  # Numero di checkpoint da mantenere

# ============================================================================
# RILEVAZIONE GPU
# ============================================================================

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "🎯 Detected $NUM_GPUS GPUs"

# Controllo memoria GPU disponibile
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_MEM_GB=$((GPU_MEM / 1024))
echo "💾 GPU Memory: ${GPU_MEM_GB}GB per GPU"

if [ "$GPU_MEM" -lt 20000 ]; then
    echo "⚠️  Warning: GPU has only ${GPU_MEM_GB}GB. Recommended: 24GB+"
    echo "   Consider reducing BATCH_SIZE_PER_GPU if you encounter OOM errors"
fi

# ============================================================================
# VALIDAZIONE DATASET
# ============================================================================

echo ""
echo "🔍 Validating dataset..."

if [ ! -f "$RAW_JSONL" ]; then
    echo "❌ Error: $RAW_JSONL not found!"
    echo "Please upload your training data to this directory."
    echo ""
    echo "You can use one of these methods:"
    echo "1. SCP: scp -P <PORT> train_raw.jsonl root@<IP>:$(pwd)/"
    echo "2. Jupyter: Upload via Jupyter interface"
    echo "3. wget: wget https://your-url.com/train_raw.jsonl"
    exit 1
fi

# Validazione formato dataset
echo "Checking dataset format..."
python3 << 'VALIDATE_DATASET'
import json
import sys
import os

try:
    jsonl_file = os.environ.get('RAW_JSONL', 'train_raw.jsonl')
    num_samples = 0
    errors = []
    
    with open(jsonl_file) as f:
        for i, line in enumerate(f, 1):
            num_samples = i
            try:
                data = json.loads(line)
                required = ['audio', 'text', 'ref_audio']
                missing = [k for k in required if k not in data]
                
                if missing:
                    errors.append(f"Line {i}: missing fields {missing}")
                    if len(errors) >= 5:  # Mostra solo primi 5 errori
                        break
                        
                # Controllo base su tipi
                if not isinstance(data.get('text'), str):
                    errors.append(f"Line {i}: 'text' must be string")
                if not isinstance(data.get('audio'), str):
                    errors.append(f"Line {i}: 'audio' must be string (path)")
                    
            except json.JSONDecodeError:
                errors.append(f"Line {i}: invalid JSON")
                if len(errors) >= 5:
                    break
    
    if errors:
        print("❌ Dataset validation failed:")
        for err in errors:
            print(f"  • {err}")
        sys.exit(1)
    
    print(f"✓ Dataset format valid ({num_samples} samples)")
    # Salva numero samples per uso successivo
    with open('/tmp/dataset_size.txt', 'w') as f:
        f.write(str(num_samples))
        
except FileNotFoundError:
    print(f"❌ File not found: {jsonl_file}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Validation error: {e}")
    sys.exit(1)
VALIDATE_DATASET

if [ $? -ne 0 ]; then
    echo ""
    echo "Please fix dataset errors before continuing."
    exit 1
fi

# Leggi numero samples dal file temporaneo
NUM_SAMPLES=$(cat /tmp/dataset_size.txt)
echo "✓ Found $NUM_SAMPLES training samples"

# ============================================================================
# CONFIGURAZIONE AUTOMATICA PER GPU
# ============================================================================

echo ""
if [ "$NUM_GPUS" -ge 8 ]; then
    # 8x RTX 3090 configuration
    BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-10}
    GRAD_ACCUM=${GRAD_ACCUM:-1}
    EPOCHS=${EPOCHS:-3}
    echo "📊 Configuration: 8-GPU setup"
elif [ "$NUM_GPUS" -ge 4 ]; then
    # 4x RTX 3090 configuration
    BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-10}
    GRAD_ACCUM=${GRAD_ACCUM:-2}
    EPOCHS=${EPOCHS:-5}
    echo "📊 Configuration: 4-GPU setup"
else
    # Fallback for fewer GPUs
    BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-8}
    GRAD_ACCUM=${GRAD_ACCUM:-4}
    EPOCHS=${EPOCHS:-8}
    echo "⚠️  Warning: Less than 4 GPUs detected. Training will be slower."
fi

# Calcola metriche dinamicamente basandosi sul dataset reale
EFFECTIVE_BATCH=$((NUM_GPUS * BATCH_SIZE_PER_GPU * GRAD_ACCUM))
STEPS_PER_EPOCH=$((NUM_SAMPLES / EFFECTIVE_BATCH))
TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))

# Stima tempo (approssimativa)
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
echo "  ├─ Dataset: $NUM_SAMPLES samples"
echo "  ├─ GPUs: $NUM_GPUS x ${GPU_MEM_GB}GB"
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

# ============================================================================
# CONTROLLO DIPENDENZE
# ============================================================================

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

# ============================================================================
# STEP 1: PREPARAZIONE DATI
# ============================================================================

echo ""
echo "=========================================="
echo "📦 Step 1/3: Preparing training data..."
echo "=========================================="

if [ ! -f "$TRAIN_JSONL" ]; then
    echo "Extracting audio codes (this may take 10-20 minutes for ${NUM_SAMPLES} samples)..."
    
    python prepare_data.py \
        --device ${DEVICE}:0 \
        --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
        --input_jsonl ${RAW_JSONL} \
        --output_jsonl ${TRAIN_JSONL}
    
    echo "✓ Data preparation completed!"
    
    # Verifica output
    NUM_LINES=$(wc -l < ${TRAIN_JSONL})
    echo "  Generated $NUM_LINES training samples"
    
    if [ "$NUM_LINES" -ne "$NUM_SAMPLES" ]; then
        echo "⚠️  Warning: Expected $NUM_SAMPLES but got $NUM_LINES samples"
        echo "   Some samples may have been skipped during processing"
    fi
else
    echo "✓ Training data already prepared (${TRAIN_JSONL})"
    NUM_LINES=$(wc -l < ${TRAIN_JSONL})
    echo "  Found $NUM_LINES training samples"
fi

# ============================================================================
# STEP 2: TRAINING
# ============================================================================

echo ""
echo "=========================================="
echo "🚀 Step 2/3: Starting multi-GPU training..."
echo "=========================================="

# Controlla se esistono checkpoint precedenti (resume support)
RESUME_CHECKPOINT=""
LATEST_CHECKPOINT=$(ls -td ${OUTPUT_DIR}/checkpoint-step-* 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo ""
    echo "⚠️  Found existing checkpoint: $(basename $LATEST_CHECKPOINT)"
    echo "Do you want to resume training from this checkpoint?"
    echo "(This will continue from where training stopped)"
    echo ""
    read -t 15 -p "Resume training? (y/N, auto-skip in 15s): " -n 1 -r REPLY || REPLY='n'
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        RESUME_CHECKPOINT="--resume_from_checkpoint ${LATEST_CHECKPOINT}"
        echo "✓ Will resume from: $(basename $LATEST_CHECKPOINT)"
    else
        echo "Starting fresh training (existing checkpoints will be preserved)"
    fi
fi

echo ""
echo "💡 Tips:"
echo "  • Press Ctrl+C to stop training"
echo "  • Checkpoints saved every $SAVE_STEPS steps"
echo "  • Validation every $EVAL_STEPS steps"
echo "  • Monitor with: tensorboard --logdir $OUTPUT_DIR"
echo "  • Only last $KEEP_CHECKPOINTS step-based checkpoints will be kept"
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
    --max_grad_norm ${MAX_GRAD_NORM} \
    --weight_decay ${WEIGHT_DECAY} \
    ${USE_8BIT_ADAM} \
    ${RESUME_CHECKPOINT} \
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

# ============================================================================
# CLEANUP VECCHI CHECKPOINT
# ============================================================================

echo ""
echo "🧹 Cleaning up old checkpoints (keeping last $KEEP_CHECKPOINTS)..."
STEP_CHECKPOINTS=$(ls -td ${OUTPUT_DIR}/checkpoint-step-* 2>/dev/null)
NUM_CHECKPOINTS=$(echo "$STEP_CHECKPOINTS" | wc -l)

if [ "$NUM_CHECKPOINTS" -gt "$KEEP_CHECKPOINTS" ]; then
    TO_DELETE=$((NUM_CHECKPOINTS - KEEP_CHECKPOINTS))
    echo "Found $NUM_CHECKPOINTS step checkpoints, removing $TO_DELETE old ones..."
    
    echo "$STEP_CHECKPOINTS" | tail -n $TO_DELETE | while read ckpt; do
        echo "  Removing: $(basename $ckpt)"
        rm -rf "$ckpt"
    done
    
    FREED_SPACE=$(du -sh ${OUTPUT_DIR} 2>/dev/null | cut -f1)
    echo "✓ Cleanup complete. Current size: $FREED_SPACE"
else
    echo "✓ Only $NUM_CHECKPOINTS checkpoints, no cleanup needed"
fi

# Lista checkpoints finali
echo ""
echo "📁 Saved checkpoints:"
ls -lh ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | awk '{print "  ", $9, "(" $5 ")"}'

# ============================================================================
# STEP 3: TEST DEL MODELLO
# ============================================================================

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

# Test con campioni dal validation set
python3 - <<EOF
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import json
import random
import os

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
    
    # Usa campioni dal validation set se possibile
    try:
        with open("${RAW_JSONL}") as f:
            all_data = [json.loads(line) for line in f]
        
        # Ultimi 5% sono validation set
        val_size = int(len(all_data) * 0.05)
        val_data = all_data[-val_size:]
        
        # Prendi 3 campioni random
        test_samples = random.sample(val_data, min(3, len(val_data)))
        test_texts = [s['text'] for s in test_samples]
        print(f"Testing with {len(test_texts)} samples from validation set")
    except:
        # Fallback a testi hardcoded
        test_texts = [
            "Buongiorno, questo è un test del modello italiano fine-tuned.",
            "La qualità della voce sintetizzata dovrebbe essere eccellente.",
            "Oggi è il sei febbraio duemila ventisei.",
        ]
        print("Testing with hardcoded samples")
    
    print("\nGenerating test audio samples...")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text[:60]}...")
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
    import traceback
    traceback.print_exc()
    print("\nYou can test manually later.")
    exit(1)
EOF

TEST_EXIT_CODE=$?

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================

echo ""
echo "=========================================="
echo "🎉 TRAINING PIPELINE COMPLETED!"
echo "=========================================="
echo ""
echo "📊 Summary:"
echo "  ├─ Dataset: $NUM_SAMPLES samples"
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
echo "🔧 Used configuration:"
echo "  Config file: ${CONFIG_FILE:-none (using defaults)}"
echo "  Learning rate: $LR"
echo "  Batch size: $BATCH_SIZE_PER_GPU x $NUM_GPUS x $GRAD_ACCUM = $EFFECTIVE_BATCH"
echo ""
echo "=========================================="

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ All done! Your model is ready to use."
else
    echo "⚠️  Training succeeded but test failed. Check the checkpoint manually."
fi

echo "=========================================="
