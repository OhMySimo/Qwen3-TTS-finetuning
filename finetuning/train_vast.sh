#!/usr/bin/env bash
# Script ottimizzato per vast.ai con 4x o 8x RTX 3090
set -e

echo "=========================================="
echo "Qwen3-TTS Multi-GPU Training Setup"
echo "=========================================="

# Configurazione
DEVICE="cuda"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="output_italian_tts"
SPEAKER_NAME="italian_multi"

# Hyperparameters ottimizzati per 15k samples
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Batch size per GPU (RTX 3090 ha 24GB, può gestire 8-12)
BATCH_SIZE_PER_GPU=10

# Gradient accumulation (effective batch = NUM_GPUS × BATCH_SIZE × GRAD_ACCUM)
GRAD_ACCUM=2

# Learning rate (scaled per batch size più grande)
LR=5e-6

# Epochs (ridotti per dataset grande)
EPOCHS=5

# Warmup steps
WARMUP=100

# Salvataggio ogni N steps
SAVE_STEPS=500
EVAL_STEPS=250

EFFECTIVE_BATCH=$((NUM_GPUS * BATCH_SIZE_PER_GPU * GRAD_ACCUM))
STEPS_PER_EPOCH=$((15000 / EFFECTIVE_BATCH))
TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))

echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Effective batch size: $EFFECTIVE_BATCH"
echo "  Learning rate: $LR"
echo "  Epochs: $EPOCHS"
echo "  Steps per epoch: ~$STEPS_PER_EPOCH"
echo "  Total training steps: ~$TOTAL_STEPS"
echo "=========================================="

# Step 1: Prepara i dati (se non già fatto)
if [ ! -f "$TRAIN_JSONL" ]; then
    echo "Step 1: Preparing training data..."
    python prepare_data.py \
        --device ${DEVICE}:0 \
        --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
        --input_jsonl ${RAW_JSONL} \
        --output_jsonl ${TRAIN_JSONL}
else
    echo "Training data already prepared, skipping..."
fi

# Step 2: Training con Accelerate
echo ""
echo "Step 2: Starting multi-GPU training..."
echo "Press Ctrl+C to stop"
echo ""

accelerate launch \
    --mixed_precision bf16 \
    --num_processes ${NUM_GPUS} \
    --multi_gpu \
    sft_12hz_optimized.py \
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
    --val_split 0.05 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved in: ${OUTPUT_DIR}"
echo "=========================================="

# Step 3: Test rapido del miglior checkpoint
echo ""
echo "Step 3: Testing best checkpoint..."
python - <<EOF
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
checkpoint_path = "${OUTPUT_DIR}/checkpoint-best"

try:
    tts = Qwen3TTSModel.from_pretrained(
        checkpoint_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    test_texts = [
        "Ciao, questo è un test del modello italiano.",
        "La qualità della voce sintetizzata dovrebbe essere eccellente.",
    ]
    
    for i, text in enumerate(test_texts):
        wavs, sr = tts.generate_custom_voice(
            text=text,
            speaker="${SPEAKER_NAME}",
        )
        output_file = f"test_output_{i+1}.wav"
        sf.write(output_file, wavs[0], sr)
        print(f"Generated: {output_file}")
    
    print("\nTest completed successfully!")
except Exception as e:
    print(f"Error during testing: {e}")
    print("You can test manually later.")
EOF

echo ""
echo "Done! Check tensorboard logs with:"
echo "  tensorboard --logdir ${OUTPUT_DIR}"
