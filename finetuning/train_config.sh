# ============================================================================
# TRAIN CONFIG - Configurazione Training Personalizzata
# ============================================================================
# 
# Questo file permette di personalizzare i parametri di training senza
# dover modificare lo script principale.
#
# Uso: Copia questo file come train_config.sh e modifica i valori
#      cp train_config.example.sh train_config.sh
#
# ============================================================================

# ----------------------------------------------------------------------------
# PERCORSI E NOMI FILE
# ----------------------------------------------------------------------------

# Dataset input (JSONL con audio, text, ref_audio)
RAW_JSONL="train_raw.jsonl"

# Dataset processato (generato automaticamente)
TRAIN_JSONL="train_with_codes.jsonl"

# Directory output per checkpoints
OUTPUT_DIR="output_italian_tts"

# Nome dello speaker nel modello fine-tuned
SPEAKER_NAME="italian_multi"

# Modelli HuggingFace
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# ----------------------------------------------------------------------------
# HYPERPARAMETERS - Training
# ----------------------------------------------------------------------------

# Learning rate (default: 5e-6)
# • 5e-6 = conservativo, stabile
# • 1e-5 = più veloce ma potenzialmente instabile
# • 2e-6 = molto conservativo per fine-tuning delicato
LR=5e-6

# Warmup steps - passi iniziali con LR crescente
# Aiuta la stabilità all'inizio del training
WARMUP=100

# Validazione split - percentuale dataset per validation
# 0.05 = 5% validation, 95% training
VAL_SPLIT=0.05

# Max gradient norm - clipping dei gradienti
# Previene gradient explosion
MAX_GRAD_NORM=1.0

# Weight decay - regolarizzazione L2
# Previene overfitting
WEIGHT_DECAY=0.01

# ----------------------------------------------------------------------------
# HYPERPARAMETERS - GPU & Batch
# ----------------------------------------------------------------------------

# Batch size per GPU (auto-determinato se non specificato)
# RTX 3090 24GB: può gestire 8-12
# Se OOM → riduci a 6-8
# Se memoria al 60% → aumenta a 12
# BATCH_SIZE_PER_GPU=10

# Gradient accumulation steps (auto-determinato se non specificato)
# Effective batch = NUM_GPUS × BATCH_SIZE × GRAD_ACCUM
# Target effective batch: 64-128 per dataset 10-20k
# GRAD_ACCUM=2

# Numero di epochs (auto-determinato se non specificato)
# • Dataset piccolo (<5k): 10-15 epochs
# • Dataset medio (5-15k): 5-8 epochs
# • Dataset grande (>15k): 3-5 epochs
# EPOCHS=5

# ----------------------------------------------------------------------------
# CHECKPOINTING & LOGGING
# ----------------------------------------------------------------------------

# Salva checkpoint ogni N steps
# Trade-off: più frequente = più sicuro ma più spazio disco
# • 500 = default, buon compromesso
# • 1000 = risparmia spazio
# • 250 = più frequente, più sicuro
SAVE_STEPS=500

# Validazione ogni N steps
# Permette di monitorare la loss durante il training
EVAL_STEPS=250

# Numero di checkpoint step-based da mantenere
# Checkpoints vecchi vengono automaticamente eliminati
# • 3 = risparmia spazio (default)
# • 5 = tiene più storia
# • 1 = minimo spazio (solo ultimo)
KEEP_CHECKPOINTS=3

# ----------------------------------------------------------------------------
# ESEMPI DI CONFIGURAZIONI PREIMPOSTATE
# ----------------------------------------------------------------------------

# === CONFIGURAZIONE 1: Training Veloce (Quick Test) ===
# Uncomment per test rapido del training
#
# LR=1e-5              # LR più alto per convergenza veloce
# EPOCHS=2             # Solo 2 epochs
# SAVE_STEPS=1000      # Salva meno spesso
# EVAL_STEPS=500       # Valida meno spesso
# KEEP_CHECKPOINTS=1   # Tiene solo ultimo checkpoint

# === CONFIGURAZIONE 2: Training Conservativo (Safe) ===
# Uncomment per training molto stabile ma lento
#
# LR=2e-6              # LR molto basso
# EPOCHS=10            # Più epochs
# SAVE_STEPS=250       # Salva più spesso
# EVAL_STEPS=125       # Valida più spesso
# WARMUP=200           # Warmup più lungo

# === CONFIGURAZIONE 3: Dataset Grande (>20k samples) ===
# Uncomment per dataset molto grandi
#
# LR=1e-5              # LR più alto (più samples)
# EPOCHS=3             # Meno epochs necessari
# BATCH_SIZE_PER_GPU=12  # Batch più grande se GPU lo supporta
# SAVE_STEPS=1000      # Salva meno spesso (molti steps)

# === CONFIGURAZIONE 4: Dataset Piccolo (<5k samples) ===
# Uncomment per dataset piccoli
#
# LR=3e-6              # LR più basso (rischio overfitting)
# EPOCHS=12            # Più epochs
# WEIGHT_DECAY=0.02    # Weight decay più alto
# VAL_SPLIT=0.1        # 10% validation (più importante)

# ----------------------------------------------------------------------------
# OPZIONI AVANZATE
# ----------------------------------------------------------------------------

# Device CUDA (di solito non serve modificare)
# DEVICE="cuda"

# ============================================================================
# NOTE:
# ============================================================================
#
# 1. Variabili commentate (#) usano i valori default dello script
# 2. Per attivare una configurazione, rimuovi il # dalla riga
# 3. Puoi mixare configurazioni diverse
# 4. I valori qui sovrascrivono quelli dello script principale
#
# ============================================================================
