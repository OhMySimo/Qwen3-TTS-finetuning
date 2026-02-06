# 🚀 Qwen3-TTS Training Package per Vast.ai

Tutto ciò di cui hai bisogno per fare fine-tuning di Qwen3-TTS su Vast.ai con 4x o 8x RTX 3090.

## 📦 Contenuto del Package

```
vast_ai_training_package/
├── README.md                      # Questo file
├── VAST_AI_SETUP_GUIDE.md         # Guida completa (leggere prima)
├── QUICK_REFERENCE.md             # Comandi rapidi
│
├── setup_vast.sh                  # Setup iniziale ambiente (eseguire per primo)
├── train_vast_optimized.sh        # Script di training ottimizzato
├── check_environment.py           # Validazione ambiente
│
├── sft_12hz.py                    # Training script (multi-GPU ottimizzato)
├── dataset.py                     # Dataset loader
├── prepare_data.py                # Preparazione dati
├── train_vast.sh                  # Training launcher (originale)
│
├── accelerate_config_4gpu.yaml    # Config per 4 GPUs
└── accelerate_config_8gpu.yaml    # Config per 8 GPUs
```

## ⚡ Quick Start (3 Steps)

### 1. Noleggia GPU su Vast.ai

- Cerca: `4x RTX 3090` o `8x RTX 3090`
- Template: **NVIDIA CUDA Development Environment**
- Disk: minimo 100GB
- Costo stimato: ~$0.48/h (4x) o ~$0.96/h (8x)

### 2. Connetti via SSH e Setup

```bash
# Connetti all'istanza Vast.ai (clicca "SSH" nel dashboard)
ssh -p <PORT> root@<IP>

# Scarica questo package
cd /workspace
# Opzione A: Se hai il file localmente
# scp -P <PORT> vast_ai_training_package.tar.gz root@<IP>:/workspace/

# Opzione B: Da un URL
wget https://your-url.com/vast_ai_training_package.tar.gz

# Estrai
tar -xzf vast_ai_training_package.tar.gz
cd vast_ai_training_package

# Esegui setup (installa dipendenze)
chmod +x setup_vast.sh
./setup_vast.sh
```

### 3. Prepara e Avvia Training

```bash
# Carica il tuo dataset (dal tuo PC)
# scp -P <PORT> train_raw.jsonl root@<IP>:/workspace/Qwen3-TTS/finetuning/

# Copia gli script nella directory di lavoro
cp *.py *.sh /workspace/Qwen3-TTS/finetuning/
cd /workspace/Qwen3-TTS/finetuning/

# Controlla che tutto sia ok
python check_environment.py

# Avvia training!
./train_vast_optimized.sh
```

## 📊 Cosa Aspettarsi

### Per 15,000 samples di training

**Con 4x RTX 3090:**
- Effective batch size: 80
- Tempo per epoch: ~25-30 minuti
- Totale (5 epochs): ~2.5 ore
- Costo: ~$1.20

**Con 8x RTX 3090:**
- Effective batch size: 80
- Tempo per epoch: ~12-15 minuti
- Totale (3 epochs): ~1.25 ore
- Costo: ~$1.20

### Output

```
output_italian_tts/
├── checkpoint-best/        # Miglior checkpoint (usa questo!)
├── checkpoint-epoch-0/
├── checkpoint-epoch-1/
├── ...
├── test_output_*.wav       # Audio di test
└── training_*.log          # Logs
```

## 🎯 Formato Dataset

Il tuo `train_raw.jsonl` deve essere così:

```jsonl
{"audio":"./audio/sample001.wav","text":"Testo italiano da sintetizzare","ref_audio":"./audio/reference.wav"}
{"audio":"./audio/sample002.wav","text":"Altro testo italiano","ref_audio":"./audio/reference.wav"}
```

**Importante:**
- ✅ Usa lo STESSO `ref_audio` per tutti i samples (5-10 secondi della voce target)
- ✅ Audio: 24kHz, mono, formato WAV
- ✅ 15,000 samples = ottimale per qualità

## 🔧 Troubleshooting Rapido

### "CUDA out of memory"
```bash
# Riduci batch size in train_vast_optimized.sh:
BATCH_SIZE_PER_GPU=6  # invece di 10
GRAD_ACCUM=4          # invece di 2
```

### "train_raw.jsonl not found"
```bash
# Assicurati di caricare il dataset:
scp -P <PORT> train_raw.jsonl root@<IP>:/workspace/Qwen3-TTS/finetuning/
```

### Training troppo lento
```bash
# Verifica GPU usage (dovrebbe essere >90%)
nvidia-smi -l 1

# Controlla che flash-attention sia installato
pip install --break-system-packages flash-attn --no-build-isolation
```

## 📚 Documentazione

- **Guida Completa**: `cat VAST_AI_SETUP_GUIDE.md`
- **Quick Reference**: `cat QUICK_REFERENCE.md`
- **Qwen3-TTS Docs**: https://github.com/QwenLM/Qwen3-TTS
- **Vast.ai Docs**: https://vast.ai/docs/

## 🧪 Test del Modello

Dopo il training:

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "output_italian_tts/checkpoint-best",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

wavs, sr = tts.generate_custom_voice(
    text="Ciao! Questo è il mio modello fine-tuned.",
    speaker="italian_multi",
)

sf.write("test.wav", wavs[0], sr)
```

## 💾 Scaricare il Modello

```bash
# Comprimi il checkpoint
tar -czf checkpoint.tar.gz output_italian_tts/checkpoint-best

# Scarica sul tuo PC
scp -P <PORT> root@<IP>:/workspace/Qwen3-TTS/finetuning/checkpoint.tar.gz .
```

## ✅ Checklist Completa

### Prima di Iniziare
- [ ] Istanza Vast.ai noleggiata (4x o 8x RTX 3090)
- [ ] SSH connection funzionante
- [ ] Dataset `train_raw.jsonl` preparato (~15k samples)
- [ ] Reference audio pronto (5-10 secondi, 24kHz)

### Durante Setup
- [ ] `setup_vast.sh` eseguito con successo
- [ ] `check_environment.py` passed tutti i test
- [ ] Dataset caricato su Vast.ai
- [ ] GPU visibili con `nvidia-smi`

### Durante Training
- [ ] Training avviato senza errori
- [ ] GPU usage >90% (verifica con `nvidia-smi`)
- [ ] Logs disponibili e training progredisce
- [ ] Checkpoint salvati periodicamente

### Dopo Training
- [ ] Training completato (5 epochs per 4x GPU, 3 per 8x)
- [ ] Test audio generati e verificati
- [ ] `checkpoint-best` scaricato su PC locale
- [ ] Istanza Vast.ai fermata (per evitare costi)

## 🎉 Success!

Se tutto è andato bene:
- ✅ Hai un modello TTS personalizzato di alta qualità
- ✅ Speso circa $1.20-$2.00
- ✅ Tempo totale: ~3 ore (setup + training + test)

## 🆘 Supporto

Problemi? Controlla:

1. **Logs**: `tail -f output_italian_tts/training_*.log`
2. **Guida completa**: `VAST_AI_SETUP_GUIDE.md`
3. **Issues GitHub**: https://github.com/QwenLM/Qwen3-TTS/issues

---

**Happy Training!** 🚀

*Package creato per ottimizzare il training di Qwen3-TTS su Vast.ai*
*Testato con: 4x e 8x RTX 3090, 15k samples italiani*
