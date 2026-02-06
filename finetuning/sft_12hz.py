# coding=utf-8
# Ottimizzato per multi-GPU training (4x o 8x RTX 3090)
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from transformers import AutoConfig
from tqdm import tqdm

target_speaker_embedding = None

def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam for memory efficiency")
    args = parser.parse_args()

    # Accelerator con logging ottimizzato
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_model_path
    )

    MODEL_PATH = args.init_model_path

    # Log info sul training
    if accelerator.is_main_process:
        num_gpus = accelerator.num_processes
        effective_batch_size = args.batch_size * num_gpus * args.gradient_accumulation_steps
        accelerator.print(f"{'='*60}")
        accelerator.print(f"Training Configuration:")
        accelerator.print(f"  Number of GPUs: {num_gpus}")
        accelerator.print(f"  Batch size per GPU: {args.batch_size}")
        accelerator.print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        accelerator.print(f"  Effective batch size: {effective_batch_size}")
        accelerator.print(f"  Learning rate: {args.lr}")
        accelerator.print(f"  Number of epochs: {args.num_epochs}")
        accelerator.print(f"  Warmup steps: {args.warmup_steps}")
        accelerator.print(f"{'='*60}\n")

    # Carica modello
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # Carica e splitta dataset
    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    
    # Train/validation split
    val_size = int(len(train_data) * args.val_split)
    train_size = len(train_data) - val_size
    
    train_dataset = TTSDataset(train_data[:train_size], qwen3tts.processor, config)
    val_dataset = TTSDataset(train_data[train_size:], qwen3tts.processor, config) if val_size > 0 else None
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=train_dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=2,
            pin_memory=True
        )

    # Optimizer con opzionale 8-bit
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                qwen3tts.model.parameters(), 
                lr=args.lr, 
                weight_decay=args.weight_decay
            )
            accelerator.print("Using 8-bit AdamW optimizer")
        except ImportError:
            accelerator.print("bitsandbytes not available, using standard AdamW")
            optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    num_training_steps = len(train_dataloader) * args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - args.warmup_steps)

    # Prepare con Accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader, scheduler
    )
    
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)

    # Training loop
    model.train()
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(
            train_dataloader, 
            disable=not accelerator.is_main_process,
            desc=f"Epoch {epoch+1}/{args.num_epochs}"
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Forward pass (codice invariato)
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                # Backward pass
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                
                # Warmup scheduler
                if global_step < args.warmup_steps:
                    lr = args.lr * (global_step + 1) / args.warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    scheduler.step()
                
                optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            # Logging
            if global_step % 10 == 0:
                avg_loss = epoch_loss / (step + 1)
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                if accelerator.is_main_process:
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch
                    }, step=global_step)

            # Validation
            if val_dataloader and global_step % args.eval_steps == 0:
                val_loss = evaluate(model, val_dataloader, accelerator)
                if accelerator.is_main_process:
                    accelerator.log({"eval/loss": val_loss}, step=global_step)
                    accelerator.print(f"Validation Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model, config, target_speaker_embedding, args, 
                            MODEL_PATH, f"checkpoint-best", accelerator
                        )
                model.train()

            # Save periodico
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                save_checkpoint(
                    model, config, target_speaker_embedding, args,
                    MODEL_PATH, f"checkpoint-step-{global_step}", accelerator
                )

        # Save a fine epoch
        if accelerator.is_main_process:
            save_checkpoint(
                model, config, target_speaker_embedding, args,
                MODEL_PATH, f"checkpoint-epoch-{epoch}", accelerator
            )

def evaluate(model, dataloader, accelerator):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            codec_ids = batch['codec_ids']
            ref_mels = batch['ref_mels']
            text_embedding_mask = batch['text_embedding_mask']
            codec_embedding_mask = batch['codec_embedding_mask']
            attention_mask = batch['attention_mask']
            codec_0_labels = batch['codec_0_labels']
            codec_mask = batch['codec_mask']

            speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype))
            input_text_ids = input_ids[:, :, 0]
            input_codec_ids = input_ids[:, :, 1]

            input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
            input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
            input_codec_embedding[:, 6, :] = speaker_embedding

            input_embeddings = input_text_embedding + input_codec_embedding

            for i in range(1, 16):
                codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                input_embeddings = input_embeddings + codec_i_embedding

            outputs = model.talker(
                inputs_embeds=input_embeddings[:, :-1, :],
                attention_mask=attention_mask[:, :-1],
                labels=codec_0_labels[:, 1:],
                output_hidden_states=True
            )

            hidden_states = outputs.hidden_states[0][-1]
            talker_hidden_states = hidden_states[codec_mask[:, 1:]]
            talker_codec_ids = codec_ids[codec_mask]

            _, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
            loss = outputs.loss + 0.3 * sub_talker_loss

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0

def save_checkpoint(model, config, target_emb, args, model_path, checkpoint_name, accelerator):
    output_dir = os.path.join(args.output_model_path, checkpoint_name)
    shutil.copytree(model_path, output_dir, dirs_exist_ok=True)

    # Update config
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = {args.speaker_name: 3000}
    talker_config["spk_is_dialect"] = {args.speaker_name: False}
    config_dict["talker_config"] = talker_config

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # Save model
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

    # Remove speaker encoder
    keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder")]
    for k in keys_to_drop:
        del state_dict[k]

    # Add speaker embedding
    weight = state_dict['talker.model.codec_embedding.weight']
    state_dict['talker.model.codec_embedding.weight'][3000] = target_emb[0].detach().to(weight.device).to(weight.dtype)
    
    save_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, save_path)
    
    accelerator.print(f"Checkpoint saved to {output_dir}")

if __name__ == "__main__":
    train()
