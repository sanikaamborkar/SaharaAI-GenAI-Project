"""
MentalBERT Fine-Tuning — Risk Level Classifier
================================================
Optimized for: 4GB VRAM / CUDA
Model:  mental/mental-bert-base-uncased
Task:   3-class risk classification — LOW / MEDIUM / HIGH
Data:   Suicide_Detection.csv (SuicideWatch / depression / teenagers)
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG — proven settings for 4GB VRAM
# ─────────────────────────────────────────────
CONFIG = {
    "model_name":       "mental/mental-bert-base-uncased",
    "max_length":       128,
    "batch_size":       4,
    "grad_accum_steps": 8,         
    "epochs":           4,
    "learning_rate":    2e-5,
    "warmup_ratio":     0.1,
    "weight_decay":     0.01,
    "fp16":             True,
    "csv_path":         r"E:\Gen-AI-Project\SahaaraAI\data\raw\Suicide_Detection.csv",
    "output_dir":       r"E:\Gen-AI-Project\SahaaraAI\models\mentalbert-risk-classifier",
    "seed":             42,
}

LABELS   = ['LOW', 'MEDIUM', 'HIGH']
LABEL2ID = {'teenagers': 0, 'depression': 1, 'SuicideWatch': 2}   # raw class -> int
ID2LABEL = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}                     # int -> risk label

# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────
class RiskDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df['text'].astype(str).tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels':         torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, device, grad_accum):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        with torch.amp.autocast('cuda', enabled=CONFIG['fp16']):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum

        if step % 100 == 0:
            vram_used = torch.cuda.memory_allocated() / 1024**3
            print(f"  Step {step:>5}/{len(loader)} | Loss: {loss.item()*grad_accum:.4f} | VRAM: {vram_used:.2f}GB")

    return total_loss / len(loader)

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            with torch.amp.autocast('cuda', enabled=CONFIG['fp16']):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=LABELS, zero_division=0)
    return acc, report

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*55}\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM  : {vram:.1f} GB")
    print(f"{'='*55}\n")

    # ── Load and prep data ──────────────────────
    print("Loading dataset...")
    df = pd.read_csv(CONFIG['csv_path'])
    df['label'] = df['class'].map(LABEL2ID)
    df = df.dropna(subset=['label'])

    # OPTIONAL: uncomment to subsample for a faster first test run
    # df = df.groupby('class').sample(n=12000, random_state=42).reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"\nLabel distribution (train):\n{train_df['class'].value_counts()}\n")

    # ── Load model ───────────────────────────────
    print(f"Loading model: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['model_name'],
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,      
    ignore_mismatched_sizes=True
).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.1f}M")

    # ── DataLoaders ───────────────────────────────
    train_dataset = RiskDataset(train_df, tokenizer, CONFIG['max_length'])
    val_dataset   = RiskDataset(val_df, tokenizer, CONFIG['max_length'])
    test_dataset  = RiskDataset(test_df, tokenizer, CONFIG['max_length'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    # ── Optimizer + Scheduler ─────────────────────
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    total_steps = (len(train_loader) // CONFIG['grad_accum_steps']) * CONFIG['epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=CONFIG['fp16'])

    print(f"Total steps    : {total_steps}")
    print(f"Effective batch: {CONFIG['batch_size'] * CONFIG['grad_accum_steps']}")
    print(f"Epochs         : {CONFIG['epochs']}\n")

    # ── Training loop ─────────────────────────────
    best_val_acc = 0
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*55}\nEPOCH {epoch+1}/{CONFIG['epochs']}\n{'='*55}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, CONFIG['grad_accum_steps'])
        val_acc, val_report = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch+1} Summary:\n  Train Loss: {train_loss:.4f}\n  Val Acc: {val_acc:.4f}")
        print(f"\nValidation Report:\n{val_report}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.cuda.empty_cache()
            gc.collect()
            model.save_pretrained(CONFIG['output_dir'], safe_serialization=False)
            tokenizer.save_pretrained(CONFIG['output_dir'])
            print(f"  Best model saved (val_acc={val_acc:.4f})")

        torch.cuda.empty_cache()

    # ── Final test evaluation ─────────────────────
    print(f"\n{'='*55}\nFINAL TEST EVALUATION\n{'='*55}")
    best_model = AutoModelForSequenceClassification.from_pretrained(CONFIG['output_dir']).to(device)
    test_acc, test_report = evaluate(best_model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}\n\n{test_report}")

    with open(os.path.join(CONFIG['output_dir'], "test_results.txt"), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n{test_report}")

    print(f"\n{'='*55}\nTraining complete!\nModel saved to: {CONFIG['output_dir']}/\nBest Val Acc: {best_val_acc:.4f}\nTest Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()