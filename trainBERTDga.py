"""
train_bert_dga.py
=================
Chạy 3 kịch bản phát hiện Word-based DGA theo bài báo:
  Scenario 1 – domain không TLD, không features
  Scenario 2 – domain có TLD,    không features
  Scenario 3 – domain có TLD,  + 11 features (chuỗi văn bản ghép)

Dữ liệu: dataout/train.csv | val.csv | test.csv
         cột: domain, family, label  (label 0=benign, 1=DGA)

Early stopping : 50 epochs max, patience=5 (theo val accuracy)
Model          : bert-base-uncased
Batch size     : 32

Output (tất cả lưu vào results/):
  results/scenario_<N>_epoch_log.txt     — log từng epoch (Train+Val metrics)
  results/scenario_<N>_final.txt         — kết quả cuối Train/Val/Test + DR
  results/summary_table3.txt             — Table 3 bài báo (evaluated on VALIDATION subset)
  results/summary_table5.txt             — Table 5 bài báo (evaluated on TESTING subset)
  models/scenario_<N>/best_model/        — checkpoint best model
"""

import os, sys, time, re, warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DATAOUT_DIR = Path(r"dataout")
DICT_DIR    = Path(r"dictionary")
MODELS_DIR  = Path(r"models")
RESULTS_DIR = Path(r"results")

BERT_MODEL  = "bert-base-uncased"
MAX_LEN     = 64
BATCH_SIZE  = 32
LR          = 2e-5
MAX_EPOCHS  = 50
PATIENCE    = 5
SEED        = 42

# ── Anti-overfitting hyperparameters (tuned after KB1/KB2 observation) ─────
WEIGHT_DECAY    = 0.05   # was 0.01 — stronger L2 regularization
DROPOUT_RATE    = 0.2    # was 0.1 (BERT default) — extra dropout on hidden+attn
LABEL_SMOOTHING = 0.05   # reduce overconfidence on training set
# ───────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()

def log(msg, logfile=None):
    print(msg)
    if logfile:
        logfile.write(msg + "\n")
        logfile.flush()

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ══════════════════════════════════════════════════════════════════════════════
#  DICTIONARY LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_dict(filename: str) -> set:
    path = DICT_DIR / filename
    if not path.exists():
        print(f"  [WARN] Dictionary not found: {path}")
        return set()
    words = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                words.add(w)
    print(f"  Loaded {len(words):,} words from {filename}")
    return words

# ══════════════════════════════════════════════════════════════════════════════
#  WORD EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def split_meaningful_words(domain_clean: str, dictionary: set):
    """Greedy backward scan — longest match ending at each position."""
    words_found, meaningful_count, total_len = [], 0, 0
    i = len(domain_clean)
    while i > 0:
        best_match, match_index = "", -1
        for j in range(0, i):
            word = domain_clean[j:i]
            if word in dictionary and len(word) > len(best_match):
                best_match, match_index = word, j
        if best_match:
            meaningful_count += 1
            total_len += len(best_match)
            words_found.insert(0, best_match)
            i = match_index + 1
        i -= 1
    return words_found, meaningful_count, total_len

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (11 features — paper Section 3.3)
# ══════════════════════════════════════════════════════════════════════════════
def strip_tld(domain: str) -> str:
    parts = domain.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else domain

def clean_alpha(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def number_to_words(n: int) -> str:
    ones = ["zero","one","two","three","four","five","six","seven","eight","nine",
            "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
            "seventeen","eighteen","nineteen"]
    tens = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
    if n < 20:   return ones[n]
    if n < 100:  return tens[n//10] + ("" if n%10==0 else ones[n%10])
    if n < 1000: return ones[n//100]+"hundred"+("" if n%100==0 else number_to_words(n%100))
    return str(n)

def extract_features(domain_no_tld: str, dicts: dict) -> dict:
    clean = clean_alpha(domain_no_tld)
    f1 = len(clean)
    words_dga,  f2, f5 = split_meaningful_words(clean, dicts["dictDGA"])
    words_nltk, f3, f6 = split_meaningful_words(clean, dicts["dictOnlyNLTK"])
    words_odga, f4, f7 = split_meaningful_words(clean, dicts["dictOnlyDGA"])
    f8  = max((len(w) for w in words_dga),  default=0)
    f9  = max((len(w) for w in words_nltk), default=0)
    f10 = max((len(w) for w in words_odga), default=0)
    f11 = "yes" if any(c.isdigit() for c in clean) else "no"
    return dict(f1=f1,f2=f2,f3=f3,f4=f4,f5=f5,f6=f6,f7=f7,f8=f8,f9=f9,f10=f10,f11=f11)

def features_to_string(feat: dict) -> str:
    parts = []
    for k, v in feat.items():
        parts.append(v if k=="f11" else number_to_words(int(v)))
    return " ".join(parts)

# ══════════════════════════════════════════════════════════════════════════════
#  BUILD INPUT STRING PER SCENARIO
# ══════════════════════════════════════════════════════════════════════════════
def build_input_string(domain: str, scenario: int, dicts: dict) -> str:
    no_tld   = strip_tld(domain)
    with_tld = domain
    if scenario == 1:
        return no_tld
    elif scenario == 2:
        return with_tld
    else:
        feat    = extract_features(no_tld, dicts)
        feat_str = features_to_string(feat)
        return with_tld + " " + feat_str

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class DGADataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred) * 100
    f1   = f1_score(y_true, y_pred, average="binary", zero_division=0) * 100
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0) * 100
    rec  = recall_score(y_true, y_pred, average="binary", zero_division=0) * 100
    cm   = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    fpr  = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0.0
    fnr  = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0.0
    return dict(acc=acc, f1=f1, prec=prec, rec=rec, fpr=fpr, fnr=fnr,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN ONE EPOCH  (with batch progress bar)
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, scheduler, scaler, epoch, total_epochs):
    model.train()
    total_loss  = 0.0
    n_batches   = len(loader)
    bar_width   = 30

    # CE loss with label smoothing — overrides BERT's internal CE
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    for step, batch in enumerate(loader, 1):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

        # Progress bar (overwrite same line)
        pct      = step / n_batches
        filled   = int(bar_width * pct)
        bar      = "█" * filled + "░" * (bar_width - filled)
        avg_loss = total_loss / step
        print(
            f"\r  Epoch {epoch:>2}/{total_epochs}  "
            f"[{bar}] {step:>4}/{n_batches}  loss={avg_loss:.4f}",
            end="", flush=True
        )

    print()  # newline after bar completes
    return total_loss / n_batches

# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

# ══════════════════════════════════════════════════════════════════════════════
#  PER-FAMILY DETECTION RATE
# ══════════════════════════════════════════════════════════════════════════════
def detection_rate_per_family(model, tokenizer, test_csv: Path, scenario: int, dicts: dict):
    """Per-family Detection Rate on the TESTING subset.

    The testing subset contains BOTH DGA samples (~900/family x 13 families) AND
    benign samples (~11,700). We compute:
        - per-DGA-family DR (TP / total DGA for that family)
        - benign-side FPR (FP / total benign rows) stored under key "_benign"
    """
    df     = pd.read_csv(test_csv)
    results = {}

    # ── Per-DGA-family Detection Rate ──────────────────────────────
    dga_df = df[df["label"] == 1].copy()
    for family, grp in dga_df.groupby("family"):
        texts  = [build_input_string(d, scenario, dicts) for d in grp["domain"].tolist()]
        labels = grp["label"].tolist()
        ds     = DGADataset(texts, labels, tokenizer)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
        y_true, y_pred = evaluate(model, loader)
        tp    = int(np.sum((y_true == 1) & (y_pred == 1)))
        total = len(y_true)
        results[family] = {"total": total, "tp": tp, "dr": tp/total*100 if total>0 else 0.0}

    # ── Benign-side classification accuracy (testing subset includes ~11,700 benign rows) ──
    # For consistency with DGA families, "tp" here means "correctly classified as benign"
    # (i.e., the True Negatives), and "dr" is TN/total_benign.
    benign_df = df[df["label"] == 0].copy()
    if len(benign_df) > 0:
        texts  = [build_input_string(d, scenario, dicts) for d in benign_df["domain"].tolist()]
        labels = benign_df["label"].tolist()
        ds     = DGADataset(texts, labels, tokenizer)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
        y_true, y_pred = evaluate(model, loader)
        tn     = int(np.sum((y_true == 0) & (y_pred == 0)))   # correctly classified as benign
        fp     = int(np.sum((y_true == 0) & (y_pred == 1)))   # benign misclassified as DGA
        total  = len(y_true)
        results["_benign"] = {
            "total": total,
            "tp":    tn,                                       # "TP" column = correct classification
            "dr":    tn/total*100 if total > 0 else 0.0,       # = TN / total_benign
            "fp":    fp,                                       # extra: # benign predicted DGA
            "fpr":   fp/total*100 if total > 0 else 0.0,       # extra: FPR
        }
    return results

# ══════════════════════════════════════════════════════════════════════════════
#  FORMAT & PRINT METRICS BLOCK
# ══════════════════════════════════════════════════════════════════════════════
def fmt_metrics_block(label: str, m: dict) -> str:
    return (
        f"  ── {label} ──\n"
        f"  {'Accuracy':<12}: {m['acc']:.2f}%\n"
        f"  {'F1-Score':<12}: {m['f1']:.2f}%\n"
        f"  {'Precision':<12}: {m['prec']:.2f}%\n"
        f"  {'Recall':<12}: {m['rec']:.2f}%\n"
        f"  {'FPR':<12}: {m['fpr']:.2f}%\n"
        f"  {'FNR':<12}: {m['fnr']:.2f}%\n"
        f"  {'TP/FP/FN/TN':<12}: {m['tp']} / {m['fp']} / {m['fn']} / {m['tn']}"
    )

def fmt_dr_block(dr_results: dict) -> str:
    lines = [f"  {'Family':<20} {'Total':>6}  {'TP':>6}  {'DR':>8}",
             "  " + "-"*46]
    # Skip the special "_benign" key when listing DGA families
    for fam, r in sorted(dr_results.items()):
        if fam == "_benign":
            continue
        lines.append(f"  {fam:<20} {r['total']:>6}  {r['tp']:>6}  {r['dr']:>7.2f}%")
    # Append benign-side classification accuracy if present
    if "_benign" in dr_results:
        b = dr_results["_benign"]
        lines.append("  " + "-"*46)
        lines.append(f"  {'benign':<20} {b['total']:>6}  {b['tp']:>6}  {b['dr']:>7.2f}%")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
#  RUN ONE SCENARIO
# ══════════════════════════════════════════════════════════════════════════════
def run_scenario(scenario: int, tokenizer, dicts: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    epoch_log_path  = RESULTS_DIR / f"scenario_{scenario}_epoch_log.txt"
    final_log_path  = RESULTS_DIR / f"scenario_{scenario}_final.txt"
    epoch_logfile   = open(epoch_log_path, "w", encoding="utf-8", buffering=1)

    hdr = "="*70
    sec_hdr = f"\n{hdr}\n  SCENARIO {scenario}\n{hdr}"
    print(sec_hdr)
    epoch_logfile.write(sec_hdr + "\n")

    # ── Load data ──────────────────────────────────────────────────
    train_df = pd.read_csv(DATAOUT_DIR / "train.csv")
    val_df   = pd.read_csv(DATAOUT_DIR / "val.csv")
    test_df  = pd.read_csv(DATAOUT_DIR / "test.csv")
    msg = f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}"
    print(msg); epoch_logfile.write(msg+"\n")

    # ── Build input strings ────────────────────────────────────────
    print("  Building input strings...", flush=True)
    t0 = time.time()
    train_texts = [build_input_string(d, scenario, dicts) for d in train_df["domain"]]
    val_texts   = [build_input_string(d, scenario, dicts) for d in val_df["domain"]]
    test_texts  = [build_input_string(d, scenario, dicts) for d in test_df["domain"]]
    msg = f"  Input strings built in {time.time()-t0:.1f}s"
    print(msg); epoch_logfile.write(msg+"\n")

    # ── Tokenize ───────────────────────────────────────────────────
    print("  Tokenizing...", flush=True)
    t0 = time.time()
    train_ds = DGADataset(train_texts, train_df["label"].tolist(), tokenizer)
    val_ds   = DGADataset(val_texts,   val_df["label"].tolist(),   tokenizer)
    test_ds  = DGADataset(test_texts,  test_df["label"].tolist(),  tokenizer)
    msg = f"  Tokenized in {time.time()-t0:.1f}s"
    print(msg); epoch_logfile.write(msg+"\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model / optimizer ──────────────────────────────────────────
    set_seed()
    # Apply higher dropout in BERT config to combat overfitting
    from transformers import BertConfig
    config = BertConfig.from_pretrained(BERT_MODEL, num_labels=2)
    config.hidden_dropout_prob          = DROPOUT_RATE
    config.attention_probs_dropout_prob = DROPOUT_RATE
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL, config=config
    ).to(DEVICE)
    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * MAX_EPOCHS
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Epoch log header ───────────────────────────────────────────
    col_hdr = (
        f"\n  {'Ep':>3}  {'TrainLoss':>10}"
        f"  {'TrACC':>7} {'TrF1':>7} {'TrFPR':>7} {'TrFNR':>7}"
        f"  {'VaACC':>7} {'VaF1':>7} {'VaFPR':>7} {'VaFNR':>7}"
        f"  {'Status'}"
    )
    sep = "  " + "-"*110
    print(col_hdr); print(sep)
    epoch_logfile.write(col_hdr + "\n" + sep + "\n")

    best_val_acc   = 0.0
    best_epoch     = 0
    patience_count = 0
    best_state     = None
    train_start    = time.time()
    epoch_history  = []   # list of dicts for saving

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler,
                                 epoch, MAX_EPOCHS)
        # Evaluate train set
        y_tr, p_tr = evaluate(model, train_loader)
        tr_m = compute_metrics(y_tr, p_tr)
        # Evaluate val set
        y_va, p_va = evaluate(model, val_loader)
        va_m = compute_metrics(y_va, p_va)

        # Early stopping logic
        if va_m["acc"] > best_val_acc:
            best_val_acc   = va_m["acc"]
            best_epoch     = epoch
            best_state     = deepcopy(model.state_dict())
            patience_count = 0
            status         = "✓ best"
        else:
            patience_count += 1
            status = f"patience {patience_count}/{PATIENCE}"

        epoch_line = (
            f"  {epoch:>3}  {train_loss:>10.4f}"
            f"  {tr_m['acc']:>6.2f}% {tr_m['f1']:>6.2f}% {tr_m['fpr']:>6.2f}% {tr_m['fnr']:>6.2f}%"
            f"  {va_m['acc']:>6.2f}% {va_m['f1']:>6.2f}% {va_m['fpr']:>6.2f}% {va_m['fnr']:>6.2f}%"
            f"  {status}"
        )
        print(epoch_line)
        epoch_logfile.write(epoch_line + "\n")

        epoch_history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "train_acc": round(tr_m["acc"],2), "train_f1": round(tr_m["f1"],2),
            "train_fpr": round(tr_m["fpr"],2), "train_fnr": round(tr_m["fnr"],2),
            "val_acc":   round(va_m["acc"],2), "val_f1":   round(va_m["f1"],2),
            "val_fpr":   round(va_m["fpr"],2), "val_fnr":  round(va_m["fnr"],2),
            "status": status,
        })

        if patience_count >= PATIENCE:
            stop_msg = f"\n  Early stopping at epoch {epoch}  (best epoch: {best_epoch})"
            print(stop_msg); epoch_logfile.write(stop_msg+"\n")
            break

    total_time = time.time() - train_start
    time_msg = f"  Training time: {total_time:.0f}s  |  Best epoch: {best_epoch}"
    print(time_msg); epoch_logfile.write(time_msg+"\n")
    epoch_logfile.close()

    # Save epoch log as CSV too
    pd.DataFrame(epoch_history).to_csv(
        RESULTS_DIR / f"scenario_{scenario}_epoch_log.csv", index=False
    )

    # ── Restore best model & save ──────────────────────────────────
    model.load_state_dict(best_state)
    save_path = MODELS_DIR / f"scenario_{scenario}" / "best_model"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  Best model saved → {save_path}")

    # ── Evaluate best model on Train / Val / Test ──────────────────
    print("  Evaluating best model on all splits...", flush=True)
    y_tr, p_tr   = evaluate(model, train_loader)
    train_m      = compute_metrics(y_tr, p_tr)
    y_va, p_va   = evaluate(model, val_loader)
    val_m        = compute_metrics(y_va, p_va)
    y_te, p_te   = evaluate(model, test_loader)
    test_m       = compute_metrics(y_te, p_te)

    # ── Per-family DR ──────────────────────────────────────────────
    print("  Computing per-family detection rates...", flush=True)
    dr_results = detection_rate_per_family(
        model, tokenizer, DATAOUT_DIR / "test.csv", scenario, dicts
    )

    # ── Build final report ─────────────────────────────────────────
    final_lines = [
        hdr,
        f"  SCENARIO {scenario} — Final Results  (best epoch: {best_epoch}, "
        f"training time: {total_time:.0f}s)",
        hdr,
        "",
        fmt_metrics_block(f"TRAIN  (best model, epoch {best_epoch})", train_m),
        "",
        fmt_metrics_block(f"VAL    (best model, epoch {best_epoch})", val_m),
        "",
        fmt_metrics_block("TEST", test_m),
        "",
        "  ── Per-family Detection Rate (testing subset) ──",
        fmt_dr_block(dr_results),
        "",
    ]
    final_report = "\n".join(final_lines)
    print("\n" + final_report)

    with open(final_log_path, "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f"  Final results saved → {final_log_path}")

    return {
        "scenario"  : scenario,
        "best_epoch": best_epoch,
        "train_time": total_time,
        "train"     : train_m,
        "val"       : val_m,
        "test"      : test_m,
        "dr"        : dr_results,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLES  (Table 3 & Table 5 bài báo)
# ══════════════════════════════════════════════════════════════════════════════
def build_summary(all_results):
    r1, r2, r3 = all_results
    lines = []

    # ── Table 3 ───────────────────────────────────────────────────
    lines += [
        "",
        "="*72,
        "  TABLE 3 — Results on 3 scenarios (evaluated on VALIDATION subset)",
        "="*72,
        f"  {'Metric':<22} {'Scenario 1':>12} {'Scenario 2':>12} {'Scenario 3':>12}",
        "  " + "-"*66,
    ]
    rows = [
        ("Epochs (best)",  r1["best_epoch"],       r2["best_epoch"],       r3["best_epoch"],       "{}",      "{}",      "{}"),
        ("Training time",  f"{r1['train_time']:.0f}s", f"{r2['train_time']:.0f}s", f"{r3['train_time']:.0f}s", "{}","{}","{}"),
        ("Accuracy",       r1["val"]["acc"],       r2["val"]["acc"],       r3["val"]["acc"],       "{:.2f}%", "{:.2f}%", "{:.2f}%"),
        ("F1-Score",       r1["val"]["f1"],        r2["val"]["f1"],        r3["val"]["f1"],        "{:.2f}%", "{:.2f}%", "{:.2f}%"),
        ("FPR",            r1["val"]["fpr"],       r2["val"]["fpr"],       r3["val"]["fpr"],       "{:.2f}%", "{:.2f}%", "{:.2f}%"),
        ("FNR",            r1["val"]["fnr"],       r2["val"]["fnr"],       r3["val"]["fnr"],       "{:.2f}%", "{:.2f}%", "{:.2f}%"),
        ("Precision",      r1["val"]["prec"],      r2["val"]["prec"],      r3["val"]["prec"],      "{:.2f}%", "{:.2f}%", "{:.2f}%"),
        ("Recall",         r1["val"]["rec"],       r2["val"]["rec"],       r3["val"]["rec"],       "{:.2f}%", "{:.2f}%", "{:.2f}%"),
    ]
    for name, v1, v2, v3, f1, f2, f3 in rows:
        s1 = f1.format(v1) if not isinstance(v1, str) else v1
        s2 = f2.format(v2) if not isinstance(v2, str) else v2
        s3 = f3.format(v3) if not isinstance(v3, str) else v3
        lines.append(f"  {name:<22} {s1:>12} {s2:>12} {s3:>12}")

    # ── Table 3b — per-split breakdown ────────────────────────────
    lines += ["", "  (Train / Test breakdown per scenario)", "  " + "-"*66]
    for split_key, split_label in [("train","TRAIN"), ("test","TEST")]:
        for metric, mkey in [("Accuracy","acc"),("F1","f1"),("FPR","fpr"),("FNR","fnr")]:
            name = f"{split_label} {metric}"
            v1 = r1[split_key][mkey]; v2 = r2[split_key][mkey]; v3 = r3[split_key][mkey]
            lines.append(f"  {name:<22} {v1:>11.2f}% {v2:>11.2f}% {v3:>11.2f}%")

    # ── Table 5 ───────────────────────────────────────────────────
    lines += [
        "",
        "="*90,
        "  TABLE 5 — Per-family Detection Rate (evaluated on TESTING subset)",
        "="*90,
        f"  {'Family':<20} {'Total':>6}  "
        f"{'TP S1':>6} {'DR S1':>8}  {'TP S2':>6} {'DR S2':>8}  {'TP S3':>6} {'DR S3':>8}",
        "  " + "-"*84,
    ]
    # Collect DGA families across the three scenarios (exclude the special _benign key)
    all_families = sorted(
        (set(r1["dr"]) | set(r2["dr"]) | set(r3["dr"])) - {"_benign"}
    )
    for fam in all_families:
        d1 = r1["dr"].get(fam, {"total":0,"tp":0,"dr":0.0})
        d2 = r2["dr"].get(fam, {"total":0,"tp":0,"dr":0.0})
        d3 = r3["dr"].get(fam, {"total":0,"tp":0,"dr":0.0})
        total = d1["total"] or d2["total"] or d3["total"]
        lines.append(
            f"  {fam:<20} {total:>6}  "
            f"{d1['tp']:>6} {d1['dr']:>7.2f}%  "
            f"{d2['tp']:>6} {d2['dr']:>7.2f}%  "
            f"{d3['tp']:>6} {d3['dr']:>7.2f}%"
        )
    # Benign-side classification accuracy on the testing subset
    if "_benign" in r1["dr"] and "_benign" in r2["dr"] and "_benign" in r3["dr"]:
        b1, b2, b3 = r1["dr"]["_benign"], r2["dr"]["_benign"], r3["dr"]["_benign"]
        lines.append("  " + "-"*84)
        lines.append(
            f"  {'benign':<20} {b1['total']:>6}  "
            f"{b1['tp']:>6} {b1['dr']:>7.2f}%  "
            f"{b2['tp']:>6} {b2['dr']:>7.2f}%  "
            f"{b3['tp']:>6} {b3['dr']:>7.2f}%"
        )
    lines.append("")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    master_log = open(RESULTS_DIR / "full_run.txt", "w", encoding="utf-8", buffering=1)

    def dual_print(msg=""):
        print(msg)
        master_log.write(msg + "\n")
        master_log.flush()

    banner = (
        "="*70 + "\n"
        "  Word-based DGA Detection — BERT (bert-base-uncased)\n"
        f"  Device     : {DEVICE}\n"
        + (f"  GPU        : {torch.cuda.get_device_name(0)}\n" if torch.cuda.is_available() else "") +
        f"  Max epochs : {MAX_EPOCHS}  |  Patience: {PATIENCE}\n"
        f"  Batch size : {BATCH_SIZE}  |  LR: {LR}  |  MaxLen: {MAX_LEN}\n"
        f"  WeightDecay: {WEIGHT_DECAY}  |  Dropout: {DROPOUT_RATE}  |  LabelSmooth: {LABEL_SMOOTHING}\n"
        f"  Started    : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        + "="*70
    )
    dual_print(banner)

    # Load dictionaries
    dual_print("\n[1/5] Loading dictionaries...")
    dicts = {
        "dictDGA":      load_dict("dictDGA.txt"),
        "dictOnlyNLTK": load_dict("dictOnlyNLTK.txt"),
        "dictOnlyDGA":  load_dict("dictOnlyDGA.txt"),
    }

    # Load tokenizer
    dual_print("\n[2/5] Loading BERT tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)

    # Run scenarios
    all_results = []
    for sc in [1, 2, 3]:
        dual_print(f"\n[{sc+2}/5] ── Scenario {sc} ──")
        result = run_scenario(sc, tokenizer, dicts)
        all_results.append(result)

    # Summary
    dual_print("\n[6/5] Building summary tables...")
    summary = build_summary(all_results)
    dual_print(summary)

    # Save summary
    t_path = RESULTS_DIR / "summary_table3_table5.txt"
    with open(t_path, "w", encoding="utf-8") as f:
        f.write(summary)
    dual_print(f"\n  Summary saved → {t_path}")
    dual_print(f"\n  Full log saved → {RESULTS_DIR / 'full_run.txt'}")
    dual_print(f"\n  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    master_log.close()

if __name__ == "__main__":
    main()