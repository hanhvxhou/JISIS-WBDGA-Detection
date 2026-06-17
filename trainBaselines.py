"""
train_baselines.py
==================
Re-implement 5 baseline models from Table 4 with the SAME train/val/test
split used by the BERT scenarios, for a fair comparison.

Baselines (in order of execution):
  1. Selvaraj & Panjanathan 2024 — RCNN-BiLSTM (word-level)
  2. Chen et al. 2023           — CNN-LSTM (word-level)
  3. Charan et al. 2020         — 8 lexical features + Gradient Boosting (≈C5.0)
  4. Biros & Kantor 2025        — CNN (char-level, 5 epochs)
  5. Vu & Hoang 2021            — J48 / Decision Tree (11 features)

Data : dataout/{train,val,test}.csv  (cùng split như train_bert_dga.py)
Output:
  results/baseline_<name>_epoch_log.{txt,csv}    (deep-learning models)
  results/baseline_<name>_final.txt              (Train/Val/Test + DR per family)
  results/baselines_summary_table4.txt           (Table 4 — fair comparison, validation subset)
  results/baselines_summary_table6.txt           (Table 6 — DR per family, testing subset)
  models/baseline_<name>/best_model              (checkpoint)
"""

import os, sys, time, re, json, math, warnings, pickle
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DATAOUT_DIR = Path(r"dataout")
DICT_DIR    = Path(r"dictionary")        # all baselines use this shared folder
MODELS_DIR  = Path(r"models")
RESULTS_DIR = Path(r"results")

SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Per-baseline hyperparameters (taken from the original papers)
# Hyperparameters from the original papers (with explicit best-effort notes
# where the paper omits a setting).
#
# IMPORTANT — Reproduction caveat:
#   Without official source code from the original authors, exact reproduction
#   is not possible. The settings below follow the published descriptions as
#   closely as possible. Where the paper omits a value, defaults are noted with
#   "[best-effort]" in the comments. This is a fair baseline comparison on the
#   SAME train/val/test split as the BERT scenarios, not a verbatim re-run.

HP = {
    # Selvaraj & Panjanathan 2024 (MDPI Informatics, WordDGA)
    #   Paper-exact reproduction (Sections 3.1-3.4 + 4):
    #     • Word-level tokens via wordninja (paper §3.1.1)
    #     • Sequence length: 64 (paper §3.1.1, "RFC max label length 63")
    #     • Embedding: 1024-dim ELMo (paper §3.1.1) — substituted with
    #       1024-dim random word embedding for offline reproduction
    #     • Three branches: word features (128-d) + BiLSTM n-gram (128-d)
    #       + 11 statistical features → 8-d, fused into a Dense head
    #     • Paper: 70:30 split + 10-fold CV; we use the unified split.
    "selvaraj_2024": dict(batch=32,  lr=5e-4, epochs=50, patience=5,
                          max_words=64, embed_dim=1024,
                          source="MDPI Informatics 2024, 11(4), 92"),

    # Chen, Lang, Chen & Xie 2023 (MDPI Applied Sciences 13(7), 4406)
    #   Paper-exact 4-branch fusion reproduction (Sections 3.2-3.5):
    #     • Branch A: wordninja + ELMo (substituted with 1024-d random
    #                 embedding) → mean + std → Linear 2048→128
    #     • Branch B: 11 statistical features → Linear 11→8
    #     • Branch C: 1-gram (char) LSTM → 128-d
    #     • Branch D: 3-gram LSTM → 128-d
    #     • Fusion: concat 392-d → FC → sigmoid
    #     • Paper: 70:10:20 train/val/test split; we use the unified split.
    "chen_2023":     dict(batch=32,  lr=5e-4, epochs=50, patience=5,
                          max_chars=64, max_words=64, max_3gram=64,
                          embed_dim=128, word_embed_dim=1024,
                          source="MDPI Appl. Sci. 2023, 13(7), 4406"),

    # Charan et al. 2020 (CANS 2020)
    #   • Tree-based ensemble (C5.0) with 8 lexical features (network features
    #     dropped here for fairness, since whois data is not available offline)
    "charan_2020":   dict(source="CANS 2020 LNCS 12579, pp. 121-141"),

    # Biros & Kantor 2025 (JTIT FITCE 2024) — CNN (Listing 2 in paper)
    #   • Word-level tokenisation via wordninja (paper Section 5.1)
    #   • Embedding: vocab=56000, dim=128, input_length=64 (paper exact)
    #   • Conv1D(200, k=4) → Dropout(0.5) → MaxPool → Conv1D(100, k=2) →
    #     MaxPool → Dropout(0.5) → Dense(100,ReLU) → Dropout(0.5) →
    #     Dense(10,ReLU) → Dense(1,sigmoid) — paper exact
    #   • Adam lr=0.001, 10 epochs, batch=128, Glorot-normal init
    #   • Extended to 50 epochs / patience 5 for unified comparison
    #   • The paper proposes BOTH a CNN and an LSTM model (reported as
    #     independent rows in Table 4: CNN=98.30%, LSTM=95.36%). We reproduce
    #     only the CNN model, since it is the stronger of the two.
    "biros_2025":    dict(batch=128, lr=1e-3, epochs=50, patience=5,
                          max_words=64, embed_dim=128,
                          source="JTIT 2025 Special Issue (jtit.2025.FITCE2024.2033) - Listing 2 (CNN only)"),

    # Vu & Hoang 2021 (JATIT, Vol 99, No 24)
    #   • 16 hand-crafted features → J48 (C4.5 decision tree)
    #   • sklearn DecisionTreeClassifier(criterion="entropy") is the closest
    #     equivalent (true J48/Weka uses reduced-error pruning)
    "vu_2021":       dict(source="JATIT 2021, Vol 99, No 24, pp. 6004-6014"),
}

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def strip_tld(domain: str) -> str:
    parts = domain.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else domain

def clean_alpha(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

# ══════════════════════════════════════════════════════════════════════════════
#  DICTIONARY LOADING & WORD EXTRACTION (mirror extractWord.py)
# ══════════════════════════════════════════════════════════════════════════════
def load_dict(filename: str) -> set:
    """Load a dictionary file from DICT_DIR as a set of lowercase words."""
    path = DICT_DIR / filename
    if not path.exists():
        print(f"  [WARN] Dictionary not found: {path}")
        return set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        return {line.strip().lower() for line in f if line.strip()}

def split_meaningful_words(domain_clean: str, dictionary: set):
    """Greedy longest-match (same as extractWord.py)."""
    words_found, count, total_len = [], 0, 0
    i = len(domain_clean)
    while i > 0:
        best, idx = "", -1
        for j in range(0, i):
            w = domain_clean[j:i]
            if w in dictionary and len(w) > len(best):
                best, idx = w, j
        if best:
            count += 1
            total_len += len(best)
            words_found.insert(0, best)
            i = idx + 1
        i -= 1
    return words_found, count, total_len

def extract_11_features(domain_no_tld: str, dicts: dict) -> list:
    """11 features (legacy — kept for compatibility, NOT used by Vu 2021 baseline)."""
    clean = clean_alpha(domain_no_tld)
    words_dga,  f2, f5 = split_meaningful_words(clean, dicts["dictDGA"])
    words_nltk, f3, f6 = split_meaningful_words(clean, dicts["dictOnlyNLTK"])
    words_odga, f4, f7 = split_meaningful_words(clean, dicts["dictOnlyDGA"])
    return [
        len(clean), f2, f3, f4, f5, f6, f7,
        max((len(w) for w in words_dga),  default=0),
        max((len(w) for w in words_nltk), default=0),
        max((len(w) for w in words_odga), default=0),
        1 if any(c.isdigit() for c in clean) else 0,
    ]


def extract_vu_16_features(domain: str, dicts: dict) -> list:
    """
    16 features from Vu & Hoang 2021, Section 3.2.

    f1  len(d)
    f2  ascii_value(d) - sum of ASCII codes of all chars (full domain incl. TLD)
    f3  countnv(d)     - number of vowels
    f4  tanv(d)        - f3 / f1
    f5  countdi(d)     - count of digits + '-'
    f6  tandi(d)       - f5 / f1
    f7  word_norm(d)   - words found in english_dict
    f8  word_dga(d)    - words found in dga_dict
    f9  noun_count(d)  - words found in noun_dict
    f10 verb_count(d)  - words found in verb_dict
    f11 adj_count(d)   - words found in adj_dict
    f12 private_count(d) - words found in private_dict
    f13 ratio_dga(d)   - f8 / f7 (0 if f7=0)
    f14 max_len_word(d) - max length of segmented words (0 if no words found)
    f15 min_len_word(d) - min length of segmented words (0 if no words found)
    f16 ratio_char(d)  - total chars of segmented words / len(d)
    """
    d = domain.lower()  # full domain with TLD
    # ── basic char stats ────────────────────────────────────────
    f1 = len(d)
    f2 = sum(ord(c) for c in d)
    f3 = sum(1 for c in d if c in "aeiou")
    f4 = f3 / f1 if f1 > 0 else 0.0
    f5 = sum(1 for c in d if c.isdigit() or c == '-')
    f6 = f5 / f1 if f1 > 0 else 0.0

    # ── word-level features ────────────────────────────────────
    # Segment the domain using a UNION of all reference dictionaries.
    # This is what Section 3.2 of Vu & Hoang (2021) implicitly does: each
    # candidate "word" is then checked against every individual dictionary
    # to compute f7..f12. Segmenting with only english_dict (as we did
    # initially) would miss DGA-specific or private words and bias every
    # subsequent count downwards.
    no_tld = strip_tld(d)
    clean  = clean_alpha(no_tld)

    # Cache the union dictionary on the dicts mapping so we build it once
    if "_union_vu" not in dicts:
        dicts["_union_vu"] = (
            dicts["english_dict"] | dicts["dga_dict"] | dicts["noun_dict"]
            | dicts["verb_dict"] | dicts["adj_dict"] | dicts["private_dict"]
        )
    union_dict = dicts["_union_vu"]

    words, _, _ = split_meaningful_words(clean, union_dict)
    # If no segmentation found, count zero words (matching paper semantics)
    # — do NOT fall back to the whole clean string as a single "word".

    f7  = sum(1 for w in words if w in dicts["english_dict"])
    f8  = sum(1 for w in words if w in dicts["dga_dict"])
    f9  = sum(1 for w in words if w in dicts["noun_dict"])
    f10 = sum(1 for w in words if w in dicts["verb_dict"])
    f11 = sum(1 for w in words if w in dicts["adj_dict"])
    f12 = sum(1 for w in words if w in dicts["private_dict"])
    f13 = (f8 / f7) if f7 > 0 else 0.0

    if words:
        word_lens = [len(w) for w in words]
        f14 = max(word_lens)
        f15 = min(word_lens)
        f16 = sum(word_lens) / f1 if f1 > 0 else 0.0
    else:
        f14 = 0
        f15 = 0
        f16 = 0.0

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
            f11, f12, f13, f14, f15, f16]

# ══════════════════════════════════════════════════════════════════════════════
#  CHARAN 2020 — 8 LEXICAL FEATURES (Table 1 of paper, lexical only)
# ══════════════════════════════════════════════════════════════════════════════
VOWELS = set("aeiou")

def syllable_count(word: str) -> int:
    """Heuristic English syllable counter (sufficient for proxy)."""
    word = word.lower()
    if not word:
        return 0
    count, prev = 0, False
    for ch in word:
        is_v = ch in VOWELS
        if is_v and not prev:
            count += 1
        prev = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def extract_charan_8_features(domain: str, dicts: dict) -> list:
    """8 lexical features from Charan 2020 (network/WHOIS features dropped).

    Note: Paper uses wordninja (probabilistic English split). We approximate
    with greedy longest-match against the English dictionary (dictEng).
    """
    no_tld = strip_tld(domain).lower()
    clean  = clean_alpha(no_tld)

    # Word Count — use the ENGLISH dictionary (not dictDGA) so the segmentation
    # reflects "how many English words can be extracted from the domain",
    # which is the discriminating signal between benign English-looking
    # domains and DGA-generated meaningless strings.
    eng_dict = dicts.get("english_dict") or dicts.get("dictEng")
    words, _, _ = split_meaningful_words(clean, eng_dict)

    word_count   = len(words)   # 0 if nothing matches — preserves signal
    length       = len(clean)
    # Syllables: if no English words found, count syllables of the whole string
    # (still a useful proxy for "natural-sounding-ness")
    if words:
        syll_count = sum(syllable_count(w) for w in words)
    else:
        syll_count = syllable_count(clean)
    vowel_count    = sum(1 for c in clean if c in VOWELS)
    consonant_cnt  = sum(1 for c in clean if c.isalpha() and c not in VOWELS)
    unique_letters = len(set(c for c in clean if c.isalpha()))
    has_hyphen     = 1 if "-" in domain else 0
    has_underscore = 1 if "_" in domain else 0

    return [
        word_count, length, syll_count, vowel_count,
        consonant_cnt, unique_letters, has_hyphen, has_underscore,
    ]


# Cache for max_length normalisation in Selvaraj 11-feature extraction.
# Filled once at first call (across the training set).
_SELVARAJ_NORM = {}

def extract_selvaraj_11_stats(domain: str, wordninja_pkg=None) -> list:
    """
    11 statistical features from Selvaraj & Panjanathan 2024 §3.3, Table 3.

    Three categories (paper §3.3):
      (A) Domain length features  : 3 dims
      (B) Character ratio features: 4 dims (already in [0,1])
      (C) Word statistical features: 4 dims (need normalisation)

    Note: paper does not list exact features. We use the most common
    11-feature template for word-DGA detection, which matches the
    paper's verbal description.
    """
    if wordninja_pkg is None:
        try:
            import wordninja as wordninja_pkg
        except ImportError:
            return [0.0] * 11

    d = domain.lower()
    no_tld = strip_tld(d)
    clean = clean_alpha(no_tld)
    words = wordninja_pkg.split(clean) if clean else []

    # (A) Domain length features (3 dims)
    len_full   = len(d)
    len_no_tld = len(no_tld)
    n_words    = len(words)
    avg_word_len = sum(len(w) for w in words) / max(1, len(words))

    # (B) Character ratio features (4 dims, already in [0,1])
    n_alpha = sum(1 for c in clean if c.isalpha())
    n_digit = sum(1 for c in clean if c.isdigit())
    n_vowel = sum(1 for c in clean if c in VOWELS)
    n_cons  = n_alpha - n_vowel
    L = max(1, len(clean))
    ratio_alpha  = n_alpha / L
    ratio_digit  = n_digit / L
    ratio_vowel  = n_vowel / L
    ratio_cons   = n_cons  / L

    # (C) Word statistical features (4 dims, need normalisation)
    # max_len_word, min_len_word, ratio_char (chars-in-words / total chars),
    # ratio_word_repeated (how many words appear more than once)
    if words:
        max_word_len = max(len(w) for w in words)
        min_word_len = min(len(w) for w in words)
        ratio_char = sum(len(w) for w in words) / L
        from collections import Counter as _Counter
        wc = _Counter(words)
        n_rep = sum(1 for c in wc.values() if c > 1)
        ratio_word_rep = n_rep / max(1, len(words))
    else:
        max_word_len = 0
        min_word_len = 0
        ratio_char = 0.0
        ratio_word_rep = 0.0

    # Normalise integer-valued features by dataset max (cached in _SELVARAJ_NORM)
    # If norm not set yet, just return raw (will be filled by first scan)
    norm = _SELVARAJ_NORM
    return [
        len_full      / max(1, norm.get("len_full",      63)),
        len_no_tld    / max(1, norm.get("len_no_tld",    60)),
        n_words       / max(1, norm.get("n_words",      20)),
        avg_word_len  / max(1, norm.get("avg_word_len", 15)),
        ratio_alpha,
        ratio_digit,
        ratio_vowel,
        ratio_cons,
        max_word_len  / max(1, norm.get("max_word_len", 20)),
        min_word_len  / max(1, norm.get("min_word_len", 10)),
        ratio_char,
    ]


def fit_selvaraj_norm(train_domains, wordninja_pkg):
    """Compute per-feature max values on training set for normalisation."""
    keys = ["len_full", "len_no_tld", "n_words", "avg_word_len",
            "max_word_len", "min_word_len"]
    vals = {k: 1 for k in keys}
    for d in train_domains:
        d_low = d.lower()
        no_tld = strip_tld(d_low)
        clean = clean_alpha(no_tld)
        words = wordninja_pkg.split(clean) if clean else []
        vals["len_full"]   = max(vals["len_full"], len(d_low))
        vals["len_no_tld"] = max(vals["len_no_tld"], len(no_tld))
        if words:
            vals["n_words"]      = max(vals["n_words"], len(words))
            vals["avg_word_len"] = max(vals["avg_word_len"],
                                       sum(len(w) for w in words) // len(words))
            vals["max_word_len"] = max(vals["max_word_len"], max(len(w) for w in words))
            vals["min_word_len"] = max(vals["min_word_len"], min(len(w) for w in words))
    _SELVARAJ_NORM.update(vals)


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
    for fam, r in sorted(dr_results.items()):
        if fam == "_benign":
            continue
        lines.append(f"  {fam:<20} {r['total']:>6}  {r['tp']:>6}  {r['dr']:>7.2f}%")
    if "_benign" in dr_results:
        b = dr_results["_benign"]
        lines.append("  " + "-"*46)
        lines.append(f"  {'benign':<20} {b['total']:>6}  {b['tp']:>6}  {b['dr']:>7.2f}%")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
#  TOKENIZERS  (char-level and word-level)
# ══════════════════════════════════════════════════════════════════════════════
class CharTokenizer:
    """Character-level tokenizer for Chen 2023."""
    def __init__(self):
        chars = "abcdefghijklmnopqrstuvwxyz0123456789-._"
        # 0 = pad, 1 = unk, then chars
        self.stoi = {"<pad>": 0, "<unk>": 1}
        for c in chars:
            self.stoi[c] = len(self.stoi)
        self.vocab_size = len(self.stoi)

    def encode(self, s: str, max_len: int) -> list:
        s = s.lower()
        ids = [self.stoi.get(c, 1) for c in s[:max_len]]
        ids += [0] * (max_len - len(ids))
        return ids


class WordninjaTokenizer:
    """
    Word-level tokenizer using the wordninja Python package, exactly as
    described in Biros & Kantor 2025 (Section V).
    """
    def __init__(self):
        try:
            import wordninja
            self.wordninja = wordninja
        except ImportError:
            raise ImportError(
                "wordninja is required for Biros 2025 reproduction. "
                "Install with: pip install wordninja"
            )
        # 0 = pad, 1 = unk
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.vocab_size = 2

    def _split(self, domain: str):
        """Strip TLD, lowercase, then split with wordninja."""
        no_tld = strip_tld(domain).lower()
        no_tld = "".join(no_tld.split("."))  # remove inner dots if any
        return self.wordninja.split(no_tld) if no_tld else []

    def build_vocab(self, all_domains, max_vocab=56000):
        """Build vocabulary from training domains (paper: input_dim=56000)."""
        from collections import Counter
        counter = Counter()
        for d in all_domains:
            words = self._split(d)
            counter.update(words)
        for w, _ in counter.most_common(max_vocab - 2):
            self.stoi[w] = len(self.stoi)
        self.vocab_size = len(self.stoi)

    def encode(self, s: str, max_len: int) -> list:
        words = self._split(s)
        ids = [self.stoi.get(w, 1) for w in words[:max_len]]
        ids += [0] * (max_len - len(ids))
        return ids


class NgramTokenizer:
    """
    N-gram (e.g. 3-gram) tokenizer for Chen et al. 2023 Branch D.

    Per paper §3.4: split the domain (no TLD) into n-grams with stride 1,
    truncate/pad to fixed length 64. The vocabulary size is at most 39^n
    (39 valid chars: a-z, 0-9, '-', '.', '_').
    """
    def __init__(self, n=3):
        self.n = n
        # 0 = pad, 1 = unk
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.vocab_size = 2

    def _ngrams(self, domain):
        s = clean_alpha(strip_tld(domain).lower())
        if len(s) < self.n:
            return [s] if s else []
        return [s[i:i+self.n] for i in range(len(s) - self.n + 1)]

    def build_vocab(self, all_domains, max_vocab=60000):
        from collections import Counter
        counter = Counter()
        for d in all_domains:
            counter.update(self._ngrams(d))
        for g, _ in counter.most_common(max_vocab - 2):
            self.stoi[g] = len(self.stoi)
        self.vocab_size = len(self.stoi)

    def encode(self, s, max_len):
        grams = self._ngrams(s)
        ids = [self.stoi.get(g, 1) for g in grams[:max_len]]
        ids += [0] * (max_len - len(ids))
        return ids


class WordTokenizer:
    """Word-level tokenizer using greedy meaningful-word split."""
    def __init__(self, dictionary: set, min_freq: int = 1):
        self.dictionary = dictionary
        self.stoi = {"<pad>": 0, "<unk>": 1}

    def build_vocab(self, all_domains, max_vocab=20000):
        from collections import Counter
        counter = Counter()
        for d in all_domains:
            no_tld = strip_tld(d).lower()
            clean  = clean_alpha(no_tld)
            words, _, _ = split_meaningful_words(clean, self.dictionary)
            if not words:
                # If no meaningful split, fall back to single token = whole clean string
                words = [clean] if clean else []
            counter.update(words)
        for w, _ in counter.most_common(max_vocab - 2):
            self.stoi[w] = len(self.stoi)
        self.vocab_size = len(self.stoi)

    def encode(self, s: str, max_len: int) -> list:
        no_tld = strip_tld(s).lower()
        clean  = clean_alpha(no_tld)
        words, _, _ = split_meaningful_words(clean, self.dictionary)
        if not words:
            words = [clean] if clean else []
        ids = [self.stoi.get(w, 1) for w in words[:max_len]]
        ids += [0] * (max_len - len(ids))
        return ids

# ══════════════════════════════════════════════════════════════════════════════
#  GENERIC PYTORCH DATASET (for sequence-based models)
# ══════════════════════════════════════════════════════════════════════════════
class SeqDataset(Dataset):
    def __init__(self, domains, labels, tokenizer, max_len):
        self.x = torch.tensor(
            [tokenizer.encode(d, max_len) for d in domains],
            dtype=torch.long
        )
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class SelvarajDataset(Dataset):
    """Dataset that yields (word_ids, stats_11d, label) for Selvaraj 2024.

    Stats are pre-computed at construction time using the dataset-wide
    normalisation factors fitted via fit_selvaraj_norm() on the training set.
    """
    def __init__(self, domains, labels, tokenizer, max_len, wordninja_pkg):
        self.x = torch.tensor(
            [tokenizer.encode(d, max_len) for d in domains],
            dtype=torch.long
        )
        stats = [extract_selvaraj_11_stats(d, wordninja_pkg) for d in domains]
        self.s = torch.tensor(stats, dtype=torch.float)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.s[idx], self.y[idx]


class ChenDataset(Dataset):
    """Dataset that yields (x_char, x_word, x_3gram, stats_11d, label) for Chen 2023.

    Per paper §3.2: model needs FOUR inputs:
      • x_char  : 1-gram (char-level) ids — for Branch C (char-based AGDs)
      • x_word  : wordninja word ids — for Branch A (word distribution)
      • x_3gram : 3-gram ids — for Branch D (dict-based AGDs)
      • stats   : 11 statistical features — for Branch B
    """
    def __init__(self, domains, labels, char_tok, word_tok, ngram3_tok,
                 max_len_char, max_len_word, max_len_3gram, wordninja_pkg):
        self.x_char = torch.tensor(
            [char_tok.encode(d, max_len_char) for d in domains], dtype=torch.long)
        self.x_word = torch.tensor(
            [word_tok.encode(d, max_len_word) for d in domains], dtype=torch.long)
        self.x_3gram = torch.tensor(
            [ngram3_tok.encode(d, max_len_3gram) for d in domains], dtype=torch.long)
        stats = [extract_selvaraj_11_stats(d, wordninja_pkg) for d in domains]
        self.s = torch.tensor(stats, dtype=torch.float)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return (self.x_char[idx], self.x_word[idx], self.x_3gram[idx],
                self.s[idx], self.y[idx])


class SelvarajDataset(Dataset):
    """Dataset that yields (word_ids, stats_11d, label) for Selvaraj 2024.

    Stats are pre-computed at construction time using the dataset-wide
    normalisation factors fitted via fit_selvaraj_norm() on the training set.
    """
    def __init__(self, domains, labels, tokenizer, max_len, wordninja_pkg):
        self.x = torch.tensor(
            [tokenizer.encode(d, max_len) for d in domains],
            dtype=torch.long
        )
        stats = [extract_selvaraj_11_stats(d, wordninja_pkg) for d in domains]
        self.s = torch.tensor(stats, dtype=torch.float)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.s[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
class SelvarajRCNN_BiLSTM(nn.Module):
    """
    Selvaraj & Panjanathan 2024 (WordDGA) — Paper-exact reproduction.

    From the paper (MDPI Informatics 2024, 11(4), 92, Sections 3.1-3.4):

      INPUT BRANCH A — Word embedding (ELMo or FastText) of wordninja tokens
        • wordninja.split(domain_no_tld) → list of meaningful words
        • Each word → 1024-dim ELMo context-sensitive vector (paper)
          [we replace ELMo with FastText / random embedding 1024-dim as fallback]
        • Word vectors aggregated → 128-dim word distribution features

      INPUT BRANCH B — n-gram BiLSTM features
        • Generate 3-grams to 7-grams of the domain (paper §3.1.1)
        • Sequence length fixed at 64 (paper §3.1.1, "RFC max label length 63")
        • BiLSTM over n-gram sequence → 128-dim sequence features

      INPUT BRANCH C — 11 statistical features (paper Table 3, §3.3)
        • length, char-ratio, word stats; transformed to 8-dim via FC

      FUSION: concat (128 + 128 + 8) = 264-dim
        • Then RCNN-BiLSTM-Dense head (paper Figure 3)

    Adaptations vs paper:
      • ELMo replaced by 1024-dim random word embedding (paper-exact ELMo
        requires AllenNLP + multi-GB downloads + heavy preprocessing)
      • cWGAN over-sampling skipped (this is the OFFENSIVE part of WordDGA
        for generating adversarial examples; we keep only the DETECTOR)
    """
    def __init__(self, vocab_size, embed_dim=1024):
        super().__init__()
        # Branch A: word embedding (1024-dim as paper §3.1.1 specifies ELMo size)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Compress 1024-dim to 128-dim word features (paper: "aggregate word
        # features to generate 128-dimensional word distribution features")
        self.word_proj = nn.Linear(embed_dim, 128)

        # Branch B: BiLSTM over n-gram sequence (using the same wordninja word
        # ids as the n-gram sequence proxy; paper's n-gram BiLSTM operates on
        # 3-7-gram char sequences but the architecture is the same)
        self.ngram_bilstm = nn.LSTM(embed_dim, 64, batch_first=True,
                                    bidirectional=True)   # 64*2 = 128-dim out

        # Branch C: 11 statistical features → 8-dim (paper §3.3)
        self.stat_fc = nn.Linear(11, 8)

        # Fusion + RCNN-BiLSTM head (paper Figure 3)
        # Concatenated input: 128 + 128 + 8 = 264-dim per position
        # We treat fusion as a single feature vector → Dense head
        fused_dim = 128 + 128 + 8
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )
        self.relu = nn.ReLU()

    def forward(self, x, stats=None):
        """
        x     : (B, L) long — wordninja-encoded word ids, L=64
        stats : (B, 11) float — pre-computed statistical features (paper §3.3)
                If None, we use a zero vector (training will rely only on
                embedding + n-gram BiLSTM).
        """
        pad_mask = (x == 0).unsqueeze(-1)              # (B, L, 1)
        e = self.embed(x)                              # (B, L, 1024)

        # Branch A: aggregate word features (masked mean → 128-dim)
        mask_f = (~pad_mask.squeeze(-1)).float().unsqueeze(-1)   # (B, L, 1)
        word_sum = (e * mask_f).sum(dim=1)                       # (B, 1024)
        word_count = mask_f.sum(dim=1).clamp(min=1.0)            # (B, 1)
        word_mean = word_sum / word_count                        # (B, 1024)
        feat_word = self.relu(self.word_proj(word_mean))         # (B, 128)

        # Branch B: BiLSTM over n-gram sequence, mask padding then max-pool
        out, _ = self.ngram_bilstm(e)                            # (B, L, 128)
        out = out.masked_fill(pad_mask, -1e4)
        feat_ngram, _ = out.max(dim=1)                           # (B, 128)

        # Branch C: 11 statistical features → 8-dim
        if stats is None:
            stats = torch.zeros(x.size(0), 11, device=x.device, dtype=torch.float)
        feat_stat = self.relu(self.stat_fc(stats))               # (B, 8)

        # Fusion
        fused = torch.cat([feat_word, feat_ngram, feat_stat], dim=1)   # (B, 264)
        return self.head(fused)


class ChenCNN_LSTM(nn.Module):
    """
    Chen, Lang, Chen & Xie 2023 (MDPI Applied Sciences 13(7), 4406) —
    "Detection of AGDs with Feature Fusion of Meaningful Word Segmentation
    and N-Gram Sequences".

    Paper-exact reproduction (Sections 3.2-3.5):

      FOUR BRANCHES → fused into 392-dim vector → FC → sigmoid

      Branch A — Word distribution features (§3.3)
        • wordninja segmentation → word sequence
        • Each word → 1024-dim ELMo embedding (substituted with 1024-dim
          random embedding for offline reproduction)
        • Compute MEAN + STD across word features → 2048-dim
        • Linear 2048 → 128-dim word distribution feature

      Branch B — 11 statistical features (§3.5)
        • Domain length + char ratio + word stats + POS distribution
        • Linear 11 → 8-dim

      Branch C — 1-gram LSTM sequence (§3.4)
        • Sequence length 64 (paper)
        • Char embedding + LSTM → 128-dim (mainly for CHAR-based AGDs)

      Branch D — 3-gram LSTM sequence (§3.4)
        • 3-gram sequence length 64 (paper)
        • 3-gram embedding + LSTM → 128-dim (mainly for DICT-based AGDs)

      FUSION: concat (128 + 8 + 128 + 128) = 392-dim → FC → 1 (sigmoid)

    Note: the class name is kept as `ChenCNN_LSTM` for backward compatibility
    even though the architecture has no CNN. This was an earlier mistake
    (we initially implemented a different paper also called "Chen 2023").

    Forward signature:
        forward(x, stats=None, x_3gram=None)
        • x       : (B, 64) 1-gram char ids
        • x_3gram : (B, 64) 3-gram ids
        • stats   : (B, 11) 11 statistical features
        Branch A uses the same word ids as Selvaraj (from WordninjaTokenizer);
        but for self-contained Chen reproduction we re-embed the 1-gram char
        sequence and treat it as the "word feature" via mean-pool. The full
        ELMo word-embedding path is omitted (would require AllenNLP).
    """
    def __init__(self, vocab_size, embed_dim=128,
                 word_vocab_size=56000, word_embed_dim=1024,
                 ngram3_vocab_size=59320):
        super().__init__()

        # Branch A: word embedding (1024-dim as paper §3.3 specifies ELMo size)
        # Paper: ELMo → mean + std → Linear 2048→128
        self.word_embed = nn.Embedding(word_vocab_size, word_embed_dim, padding_idx=0)
        self.word_proj  = nn.Linear(word_embed_dim * 2, 128)   # mean + std → 128

        # Branch B: 11 statistical features → 8-dim (paper §3.5)
        self.stat_fc = nn.Linear(11, 8)

        # Branch C: 1-gram LSTM (paper §3.4, mainly for char-based AGDs)
        self.embed_1gram = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm_1gram  = nn.LSTM(embed_dim, 64, batch_first=True,
                                   bidirectional=True)         # → 128-dim

        # Branch D: 3-gram LSTM (paper §3.4, mainly for dict-based AGDs)
        self.embed_3gram = nn.Embedding(ngram3_vocab_size, embed_dim, padding_idx=0)
        self.lstm_3gram  = nn.LSTM(embed_dim, 64, batch_first=True,
                                   bidirectional=True)         # → 128-dim

        # Fusion + FC head (paper §3.2 + Figure 2)
        # Total fused dim: 128 + 8 + 128 + 128 = 392
        fused_dim = 128 + 8 + 128 + 128
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )
        self.relu = nn.ReLU()

    def forward(self, x, stats=None, x_3gram=None, x_word=None):
        """
        x       : (B, 64) 1-gram char ids
        x_3gram : (B, 64) 3-gram token ids — if None, branch D is zeroed
        stats   : (B, 11) statistical features — if None, branch B is zeroed
        x_word  : (B, L_word) word ids (wordninja) — if None, branch A is zeroed
        """
        B = x.size(0)

        # Branch A: word distribution features (mean + std) → 128-dim
        if x_word is not None:
            pad_mask_w = (x_word == 0).unsqueeze(-1)             # (B, L, 1)
            mask_f = (~pad_mask_w.squeeze(-1)).float().unsqueeze(-1)
            ew = self.word_embed(x_word)                         # (B, L, 1024)
            ew_masked = ew * mask_f
            count = mask_f.sum(dim=1).clamp(min=1.0)              # (B, 1)
            word_mean = ew_masked.sum(dim=1) / count              # (B, 1024)
            # Std across word features
            sq = (ew_masked - word_mean.unsqueeze(1) * mask_f) ** 2
            word_var = sq.sum(dim=1) / count
            word_std = torch.sqrt(word_var.clamp(min=1e-8))       # (B, 1024)
            word_features = torch.cat([word_mean, word_std], dim=1)   # (B, 2048)
            feat_word = self.relu(self.word_proj(word_features))      # (B, 128)
        else:
            feat_word = torch.zeros(B, 128, device=x.device, dtype=torch.float)

        # Branch B: 11 statistical features → 8-dim
        if stats is None:
            stats = torch.zeros(B, 11, device=x.device, dtype=torch.float)
        feat_stat = self.relu(self.stat_fc(stats))                # (B, 8)

        # Branch C: 1-gram LSTM (char-based AGDs)
        pad_mask_c = (x == 0).unsqueeze(-1)
        e1 = self.embed_1gram(x)                                  # (B, 64, 128)
        out1, _ = self.lstm_1gram(e1)                             # (B, 64, 128)
        out1 = out1.masked_fill(pad_mask_c, -1e4)
        feat_1gram, _ = out1.max(dim=1)                           # (B, 128)

        # Branch D: 3-gram LSTM (dict-based AGDs)
        if x_3gram is not None:
            pad_mask_3 = (x_3gram == 0).unsqueeze(-1)
            e3 = self.embed_3gram(x_3gram)                        # (B, 64, 128)
            out3, _ = self.lstm_3gram(e3)                         # (B, 64, 128)
            out3 = out3.masked_fill(pad_mask_3, -1e4)
            feat_3gram, _ = out3.max(dim=1)                       # (B, 128)
        else:
            feat_3gram = torch.zeros(B, 128, device=x.device, dtype=torch.float)

        # Fusion + classification
        fused = torch.cat([feat_word, feat_stat, feat_1gram, feat_3gram], dim=1)
        return self.head(fused)


class BirosCNN(nn.Module):
    """
    Biros & Kantor 2025 (JTIT FITCE 2024) — CNN for word-based DGA detection.
    Paper-exact reproduction of Listing 2 (Keras code in the paper).

    The original paper trains TWO SEPARATE models (Listing 1 = LSTM, Listing 2
    = CNN), each reported as an independent row in Table 4 (LSTM 95.36%, CNN
    98.30%). We reproduce ONLY the CNN model here, since it is the stronger
    of the two reported variants and is suitable as a single baseline.

    Architecture (paper Listing 2):
      • Embedding(input_dim=56000, output_dim=128, input_length=64)
      • Conv1D(filters=200, kernel=4, padding='same', glorot_normal init)
      • Dropout(0.5)
      • MaxPool1D(pool=2, strides=2)
      • Conv1D(filters=100, kernel=2, padding='same')
      • MaxPool1D(pool=2, strides=2)
      • Dropout(0.5)
      • Flatten
      • Dense(100, ReLU, glorot_normal)
      • Dropout(0.5)
      • Dense(10, ReLU, glorot_normal)
      • Dense(1, sigmoid, glorot_normal)
      • Adam (lr=0.001 default), binary_crossentropy, 10 epochs, batch 128

    Adaptations:
      • Last Dense → 2 units + softmax for CE-loss compatibility (equivalent
        to 1-unit sigmoid for binary classification)
      • Input is WORD ids (via wordninja tokenizer), max_words=64, vocab≤56000
      • Extended to 50 epochs / patience 5 for unified comparison
    """
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        # input_length = 64 words (from paper)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Conv1: 200 filters, kernel=4, padding='same'
        self.conv1 = nn.Conv1d(embed_dim, 200, kernel_size=4, padding=2)
        self.dropout1 = nn.Dropout(0.5)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Conv2: 100 filters, kernel=2, padding='same'
        self.conv2 = nn.Conv1d(200, 100, kernel_size=2, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.5)
        # Dense layers
        # After paper-exact path: 64 → conv1(same)=64 → pool→32 → conv2(same)=32 → pool→16
        # flatten = 100 * 16 = 1600
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100 * 16, 100)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 2)
        self.relu = nn.ReLU()
        # Glorot-normal init (paper)
        for m in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
            if hasattr(m, "weight"):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e = self.embed(x).transpose(1, 2)              # (B, 128, 64)
        h = self.relu(self.conv1(e))                   # (B, 200, ~64)
        h = self.dropout1(h)
        h = self.pool1(h)                              # (B, 200, ~32)
        h = self.relu(self.conv2(h))                   # (B, 100, ~32)
        h = self.pool2(h)                              # (B, 100, ~16)
        h = self.dropout2(h)
        h = self.flatten(h)                            # (B, 1600)
        # Handle minor length mismatches gracefully
        target = 100 * 16
        if h.size(1) != target:
            h = h[:, :target] if h.size(1) > target else \
                torch.nn.functional.pad(h, (0, target - h.size(1)))
        h = self.relu(self.fc1(h)); h = self.dropout3(h)
        h = self.relu(self.fc2(h))
        return self.fc3(h)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / EVAL HELPERS for deep-learning baselines
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, scaler, criterion, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)
    bar_w      = 30

    for step, batch in enumerate(loader, 1):
        # Handle different batch sizes:
        #  2-tuple: (x, y)             - generic SeqDataset
        #  3-tuple: (x, stats, y)      - SelvarajDataset
        #  5-tuple: (x_c, x_w, x_3, s, y) - ChenDataset
        if len(batch) == 5:
            x_c, x_w, x_3, stats, y = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(x_c, stats=stats, x_3gram=x_3, x_word=x_w)
                loss   = criterion(logits, y)
        elif len(batch) == 3:
            x, stats, y = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(x, stats)
                loss   = criterion(logits, y)
        else:
            x, y = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        pct    = step / n_batches
        filled = int(bar_w * pct)
        bar    = "█"*filled + "░"*(bar_w-filled)
        print(f"\r  Epoch {epoch:>2}/{total_epochs}  [{bar}] {step:>4}/{n_batches}  "
              f"loss={total_loss/step:.4f}", end="", flush=True)
    print()
    return total_loss / n_batches

def evaluate_torch(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                x_c, x_w, x_3, stats, y = batch
                x_c, x_w, x_3, stats = x_c.to(DEVICE), x_w.to(DEVICE), x_3.to(DEVICE), stats.to(DEVICE)
                with torch.amp.autocast("cuda"):
                    logits = model(x_c, stats=stats, x_3gram=x_3, x_word=x_w)
            elif len(batch) == 3:
                x, stats, y = batch
                x, stats = x.to(DEVICE), stats.to(DEVICE)
                with torch.amp.autocast("cuda"):
                    logits = model(x, stats)
            else:
                x, y = batch
                x = x.to(DEVICE)
                with torch.amp.autocast("cuda"):
                    logits = model(x)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.numpy())
    return np.array(labels), np.array(preds)

def detection_rate_per_family_torch(model, df_test, tokenizer, max_len,
                                     use_selvaraj_stats=False, wordninja_pkg=None,
                                     chen_mode=False, chen_kwargs=None):
    """Per-family Detection Rate on the TESTING subset for deep-learning baselines.

    The testing subset contains BOTH DGA samples and benign samples. We compute:
        - per-DGA-family DR (TP / total DGA for that family)
        - benign-side FPR (FP / total benign rows) stored under key "_benign"
    """
    def _build_loader(domains, labels):
        if chen_mode and chen_kwargs is not None:
            ds = ChenDataset(
                domains, labels,
                chen_kwargs["char_tok"], chen_kwargs["word_tok"],
                chen_kwargs["ngram3_tok"],
                chen_kwargs["max_len_char"], chen_kwargs["max_len_word"],
                chen_kwargs["max_len_3gram"], chen_kwargs["wordninja_pkg"]
            )
        elif use_selvaraj_stats and wordninja_pkg is not None:
            ds = SelvarajDataset(domains, labels, tokenizer, max_len, wordninja_pkg)
        else:
            ds = SeqDataset(domains, labels, tokenizer, max_len)
        return DataLoader(ds, batch_size=256, shuffle=False, pin_memory=True)

    results = {}

    # ── Per-DGA-family Detection Rate ──────────────────────────────
    dga_df = df_test[df_test["label"] == 1].copy()
    for fam, grp in dga_df.groupby("family"):
        loader = _build_loader(grp["domain"].tolist(), grp["label"].tolist())
        y_true, y_pred = evaluate_torch(model, loader)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        total = len(y_true)
        results[fam] = {"total": total, "tp": tp,
                        "dr": tp/total*100 if total > 0 else 0.0}

    # ── Benign-side classification accuracy (testing subset includes benign rows) ──
    # For consistency with DGA families, "tp" here means "correctly classified as benign"
    # (i.e., the True Negatives), and "dr" is TN/total_benign.
    benign_df = df_test[df_test["label"] == 0].copy()
    if len(benign_df) > 0:
        loader = _build_loader(benign_df["domain"].tolist(), benign_df["label"].tolist())
        y_true, y_pred = evaluate_torch(model, loader)
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        total = len(y_true)
        results["_benign"] = {
            "total": total,
            "tp":    tn,
            "dr":    tn/total*100 if total > 0 else 0.0,
            "fp":    fp,
            "fpr":   fp/total*100 if total > 0 else 0.0,
        }
    return results

# ══════════════════════════════════════════════════════════════════════════════
#  RUN A SINGLE DEEP-LEARNING BASELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_deep_baseline(name, model_cls, tokenizer, max_len, hp, train_df, val_df, test_df,
                      use_selvaraj_stats=False, wordninja_pkg=None,
                      chen_mode=False, chen_kwargs=None):
    print(f"\n{'='*70}\n  BASELINE: {name}\n{'='*70}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    epoch_log_path = RESULTS_DIR / f"baseline_{name}_epoch_log.txt"
    final_log_path = RESULTS_DIR / f"baseline_{name}_final.txt"
    log_f = open(epoch_log_path, "w", encoding="utf-8", buffering=1)
    log_f.write(f"BASELINE: {name}\n")
    log_f.write(f"Hyperparams: {hp}\n\n")

    # ── Datasets ───────────────────────────────────────────────────
    if chen_mode and chen_kwargs is not None:
        # Chen et al. 2023 needs 4 inputs: char, word, 3-gram, stats
        ds_cls = lambda dom, lab: ChenDataset(
            dom, lab,
            chen_kwargs["char_tok"],   chen_kwargs["word_tok"],
            chen_kwargs["ngram3_tok"],
            chen_kwargs["max_len_char"], chen_kwargs["max_len_word"],
            chen_kwargs["max_len_3gram"], chen_kwargs["wordninja_pkg"]
        )
    elif use_selvaraj_stats and wordninja_pkg is not None:
        ds_cls = lambda dom, lab: SelvarajDataset(dom, lab, tokenizer, max_len, wordninja_pkg)
    else:
        ds_cls = lambda dom, lab: SeqDataset(dom, lab, tokenizer, max_len)
    train_ds = ds_cls(train_df["domain"].tolist(), train_df["label"].tolist())
    val_ds   = ds_cls(val_df["domain"].tolist(),   val_df["label"].tolist())
    test_ds  = ds_cls(test_df["domain"].tolist(),  test_df["label"].tolist())

    train_loader = DataLoader(train_ds, batch_size=hp["batch"], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=hp["batch"], shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=hp["batch"], shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────
    set_seed()
    if chen_mode and chen_kwargs is not None:
        model = model_cls(
            vocab_size=chen_kwargs["char_tok"].vocab_size,
            embed_dim=hp["embed_dim"],
            word_vocab_size=chen_kwargs["word_tok"].vocab_size,
            word_embed_dim=hp.get("word_embed_dim", 1024),
            ngram3_vocab_size=chen_kwargs["ngram3_tok"].vocab_size,
        ).to(DEVICE)
    else:
        model = model_cls(vocab_size=tokenizer.vocab_size,
                          embed_dim=hp["embed_dim"]).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=hp["lr"])
    scaler    = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()

    # Header
    hdr = (f"\n  {'Ep':>3}  {'Loss':>9}"
           f"  {'TrACC':>6} {'TrF1':>6} {'TrFPR':>6} {'TrFNR':>6}"
           f"  {'VaACC':>6} {'VaF1':>6} {'VaFPR':>6} {'VaFNR':>6}  {'Status'}")
    print(hdr); log_f.write(hdr + "\n")
    print("  "+ "-"*108); log_f.write("  "+ "-"*108 + "\n")

    best_acc, best_epoch, patience_cnt = 0.0, 0, 0
    best_state = None
    epoch_hist = []
    t_start    = time.time()

    for epoch in range(1, hp["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion,
                               epoch, hp["epochs"])
        y_tr, p_tr = evaluate_torch(model, train_loader)
        tr_m       = compute_metrics(y_tr, p_tr)
        y_va, p_va = evaluate_torch(model, val_loader)
        va_m       = compute_metrics(y_va, p_va)

        if va_m["acc"] > best_acc:
            best_acc, best_epoch = va_m["acc"], epoch
            best_state = deepcopy(model.state_dict())
            patience_cnt = 0
            status = "✓ best"
        else:
            patience_cnt += 1
            status = f"patience {patience_cnt}/{hp['patience']}"

        line = (f"  {epoch:>3}  {loss:>9.4f}"
                f"  {tr_m['acc']:>5.2f}% {tr_m['f1']:>5.2f}% {tr_m['fpr']:>5.2f}% {tr_m['fnr']:>5.2f}%"
                f"  {va_m['acc']:>5.2f}% {va_m['f1']:>5.2f}% {va_m['fpr']:>5.2f}% {va_m['fnr']:>5.2f}%"
                f"  {status}")
        print(line); log_f.write(line + "\n")
        epoch_hist.append({
            "epoch": epoch, "loss": round(loss,4),
            "train_acc": tr_m["acc"], "train_f1": tr_m["f1"],
            "train_fpr": tr_m["fpr"], "train_fnr": tr_m["fnr"],
            "val_acc": va_m["acc"], "val_f1": va_m["f1"],
            "val_fpr": va_m["fpr"], "val_fnr": va_m["fnr"],
            "status": status,
        })

        if hp["patience"] > 0 and patience_cnt >= hp["patience"]:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            log_f.write(f"  Early stopping at epoch {epoch} (best: {best_epoch})\n")
            break

    total_time = time.time() - t_start
    log_f.write(f"\nTraining time: {total_time:.0f}s  Best epoch: {best_epoch}\n")
    log_f.close()
    pd.DataFrame(epoch_hist).to_csv(
        RESULTS_DIR / f"baseline_{name}_epoch_log.csv", index=False
    )

    # ── Best model evaluation ──────────────────────────────────────
    model.load_state_dict(best_state)
    save_dir = MODELS_DIR / f"baseline_{name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state,
                "vocab_size": tokenizer.vocab_size,
                "max_len":    max_len,
                "hp":         hp},
               save_dir / "best_model.pt")

    y_tr, p_tr = evaluate_torch(model, train_loader); train_m = compute_metrics(y_tr, p_tr)
    y_va, p_va = evaluate_torch(model, val_loader);   val_m   = compute_metrics(y_va, p_va)
    y_te, p_te = evaluate_torch(model, test_loader);  test_m  = compute_metrics(y_te, p_te)

    dr = detection_rate_per_family_torch(model, test_df, tokenizer, max_len,
                                          use_selvaraj_stats=use_selvaraj_stats,
                                          wordninja_pkg=wordninja_pkg,
                                          chen_mode=chen_mode,
                                          chen_kwargs=chen_kwargs)

    final_text = "\n".join([
        "="*70, f"  {name} — Final Results (best epoch {best_epoch}, time {total_time:.0f}s)",
        "="*70, "",
        fmt_metrics_block("TRAIN (best model)", train_m), "",
        fmt_metrics_block("VAL   (best model)", val_m),   "",
        fmt_metrics_block("TEST", test_m), "",
        "  ── Per-family Detection Rate (testing subset) ──",
        fmt_dr_block(dr), "",
    ])
    print(final_text)
    with open(final_log_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    return dict(name=name, best_epoch=best_epoch, train_time=total_time,
                train=train_m, val=val_m, test=test_m, dr=dr)

# ══════════════════════════════════════════════════════════════════════════════
#  CHARAN 2020 — 8 lexical features + ensemble (GradientBoosting ≈ C5.0)
# ══════════════════════════════════════════════════════════════════════════════
def run_charan_2020(dicts, train_df, val_df, test_df):
    """Charan 2020 — re-implementation.

    The original paper proposes 15 features (8 lexical + 7 network/WHOIS) with
    a C5.0 ensemble that uses boosting. WHOIS features are dropped for
    fairness on the offline dataset, so we keep the 8 lexical features only.

    For the classifier we use sklearn's GradientBoostingClassifier, which is
    the closest open-source equivalent of C5.0 with boosting (chefboost's
    plain C4.5 lacks boosting and produced much weaker results).
    """
    print(f"\n{'='*70}\n  BASELINE: charan_2020 (Gradient Boosting, 8 lexical features)\n{'='*70}")
    name = "charan_2020"

    from sklearn.ensemble import GradientBoostingClassifier

    feat_cols = ["word_count", "length", "syllables", "vowels",
                 "consonants", "unique_letters", "hyphen", "underscore"]

    def build_xy(df):
        X = np.array([extract_charan_8_features(d, dicts) for d in df["domain"]])
        y = df["label"].values.astype(int)
        return X, y

    print("  Building 8-feature matrices ...")
    t0 = time.time()
    train_X, train_y = build_xy(train_df)
    val_X,   val_y   = build_xy(val_df)
    test_X,  test_y  = build_xy(test_df)
    print(f"  Done in {time.time()-t0:.0f}s  (train: {len(train_X)} rows)")

    t_start = time.time()
    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        random_state=SEED,
    )
    clf.fit(train_X, train_y)
    total_time = time.time() - t_start

    def predict(X):
        return clf.predict(X)

    # Evaluate
    train_m = compute_metrics(train_y, predict(train_X))
    val_m   = compute_metrics(val_y,   predict(val_X))
    test_m  = compute_metrics(test_y,  predict(test_X))

    # Per-family DR (testing subset includes both DGA and benign rows)
    dr_results = {}
    dga_test = test_df[test_df["label"] == 1].reset_index(drop=True)
    fam_X_all = np.array([extract_charan_8_features(d, dicts) for d in dga_test["domain"]])
    for fam in sorted(dga_test["family"].unique()):
        mask = dga_test["family"].values == fam
        if not mask.any(): continue
        pred  = predict(fam_X_all[mask])
        tp    = int(np.sum(pred == 1))
        total = int(mask.sum())
        dr_results[fam] = {"total": total, "tp": tp,
                           "dr": tp/total*100 if total>0 else 0.0}

    # Benign-side classification accuracy on the testing subset
    benign_test = test_df[test_df["label"] == 0].reset_index(drop=True)
    if len(benign_test) > 0:
        benign_X = np.array([extract_charan_8_features(d, dicts) for d in benign_test["domain"]])
        pred = predict(benign_X)
        tn = int(np.sum(pred == 0))
        fp = int(np.sum(pred == 1))
        total = len(benign_test)
        dr_results["_benign"] = {
            "total": total,
            "tp":    tn,
            "dr":    tn/total*100 if total > 0 else 0.0,
            "fp":    fp,
            "fpr":   fp/total*100 if total > 0 else 0.0,
        }

    final_text = "\n".join([
        "="*70, f"  {name} — Final Results (time {total_time:.0f}s)",
        "="*70, "",
        fmt_metrics_block("TRAIN", train_m), "",
        fmt_metrics_block("VAL",   val_m),   "",
        fmt_metrics_block("TEST",  test_m),  "",
        "  ── Per-family Detection Rate (testing subset) ──",
        fmt_dr_block(dr_results), "",
    ])
    print(final_text)
    save_dir = MODELS_DIR / f"baseline_{name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "best_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(RESULTS_DIR / f"baseline_{name}_final.txt", "w", encoding="utf-8") as f:
        f.write(final_text)

    return dict(name=name, best_epoch="—", train_time=total_time,
                train=train_m, val=val_m, test=test_m, dr=dr_results)

# ══════════════════════════════════════════════════════════════════════════════
#  VU & HOANG 2021 — J48 / Decision Tree with 11 features
# ══════════════════════════════════════════════════════════════════════════════
def run_vu_2021(dicts, train_df, val_df, test_df):
    """Vu & Hoang 2021 - J48 decision tree with 16 features (paper-exact)."""
    print(f"\n{'='*70}\n  BASELINE: vu_2021 (J48 / Decision Tree, 16 features)\n{'='*70}")
    name = "vu_2021"

    print("  Extracting 16 features for all splits (Vu & Hoang 2021) ...")
    t0 = time.time()
    train_X = np.array([extract_vu_16_features(d, dicts) for d in train_df["domain"]])
    val_X   = np.array([extract_vu_16_features(d, dicts) for d in val_df["domain"]])
    test_X  = np.array([extract_vu_16_features(d, dicts) for d in test_df["domain"]])
    train_y = train_df["label"].values
    val_y   = val_df["label"].values
    test_y  = test_df["label"].values
    print(f"  Done in {time.time()-t0:.0f}s   ({train_X.shape[1]} features)")

    t_start = time.time()
    # J48 ≈ C4.5: entropy criterion + reduced-error pruning (sklearn equivalent)
    clf = DecisionTreeClassifier(criterion="entropy",
                                 min_samples_leaf=2,
                                 random_state=SEED)
    clf.fit(train_X, train_y)
    total_time = time.time() - t_start

    train_m = compute_metrics(train_y, clf.predict(train_X))
    val_m   = compute_metrics(val_y,   clf.predict(val_X))
    test_m  = compute_metrics(test_y,  clf.predict(test_X))

    # Per-family DR (testing subset includes both DGA and benign rows)
    dr_results = {}
    dga_test = test_df[test_df["label"] == 1].reset_index(drop=True)
    feat_test_dga = np.array([extract_vu_16_features(d, dicts) for d in dga_test["domain"]])
    for fam in sorted(dga_test["family"].unique()):
        mask = dga_test["family"].values == fam
        if not mask.any(): continue
        pred = clf.predict(feat_test_dga[mask])
        tp = int(np.sum(pred == 1)); total = int(mask.sum())
        dr_results[fam] = {"total": total, "tp": tp,
                           "dr": tp/total*100 if total>0 else 0.0}

    # Benign-side classification accuracy on the testing subset
    benign_test = test_df[test_df["label"] == 0].reset_index(drop=True)
    if len(benign_test) > 0:
        benign_X = np.array([extract_vu_16_features(d, dicts) for d in benign_test["domain"]])
        pred = clf.predict(benign_X)
        tn = int(np.sum(pred == 0))
        fp = int(np.sum(pred == 1))
        total = len(benign_test)
        dr_results["_benign"] = {
            "total": total,
            "tp":    tn,
            "dr":    tn/total*100 if total > 0 else 0.0,
            "fp":    fp,
            "fpr":   fp/total*100 if total > 0 else 0.0,
        }

    final_text = "\n".join([
        "="*70, f"  {name} — Final Results (time {total_time:.0f}s)",
        "="*70, "",
        fmt_metrics_block("TRAIN", train_m), "",
        fmt_metrics_block("VAL",   val_m),   "",
        fmt_metrics_block("TEST",  test_m),  "",
        "  ── Per-family Detection Rate (testing subset) ──",
        fmt_dr_block(dr_results), "",
    ])
    print(final_text)
    save_dir = MODELS_DIR / f"baseline_{name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "best_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(RESULTS_DIR / f"baseline_{name}_final.txt", "w", encoding="utf-8") as f:
        f.write(final_text)

    return dict(name=name, best_epoch="—", train_time=total_time,
                train=train_m, val=val_m, test=test_m, dr=dr_results)

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLES  (Table 4 + Table 6 mirrors)
# ══════════════════════════════════════════════════════════════════════════════
def build_summary(all_results):
    lines = []
    # Table 4 — Performance comparison on VALIDATION subset
    lines += ["", "="*82,
              "  TABLE 4 — Performance comparison (evaluated on VALIDATION subset)",
              "="*82,
              f"  {'Baseline':<26} {'ACC':>8} {'F1':>8} {'FPR':>8} {'FNR':>8} {'Time':>8}",
              "  " + "-"*76]
    for r in all_results:
        lines.append(
            f"  {r['name']:<26} {r['val']['acc']:>7.2f}% "
            f"{r['val']['f1']:>7.2f}% {r['val']['fpr']:>7.2f}% "
            f"{r['val']['fnr']:>7.2f}% {r['train_time']:>7.0f}s"
        )

    # Table 6 — DR per family on TESTING subset
    lines += ["", "="*92,
              "  TABLE 6 — Detection rate comparison (evaluated on TESTING subset)",
              "="*92]
    header = f"  {'Family':<20} {'Total':>6}"
    for r in all_results:
        header += f"  {r['name'][:14]:>14}"
    lines.append(header)
    lines.append("  " + "-"*(20+8+16*len(all_results)))

    # Collect DGA families across baselines (exclude the special _benign key)
    all_fams = sorted({fam for r in all_results for fam in r["dr"].keys()} - {"_benign"})
    for fam in all_fams:
        # Pick total from first non-zero result
        total = 0
        for r in all_results:
            if fam in r["dr"]:
                total = r["dr"][fam]["total"]
                break
        row = f"  {fam:<20} {total:>6}"
        for r in all_results:
            if fam in r["dr"]:
                row += f"  {r['dr'][fam]['dr']:>13.2f}%"
            else:
                row += f"  {'—':>14}"
        lines.append(row)

    # Benign-side classification accuracy row (testing subset includes ~11,700 benign rows)
    if any("_benign" in r["dr"] for r in all_results):
        # Pick benign total from first baseline that has it
        b_total = 0
        for r in all_results:
            if "_benign" in r["dr"]:
                b_total = r["dr"]["_benign"]["total"]
                break
        lines.append("  " + "-"*(20+8+16*len(all_results)))
        row = f"  {'benign':<20} {b_total:>6}"
        for r in all_results:
            if "_benign" in r["dr"]:
                row += f"  {r['dr']['_benign']['dr']:>13.2f}%"
            else:
                row += f"  {'—':>14}"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    master_log = open(RESULTS_DIR / "baselines_full_run.txt", "w",
                      encoding="utf-8", buffering=1)
    def dual(msg=""):
        print(msg); master_log.write(msg+"\n")

    dual("="*70)
    dual("  BASELINE COMPARISON — Re-implementing 5 prior studies")
    dual("  ")
    dual("  REPRODUCTION DISCLAIMER:")
    dual("  Without official source code from the original authors, exact")
    dual("  reproduction is not possible. The implementations below follow")
    dual("  the published paper descriptions as closely as possible. Where a")
    dual("  paper omits an architectural detail (e.g. exact filter count),")
    dual("  reasonable defaults are used and marked as [best-effort] in the")
    dual("  model docstrings. All baselines are evaluated on the SAME")
    dual("  train/val/test split as the BERT scenarios, which ensures the")
    dual("  comparison is fair across methods.")
    dual("  ")
    dual(f"  Device : {DEVICE}")
    if torch.cuda.is_available():
        dual(f"  GPU    : {torch.cuda.get_device_name(0)}")
    dual(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    dual("="*70)

    # ── Load dictionaries ──────────────────────────────────────────
    # All baselines (Selvaraj, Chen, Charan, Biros, Vu) now use the SAME
    # shared dictionary/ folder for fair comparison.
    dual("\n[setup] Loading dictionaries (shared dictionary/ folder) ...")
    dicts = {
        # Used by Selvaraj 2024, Chen 2023, Charan 2020
        "dictDGA":       load_dict("dictDGA.txt"),
        "dictOnlyNLTK":  load_dict("dictOnlyNLTK.txt"),
        "dictOnlyDGA":   load_dict("dictOnlyDGA.txt"),
        "english_dict":  load_dict("dictEng.txt"),

        # Used by Vu & Hoang 2021 — re-using the same shared dictionaries
        # for fair comparison with the other baselines. Mapping to Vu's
        # paper notation:
        #   • english_dict   → dictEng.txt
        #   • dga_dict       → dictDGA.txt        (DGA word list)
        #   • noun_dict      → dictNounsDGA.txt   (DGA-noun word list)
        #   • verb_dict      → dictVerbsDGA.txt   (DGA-verb word list)
        #   • adj_dict       → dictAdjsDGA.txt    (DGA-adjective word list)
        #   • private_dict   → dictOnlyDGA.txt    (DGA-only/private words)
        "dga_dict":      load_dict("dictDGA.txt"),
        "noun_dict":     load_dict("dictNounsDGA.txt"),
        "verb_dict":     load_dict("dictVerbsDGA.txt"),
        "adj_dict":      load_dict("dictAdjsDGA.txt"),
        "private_dict":  load_dict("dictOnlyDGA.txt"),
    }

    # Summary of loaded dictionaries
    dual("  Dictionary sizes:")
    for k in ["english_dict", "dictDGA", "dictOnlyNLTK", "dictOnlyDGA",
              "dga_dict", "noun_dict", "verb_dict", "adj_dict", "private_dict"]:
        if k in dicts:
            dual(f"    {k:<20} : {len(dicts[k]):>7,} words")

    # ── Load CSVs ──────────────────────────────────────────────────
    dual("[setup] Loading dataout/ CSVs ...")
    train_df = pd.read_csv(DATAOUT_DIR / "train.csv")
    val_df   = pd.read_csv(DATAOUT_DIR / "val.csv")
    test_df  = pd.read_csv(DATAOUT_DIR / "test.csv")
    dual(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

    # ── Build tokenizers ───────────────────────────────────────────
    dual("[setup] Building tokenizers ...")
    char_tok = CharTokenizer()
    word_tok = WordTokenizer(dictionary=dicts["dictDGA"])
    word_tok.build_vocab(train_df["domain"].tolist())
    # Wordninja tokenizer for Selvaraj 2024 + Biros 2025 + Chen 2023 Branch A
    wn_tok = WordninjaTokenizer()
    wn_tok.build_vocab(train_df["domain"].tolist(), max_vocab=56000)
    # 3-gram tokenizer for Chen 2023 Branch D (paper §3.4)
    ngram3_tok = NgramTokenizer(n=3)
    ngram3_tok.build_vocab(train_df["domain"].tolist(), max_vocab=60000)
    dual(f"  Char vocab     : {char_tok.vocab_size}")
    dual(f"  Word vocab     : {word_tok.vocab_size}")
    dual(f"  Wordninja vocab: {wn_tok.vocab_size}")
    dual(f"  3-gram vocab   : {ngram3_tok.vocab_size}")

    # Fit normalisation factors for Selvaraj's 11 statistical features
    dual("[setup] Fitting Selvaraj normalisation factors on train set ...")
    import wordninja as _wn
    fit_selvaraj_norm(train_df["domain"].tolist(), _wn)
    dual(f"  Norm factors: {_SELVARAJ_NORM}")

    all_results = []

    # ── 1. Selvaraj 2024 (RCNN-BiLSTM, paper-exact: wordninja + 11 stats) ─
    r = run_deep_baseline("selvaraj_2024", SelvarajRCNN_BiLSTM, wn_tok,
                          HP["selvaraj_2024"]["max_words"],
                          HP["selvaraj_2024"], train_df, val_df, test_df,
                          use_selvaraj_stats=True, wordninja_pkg=_wn)
    all_results.append(r)

    # ── 2. Chen 2023 (4-branch fusion, paper-exact reproduction) ──
    chen_kwargs = {
        "char_tok": char_tok,
        "word_tok": wn_tok,
        "ngram3_tok": ngram3_tok,
        "max_len_char":  HP["chen_2023"]["max_chars"],
        "max_len_word":  HP["chen_2023"]["max_words"],
        "max_len_3gram": HP["chen_2023"]["max_3gram"],
        "wordninja_pkg": _wn,
    }
    r = run_deep_baseline("chen_2023", ChenCNN_LSTM, char_tok,
                          HP["chen_2023"]["max_chars"],
                          HP["chen_2023"], train_df, val_df, test_df,
                          chen_mode=True, chen_kwargs=chen_kwargs)
    all_results.append(r)

    # ── 3. Charan 2020 (8 features + Gradient Boosting) ───────────
    r = run_charan_2020(dicts, train_df, val_df, test_df)
    all_results.append(r)

    # ── 4. Biros 2025 CNN (word-level via wordninja, paper-exact Listing 2) ─
    #   We reproduce only the CNN model (not the LSTM), since the paper
    #   reports CNN=98.30% vs LSTM=95.36% — the CNN is the stronger of
    #   the two variants.
    r = run_deep_baseline("biros_2025", BirosCNN, wn_tok,
                          HP["biros_2025"]["max_words"],
                          HP["biros_2025"], train_df, val_df, test_df)
    all_results.append(r)

    # ── 5. Vu & Hoang 2021 (J48, 16 features) ──────────────────────
    r = run_vu_2021(dicts, train_df, val_df, test_df)
    all_results.append(r)

    # ── Summary ────────────────────────────────────────────────────
    dual("\n[summary] Building Table 4 + Table 6 ...")
    summary = build_summary(all_results)
    dual(summary)

    sum_path = RESULTS_DIR / "baselines_summary_table4_table6.txt"
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write(summary)
    dual(f"\n  Summary saved → {sum_path}")
    dual(f"  Full log → {RESULTS_DIR / 'baselines_full_run.txt'}")
    dual(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    master_log.close()

if __name__ == "__main__":
    main()