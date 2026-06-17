"""
Dataset preparation script for DGA detection.

Structure:
  A/                  <- 13 DGA family .txt files
  benign.txt          <- benign domains (same level as A/)
  dataout/
    train.csv         <- 70% of combined pool
    val.csv           <- 15% of combined pool
    test.csv          <- 15% of combined pool

Chiến lược chống leakage tuyệt đối:
  1. Dedup nội bộ từng file (xóa domain trùng trong cùng file).
  2. Dedup toàn cục: domain đã xuất hiện ở họ trước thì bỏ khỏi họ sau
     (thứ tự xử lý: DGA files theo alphabet, benign cuối cùng).
  3. Với mỗi nguồn, shuffle 1 lần rồi slice liên tiếp → train/val/test
     không bao giờ chứa cùng 1 domain.

Nếu sau dedup toàn cục file DGA còn < DGA_POOL mẫu: dùng tất cả còn lại,
chia 70/15/15 — đủ số lượng tối đa có thể.
"""

import random
import csv
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────────────
INPUT_DIR   = Path(r"D:\job\pycharm\Word-BasedDGA\JISIS\wordBased")
BENIGN_FILE = Path(r"benign.txt")
DATAOUT_DIR = Path(r"dataout")

SEED        = 42
DGA_POOL    = 6000   # samples per DGA family going into dataout
BENIGN_POOL = 78000  # benign samples going into dataout

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)

LABEL_DGA    = 1
LABEL_BENIGN = 0
# ────────────────────────────────────────────────────────────────────────────


def read_domains(path: Path) -> list[str]:
    """Read unique, non-empty, stripped lines — preserving first-occurrence order."""
    seen = set()
    result = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            d = line.strip()
            if d and d not in seen:
                seen.add(d)
                result.append(d)
    return result


def split_pool(domains: list[str], seed: int):
    """
    Shuffle then slice into (train, val, test) lists.
    Slice is contiguous → zero overlap by construction.
    """
    n = len(domains)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    t = int(n * TRAIN_RATIO)
    v = int(n * VAL_RATIO)
    tr = [domains[i] for i in idx[:t]]
    va = [domains[i] for i in idx[t:t+v]]
    te = [domains[i] for i in idx[t+v:]]
    return tr, va, te


def write_csv(path: Path, rows: list[tuple]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "family", "label"])
        writer.writerows(rows)


def main():
    DATAOUT_DIR.mkdir(parents=True, exist_ok=True)

    train_rows: list[tuple[str, str, int]] = []
    val_rows:   list[tuple[str, str, int]] = []
    test_rows:  list[tuple[str, str, int]] = []

    # Global seen set — bất kỳ domain nào đã được phân vào 1 split
    # thì không bao giờ xuất hiện ở split khác, kể cả từ nguồn khác.
    global_seen: set[str] = set()

    def add_to_splits(domains: list[str], family: str, label: int, pool_size: int):
        """
        Dedup against global_seen, take up to pool_size,
        split 70/15/15, append to train/val/test rows.
        Returns actual pool size used.
        """
        # Loại bỏ domain đã dùng ở nguồn khác
        clean = [d for d in domains if d not in global_seen]

        if not clean:
            return 0

        # Lấy tối đa pool_size
        pool = clean[:pool_size] if len(clean) >= pool_size else clean

        tr, va, te = split_pool(pool, SEED)

        for d in tr:
            train_rows.append((d, family, label))
        for d in va:
            val_rows.append((d, family, label))
        for d in te:
            test_rows.append((d, family, label))

        # Đánh dấu đã dùng
        global_seen.update(pool)
        return len(pool)

    # ── DGA families ────────────────────────────────────────────────────────
    dga_files = sorted(INPUT_DIR.glob("*.txt"))
    if not dga_files:
        raise FileNotFoundError(f"No DGA .txt files found in {INPUT_DIR}/")

    for fpath in dga_files:
        family = fpath.stem
        domains = read_domains(fpath)  # đã dedup nội bộ

        if not domains:
            print(f"[WARN] {family}: empty file, skipping.")
            continue

        # Shuffle với seed riêng của từng họ trước khi dedup toàn cục
        rng = random.Random(SEED + hash(family))
        rng.shuffle(domains)

        used = add_to_splits(domains, family, LABEL_DGA, DGA_POOL)

        available_after_dedup = sum(1 for d in domains if d not in global_seen) + used
        note = (f"{len(domains)} unique rows, "
                f"after global dedup → pool={used}"
                + (f" (< {DGA_POOL}, dùng tất cả còn lại)" if used < DGA_POOL else ""))
        print(f"[DGA ] {family}: {note}")

    # ── Benign ──────────────────────────────────────────────────────────────
    if not BENIGN_FILE.exists():
        raise FileNotFoundError(f"{BENIGN_FILE} not found.")

    benign_domains = read_domains(BENIGN_FILE)  # đã dedup nội bộ
    random.Random(SEED).shuffle(benign_domains)

    used_b = add_to_splits(benign_domains, "benign", LABEL_BENIGN, BENIGN_POOL)
    note_b = (f"{len(benign_domains)} unique rows → pool={used_b}"
              + (f" (< {BENIGN_POOL})" if used_b < BENIGN_POOL else ""))
    print(f"[BEN ] benign: {note_b}")

    # ── Shuffle final splits & write ────────────────────────────────────────
    rng_final = random.Random(SEED)
    rng_final.shuffle(train_rows)
    rng_final.shuffle(val_rows)
    rng_final.shuffle(test_rows)

    write_csv(DATAOUT_DIR / "train.csv", train_rows)
    write_csv(DATAOUT_DIR / "val.csv",   val_rows)
    write_csv(DATAOUT_DIR / "test.csv",  test_rows)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────")
    print(f"  dataout/train.csv : {len(train_rows):>7,} rows")
    print(f"  dataout/val.csv   : {len(val_rows):>7,} rows")
    print(f"  dataout/test.csv  : {len(test_rows):>7,} rows")
    print(f"  Total domains used: {len(global_seen):>7,}")

    # Leakage check — phải luôn = 0 với logic trên
    print("\n── Leakage check ────────────────────────────────")
    tr_set = {r[0] for r in train_rows}
    va_set = {r[0] for r in val_rows}
    te_set = {r[0] for r in test_rows}
    tv = len(tr_set & va_set)
    tt = len(tr_set & te_set)
    vt = len(va_set & te_set)
    print(f"  train ∩ val  : {tv} domains")
    print(f"  train ∩ test : {tt} domains")
    print(f"  val   ∩ test : {vt} domains")
    if tv == 0 and tt == 0 and vt == 0:
        print("  ✓ Zero leakage confirmed.")
    else:
        print("  [!!] Leakage detected — investigate source files.")

    print("\nDone.")


if __name__ == "__main__":
    main()