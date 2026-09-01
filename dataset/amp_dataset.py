"""
AMP-Diff dataset classes for pretraining (AMPSphere) and finetuning (finetune.csv).

- No IMGT padding: sequences stay at natural length, padded to batch max_len
- No HEAVY_CDR_INDEX constant: region labels come from motif scanning
- Single-sequence records only (no H/L chain logic)
- AMPSphere data loaded from SQLite; finetune data from CSV
"""

import os
import random
import sqlite3
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)

from utils.tokenizer import Tokenizer
from utils.amp_motifs import (
    generate_region_index, generate_mask, load_active_motifs,
    load_hemolytic_motifs, parse_motif_types
)


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def _pad2d(tensors, pad_val, target_len=None):
    """Pad a list of 1-D tensors to target_len (or batch max if None)."""
    L = target_len if target_len is not None else max(len(t) for t in tensors)
    out = torch.full((len(tensors), L), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :len(t)] = t
    return out


# ---------------------------------------------------------------------------
# AMPSphere pretrain dataset
# ---------------------------------------------------------------------------

class AMPSphereDataset(Dataset):
    """Pretraining dataset loaded from AMPSphere SQLite database.

    Queries all sequences from the AMP table and computes per-sequence
    motif region labels on the fly.

    Args:
        db_path:          path to AMPSphere_latest.sqlite
        active_motifs:    dict from load_active_motifs()
        hemolytic_motifs: list from load_hemolytic_motifs()
        max_len:          truncate/pad sequences to this length
        mode:             'de' or 'inp' (controls which positions are fixed)
        min_len:          minimum sequence length to include
        split_ratio:      train/val split ratio
        split:            'train' or 'val'
        seed:             random seed for split
    """

    def __init__(
        self,
        db_path: str,
        active_motifs: dict,
        hemolytic_motifs: List[str],
        max_len: int = 100,
        mode: str = "de",
        min_len: int = 5,
        split_ratio: float = 0.95,
        split: str = "train",
        seed: int = 2023,
        require_active_motif: bool = False,
        min_optimizable: int = 1,
    ):
        super().__init__()
        self.active_motifs = active_motifs
        self.hemolytic_motifs = hemolytic_motifs
        self.max_len = max_len
        self.mode = mode
        self.tokenizer = Tokenizer()

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT sequence FROM AMP").fetchall()
        conn.close()

        raw_sequences = [r[0] for r in rows if min_len <= len(r[0]) <= max_len]
        sequences = []
        for seq in raw_sequences:
            region_index = generate_region_index(
                seq, self.active_motifs, self.hemolytic_motifs, self.mode
            )
            active_positions = sum(1 for r in region_index if r == 1)
            optimizable_positions = sum(1 for r in region_index if r == 0)
            if require_active_motif and active_positions == 0:
                continue
            if optimizable_positions < min_optimizable:
                continue
            sequences.append(seq)

        rng = random.Random(seed)
        rng.shuffle(sequences)
        split_idx = int(len(sequences) * split_ratio)
        if split == "train":
            self.sequences = sequences[:split_idx]
        else:
            self.sequences = sequences[split_idx:]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        region_index = generate_region_index(
            seq, self.active_motifs, self.hemolytic_motifs, self.mode
        )
        mask = generate_mask(region_index, self.mode)
        tokens = self.tokenizer.seq2idx(seq)  # (L,)
        return {
            "seq": seq,
            "seq_tokens": tokens,
            "region_index": torch.tensor(region_index, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "length": len(seq),
        }



# ---------------------------------------------------------------------------
# TSV pretrain dataset
# ---------------------------------------------------------------------------

class AMPTSVDataset(Dataset):
    """Pretraining dataset loaded from a TSV file (e.g., AMP.tsv from AMPSphere).

    Expects a tab-separated file with at least a 'sequence' column.
    Equivalent to AMPSphereDataset but reads from .tsv instead of SQLite.

    Args:
        tsv_path:         path to AMP.tsv
        active_motifs:    dict from load_active_motifs()
        hemolytic_motifs: list from load_hemolytic_motifs()
        max_len:          truncate/pad sequences to this length
        mode:             'de' or 'inp' (controls which positions are fixed)
        min_len:          minimum sequence length to include
        split_ratio:      train/val split ratio
        split:            'train' or 'val'
        seed:             random seed for split
    """

    def __init__(
        self,
        tsv_path: str,
        active_motifs: dict,
        hemolytic_motifs: List[str],
        max_len: int = 100,
        mode: str = "de",
        min_len: int = 5,
        split_ratio: float = 0.95,
        split: str = "train",
        seed: int = 2023,
        require_active_motif: bool = False,
        min_optimizable: int = 1,
    ):
        super().__init__()
        self.active_motifs = active_motifs
        self.hemolytic_motifs = hemolytic_motifs
        self.max_len = max_len
        self.mode = mode
        self.tokenizer = Tokenizer()

        df = pd.read_csv(tsv_path, sep="	")
        df = df.dropna(subset=["sequence"])
        raw_sequences = df.loc[df["sequence"].str.len().between(min_len, max_len), "sequence"].tolist()
        sequences = []
        for seq in raw_sequences:
            region_index = generate_region_index(
                seq, self.active_motifs, self.hemolytic_motifs, self.mode
            )
            active_positions = sum(1 for r in region_index if r == 1)
            optimizable_positions = sum(1 for r in region_index if r == 0)
            if require_active_motif and active_positions == 0:
                continue
            if optimizable_positions < min_optimizable:
                continue
            sequences.append(seq)

        rng = random.Random(seed)
        rng.shuffle(sequences)
        split_idx = int(len(sequences) * split_ratio)
        if split == "train":
            self.sequences = sequences[:split_idx]
        else:
            self.sequences = sequences[split_idx:]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        region_index = generate_region_index(
            seq, self.active_motifs, self.hemolytic_motifs, self.mode
        )
        mask = generate_mask(region_index, self.mode)
        tokens = self.tokenizer.seq2idx(seq)
        return {
            "seq": seq,
            "seq_tokens": tokens,
            "region_index": torch.tensor(region_index, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "length": len(seq),
        }


# ---------------------------------------------------------------------------
# CSV pretrain dataset
# ---------------------------------------------------------------------------

class AMPPretrainCSVDataset(Dataset):
    """Sequence-only pretraining dataset loaded from a CSV file.

    Expects a `sequence` column. Optional filters make stage-2 pretraining
    match the AmpDiff conditional task more closely:
      - require_active_motif: keep only sequences with at least one active motif
      - min_optimizable: keep only sequences with enough non-fixed positions
    """

    def __init__(
        self,
        csv_path: str,
        active_motifs: dict,
        hemolytic_motifs: List[str],
        max_len: int = 100,
        mode: str = "de",
        min_len: int = 5,
        split_ratio: float = 0.95,
        split: str = "train",
        seed: int = 2023,
        require_active_motif: bool = False,
        min_optimizable: int = 1,
    ):
        super().__init__()
        self.active_motifs = active_motifs
        self.hemolytic_motifs = hemolytic_motifs
        self.max_len = max_len
        self.mode = mode
        self.tokenizer = Tokenizer()

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["sequence"])
        raw_sequences = df.loc[
            df["sequence"].str.len().between(min_len, max_len), "sequence"
        ].tolist()

        sequences = []
        for seq in raw_sequences:
            region_index = generate_region_index(
                seq, self.active_motifs, self.hemolytic_motifs, self.mode
            )
            active_positions = sum(1 for r in region_index if r == 1)
            optimizable_positions = sum(1 for r in region_index if r == 0)
            if require_active_motif and active_positions == 0:
                continue
            if optimizable_positions < min_optimizable:
                continue
            sequences.append(seq)

        rng = random.Random(seed)
        rng.shuffle(sequences)
        split_idx = int(len(sequences) * split_ratio)
        if split == "train":
            self.sequences = sequences[:split_idx]
        else:
            self.sequences = sequences[split_idx:]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        region_index = generate_region_index(
            seq, self.active_motifs, self.hemolytic_motifs, self.mode
        )
        mask = generate_mask(region_index, self.mode)
        tokens = self.tokenizer.seq2idx(seq)
        return {
            "seq": seq,
            "seq_tokens": tokens,
            "region_index": torch.tensor(region_index, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "length": len(seq),
        }


# ---------------------------------------------------------------------------
# Finetune dataset
# ---------------------------------------------------------------------------

class AMPFinetuneDataset(Dataset):
    """Finetuning dataset loaded from finetune.csv.

    Expects columns: 'sequence' and 'value' (log activity score).

    Args:
        csv_path:         path to finetune.csv
        active_motifs:    dict from load_active_motifs()
        hemolytic_motifs: list from load_hemolytic_motifs()
        max_len:          truncate/pad sequences to this length
        mode:             'de' or 'inp'
        min_len:          minimum sequence length
        split_ratio:      train/val split
        split:            'train' or 'val'
        seed:             random seed for split
    """

    def __init__(
        self,
        csv_path: str,
        active_motifs: dict,
        hemolytic_motifs: List[str],
        max_len: int = 100,
        mode: str = "de",
        min_len: int = 5,
        split_ratio: float = 0.95,
        split: str = "train",
        seed: int = 2023,
        require_active_motif: bool = False,
        min_optimizable: int = 1,
    ):
        super().__init__()
        self.active_motifs = active_motifs
        self.hemolytic_motifs = hemolytic_motifs
        self.max_len = max_len
        self.mode = mode
        self.tokenizer = Tokenizer()

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["sequence"])
        df = df[df["sequence"].str.len().between(min_len, max_len)]
        keep_indices = []
        for idx, seq in df["sequence"].items():
            region_index = generate_region_index(
                seq, self.active_motifs, self.hemolytic_motifs, self.mode
            )
            active_positions = sum(1 for r in region_index if r == 1)
            optimizable_positions = sum(1 for r in region_index if r == 0)
            if require_active_motif and active_positions == 0:
                continue
            if optimizable_positions < min_optimizable:
                continue
            keep_indices.append(idx)
        df = df.loc[keep_indices].reset_index(drop=True)

        indices = list(range(len(df)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        split_idx = int(len(indices) * split_ratio)
        if split == "train":
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]

        self.sequences = df.loc[indices, "sequence"].tolist()
        if "value" in df.columns:
            self.values = df.loc[indices, "value"].tolist()
        else:
            self.values = [0.0] * len(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        region_index = generate_region_index(
            seq, self.active_motifs, self.hemolytic_motifs, self.mode
        )
        mask = generate_mask(region_index, self.mode)
        tokens = self.tokenizer.seq2idx(seq)
        return {
            "seq": seq,
            "seq_tokens": tokens,
            "region_index": torch.tensor(region_index, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "activity_value": torch.tensor(self.values[idx], dtype=torch.float32),
            "length": len(seq),
        }


# ---------------------------------------------------------------------------
# Mask collater (replaces OasHeavyMaskCollater / OasCamelCollater)
# ---------------------------------------------------------------------------

class AMP_MaskCollater:
    """Batches AMP sequences with OA-ARMS diffusion masking.

    Masking strategy (from Hoogeboom et al. OA-ARDM):
      - D = sequence length
      - t = random timestep in [1, D)
      - num_mask = D - t + 1
      - Masked positions are randomly selected from framework positions only
        (motif positions are never masked — they are always kept fixed)

    Output tuple:
        (src, tgt, region, masks, motif_masks, timesteps)
        src:         (B, max_len) masked input token ids
        tgt:         (B, max_len) target token ids
        region:      (B, max_len) region labels (0/1/2)
        masks:       (B, max_len) bool, True = masked
        motif_masks: (B, max_len) bool, True = motif (CDR analog, fixed)
        timesteps:   (B,) number of masked positions
    """

    def __init__(self, tokenizer: Optional[Tokenizer] = None, max_len: Optional[int] = None):
        self.tokenizer = tokenizer or Tokenizer()
        self.max_len = max_len

    def __call__(self, batch: List[Dict]) -> tuple:
        tokenizer = self.tokenizer
        mask_id = torch.tensor(tokenizer.idx_msk, dtype=torch.int64)
        pad_id = tokenizer.idx_pad

        H_tokenized = [item["seq_tokens"] for item in batch]
        H_region = [item["region_index"] for item in batch]
        H_motif_mask = [~item["mask"] for item in batch]  # True = fixed motif position

        max_len = self.max_len if self.max_len is not None else max(len(t) for t in H_tokenized)

        H_src, H_masks, H_timesteps = [], [], []

        for i, h_x in enumerate(H_tokenized):
            D = len(h_x)
            motif_mask = H_motif_mask[i]  # (D,) True = fixed

            if D <= 1:
                t = 1
            else:
                t = np.random.randint(1, D)
            num_mask = D - t + 1

            # Sample mask positions from framework positions only
            framework_positions = np.where(~motif_mask.numpy()[:D])[0]
            if len(framework_positions) == 0:
                # All positions are motif-fixed; no masking possible
                mask_arr = np.array([], dtype=np.int64)
            else:
                n_to_mask = min(num_mask, len(framework_positions))
                mask_arr = np.random.choice(framework_positions, n_to_mask, replace=False)

            h_index_arr = np.arange(max_len)
            h_mask = np.isin(h_index_arr, mask_arr).reshape(h_index_arr.shape)
            h_mask = torch.tensor(h_mask, dtype=torch.bool)

            h_x_t = h_x.clone()
            h_x_t[h_mask[:D]] = mask_id
            # Positions beyond sequence length: pad
            if D < max_len:
                pad_ext = torch.full((max_len - D,), pad_id, dtype=torch.int64)
                h_x_padded = torch.cat([h_x_t, pad_ext])
                h_tgt_padded = torch.cat([h_x, pad_ext])
            else:
                h_x_padded = h_x_t
                h_tgt_padded = h_x

            H_src.append(h_x_padded)
            H_masks.append(h_mask)
            H_timesteps.append(h_mask[:D].sum().item())

        H_src = torch.stack(H_src)                                   # (B, max_len)
        H_tgt = _pad2d(H_tokenized, pad_id, max_len).long()          # (B, max_len)
        H_region = _pad2d(H_region, 0, max_len).long()               # (B, max_len)
        H_masks = torch.stack(H_masks).bool()                        # (B, max_len)
        H_motif = _pad2d(
            [(~item["mask"]).long() for item in batch], 0, max_len
        ).bool()  # (B, max_len) True = fixed motif
        H_timesteps = torch.tensor(H_timesteps, dtype=torch.long)    # (B,)
        H_seq_lengths = torch.tensor(
            [item["length"] for item in batch], dtype=torch.long
        )  # (B,) actual unpadded sequence lengths

        return H_src, H_tgt, H_region, H_masks, H_motif, H_timesteps, H_seq_lengths


# ---------------------------------------------------------------------------
# Dataset factory used by amp_train.py / amp_finetune.py
# ---------------------------------------------------------------------------

def get_amp_dataset(
    data_path: str,
    dataset_type: str,
    active_motifs: dict,
    hemolytic_motifs: List[str],
    max_len: int = 100,
    mode: str = "de",
    split_ratio: float = 0.95,
    seed: int = 2023,
    require_active_motif: bool = False,
    min_optimizable: int = 1,
):
    """Return {'train': dataset, 'val': dataset} for the given data source.

    dataset_type: 'pretrain' (SQLite/TSV/CSV, auto-detected by extension) or 'finetune' (CSV)
    """
    common = dict(
        active_motifs=active_motifs,
        hemolytic_motifs=hemolytic_motifs,
        max_len=max_len,
        mode=mode,
        split_ratio=split_ratio,
        seed=seed,
        require_active_motif=require_active_motif,
        min_optimizable=min_optimizable,
    )
    if dataset_type == "pretrain":
        if data_path.endswith(".tsv"):
            cls = AMPTSVDataset
            kwargs = {"tsv_path": data_path}
        elif data_path.endswith(".csv"):
            cls = AMPPretrainCSVDataset
            kwargs = {"csv_path": data_path}
        else:
            cls = AMPSphereDataset
            kwargs = {"db_path": data_path}
    elif dataset_type == "finetune":
        cls = AMPFinetuneDataset
        kwargs = {"csv_path": data_path}
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return {
        "train": cls(**kwargs, **common, split="train"),
        "val": cls(**kwargs, **common, split="val"),
    }
