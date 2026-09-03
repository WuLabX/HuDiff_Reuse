"""
AMP-Diff pretraining script.

Training setup:
  - Dataset: AMP sequence datasets with motif-derived region labels
  - Collater: AMP_MaskCollater
  - Region embedding: n_region=3 (motif-based) instead of 7 (IMGT-based)
  - No chain_type embedding (AMP is single-chain)
  - Motif type filtering via --motif-type flag
"""

import os
import argparse
import shutil
import sys
import yaml
from easydict import EasyDict

current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

import torch
import torch.utils.tensorboard
import numpy as np
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dataset.Generalizability.amp_dataset import AMP_MaskCollater, get_amp_dataset
from utils.Generalizability.train_utils import model_selected, optimizer_selected, scheduler_selected
from utils.Generalizability.misc import seed_all, get_new_log_dir, get_logger, inf_iterator, count_parameters
from utils.Generalizability.loss import OasMaskedHeavyCrossEntropyLoss, MaskedAccuracy
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score
)
from utils.Generalizability.amp_motifs import load_active_motifs, load_hemolytic_motifs, parse_motif_types
from utils.Generalizability.amp_scorers import build_scorer_from_args


def convert_multi_gpu_checkpoint_to_single_gpu(checkpoint):
    if "module" in list(checkpoint["model"].keys())[0]:
        new_state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint["model"].items()
        }
        checkpoint["model"] = new_state_dict
    return checkpoint["model"]


def get_cfg(path, default=None):
    cur = config
    for part in path.split("."):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def masked_topk_accuracy(pred, tgt, mask, k):
    mask = mask.bool()
    if mask.sum().item() == 0:
        return 0.0
    logits = pred[:, :, :20]
    masked_tgt = torch.masked_select(tgt, mask)
    topk = logits.topk(k, dim=-1).indices
    masked_topk = topk[mask]
    return (masked_topk == masked_tgt.unsqueeze(-1)).any(dim=-1).float().mean().item()


def batch_mask_stats(masks, motif_masks, seq_lengths):
    valid = (
        torch.arange(masks.size(1), device=masks.device).unsqueeze(0)
        < seq_lengths.unsqueeze(1)
    )
    valid_tokens = valid.sum().clamp_min(1)
    masked_per_seq = masks[:, :].sum(dim=1).float()
    motif_per_seq = (motif_masks & valid).sum(dim=1).float()
    seq_len_f = seq_lengths.float().clamp_min(1)
    return {
        "masked_per_seq": masked_per_seq.mean().item(),
        "zero_mask_fraction": (masked_per_seq == 0).float().mean().item(),
        "motif_fixed_fraction": (motif_masks & valid).sum().float().div(valid_tokens).item(),
        "no_motif_fraction": (motif_per_seq == 0).float().mean().item(),
        "motif_per_seq": motif_per_seq.mean().item(),
        "seq_len": seq_len_f.mean().item(),
    }


def reset_train_window():
    return {
        "loss": 0.0,
        "nll": 0.0,
        "motif_loss": 0.0,
        "scorer_loss": 0.0,
        "acc": 0.0,
        "top3_acc": 0.0,
        "top5_acc": 0.0,
        "masked_per_seq": 0.0,
        "zero_mask_fraction": 0.0,
        "motif_fixed_fraction": 0.0,
        "no_motif_fraction": 0.0,
        "motif_per_seq": 0.0,
        "seq_len": 0.0,
        "steps": 0,
        "true": [],
        "pred": [],
    }


def update_train_window(
    window, loss, nll, motif_loss, scorer_loss, acc, top3, top5,
    stats, pred, tgt, mask
):
    window["loss"] += float(loss)
    window["nll"] += float(nll)
    window["motif_loss"] += float(motif_loss)
    window["scorer_loss"] += float(scorer_loss)
    window["acc"] += float(acc)
    window["top3_acc"] += float(top3)
    window["top5_acc"] += float(top5)
    for key in (
        "masked_per_seq", "zero_mask_fraction", "motif_fixed_fraction",
        "no_motif_fraction", "motif_per_seq", "seq_len"
    ):
        window[key] += float(stats[key])
    window["steps"] += 1

    mask = mask.bool()
    if mask.sum().item() > 0:
        pred20 = pred[:, :, :20].argmax(dim=-1)
        true_flat = torch.masked_select(tgt, mask).detach().cpu()
        pred_flat = torch.masked_select(pred20, mask).detach().cpu()
        aa_mask = true_flat < 20
        if aa_mask.any():
            window["true"].append(true_flat[aa_mask])
            window["pred"].append(pred_flat[aa_mask])


def flush_train_window(window, it, force=False):
    if window["steps"] == 0:
        return window
    if not force and window["steps"] < args.train_window:
        return window

    n = window["steps"]
    true = torch.cat(window["true"]).numpy() if window["true"] else np.array([])
    pred = torch.cat(window["pred"]).numpy() if window["pred"] else np.array([])
    labels = sorted(set(true.tolist())) if len(true) else []
    f1 = f1_score(true, pred, average="macro", zero_division=0, labels=labels) if len(true) else 0.0
    precision = precision_score(true, pred, average="macro", zero_division=0, labels=labels) if len(true) else 0.0
    recall = recall_score(true, pred, average="macro", zero_division=0, labels=labels) if len(true) else 0.0

    for key in (
        "loss", "nll", "motif_loss", "scorer_loss", "acc", "top3_acc",
        "top5_acc", "masked_per_seq", "zero_mask_fraction",
        "motif_fixed_fraction", "no_motif_fraction", "motif_per_seq", "seq_len"
    ):
        writer.add_scalar(f"train_window/{key}", window[key] / n, it)
    writer.add_scalar("train_window/f1", f1, it)
    writer.add_scalar("train_window/precision", precision, it)
    writer.add_scalar("train_window/recall", recall, it)
    writer.flush()
    logger.info(
        f"Train window ending {it}: loss={window['loss']/n:.4f} "
        f"nll={window['nll']/n:.4f} acc={window['acc']/n:.4f} "
        f"top3={window['top3_acc']/n:.4f} top5={window['top5_acc']/n:.4f} "
        f"f1={f1:.4f} prec={precision:.4f} rec={recall:.4f}"
    )
    return reset_train_window()


def train(it):
    sum_loss = sum_nll = sum_cdr_loss = sum_acc = sum_top3 = sum_top5 = 0.0
    sum_roc = sum_scorer_loss = 0.0
    stat_sums = {
        "masked_per_seq": 0.0,
        "zero_mask_fraction": 0.0,
        "motif_fixed_fraction": 0.0,
        "no_motif_fraction": 0.0,
        "motif_per_seq": 0.0,
        "seq_len": 0.0,
    }
    model.train()
    for _ in range(config.train.batch_acc):
        optimizer.zero_grad()
        (H_src, H_tgt, H_region, H_masks, H_motif_masks, H_timesteps, H_seq_lengths) = next(train_iterator)
        H_src = H_src.to(device)
        H_tgt = H_tgt.to(device)
        H_region = H_region.to(device)
        H_masks = H_masks.to(device)
        H_motif_masks = H_motif_masks.to(device)
        H_timesteps = H_timesteps.to(device)
        H_seq_lengths = H_seq_lengths.to(device)

        H_pred = model(H_src, H_region, H_chn_type=None)

        H_loss, H_nll, H_motif_loss = cross_loss(
            H_pred, H_tgt, H_masks, H_motif_masks, H_timesteps, H_seq_lengths
        )

        scorer_loss = torch.zeros(1, device=device)
        if scorer is not None:
            logits_aa = H_pred[:, :, :20]
            soft_probs = torch.softmax(logits_aa, dim=-1)
            scorer_loss = scorer.combined_guidance_loss(soft_probs, mode=args.mode)

        if args.train_loss == "fr":
            loss = H_loss + lam_act * scorer_loss
        else:  # 'all'
            loss = H_loss + H_motif_loss + lam_act * scorer_loss

        loss.backward()
        if config.train.clip_norm > 0:
            clip_grad_norm_(model.parameters(), config.train.clip_norm)
        optimizer.step()

        loss_item = loss.item()
        nll_item = H_nll.item()
        motif_loss_item = H_motif_loss.item()
        scorer_loss_item = scorer_loss.item() if torch.is_tensor(scorer_loss) else scorer_loss

        sum_loss += loss_item
        sum_nll += nll_item
        sum_cdr_loss += motif_loss_item
        sum_scorer_loss += scorer_loss_item
        H_acc, roc_auc = mask_acc(H_pred, H_tgt, H_masks)
        acc_item = H_acc.item()
        top3_item = masked_topk_accuracy(H_pred, H_tgt, H_masks, 3)
        top5_item = masked_topk_accuracy(H_pred, H_tgt, H_masks, 5)
        sum_acc += acc_item
        sum_top3 += top3_item
        sum_top5 += top5_item
        sum_roc += roc_auc
        stats = batch_mask_stats(H_masks, H_motif_masks, H_seq_lengths)
        for key in stat_sums:
            stat_sums[key] += stats[key]
        update_train_window(
            train_window, loss_item, nll_item, motif_loss_item, scorer_loss_item,
            acc_item, top3_item, top5_item, stats, H_pred, H_tgt, H_masks
        )

    n = config.train.batch_acc
    logger.info(
        f"Train iter {it}: loss={sum_loss/n:.4f} nll={sum_nll/n:.4f} "
        f"motif_loss={sum_cdr_loss/n:.4f} scorer={sum_scorer_loss/n:.4f} "
        f"acc={sum_acc/n:.4f} top3={sum_top3/n:.4f} top5={sum_top5/n:.4f} "
        f"masked={stat_sums['masked_per_seq']/n:.2f} "
        f"motif_frac={stat_sums['motif_fixed_fraction']/n:.3f} "
        f"no_motif={stat_sums['no_motif_fraction']/n:.3f} roc={sum_roc/n:.4f}"
    )
    writer.add_scalar("train/loss", sum_loss / n, it)
    writer.add_scalar("train/nll", sum_nll / n, it)
    writer.add_scalar("train/motif_loss", sum_cdr_loss / n, it)
    writer.add_scalar("train/scorer_loss", sum_scorer_loss / n, it)
    writer.add_scalar("train/acc", sum_acc / n, it)
    writer.add_scalar("train/top3_acc", sum_top3 / n, it)
    writer.add_scalar("train/top5_acc", sum_top5 / n, it)
    writer.add_scalar("train/masked_per_seq", stat_sums["masked_per_seq"] / n, it)
    writer.add_scalar("train/zero_mask_fraction", stat_sums["zero_mask_fraction"] / n, it)
    writer.add_scalar("train/motif_fixed_fraction", stat_sums["motif_fixed_fraction"] / n, it)
    writer.add_scalar("train/no_motif_fraction", stat_sums["no_motif_fraction"] / n, it)
    writer.add_scalar("train/motif_per_seq", stat_sums["motif_per_seq"] / n, it)
    writer.add_scalar("train/seq_len", stat_sums["seq_len"] / n, it)
    writer.add_scalar("train/optimizer_updates", (it + 1) * n, it)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], it)
    writer.flush()
    return flush_train_window(train_window, it)


def save_checkpoint(path, it):
    state_model = model.module if hasattr(model, "module") else model
    torch.save({
        "config": config,
        "model": state_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": it,
    }, path)


def valid(it):
    sum_loss = sum_nll = sum_cdr_loss = 0.0
    model.eval()
    n = len(val_loader)
    all_true  = []   # masked true token indices  (int)
    all_probs = []   # masked softmax probs over all classes (float)
    stat_sums = {
        "masked_per_seq": 0.0,
        "zero_mask_fraction": 0.0,
        "motif_fixed_fraction": 0.0,
        "no_motif_fraction": 0.0,
        "motif_per_seq": 0.0,
        "seq_len": 0.0,
    }
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val"):
            (H_src, H_tgt, H_region, H_masks, H_motif_masks, H_timesteps, H_seq_lengths) = batch
            H_src = H_src.to(device)
            H_tgt = H_tgt.to(device)
            H_region = H_region.to(device)
            H_masks = H_masks.to(device)
            H_motif_masks = H_motif_masks.to(device)
            H_timesteps = H_timesteps.to(device)
            H_seq_lengths = H_seq_lengths.to(device)

            H_pred = model(H_src, H_region, H_chn_type=None)
            H_loss, H_nll, H_motif_loss = cross_loss(
                H_pred, H_tgt, H_masks, H_motif_masks, H_timesteps, H_seq_lengths
            )
            loss = H_loss + H_motif_loss
            sum_loss += loss.item()
            sum_nll += H_nll.item()
            sum_cdr_loss += H_motif_loss.item()

            # Collect masked positions for epoch-level metric computation
            mask_flat  = H_masks.bool()                                   # (B, L)
            true_flat  = torch.masked_select(H_tgt, mask_flat)           # (T,)
            probs_full = torch.softmax(H_pred, dim=-1)                   # (B, L, C)
            mask_exp   = mask_flat.unsqueeze(-1).expand_as(probs_full)   # (B, L, C)
            probs_flat = probs_full[mask_exp].view(-1, H_pred.size(-1))  # (T, C)
            all_true.append(true_flat.cpu())
            all_probs.append(probs_flat.cpu())
            stats = batch_mask_stats(H_masks, H_motif_masks, H_seq_lengths)
            for key in stat_sums:
                stat_sums[key] += stats[key]

    # ── epoch-level metrics ───────────────────────────────────────────────────
    all_true  = torch.cat(all_true).numpy() if all_true else torch.empty(0).numpy()
    all_probs = torch.cat(all_probs).numpy() if all_probs else torch.empty(0, 23).numpy()
    all_pred  = all_probs.argmax(axis=1)             # (N_masked,)

    # Restrict to the 20 standard AA classes (indices 0-19) for AUC.
    # Targets should only ever be 0-19 in well-formed sequences; filter
    # out any stray special tokens just in case.
    aa_mask = all_true < 20
    t20     = all_true[aa_mask]             # true labels in [0, 19]
    p20     = all_probs[aa_mask][:, :20]   # probs for 20 AAs
    # Re-normalise so rows sum to 1 after dropping special-token columns
    p20     = p20 / p20.sum(axis=1, keepdims=True).clip(1e-9)
    pred20  = p20.argmax(axis=1)

    val_acc = float((all_pred == all_true).mean()) if len(all_true) else 0.0

    # AUC: one-vs-rest macro over 20 AA classes
    present_classes = sorted(set(t20.tolist()))
    try:
        val_auc = roc_auc_score(
            t20, p20,
            multi_class="ovr", average="macro",
            labels=list(range(20))
        )
    except ValueError:
        val_auc = float("nan")

    val_f1        = f1_score(t20, pred20, average="macro", zero_division=0, labels=present_classes) if len(t20) else 0.0
    val_precision = precision_score(t20, pred20, average="macro", zero_division=0, labels=present_classes) if len(t20) else 0.0
    val_recall    = recall_score(t20, pred20, average="macro", zero_division=0, labels=present_classes) if len(t20) else 0.0
    if len(t20):
        top3 = (p20.argsort(axis=1)[:, -3:] == t20[:, None]).any(axis=1).mean()
        top5 = (p20.argsort(axis=1)[:, -5:] == t20[:, None]).any(axis=1).mean()
    else:
        top3 = top5 = 0.0

    mean_loss = sum_loss / n
    scheduler.step(mean_loss)
    logger.info(
        f"Val iter {it}: loss={mean_loss:.4f} nll={sum_nll/n:.4f} "
        f"motif_loss={sum_cdr_loss/n:.4f} "
        f"acc={val_acc:.4f} top3={top3:.4f} top5={top5:.4f} auc={val_auc:.4f} "
        f"f1={val_f1:.4f} prec={val_precision:.4f} rec={val_recall:.4f}"
    )
    writer.add_scalar("val/loss",      mean_loss,     it)
    writer.add_scalar("val/nll",       sum_nll / n,   it)
    writer.add_scalar("val/acc",       val_acc,       it)
    writer.add_scalar("val/top3_acc",  top3,          it)
    writer.add_scalar("val/top5_acc",  top5,          it)
    writer.add_scalar("val/auc",       val_auc,       it)
    writer.add_scalar("val/f1_score",  val_f1,        it)
    writer.add_scalar("val/precision", val_precision, it)
    writer.add_scalar("val/recall",    val_recall,    it)
    writer.add_scalar("val/masked_per_seq", stat_sums["masked_per_seq"] / n, it)
    writer.add_scalar("val/zero_mask_fraction", stat_sums["zero_mask_fraction"] / n, it)
    writer.add_scalar("val/motif_fixed_fraction", stat_sums["motif_fixed_fraction"] / n, it)
    writer.add_scalar("val/no_motif_fraction", stat_sums["no_motif_fraction"] / n, it)
    writer.add_scalar("val/motif_per_seq", stat_sums["motif_per_seq"] / n, it)
    writer.add_scalar("val/seq_len", stat_sums["seq_len"] / n, it)
    writer.flush()
    return mean_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to AMPSphere_latest.sqlite")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, default="logs/amp_pretrain")
    parser.add_argument("--train_loss", type=str, default="all",
                        choices=["fr", "all"],
                        help="'fr'=framework only; 'all'=framework+motif")
    parser.add_argument("--motif_type", type=str, default="prosite,regular,merci",
                        help="Comma-separated motif types to enable: "
                             "prosite,regular,merci,none")
    parser.add_argument("--mode", type=str, default="de", choices=["de", "inp"])
    parser.add_argument("--resume", type=eval, default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Load model weights only; keep current config and optimizer.")
    parser.add_argument(
        "--scorer", type=str, default="none",
        help="One activity guidance scorer: pepnet, amppred_mfa, "
             "iamp_attenpred, unidl4biopep; optional hemopi2 in inp mode",
    )
    parser.add_argument("--pepnet_root", type=str, default="/mnt/wucy/WUCHUYA/PepNet")
    parser.add_argument("--hemopi2_root", type=str, default="/mnt/wucy/WUCHUYA/hemopi2")
    parser.add_argument("--lam_act", type=float, default=1.0,
                        help="Weight for scorer guidance loss")
    parser.add_argument("--require_active_motif", type=eval, default=False,
                        help="Keep only sequences with at least one active motif.")
    parser.add_argument("--min_optimizable", type=int, default=1,
                        help="Minimum non-fixed positions required per sequence.")
    parser.add_argument("--ckpt_subdir", type=str, default=None,
                        help="Canonical checkpoint subdirectory under checkpoints/.")
    parser.add_argument("--ckpt_name", type=str, default="pretrain.pt",
                        help="Canonical checkpoint file name.")
    parser.add_argument("--save_milestone_every", type=int, default=0,
                        help="Also save numbered checkpoints every N iterations; 0 disables.")
    parser.add_argument("--train_window", type=int, default=100,
                        help="Number of train steps per smoothed train_window/* metric.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.resume:
        with open(args.config_path, "r") as f:
            config = EasyDict(yaml.safe_load(f))
        start_iter = 0
    else:
        assert args.checkpoint, "Checkpoint path required for resume."
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = ckpt["config"]
        start_iter = int(ckpt.get("iteration", -1)) + 1

    # Resolve motif file paths relative to script or absolute
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    motif_active = os.path.join(base_dir, config.amp.motif_active_file)
    motif_hem = os.path.join(base_dir, config.amp.motif_hemolytic_file)

    enabled_types = parse_motif_types(args.motif_type)
    active_motifs = load_active_motifs(motif_active, enabled_types)
    hemolytic_motifs = load_hemolytic_motifs(motif_hem)

    scorer = build_scorer_from_args(
        args.scorer,
        pepnet_root=args.pepnet_root,
        hemopi2_root=args.hemopi2_root,
        device=device,
    )
    lam_act = args.lam_act

    version = f"amp_pretrain_{args.mode}_{args.train_loss}"
    log_dir = get_new_log_dir(root=args.log_path, prefix=version)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = get_logger("train", log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)

    shutil.copyfile(args.config_path, os.path.join(log_dir, os.path.basename(args.config_path)))
    shutil.copyfile(os.path.abspath(__file__), os.path.join(log_dir, "amp_train.py"))

    seed_all(config.train.seed)

    subsets = get_amp_dataset(
        data_path=args.data_path,
        dataset_type="pretrain",
        active_motifs=active_motifs,
        hemolytic_motifs=hemolytic_motifs,
        max_len=config.model.max_len,
        mode=args.mode,
        split_ratio=get_cfg("train.split_ratio", 0.95),
        seed=config.train.seed,
        require_active_motif=args.require_active_motif,
        min_optimizable=args.min_optimizable,
    )
    collater = AMP_MaskCollater(max_len=config.model.max_len)
    train_iterator = inf_iterator(DataLoader(
        subsets["train"],
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=collater,
    ))
    val_loader = DataLoader(
        subsets["val"],
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        collate_fn=collater,
    )
    logger.info(f"Train: {len(subsets['train'])}  Val: {len(subsets['val'])}")

    model = model_selected(config).to(device)
    if args.resume:
        model.load_state_dict(convert_multi_gpu_checkpoint_to_single_gpu(ckpt))
    elif args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(convert_multi_gpu_checkpoint_to_single_gpu(init_ckpt))
        logger.info(f"Initialized model weights from {args.init_checkpoint}")

    visible_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    use_data_parallel = (
        device == "cuda"
        and torch.cuda.device_count() > 1
        and os.environ.get("AMPDIFF_DATA_PARALLEL", "0") == "1"
    )
    if use_data_parallel:
        model = torch.nn.DataParallel(model)
        logger.info(
            f"Using torch.nn.DataParallel on {torch.cuda.device_count()} visible GPUs "
            f"(CUDA_VISIBLE_DEVICES={visible_cuda or 'all'})"
        )

    optimizer = optimizer_selected(config.train.optimizer, model)
    scheduler = scheduler_selected(config.train.scheduler, optimizer)
    cross_loss = OasMaskedHeavyCrossEntropyLoss()
    mask_acc = MaskedAccuracy()

    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        logger.info(f"Resuming from {args.checkpoint} at iter {start_iter}")

    logger.info(f"Trainable parameters: {count_parameters(model)/1e6:.4f} M")
    best_val_loss = float("inf")
    best_iter = 0
    train_window = reset_train_window()
    canonical_subdir = args.ckpt_subdir or args.mode
    canonical_dir = os.path.join(base_dir, "checkpoints", canonical_subdir)
    os.makedirs(canonical_dir, exist_ok=True)
    canonical_path = os.path.join(canonical_dir, args.ckpt_name)

    for it in range(start_iter, config.train.max_iter + 1):
        train_window = train(it)
        if it % config.train.valid_step == 0 or it == config.train.max_iter:
            val_loss = valid(it)
            last_ckpt = os.path.join(ckpt_dir, "last.pt")
            save_checkpoint(last_ckpt, it)
            logger.info("Updated run last checkpoint -> checkpoints/last.pt")

            if args.save_milestone_every > 0 and it % args.save_milestone_every == 0:
                milestone_ckpt = os.path.join(ckpt_dir, f"{it}.pt")
                shutil.copyfile(last_ckpt, milestone_ckpt)
                logger.info(f"Saved milestone checkpoint -> checkpoints/{it}.pt")

            if val_loss < best_val_loss:
                best_val_loss, best_iter = val_loss, it
                logger.info(f"Best val loss: {best_val_loss:.6f} at iter {best_iter}")
                best_ckpt = os.path.join(ckpt_dir, "best.pt")
                shutil.copyfile(last_ckpt, best_ckpt)
                shutil.copyfile(best_ckpt, canonical_path)
                logger.info("Updated run best checkpoint -> checkpoints/best.pt")
                logger.info(
                    f"Canonical pretrain ckpt -> checkpoints/Generalizability/{canonical_subdir}/{args.ckpt_name}"
                )
            else:
                logger.info(
                    f"Val loss not improved. Best: {best_val_loss:.6f} at {best_iter}"
                )
    flush_train_window(train_window, config.train.max_iter, force=True)
