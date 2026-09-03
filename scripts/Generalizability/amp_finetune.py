"""
AMP-Diff finetuning script.

Finetuning setup:
  - Dataset: AMPFinetuneDataset (CSV)
  - Scorer: one selectable activity predictor, plus optional HemoPI2
  - Loss: diffusion_loss + lambda_activity * activity_loss [+ lambda_hem * hemolysis_loss]
  - Motif type and guidance predictor controlled via CLI flags
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
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score
)

from dataset.Generalizability.amp_dataset import AMP_MaskCollater, get_amp_dataset
from utils.Generalizability.train_utils import model_selected, optimizer_selected, scheduler_selected
from utils.Generalizability.misc import seed_all, get_new_log_dir, get_logger, inf_iterator, count_parameters
from utils.Generalizability.loss import OasMaskedHeavyCrossEntropyLoss, MaskedAccuracy
from utils.Generalizability.amp_motifs import load_active_motifs, load_hemolytic_motifs, parse_motif_types
from utils.Generalizability.amp_scorers import ACTIVITY_SCORERS, build_scorer_from_args, parse_scorer_names


def convert_multi_gpu_checkpoint_to_single_gpu(checkpoint):
    if "module" in list(checkpoint["model"].keys())[0]:
        new_state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint["model"].items()
        }
        checkpoint["model"] = new_state_dict
    return checkpoint["model"]


def finetune(it):
    sum_loss = sum_diff_loss = sum_nll = sum_scorer_loss = sum_acc = 0.0
    model.train()

    for _ in range(config.finetune.batch_acc):
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

        diff_loss, H_nll, _ = cross_loss(H_pred, H_tgt, H_masks, H_motif_masks, H_timesteps, H_seq_lengths)

        scorer_loss = torch.zeros(1, device=device)
        if scorer is not None:
            # Compute soft distribution over 20 AAs from model logits
            # H_pred: (B, L, n_tokens); take first 20 dims (standard AAs)
            logits_aa = H_pred[:, :, :20]  # (B, L, 20)
            soft_probs = torch.softmax(logits_aa, dim=-1)
            scorer_loss = scorer.combined_guidance_loss(
                soft_probs, mode=args.mode,
                include_hemolysis=args.hemolysis_guidance,
                lengths=H_seq_lengths,
            )

        loss = diff_loss + lam_act * scorer_loss
        loss.backward()
        if config.finetune.clip_norm > 0:
            clip_grad_norm_(model.parameters(), config.finetune.clip_norm)
        optimizer.step()

        H_acc, _ = mask_acc(H_pred, H_tgt, H_masks)

        sum_loss += loss.item()
        sum_diff_loss += diff_loss.item()
        sum_nll += H_nll.item()
        sum_scorer_loss += scorer_loss.item() if torch.is_tensor(scorer_loss) else scorer_loss
        sum_acc += H_acc.item()

    n = config.finetune.batch_acc
    logger.info(
        f"Finetune iter {it}: total={sum_loss/n:.4f} nll={sum_nll/n:.4f} "
        f"diff={sum_diff_loss/n:.4f} scorer={sum_scorer_loss/n:.4f} "
        f"acc={sum_acc/n:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
    )
    writer.add_scalar("finetune/loss", sum_loss / n, it)
    writer.add_scalar("finetune/diff_loss", sum_diff_loss / n, it)
    writer.add_scalar("finetune/nll", sum_nll / n, it)
    writer.add_scalar("finetune/scorer_loss", sum_scorer_loss / n, it)
    writer.add_scalar("finetune/acc", sum_acc / n, it)
    writer.add_scalar("finetune/lr", optimizer.param_groups[0]["lr"], it)
    writer.flush()


def finetune_val(it):
    sum_loss = sum_nll = 0.0
    model.eval()
    n = len(val_loader)
    all_true = []
    all_probs = []
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
            diff_loss, H_nll, _ = cross_loss(H_pred, H_tgt, H_masks, H_motif_masks, H_timesteps, H_seq_lengths)
            sum_loss += diff_loss.item()
            sum_nll += H_nll.item()

            # Collect masked positions for epoch-level metric computation
            mask_flat  = H_masks.bool()                                   # (B, L)
            true_flat  = torch.masked_select(H_tgt, mask_flat)           # (T,)
            probs_full = torch.softmax(H_pred, dim=-1)                   # (B, L, C)
            mask_exp   = mask_flat.unsqueeze(-1).expand_as(probs_full)   # (B, L, C)
            probs_flat = probs_full[mask_exp].view(-1, H_pred.size(-1))  # (T, C)
            all_true.append(true_flat.cpu())
            all_probs.append(probs_flat.cpu())

    # ── epoch-level metrics ───────────────────────────────────────────────────
    all_true  = torch.cat(all_true).numpy()          # (N_masked,)
    all_probs = torch.cat(all_probs).numpy()         # (N_masked, 23)
    all_pred  = all_probs.argmax(axis=1)             # (N_masked,)

    # Restrict to the 20 standard AA classes (indices 0-19) for AUC.
    aa_mask = all_true < 20
    t20     = all_true[aa_mask]
    p20     = all_probs[aa_mask][:, :20]
    p20     = p20 / p20.sum(axis=1, keepdims=True).clip(1e-9)
    pred20  = p20.argmax(axis=1)

    val_acc = float((all_pred == all_true).mean())

    present_classes = sorted(set(t20.tolist()))
    try:
        val_auc = roc_auc_score(
            t20, p20,
            multi_class="ovr", average="macro",
            labels=list(range(20))
        )
    except ValueError:
        val_auc = float("nan")

    val_f1        = f1_score(t20, pred20, average="macro", zero_division=0, labels=present_classes)
    val_precision = precision_score(t20, pred20, average="macro", zero_division=0, labels=present_classes)
    val_recall    = recall_score(t20, pred20, average="macro", zero_division=0, labels=present_classes)

    mean_loss = sum_loss / n
    scheduler.step(mean_loss)
    logger.info(
        f"Val iter {it}: loss={mean_loss:.4f} nll={sum_nll/n:.4f} "
        f"acc={val_acc:.4f} auc={val_auc:.4f} "
        f"f1={val_f1:.4f} prec={val_precision:.4f} rec={val_recall:.4f}"
    )
    writer.add_scalar("val/loss",      mean_loss,     it)
    writer.add_scalar("val/nll",       sum_nll / n,   it)
    writer.add_scalar("val/acc",       val_acc,       it)
    writer.add_scalar("val/auc",       val_auc,       it)
    writer.add_scalar("val/f1_score",  val_f1,        it)
    writer.add_scalar("val/precision", val_precision, it)
    writer.add_scalar("val/recall",    val_recall,    it)
    writer.flush()
    return mean_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to finetune.csv")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, default="logs/amp_finetune")
    parser.add_argument("--mode", type=str, default="de", choices=["de", "inp"],
                        help="de=fix active motif; inp=fix both motifs")
    parser.add_argument(
        "--guidance_scorer", type=str, default=None, choices=ACTIVITY_SCORERS,
        help="The single activity predictor used for gradient guidance",
    )
    parser.add_argument(
        "--scorer", type=str, default=None,
        help="Deprecated compatibility alias; must contain exactly one activity scorer",
    )
    parser.add_argument(
        "--hemolysis-guidance", "--hemolysis_guidance",
        dest="hemolysis_guidance", action=argparse.BooleanOptionalAction, default=False,
        help="Also apply HemoPI2 hemolysis guidance in inp mode",
    )
    parser.add_argument("--motif_type", type=str, default="prosite,regular,merci",
                        help="Comma-separated motif types: prosite,regular,merci,none")
    parser.add_argument("--pepnet_root", type=str, default="/mnt/wucy/WUCHUYA/PepNet")
    parser.add_argument("--pepnet_ckpt", type=str, default=None,
                        help="Override PepNet checkpoint path")
    parser.add_argument("--hemopi2_root", type=str, default="/mnt/wucy/WUCHUYA/hemopi2")
    parser.add_argument("--amppred_root", type=str, default="/mnt/wucy/WUCHUYA/AMPpred-MFA")
    parser.add_argument("--iamp_root", type=str, default="/mnt/wucy/WUCHUYA/iAMP-Attenpred")
    parser.add_argument("--unidl_root", type=str, default="/mnt/wucy/WUCHUYA/UniDL4BioPep")
    parser.add_argument("--pretrain_ckpt", type=str, default="",
                        help="Path to pretrained checkpoint. Defaults to checkpoints/Generalizability/<mode>/pretrain.pt")
    parser.add_argument("--resume", type=eval, default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.guidance_scorer is None:
        legacy = parse_scorer_names(args.scorer or "pepnet")
        selected = [name for name in ACTIVITY_SCORERS if name in legacy]
        if len(selected) != 1:
            parser.error("--scorer compatibility mode requires exactly one activity scorer")
        args.guidance_scorer = selected[0]
        if args.scorer is not None:
            args.hemolysis_guidance = "hemopi2" in legacy

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.resume:
        with open(args.config_path, "r") as f:
            config = EasyDict(yaml.safe_load(f))
    else:
        assert args.checkpoint
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = ckpt["config"]

    lam_act = getattr(config.finetune, "lambda_activity", 1.0)
    lam_hem = getattr(config.finetune, "lambda_hemolysis", 1.0)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    motif_active = os.path.join(base_dir, config.amp.motif_active_file)
    motif_hem_path = os.path.join(base_dir, config.amp.motif_hemolytic_file)

    enabled_types = parse_motif_types(args.motif_type)
    active_motifs = load_active_motifs(motif_active, enabled_types)
    hemolytic_motifs = load_hemolytic_motifs(motif_hem_path)

    version = f"amp_finetune_{args.mode}_guidance{args.guidance_scorer}"
    log_dir = get_new_log_dir(root=args.log_path, prefix=version)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = get_logger("finetune", log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)

    shutil.copyfile(args.config_path, os.path.join(log_dir, os.path.basename(args.config_path)))
    shutil.copyfile(os.path.abspath(__file__), os.path.join(log_dir, "amp_finetune.py"))

    seed_all(config.finetune.seed)

    subsets = get_amp_dataset(
        data_path=args.data_path,
        dataset_type="finetune",
        active_motifs=active_motifs,
        hemolytic_motifs=hemolytic_motifs,
        max_len=config.model.max_len,
        mode=args.mode,
    )
    collater = AMP_MaskCollater(max_len=config.model.max_len)
    train_iterator = inf_iterator(DataLoader(
        subsets["train"],
        batch_size=config.finetune.batch_size,
        shuffle=True,
        num_workers=config.finetune.num_workers,
        collate_fn=collater,
    ))
    val_loader = DataLoader(
        subsets["val"],
        batch_size=config.finetune.batch_size,
        num_workers=config.finetune.num_workers,
        collate_fn=collater,
    )
    logger.info(f"Train: {len(subsets['train'])}  Val: {len(subsets['val'])}")

    # Load pretrained model
    if args.pretrain_ckpt:
        pretrain_ckpt_path = args.pretrain_ckpt
    else:
        pretrain_ckpt_path = os.path.join(
            base_dir, "checkpoints", args.mode, "pretrain.pt"
        )
    pretrain_ckpt = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=False)
    model = model_selected(pretrain_ckpt["config"]).to(device)
    model.load_state_dict(convert_multi_gpu_checkpoint_to_single_gpu(pretrain_ckpt))
    logger.info(f"Loaded pretrained model from {pretrain_ckpt_path}")

    if args.resume:
        model.load_state_dict(convert_multi_gpu_checkpoint_to_single_gpu(
            torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ))

    # Load scorer (frozen; used only for gradient signal during training)
    guidance_names = args.guidance_scorer
    if args.hemolysis_guidance and args.mode == "inp":
        guidance_names += ",hemopi2"
    scorer = build_scorer_from_args(
        guidance_names,
        guidance_activity=args.guidance_scorer,
        pepnet_root=args.pepnet_root,
        hemopi2_root=args.hemopi2_root,
        amppred_root=args.amppred_root,
        iamp_root=args.iamp_root,
        unidl_root=args.unidl_root,
        pepnet_ckpt=args.pepnet_ckpt,
        device=device,
    )
    if scorer is not None:
        logger.info(
            f"Activity guidance={scorer.guidance_activity}; "
            f"hemolysis_guidance={args.hemolysis_guidance and scorer.esm_model is not None}"
        )

    optimizer = optimizer_selected(config.finetune.optimizer, model)
    scheduler = scheduler_selected(config.finetune.scheduler, optimizer)
    cross_loss = OasMaskedHeavyCrossEntropyLoss()
    mask_acc = MaskedAccuracy()

    logger.info(f"Trainable parameters: {count_parameters(model)/1e6:.4f} M")
    best_val_loss = float("inf")
    best_iter = 0

    for it in range(config.finetune.max_iter + 1):
        finetune(it)
        if it % config.finetune.valid_step == 0 and it != 0 or it == config.finetune.max_iter:
            val_loss = finetune_val(it)
            if val_loss < best_val_loss:
                best_val_loss, best_iter = val_loss, it
                logger.info(f"Best val loss: {best_val_loss:.6f} at iter {best_iter}")
                iter_ckpt = os.path.join(ckpt_dir, f"{it}.pt")
                torch.save({
                    "config": config,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "iteration": it,
                }, iter_ckpt)
                canonical_dir = os.path.join(base_dir, "checkpoints", args.mode)
                os.makedirs(canonical_dir, exist_ok=True)
                canonical_path = os.path.join(canonical_dir, f"{args.mode}.pt")
                shutil.copyfile(iter_ckpt, canonical_path)
                logger.info(f"Canonical finetune ckpt -> checkpoints/Generalizability/{args.mode}/{args.mode}.pt")
            else:
                logger.info(
                    f"Val loss not improved. Best: {best_val_loss:.6f} at {best_iter}"
                )
