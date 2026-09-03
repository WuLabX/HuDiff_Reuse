"""
AMP-Diff sampling script.

Sampling setup:
  - Uses amp_batch_input() with motif-based fixed/optimizable positions
  - Validity check is standard amino-acid characters
  - Output: FASTA + TSV with predictor-specific score columns
  - One activity predictor selected via --guidance_scorer
  - The other three activity predictors are independent post-generation evaluators
  - Motif types selectable via --motif_type flag
  - Mode: de (fix active motif) or inp (fix both motifs)
"""

import os
import sys
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

from utils.Generalizability.tokenizer import Tokenizer
from utils.Generalizability.train_utils import model_selected
from utils.Generalizability.misc import get_new_log_dir, get_logger, seed_all
from utils.Generalizability.amp_motifs import (
    load_active_motifs, load_hemolytic_motifs, parse_motif_types,
    generate_region_index, generate_mask
)
from utils.Generalizability.amp_scorers import (
    ACTIVITY_SCORERS, build_scorer_from_args, parse_scorer_names,
    resolve_independent_scorers,
)

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def load_checkpoint(ckpt_path: str, device: str):
    """Load model from checkpoint. Supports both pretrain and finetune checkpoints."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = model_selected(config).to(device)
    state = ckpt["model"]
    if "module." in list(state.keys())[0]:
        state = {k.replace("module.", ""): v for k, v in state.items()}
    # Handle finetune checkpoint key nesting
    if any(k.startswith("infilling_pretrain.") for k in state.keys()):
        state = {k.replace("infilling_pretrain.", ""): v
                 for k, v in state.items() if k.startswith("infilling_pretrain.")}
    model.load_state_dict(state)
    model.eval()
    return model, config


def amp_batch_input(
    sequence: str,
    active_motifs: dict,
    hemolytic_motifs: list,
    mode: str,
    batch_size: int,
    tokenizer: Tokenizer,
    max_len: int = 100,
):
    """Prepare batched input tensors for sampling.

    Returns:
        pad_tokens:   (batch_size, max_len) token ids, masked at framework positions
        pad_region:   (batch_size, max_len) region index (0/1/2)
        optimize_pos: array of position indices that will be sampled
        tokenizer:    Tokenizer instance
    """
    seq_len = len(sequence)
    region_index = generate_region_index(sequence, active_motifs, hemolytic_motifs, mode)
    mask = generate_mask(region_index, mode)  # True = optimize

    tokens = tokenizer.seq2idx(sequence)  # (seq_len,)
    region_t = torch.tensor(region_index, dtype=torch.long)

    # Pad to max_len
    pad_len = max(max_len, seq_len)
    pad_tokens = torch.full((pad_len,), tokenizer.idx_pad, dtype=torch.long)
    pad_tokens[:seq_len] = tokens

    pad_region = torch.zeros(pad_len, dtype=torch.long)
    pad_region[:seq_len] = region_t

    # Apply mask tokens to optimize positions
    optimize_pos = np.array([i for i, m in enumerate(mask) if m], dtype=np.int64)
    pad_tokens[optimize_pos] = tokenizer.idx_msk

    # Expand to batch
    pad_tokens = pad_tokens.unsqueeze(0).expand(batch_size, -1).clone()
    pad_region = pad_region.unsqueeze(0).expand(batch_size, -1).clone()

    return pad_tokens, pad_region, optimize_pos


def is_valid_sequence(seq: str) -> bool:
    return len(seq) > 0 and all(aa in VALID_AAS for aa in seq)


def seqs_to_fasta(sequences: list, names: list, save_path: str):
    records = [SeqRecord(Seq(seq), id=name, description="") for seq, name in zip(sequences, names)]
    with open(save_path, "w") as f:
        SeqIO.write(records, f, "fasta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (.pt)")
    parser.add_argument("--input_fasta", type=str, required=True,
                        help="Input FASTA of seed AMP sequences to optimize")
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
        "--independent_scorers", type=str, default="auto",
        help="Independent activity evaluators; auto selects the other three",
    )
    parser.add_argument(
        "--hemolysis-guidance", "--hemolysis_guidance",
        dest="hemolysis_guidance", action=argparse.BooleanOptionalAction, default=False,
        help="Also apply HemoPI2 hemolysis guidance in inp mode",
    )
    parser.add_argument(
        "--eval-hemolysis", "--eval_hemolysis",
        dest="eval_hemolysis", action=argparse.BooleanOptionalAction, default=False,
        help="Include HemoPI2 as a post-generation evaluator",
    )
    parser.add_argument("--motif_type", type=str, default="prosite,regular,merci",
                        help="Enabled motif types: prosite,regular,merci,none")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (>1=diverse, <1=conservative)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of designs per input sequence")
    parser.add_argument("--n_rounds", type=int, default=1,
                        help="Number of independent sampling rounds")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--sample_order", type=str, default="shuffle",
                        choices=["shuffle", "sequential"])
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--pepnet_root", type=str, default="/mnt/wucy/WUCHUYA/PepNet")
    parser.add_argument("--hemopi2_root", type=str, default="/mnt/wucy/WUCHUYA/hemopi2")
    parser.add_argument("--amppred_root", type=str, default="/mnt/wucy/WUCHUYA/AMPpred-MFA")
    parser.add_argument("--iamp_root", type=str, default="/mnt/wucy/WUCHUYA/iAMP-Attenpred")
    parser.add_argument("--unidl_root", type=str, default="/mnt/wucy/WUCHUYA/UniDL4BioPep")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Scale factor for gradient guidance signal")
    args = parser.parse_args()

    if args.guidance_scorer is None:
        legacy = parse_scorer_names(args.scorer or "pepnet")
        selected = [name for name in ACTIVITY_SCORERS if name in legacy]
        if len(selected) != 1:
            parser.error("--scorer compatibility mode requires exactly one activity scorer")
        args.guidance_scorer = selected[0]
        if args.scorer is not None:
            args.hemolysis_guidance = "hemopi2" in legacy
    try:
        independent_names = resolve_independent_scorers(
            args.guidance_scorer, args.independent_scorers
        )
    except ValueError as exc:
        parser.error(str(exc))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, config = load_checkpoint(args.ckpt, device)
    max_len = config.model.max_len

    # Resolve motif files
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    motif_active_path = os.path.join(base_dir, config.amp.motif_active_file)
    motif_hem_path = os.path.join(base_dir, config.amp.motif_hemolytic_file)

    enabled_types = parse_motif_types(args.motif_type)
    active_motifs = load_active_motifs(motif_active_path, enabled_types)
    hemolytic_motifs = load_hemolytic_motifs(motif_hem_path)

    # Load scorers for gradient guidance
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
        device=device,
    )
    evaluator_names = list(independent_names)
    if args.eval_hemolysis:
        evaluator_names.append("hemopi2")
    evaluator = build_scorer_from_args(
        ",".join(evaluator_names) or "none",
        infer_single_guidance=False,
        pepnet_root=args.pepnet_root,
        hemopi2_root=args.hemopi2_root,
        amppred_root=args.amppred_root,
        iamp_root=args.iamp_root,
        unidl_root=args.unidl_root,
        device=device,
    )

    # Output directory
    ckpt_dir = os.path.dirname(os.path.dirname(args.ckpt))
    tag = (f"rounds{args.n_rounds}_mode{args.mode}_guidance{args.guidance_scorer}"
           f"_temp{args.temperature}")
    root_log_dir = args.output or get_new_log_dir(root=ckpt_dir, prefix=tag)
    os.makedirs(root_log_dir, exist_ok=True)
    logger = get_logger("sample", root_log_dir)
    logger.info(f"Output: {root_log_dir}")
    logger.info(args)
    logger.info(f"Activity guidance predictor: {args.guidance_scorer}")
    logger.info(f"Independent activity predictors: {','.join(independent_names)}")

    tokenizer = Tokenizer()

    # Read input sequences
    input_seqs = []
    input_names = []
    for record in SeqIO.parse(args.input_fasta, "fasta"):
        input_seqs.append(str(record.seq))
        input_names.append(record.id)

    logger.info(f"Input sequences: {len(input_seqs)}")

    all_generated = []   # list of (name, original_seq, generated_seq, round)

    for round_idx in range(args.n_rounds):
        current_seed = args.seed + round_idx
        seed_all(current_seed)

        subdir = f"Seed{current_seed}_Mode{args.mode}_Temp{args.temperature}"
        round_dir = os.path.join(root_log_dir, subdir)
        os.makedirs(round_dir, exist_ok=True)
        round_logger = get_logger(f"round_{round_idx}", round_dir)
        round_logger.info(f"Round {round_idx+1}/{args.n_rounds}")

        csv_path = os.path.join(round_dir, "sample_amp_result.csv")
        with open(csv_path, "w") as f:
            f.write("type,name,sequence,seed,temperature\n")

        generated_seqs = []
        generated_names = []

        for idx, (seed_seq, seed_name) in enumerate(
            tqdm(zip(input_seqs, input_names), total=len(input_seqs), desc=f"Round {round_idx+1}")
        ):
            # Record seed sequence
            with open(csv_path, "a") as f:
                f.write(f"seed,{seed_name},{seed_seq},{current_seed},{args.temperature}\n")

            pad_tokens, pad_region, optimize_pos = amp_batch_input(
                seed_seq, active_motifs, hemolytic_motifs, args.mode,
                args.batch_size, tokenizer, max_len
            )

            if args.sample_order == "shuffle":
                np.random.shuffle(optimize_pos)

            remaining = args.num_samples

            with torch.no_grad():
                for pos_i in optimize_pos:
                    # Standard autoregressive decoding
                    pred = model(
                        pad_tokens.to(device),
                        pad_region.to(device),
                        H_chn_type=None
                    )  # (B, L, n_tokens)
                    logits = pred[:, pos_i, :20]  # restrict to 20 standard AAs
                    if args.temperature != 1.0:
                        logits = logits / args.temperature

                    # Gradient guidance (if scorer enabled)
                    if scorer is not None and args.guidance_scale > 0:
                        soft_all = torch.softmax(
                            pred[:, :, :20] / max(args.temperature, 0.1), dim=-1
                        )  # (B, L, 20)
                        with torch.enable_grad():
                            soft_all_g = soft_all.detach().requires_grad_(True)
                            guidance_loss = scorer.combined_guidance_loss(
                                soft_all_g, mode=args.mode,
                                include_hemolysis=args.hemolysis_guidance,
                                lengths=torch.full(
                                    (soft_all_g.shape[0],), len(seed_seq),
                                    dtype=torch.long, device=soft_all_g.device,
                                ),
                            )
                            guidance_loss.backward()
                        if soft_all_g.grad is not None:
                            # combined_guidance_loss is minimized: descend its gradient
                            # so activity increases (and inp hemolysis decreases).
                            logits = logits - args.guidance_scale * soft_all_g.grad[:, pos_i, :]

                    probs = torch.softmax(logits, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    pad_tokens[:, pos_i] = sampled

            # Decode generated sequences
            for b in range(args.batch_size):
                if remaining <= 0:
                    break
                gen_seq = tokenizer.idx2seq(pad_tokens[b])
                if is_valid_sequence(gen_seq):
                    gen_name = f"{seed_name}_amp_sample_{idx}_{b}"
                    with open(csv_path, "a") as f:
                        f.write(f"generated,{gen_name},{gen_seq},{current_seed},{args.temperature}\n")
                    generated_seqs.append(gen_seq)
                    generated_names.append(gen_name)
                    remaining -= 1

        # Save FASTA for this round
        fasta_path = os.path.join(round_dir, "generated_amps.fasta")
        seqs_to_fasta(generated_seqs, generated_names, fasta_path)
        round_logger.info(f"Generated {len(generated_seqs)} sequences -> {fasta_path}")

        # Post-generation scoring
        if evaluator is not None and generated_seqs:
            round_logger.info("Running independent post-generation evaluation...")
            score_df = evaluator.evaluate(generated_seqs)
            score_df.insert(0, "name", generated_names)
            score_df.insert(1, "guidance_activity_scorer", args.guidance_scorer)
            score_path = os.path.join(round_dir, "scores.tsv")
            score_df.to_csv(score_path, sep="\t", index=False)
            round_logger.info(f"Scores saved to {score_path}")
            for column in [c for c in score_df.columns if c.endswith("_score")]:
                round_logger.info(f"Mean {column}: {score_df[column].mean():.4f}")

        all_generated.extend(zip(generated_names, generated_seqs))

    logger.info(f"Total generated: {len(all_generated)} sequences across {args.n_rounds} rounds")
