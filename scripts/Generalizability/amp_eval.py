"""
AMP-Diff evaluation script.

Evaluates generated AMP sequences with any registered activity/hemolysis predictors.
All scorers are toggled at runtime via --scorer flag.

Usage:
    python amp_eval.py --fasta generated_amps.fasta --scorer pepnet,amppred_mfa,iamp_attenpred,unidl4biopep
    python amp_eval.py --fasta generated_amps.fasta --scorer pepnet
    python amp_eval.py --fasta generated_amps.fasta --scorer hemopi2
"""

import os
import sys
import argparse
from Bio import SeqIO

current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

from utils.Generalizability.amp_scorers import build_scorer_from_args
from utils.Generalizability.misc import get_logger


def load_fasta(fasta_path: str):
    sequences, names = [], []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        names.append(record.id)
    return sequences, names


def evaluate(
    fasta_path: str,
    scorer_arg: str,
    output_dir: str = None,
    pepnet_root: str = "/mnt/wucy/WUCHUYA/PepNet",
    hemopi2_root: str = "/mnt/wucy/WUCHUYA/hemopi2",
    amppred_root: str = "/mnt/wucy/WUCHUYA/AMPpred-MFA",
    iamp_root: str = "/mnt/wucy/WUCHUYA/iAMP-Attenpred",
    unidl_root: str = "/mnt/wucy/WUCHUYA/UniDL4BioPep",
):
    """Evaluate sequences in a FASTA file.

    Args:
        fasta_path:   path to generated sequences FASTA
        scorer_arg:   comma-separated registered scorers
        output_dir:   directory to write output CSV (default: same dir as fasta)
        pepnet_root:  path to PepNet repo
        hemopi2_root: path to hemopi2 repo

    Returns:
        pandas DataFrame with evaluation results
    """
    if output_dir is None:
        output_dir = os.path.dirname(fasta_path)

    logger = get_logger("amp_eval", output_dir)
    sequences, names = load_fasta(fasta_path)
    logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")

    scorer = build_scorer_from_args(
        scorer_arg,
        infer_single_guidance=False,
        pepnet_root=pepnet_root,
        hemopi2_root=hemopi2_root,
        amppred_root=amppred_root,
        iamp_root=iamp_root,
        unidl_root=unidl_root,
    )

    if scorer is None:
        logger.warning("No scorer enabled (--scorer none). Nothing to evaluate.")
        return None

    logger.info(f"Evaluating with: {scorer_arg}")
    df = scorer.evaluate(sequences)
    df.insert(0, "name", names)

    out_path = os.path.join(output_dir, "amp_eval_results.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")

    # Log summary stats
    for col in [column for column in df.columns if column.endswith("_score")]:
        if col in df.columns:
            logger.info(f"  {col}: mean={df[col].mean():.4f}  std={df[col].std():.4f}  "
                        f"min={df[col].min():.4f}  max={df[col].max():.4f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated AMP sequences.")
    parser.add_argument("--fasta", type=str, required=True,
                        help="Path to generated sequences FASTA file")
    parser.add_argument(
        "--scorer", type=str,
        default="pepnet,amppred_mfa,iamp_attenpred,unidl4biopep",
        help="Comma-separated scorers (all four activity predictors and/or hemopi2)",
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as FASTA)")
    parser.add_argument("--pepnet_root", type=str, default="/mnt/wucy/WUCHUYA/PepNet")
    parser.add_argument("--hemopi2_root", type=str, default="/mnt/wucy/WUCHUYA/hemopi2")
    parser.add_argument("--amppred_root", type=str, default="/mnt/wucy/WUCHUYA/AMPpred-MFA")
    parser.add_argument("--iamp_root", type=str, default="/mnt/wucy/WUCHUYA/iAMP-Attenpred")
    parser.add_argument("--unidl_root", type=str, default="/mnt/wucy/WUCHUYA/UniDL4BioPep")
    args = parser.parse_args()

    evaluate(
        fasta_path=args.fasta,
        scorer_arg=args.scorer,
        output_dir=args.output_dir,
        pepnet_root=args.pepnet_root,
        hemopi2_root=args.hemopi2_root,
        amppred_root=args.amppred_root,
        iamp_root=args.iamp_root,
        unidl_root=args.unidl_root,
    )
