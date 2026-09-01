"""
Motif loading and mask generation for AMP-Diff.

Active motif file (AMP_act_Motifs.csv):
  - 'pattern': PROSITE patterns or exact k-mer sequences
  - 'type': categorizes as PROSITE Pattern, MERCI Motif, K-mer Motif, etc.

Hemolytic negative motif file (AMP_hom_negMotifs.csv):
  - 'motif': pre-joined k-mers (one per row)

Runtime flag (--motif-type): comma-separated subset of {prosite, regular, merci, none}
"""

import re
from typing import Dict, List, Set
import torch

# Type classification keywords
_PROSITE_KEYS = ('PROSITE', 'Cysteine Regex')
_MERCI_KEYS = ('MERCI',)
# Everything else (K-mer, Literature, etc.) falls into 'regular'


def _prosite_to_regex(pattern: str) -> re.Pattern:
    """Convert a PROSITE pattern string to a compiled regex.

    PROSITE syntax:
      A-x(0,1)-K-[HR]-x(2)-... where
        x        -> any amino acid (.)
        x(n)     -> .{n}
        x(n,m)   -> .{n,m}
        [ABC]    -> character class
        -        -> separator (ignored)
    """
    parts = pattern.split('-')
    regex_parts = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # x(n) or x(n,m)
        m = re.match(r'^[xX]\((\d+)(?:,(\d+))?\)$', part)
        if m:
            lo, hi = m.group(1), m.group(2)
            regex_parts.append(f'.{{{lo},{hi}}}' if hi else f'.{{{lo}}}')
            continue
        if part.lower() == 'x':
            regex_parts.append('.')
            continue
        if part.startswith('[') and ']' in part:
            regex_parts.append(part)
            continue
        regex_parts.append(re.escape(part))
    return re.compile(''.join(regex_parts))


def load_active_motifs(
    csv_path: str,
    enabled_types: Set[str] = None
) -> Dict[str, list]:
    """Parse AMP_act_Motifs.csv.

    Columns: 'pattern' (motif sequence / PROSITE pattern), 'type' (type / family).

    Args:
        csv_path: path to AMP_act_Motifs.csv
        enabled_types: subset of {'prosite', 'regular', 'merci'}.
                       None = all enabled.

    Returns:
        {'exact': [str, ...], 'prosite': [re.Pattern, ...]}
    """
    import csv
    if enabled_types is None:
        enabled_types = {'prosite', 'regular', 'merci'}

    exact_motifs: List[str] = []
    prosite_patterns: List[re.Pattern] = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            motif = row['pattern'].strip()
            type_family = row['type'].strip()
            if not motif or not type_family:
                continue

            is_prosite = any(k in type_family for k in _PROSITE_KEYS)
            is_merci = any(k in type_family for k in _MERCI_KEYS)

            if is_prosite:
                if 'prosite' in enabled_types:
                    try:
                        prosite_patterns.append(_prosite_to_regex(motif))
                    except Exception:
                        pass
            elif is_merci:
                if 'merci' in enabled_types:
                    exact_motifs.append(motif)
            else:
                if 'regular' in enabled_types:
                    exact_motifs.append(motif)

    return {'exact': exact_motifs, 'prosite': prosite_patterns}


def load_hemolytic_motifs(csv_path: str) -> List[str]:
    """Parse AMP_hom_negMotifs.csv.

    Column 'motif': pre-joined k-mer strings (e.g. 'SKIKK').
    """
    import csv
    motifs: List[str] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            motif = row['motif'].strip()
            if motif:
                motifs.append(motif)
    return motifs


def find_motif_positions(
    sequence: str,
    exact_motifs: List[str],
    prosite_patterns: List[re.Pattern]
) -> List[int]:
    """Return sorted list of 0-based position indices covered by any motif match."""
    covered: Set[int] = set()
    for motif in exact_motifs:
        start = 0
        while True:
            idx = sequence.find(motif, start)
            if idx == -1:
                break
            covered.update(range(idx, idx + len(motif)))
            start = idx + 1
    for pattern in prosite_patterns:
        for m in pattern.finditer(sequence):
            covered.update(range(m.start(), m.end()))
    return sorted(covered)


def generate_region_index(
    sequence: str,
    active_motifs: Dict[str, list],
    hemolytic_motifs: List[str],
    mode: str = 'de'
) -> List[int]:
    """Per-position region label for a given peptide sequence.

    Labels:
      0 = framework (optimized during generation)
      1 = active antimicrobial motif (always fixed)
      2 = hemolytic-negative motif (fixed only in 'inp' mode)

    Active motif positions (label=1) override hemolytic positions (label=2).
    """
    n = len(sequence)
    region = [0] * n

    # Hemolytic (lower priority) first
    for pos in find_motif_positions(sequence, hemolytic_motifs, []):
        if pos < n:
            region[pos] = 2

    # Active motif (higher priority) overwrites
    for pos in find_motif_positions(
        sequence, active_motifs['exact'], active_motifs['prosite']
    ):
        if pos < n:
            region[pos] = 1

    return region


def generate_mask(region_index: List[int], mode: str) -> List[bool]:
    """Per-position boolean mask. True = optimize (will be diffusion-masked).

    de:  only framework positions (0) are regenerated; active motif (1) fixed
    inp: framework (0) regenerated; both motif types (1 and 2) fixed
    """
    if mode == 'de':
        return [r == 0 for r in region_index]
    elif mode == 'inp':
        return [r == 0 for r in region_index]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'de' or 'inp'.")


def parse_motif_types(s: str) -> Set[str]:
    """Parse --motif-type CLI string to a set of enabled type names."""
    if s.strip().lower() == 'none':
        return set()
    return {t.strip().lower() for t in s.split(',')}
