# Deduplicated AMP Merge

Generated: 2026-06-19T15:13:44.855419+00:00

This directory contains a sequence-deduplicated merge of AMP records from the
database collection. Original database folders were not modified.

Excluded databases: AMPSphere
Excluded sequence keys: 863498

## Files

- `amp_peptide_master.csv`: one row per normalized sequence.
- `amp_evidence_long.csv`: all parsed source rows and experiment/model evidence.
- `amp_confirmed_binary.csv`: confirmed records with binary activity or hemolysis labels.
- `amp_confirmed_continuous.csv`: confirmed records with raw continuous activity or hemolysis values.
- `amp_predicted_binary.csv`: predicted/model records with binary labels.
- `amp_predicted_continuous.csv`: predicted/model records with raw continuous values.
- `amp_unclassified_review.csv`: records retained because evidence type is unclear.
- `amp_combo_9_counts.csv`: activity class by hemolysis class counts at unique-sequence level.
- `merge_report.json`: input coverage, counts, parse errors and rules.

## Counts

- Parsed evidence rows: 622961
- Unique normalized sequences: 118888
- Parsed files: 107
- Files with parse errors: 0

## Notes

The merge preserves duplicate source records in the evidence table. MIC, HC50,
concentration, percentage and comparator strings are kept raw; no cross-unit
normalization is performed. Toxicity/hemolysis labels are not used as the AMP
positive/negative label.
