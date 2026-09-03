# HuDiff Robustness Raw Data
This directory is reserved for the HuDiff release data used by the robustness
module. The files can be downloaded from Hugging Face:

https://huggingface.co/cloud77/HuDiff

## Files

The following files are expected under `dataset/Robustness/raw_data/`:

- `oas_pair_human_data/newprocessed/train_processed_pad_filter.lmdb`: LMDB table of processed paired human antibody VH/VL training records.
- `oas_pair_human_data/newprocessed/train_processed_pad_filter.lmdb-lock`: LMDB lock file associated with the paired human antibody training database.
- `oas_pair_human_data/newprocessed/oas_pair_index_pad_filter.pt`: PyTorch index file for the processed paired human antibody LMDB records.
- `oas_pair_mouse_data/newprocessed/train_processed_pad_filter.lmdb`: LMDB table of processed paired mouse antibody VH/VL training records.
- `oas_pair_mouse_data/newprocessed/train_processed_pad_filter.lmdb-lock`: LMDB lock file associated with the paired mouse antibody training database.
- `oas_pair_mouse_data/newprocessed/oas_pair_index_pad_filter.pt`: PyTorch index file for the processed paired mouse antibody LMDB records.
- `oas_heavy_human_data/heavy_test_nano.lmdb`: LMDB table of processed human heavy-chain records used by heavy-chain/nanobody evaluation workflows.
- `oas_heavy_human_data/heavy_test_nano.lmdb-lock`: LMDB lock file associated with the human heavy-chain LMDB.
- `oas_heavy_human_data/heavy_nano_idx.pt`: PyTorch index file for the processed human heavy-chain LMDB records.
- `oas_vhh_data/vhh_test_nano.lmdb`: LMDB table of processed VHH/nanobody records used by nanobody robustness workflows.
- `oas_vhh_data/vhh_test_nano.lmdb-lock`: LMDB lock file associated with the VHH/nanobody LMDB.
- `oas_vhh_data/vhh_nano_idx.pt`: PyTorch index file for the processed VHH/nanobody LMDB records.

## Notes

These files are runtime dataset artifacts for HuDiff-style robustness training,
finetuning, and evaluation. They are separate from the small benchmark inputs in
`data/Robustness/`, which contain the chicken, rabbit, BH1, Emicizumab, shark349,
HuAb348_H, and Humab25_H robustness cases.

Do not place generated robustness outputs here. New robustness outputs should be
written under `results/Robustness/`.
