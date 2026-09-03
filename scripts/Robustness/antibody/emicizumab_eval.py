"""Antibody humanization evaluation for Emicizumab tasks 1 and 2.

The available groups are only:
  - Emicizumab: parental/input antibody sequences
  - sample: HuDiff-Ab generated sequences

Emicizumab parent rows contain full-length heavy/light chains, so parent and
sample sequences are normalized to variable domains before scoring.
"""

import os
import sys

import pandas as pd
from abnumber import Chain

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.Robustness.misc import get_logger
from antibody_scripts.HuAb348_eval import ABNATIV_DIR, OASIS_DB_PATH
from antibody_scripts.chicken_eval import (
    _get_abnativ_exec,
    _score_abnativ_group,
    _score_t20_group,
    _score_oasis_group,
    _score_germline_group,
    _score_sample_preservation,
    _score_ablstm_group,
    _standardize_pair_df,
)


def _variable_domain(seq):
    seq = str(seq).strip().upper()
    return str(Chain(seq, scheme='imgt').seq)


def _standardize_emicizumab_df(df):
    df = _standardize_pair_df(df)
    df['hseq'] = df['hseq'].map(_variable_domain)
    df['lseq'] = df['lseq'].map(_variable_domain)
    return df


def main(sample_path):
    base_dir = os.path.dirname(sample_path)
    eval_dir = os.path.join(base_dir, 'evaluation_results')
    dir_paths = {
        'AbNatiV': os.path.join(eval_dir, '1_AbNatiV'),
        'T20': os.path.join(eval_dir, '2_T20'),
        'OASis': os.path.join(eval_dir, '3_OASis'),
        'Germline_identity': os.path.join(eval_dir, '4_Germline_Identity'),
        'Preservation': os.path.join(eval_dir, '5_Preservation'),
        'ABLSTM': os.path.join(eval_dir, '7_ABLSTM'),
    }
    for path in dir_paths.values():
        os.makedirs(path, exist_ok=True)

    logger = get_logger('emicizumab_eval', eval_dir, log_name='eval_log.txt')
    logger.info('=' * 80)
    logger.info('Emicizumab Antibody Humanization Evaluation')
    logger.info(f'Sample file: {sample_path}')
    logger.info(f'Output directory: {eval_dir}')

    sample_all_df = pd.read_csv(sample_path)
    parent_df = _standardize_emicizumab_df(
        sample_all_df[sample_all_df['Specific'].isin(['Emicizumab', 'emicizumab', 'mouse'])]
    )
    sample_df = _standardize_emicizumab_df(sample_all_df[sample_all_df['Specific'] == 'humanization'])
    logger.info(f'Loaded {len(parent_df)} Emicizumab parental sequences')
    logger.info(f'Loaded {len(sample_df)} generated sample sequences')

    logger.info('1. AbNatiV Scoring...')
    abnativ_exec = _get_abnativ_exec(logger)
    parent_vh, parent_vl = _score_abnativ_group('Emicizumab', parent_df, dir_paths['AbNatiV'], abnativ_exec, logger)
    sample_vh, sample_vl = _score_abnativ_group('sample', sample_df, dir_paths['AbNatiV'], abnativ_exec, logger)
    pd.DataFrame({
        'Type': ['Emicizumab', 'Sample'],
        'VH_Mean': [parent_vh, sample_vh],
        'VL_Mean': [parent_vl, sample_vl],
        'VH_Improve_vs_Emicizumab': [None, sample_vh - parent_vh],
        'VL_Improve_vs_Emicizumab': [None, sample_vl - parent_vl],
    }).to_csv(os.path.join(dir_paths['AbNatiV'], 'abnativ_summary.csv'), index=False)

    logger.info('2. T20 Scoring...')
    parent_t20_h, parent_t20_l = _score_t20_group('Emicizumab', parent_df, dir_paths['T20'], logger)
    sample_t20_h, sample_t20_l = _score_t20_group('sample', sample_df, dir_paths['T20'], logger)
    pd.DataFrame({
        'Type': ['Emicizumab', 'Sample'],
        'H_T20_Mean': [parent_t20_h, sample_t20_h],
        'L_T20_Mean': [parent_t20_l, sample_t20_l],
        'H_T20_Improve_vs_Emicizumab': [None, sample_t20_h - parent_t20_h],
        'L_T20_Improve_vs_Emicizumab': [None, sample_t20_l - parent_t20_l],
    }).to_csv(os.path.join(dir_paths['T20'], 't20_summary.csv'), index=False)

    logger.info('3. OASis Scoring...')
    parent_oasis = _score_oasis_group('Emicizumab', parent_df, dir_paths['OASis'], logger)
    sample_oasis = _score_oasis_group('sample', sample_df, dir_paths['OASis'], logger)
    pd.DataFrame({
        'Type': ['Emicizumab', 'Sample'],
        'OASis_50%_Mean': [parent_oasis, sample_oasis],
        'OASis_Improve_vs_Emicizumab': [
            None,
            sample_oasis - parent_oasis if sample_oasis is not None and parent_oasis is not None else None,
        ],
        'Status': [
            'Skipped' if parent_oasis is None else 'Done',
            'Skipped' if sample_oasis is None else 'Done',
        ],
        'Reason': [
            '' if parent_oasis is not None else f'OASis database not found at {OASIS_DB_PATH}',
            '' if sample_oasis is not None else f'OASis database not found at {OASIS_DB_PATH}',
        ],
    }).to_csv(os.path.join(dir_paths['OASis'], 'oasis_summary.csv'), index=False)

    logger.info('4. Germline Identity Scoring...')
    parent_gi_h, parent_gi_l = _score_germline_group('Emicizumab', parent_df, dir_paths['Germline_identity'], logger)
    sample_gi_h, sample_gi_l = _score_germline_group('sample', sample_df, dir_paths['Germline_identity'], logger)
    pd.DataFrame({
        'Type': ['Emicizumab', 'Sample'],
        'H_Germline_Identity': [parent_gi_h, sample_gi_h],
        'L_Germline_Identity': [parent_gi_l, sample_gi_l],
        'H_GI_Improve_vs_Emicizumab': [None, sample_gi_h - parent_gi_h],
        'L_GI_Improve_vs_Emicizumab': [None, sample_gi_l - parent_gi_l],
    }).to_csv(os.path.join(dir_paths['Germline_identity'], 'germline_identity_summary.csv'), index=False)

    logger.info('5. Preservation Scoring...')
    _score_sample_preservation(parent_df, sample_df, dir_paths['Preservation'], logger)

    logger.info('7. ABLSTM Scoring...')
    parent_ablstm = _score_ablstm_group('Emicizumab', parent_df, dir_paths['ABLSTM'], logger)
    sample_ablstm = _score_ablstm_group('sample', sample_df, dir_paths['ABLSTM'], logger)
    pd.DataFrame({
        'Type': ['Emicizumab', 'Sample'],
        'ABLSTM_H_Mean': [parent_ablstm, sample_ablstm],
        'ABLSTM_Improve_vs_Emicizumab': [None, sample_ablstm - parent_ablstm],
    }).to_csv(os.path.join(dir_paths['ABLSTM'], 'ablstm_summary.csv'), index=False)

    logger.info('Emicizumab evaluation complete.')
    return eval_dir


if __name__ == '__main__':
    current_path = os.getenv('PATH', '')
    current_pythonpath = os.getenv('PYTHONPATH', '')
    abnativ_lib = '/mnt/wucy/miniconda3/envs/abnativ/lib'
    abnativ_bin = '/mnt/wucy/miniconda3/envs/abnativ/bin'
    biophi_bin = '/mnt/wucy/miniconda3/envs/biophi/bin'
    if os.path.exists(abnativ_lib):
        os.environ['LD_LIBRARY_PATH'] = f"{abnativ_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    if os.path.exists(abnativ_bin):
        os.environ['PATH'] = f"{abnativ_bin}:{current_path}"
    if os.path.exists(biophi_bin):
        os.environ['PATH'] = f"{biophi_bin}:{os.environ.get('PATH', '')}"
    if os.path.exists(ABNATIV_DIR):
        os.environ['PYTHONPATH'] = f"{ABNATIV_DIR}:{current_pythonpath}"
    if len(sys.argv) < 2:
        print('Usage: python scripts/Robustness/antibody/emicizumab_eval.py <sample_humanization_result.csv>')
        sys.exit(1)
    main(sys.argv[1])
