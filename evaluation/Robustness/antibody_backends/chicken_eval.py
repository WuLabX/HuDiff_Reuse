"""
Antibody humanization evaluation for the chicken dataset.

This mirrors the HuAb348 evaluation layout, but the available groups are only:
  - chicken: parental/input chicken antibody sequences
  - sample: HuDiff-Ab generated sequences
"""

import os
import shutil
import subprocess
import tempfile
import sys

import numpy as np
import pandas as pd
from abnumber import Chain

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from evaluation.Robustness.antibody_backends.HuAb348_eval import (
    ABNATIV_DIR,
    BIOPHI_DIR,
    OASIS_DB_PATH,
    cal_all_preservation,
    cal_germline_identity_single,
    cal_vernier_preservation,
    detect_light_chain_type,
    run_ablstm_eval,
    run_abnativ,
    run_oasis,
    seqs_to_fasta,
    seqs_to_paired_fasta,
)
from evaluation.Robustness.T20_eval import main as t20_main
from utils.Robustness.misc import get_logger


def _mean(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(values)) if values else None


def _col(df, *names):
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f'None of columns {names} found in {list(df.columns)}')


def _standardize_pair_df(df):
    df = df.copy()
    h_col = _col(df, 'hseq', 'h_seq')
    l_col = _col(df, 'lseq', 'l_seq')
    if h_col != 'hseq':
        df['hseq'] = df[h_col]
    if l_col != 'lseq':
        df['lseq'] = df[l_col]
    if 'name' not in df.columns:
        df['name'] = [f'chicken_{i}' for i in range(len(df))]
    return df.reset_index(drop=True)


def _get_abnativ_exec(logger):
    abnativ_exec = os.path.expanduser('/mnt/wucy/miniconda3/envs/abnativ/bin/abnativ')
    if os.path.isfile(abnativ_exec):
        logger.info(f'Using Conda AbNatiV: {abnativ_exec}')
        return abnativ_exec
    abnativ_exec = shutil.which('abnativ')
    if abnativ_exec:
        logger.info(f'Using PATH AbNatiV: {abnativ_exec}')
        return abnativ_exec
    for candidate in (os.path.join(ABNATIV_DIR, 'bin', 'abnativ'), os.path.join(ABNATIV_DIR, 'abnativ')):
        if os.path.isfile(candidate):
            logger.info(f'Using local AbNatiV: {candidate}')
            return candidate
    raise FileNotFoundError('AbNatiV executable was not found')


def _score_abnativ_group(label, df, out_dir, abnativ_exec, logger):
    group_dir = os.path.join(out_dir, label)
    os.makedirs(group_dir, exist_ok=True)

    vh_oid = f'{label}_vh'
    vh_score_file = os.path.join(group_dir, f'{vh_oid}_abnativ_seq_scores.csv')
    if not os.path.isfile(vh_score_file):
        vh_fa = os.path.join(group_dir, f'{label}_vh.fasta')
        seqs_to_fasta(df['hseq'].tolist(), [f'{label}_{i}' for i in range(len(df))], vh_fa)
        run_abnativ(abnativ_exec, 'VH', vh_fa, group_dir, vh_oid)
    vh_df = pd.read_csv(vh_score_file)
    vh_mean = float(vh_df['AbNatiV VH Score'].mean())

    light_scores = []
    light_types = [detect_light_chain_type(seq) for seq in df['lseq'].tolist()]
    for lt in ['VKappa', 'VLambda']:
        indices = [i for i, t in enumerate(light_types) if t == lt]
        if not indices:
            continue
        oid = f'{label}_{lt.lower()}'
        score_file = os.path.join(group_dir, f'{oid}_abnativ_seq_scores.csv')
        if not os.path.isfile(score_file):
            l_fa = os.path.join(group_dir, f'{label}_{lt.lower()}.fasta')
            seqs_to_fasta([df.iloc[i]['lseq'] for i in indices], [f'{label}_{i}' for i in indices], l_fa)
            run_abnativ(abnativ_exec, lt, l_fa, group_dir, oid)
        score_df = pd.read_csv(score_file)
        score_col = [c for c in score_df.columns if 'AbNatiV' in c and 'Score' in c][0]
        light_scores.extend(score_df[score_col].tolist())

    vl_mean = _mean(light_scores)
    logger.info(f'  {label} AbNatiV: VH={vh_mean:.4f}, VL={vl_mean:.4f}')
    return vh_mean, vl_mean


def _score_t20_group(label, df, out_dir, logger):
    final_path = os.path.join(out_dir, f'{label}_t20_score.csv')
    if os.path.isfile(final_path):
        t20_df = pd.read_csv(final_path)
    else:
        tmp_dir = os.path.join(out_dir, f'_{label}_t20_tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_csv = os.path.join(tmp_dir, f'{label}_t20_input.csv')
        t20_input = pd.DataFrame({
            'Specific': ['humanization'] * len(df),
            'name': df['name'].astype(str).tolist(),
            'hseq': df['hseq'].tolist(),
            'lseq': df['lseq'].tolist(),
        })
        t20_input.to_csv(tmp_csv, index=False)
        t20_path = t20_main(tmp_csv)
        shutil.copy(t20_path, final_path)
        t20_df = pd.read_csv(final_path)
    h_mean = float(t20_df['h_score'].mean())
    l_mean = float(t20_df['l_score'].mean())
    logger.info(f'  {label} T20: H={h_mean:.4f}, L={l_mean:.4f}')
    return h_mean, l_mean


def _score_oasis_group(label, df, out_dir, logger):
    if not os.path.isfile(OASIS_DB_PATH):
        logger.warning(f'  Skipping {label} OASis: database not found at {OASIS_DB_PATH}')
        return None
    biophi_exec = shutil.which('biophi') or os.path.join(BIOPHI_DIR, 'bin', 'biophi')
    fasta_path = os.path.join(out_dir, f'{label}_paired.fasta')
    xlsx_path = os.path.join(out_dir, f'{label}_oasis.xlsx')
    if not os.path.isfile(xlsx_path):
        seqs_to_paired_fasta(df['hseq'].tolist(), df['lseq'].tolist(), df['name'].astype(str).tolist(), fasta_path)
        run_oasis(biophi_exec, fasta_path, OASIS_DB_PATH, xlsx_path)
    res_df = pd.read_excel(xlsx_path, sheet_name='OASis Curves', index_col=0)
    value = float(res_df['50%'].mean()) if '50%' in res_df.columns else None
    logger.info(f'  {label} OASis 50%: {value:.4f}' if value is not None else f'  {label} OASis 50%: NA')
    return value


def _score_germline_group(label, df, out_dir, logger):
    out_csv = os.path.join(out_dir, f'{label}_germline_identity.csv')
    progress_csv = out_csv + '.progress.csv'
    current_txt = out_csv + '.current.txt'
    if os.path.isfile(out_csv):
        gi_df = pd.read_csv(out_csv)
    else:
        if os.path.isfile(progress_csv) and os.path.getsize(progress_csv) > 0:
            gi_df = pd.read_csv(progress_csv)
            rows = gi_df.to_dict('records')
        else:
            rows = []
        completed = set(str(r['name']) for r in rows if 'name' in r)
        total = len(df)
        for idx, row in df.reset_index(drop=True).iterrows():
            name = str(row['name'])
            if name in completed:
                continue
            with open(current_txt, 'w') as f:
                f.write(f'{label}\t{idx + 1}/{total}\t{name}\n')
            rows.append({
                'name': name,
                'h_germline_identity': cal_germline_identity_single(row['hseq']),
                'l_germline_identity': cal_germline_identity_single(row['lseq']),
            })
            pd.DataFrame(rows).to_csv(progress_csv, index=False)
            if (len(rows) % 25 == 0) or (len(rows) == total):
                logger.info(f'  {label} Germline progress: {len(rows)}/{total}')
        gi_df = pd.DataFrame(rows)
        gi_df.to_csv(out_csv, index=False)
        if os.path.isfile(current_txt):
            os.remove(current_txt)
    h_mean = float(gi_df['h_germline_identity'].mean())
    l_mean = float(gi_df['l_germline_identity'].mean())
    logger.info(f'  {label} Germline Identity: H={h_mean:.4f}, L={l_mean:.4f}')
    return h_mean, l_mean


def _parent_name(sample_name):
    sample_name = str(sample_name)
    if '_human_sample_' in sample_name:
        return sample_name.split('_human_sample_')[0]
    return sample_name


def _score_sample_preservation(chicken_df, sample_df, out_dir, logger):
    out_csv = os.path.join(out_dir, 'sample_preservation.csv')
    if os.path.isfile(out_csv):
        pres_df = pd.read_csv(out_csv)
    else:
        parent_map = {
            str(row['name']): (row['hseq'], row['lseq'])
            for _, row in chicken_df.iterrows()
        }
        rows = []
        for _, row in sample_df.iterrows():
            pname = _parent_name(row['name'])
            if pname not in parent_map:
                continue
            parent_h, parent_l = parent_map[pname]
            try:
                sample_h = Chain(row['hseq'], scheme='kabat')
                sample_l = Chain(row['lseq'], scheme='kabat')
                parent_h_chain = Chain(parent_h, scheme='kabat')
                parent_l_chain = Chain(parent_l, scheme='kabat')
                rows.append({
                    'name': row['name'],
                    'parent_name': pname,
                    'h_all_preservation': cal_all_preservation(sample_h, parent_h_chain),
                    'l_all_preservation': cal_all_preservation(sample_l, parent_l_chain),
                    'h_vernier_preservation': cal_vernier_preservation(sample_h, parent_h_chain),
                    'l_vernier_preservation': cal_vernier_preservation(sample_l, parent_l_chain),
                })
            except Exception:
                continue
        pres_df = pd.DataFrame(rows)
        pres_df.to_csv(out_csv, index=False)

    summary = pd.DataFrame({
        'Type': ['Sample'],
        'H_All_Preservation': [pres_df['h_all_preservation'].mean()],
        'L_All_Preservation': [pres_df['l_all_preservation'].mean()],
        'H_Vernier_Preservation': [pres_df['h_vernier_preservation'].mean()],
        'L_Vernier_Preservation': [pres_df['l_vernier_preservation'].mean()],
    })
    summary.to_csv(os.path.join(out_dir, 'preservation_summary.csv'), index=False)
    logger.info(
        '  sample Preservation: '
        f"H_all={summary.iloc[0]['H_All_Preservation']:.4f}, "
        f"L_all={summary.iloc[0]['L_All_Preservation']:.4f}"
    )


def _score_ablstm_group(label, df, out_dir, logger):
    out_csv = os.path.join(out_dir, f'{label}_ablstm_score.csv')
    if os.path.isfile(out_csv):
        score_df = pd.read_csv(out_csv)
    else:
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        hseqs = df['hseq'].astype(str).tolist()
        scores = [np.nan] * len(hseqs)
        valid_indices = [
            i for i, seq in enumerate(hseqs)
            if seq and set(seq.upper()).issubset(valid_aa)
        ]
        skipped = len(hseqs) - len(valid_indices)
        if skipped:
            logger.info(f'  {label} ABLSTM: skipped {skipped} non-standard heavy-chain sequences')
        if valid_indices:
            valid_scores = run_ablstm_eval([hseqs[i].upper() for i in valid_indices])
            for idx, score in zip(valid_indices, valid_scores):
                scores[idx] = score
        score_df = pd.DataFrame({'Sample': range(len(scores)), 'H_ABLSTM_Score': scores})
        score_df.to_csv(out_csv, index=False)
    score_df['H_ABLSTM_Score'] = pd.to_numeric(score_df['H_ABLSTM_Score'], errors='coerce')
    mean_score = float(score_df['H_ABLSTM_Score'].mean())
    logger.info(f'  {label} ABLSTM: H={mean_score:.4f}')
    return mean_score


def main(sample_path):
    base_dir = os.path.dirname(sample_path)
    eval_dir = os.path.join(base_dir, 'evaluation_results')
    dir_paths = {
        'AbNatiV': os.path.join(eval_dir, '1_AbNatiV'),
        'T20': os.path.join(eval_dir, '2_T20'),
        'OASis': os.path.join(eval_dir, '3_OASis'),
        'Germline_identity': os.path.join(eval_dir, '4_Germline_Identity'),
        'Preservation': os.path.join(eval_dir, '5_Preservation'),
        'Mutation_precision': os.path.join(eval_dir, '6_Mutation_Precision'),
        'ABLSTM': os.path.join(eval_dir, '7_ABLSTM'),
    }
    for path in dir_paths.values():
        os.makedirs(path, exist_ok=True)

    logger = get_logger('chicken_eval', eval_dir, log_name='eval_log.txt')
    logger.info('=' * 80)
    logger.info('Chicken Antibody Humanization Evaluation')
    logger.info(f'Sample file: {sample_path}')
    logger.info(f'Output directory: {eval_dir}')

    sample_all_df = pd.read_csv(sample_path)
    chicken_df = _standardize_pair_df(sample_all_df[sample_all_df['Specific'].isin(['chicken', 'mouse'])])
    sample_df = _standardize_pair_df(sample_all_df[sample_all_df['Specific'] == 'humanization'])
    logger.info(f'Loaded {len(chicken_df)} chicken parental sequences')
    logger.info(f'Loaded {len(sample_df)} generated sample sequences')

    logger.info('1. AbNatiV Scoring...')
    abnativ_exec = _get_abnativ_exec(logger)
    chicken_vh, chicken_vl = _score_abnativ_group('chicken', chicken_df, dir_paths['AbNatiV'], abnativ_exec, logger)
    sample_vh, sample_vl = _score_abnativ_group('sample', sample_df, dir_paths['AbNatiV'], abnativ_exec, logger)
    pd.DataFrame({
        'Type': ['Chicken', 'Sample'],
        'VH_Mean': [chicken_vh, sample_vh],
        'VL_Mean': [chicken_vl, sample_vl],
        'VH_Improve_vs_Chicken': [None, sample_vh - chicken_vh],
        'VL_Improve_vs_Chicken': [None, sample_vl - chicken_vl],
    }).to_csv(os.path.join(dir_paths['AbNatiV'], 'abnativ_summary.csv'), index=False)

    logger.info('2. T20 Scoring...')
    chicken_t20_h, chicken_t20_l = _score_t20_group('chicken', chicken_df, dir_paths['T20'], logger)
    sample_t20_h, sample_t20_l = _score_t20_group('sample', sample_df, dir_paths['T20'], logger)
    pd.DataFrame({
        'Type': ['Chicken', 'Sample'],
        'H_T20_Mean': [chicken_t20_h, sample_t20_h],
        'L_T20_Mean': [chicken_t20_l, sample_t20_l],
        'H_T20_Improve_vs_Chicken': [None, sample_t20_h - chicken_t20_h],
        'L_T20_Improve_vs_Chicken': [None, sample_t20_l - chicken_t20_l],
    }).to_csv(os.path.join(dir_paths['T20'], 't20_summary.csv'), index=False)

    logger.info('3. OASis Scoring...')
    chicken_oasis = _score_oasis_group('chicken', chicken_df, dir_paths['OASis'], logger)
    sample_oasis = _score_oasis_group('sample', sample_df, dir_paths['OASis'], logger)
    pd.DataFrame({
        'Type': ['Chicken', 'Sample'],
        'OASis_50%_Mean': [chicken_oasis, sample_oasis],
        'OASis_Improve_vs_Chicken': [
            None,
            sample_oasis - chicken_oasis if sample_oasis is not None and chicken_oasis is not None else None,
        ],
        'Status': [
            'Skipped' if chicken_oasis is None else 'Done',
            'Skipped' if sample_oasis is None else 'Done',
        ],
        'Reason': [
            '' if chicken_oasis is not None else f'OASis database not found at {OASIS_DB_PATH}',
            '' if sample_oasis is not None else f'OASis database not found at {OASIS_DB_PATH}',
        ],
    }).to_csv(os.path.join(dir_paths['OASis'], 'oasis_summary.csv'), index=False)

    logger.info('4. Germline Identity Scoring...')
    chicken_gi_h, chicken_gi_l = _score_germline_group('chicken', chicken_df, dir_paths['Germline_identity'], logger)
    sample_gi_h, sample_gi_l = _score_germline_group('sample', sample_df, dir_paths['Germline_identity'], logger)
    pd.DataFrame({
        'Type': ['Chicken', 'Sample'],
        'H_Germline_Identity': [chicken_gi_h, sample_gi_h],
        'L_Germline_Identity': [chicken_gi_l, sample_gi_l],
        'H_GI_Improve_vs_Chicken': [None, sample_gi_h - chicken_gi_h],
        'L_GI_Improve_vs_Chicken': [None, sample_gi_l - chicken_gi_l],
    }).to_csv(os.path.join(dir_paths['Germline_identity'], 'germline_identity_summary.csv'), index=False)

    logger.info('5. Preservation Scoring...')
    _score_sample_preservation(chicken_df, sample_df, dir_paths['Preservation'], logger)

    logger.info('6. Mutation Precision Scoring...')
    pd.DataFrame({
        'Metric': ['Mutation precision'],
        'Status': ['Skipped'],
        'Reason': ['Chicken dataset has no paired experimental humanized reference; only chicken and sample groups are available.'],
    }).to_csv(os.path.join(dir_paths['Mutation_precision'], 'mutation_precision_summary.csv'), index=False)

    logger.info('7. ABLSTM Scoring...')
    chicken_ablstm = _score_ablstm_group('chicken', chicken_df, dir_paths['ABLSTM'], logger)
    sample_ablstm = _score_ablstm_group('sample', sample_df, dir_paths['ABLSTM'], logger)
    pd.DataFrame({
        'Type': ['Chicken', 'Sample'],
        'ABLSTM_H_Mean': [chicken_ablstm, sample_ablstm],
        'ABLSTM_Improve_vs_Chicken': [None, sample_ablstm - chicken_ablstm],
    }).to_csv(os.path.join(dir_paths['ABLSTM'], 'ablstm_summary.csv'), index=False)

    logger.info('Chicken evaluation complete.')
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
        print('Usage: python scripts/Robustness/antibody/chicken_eval.py <sample_humanization_result.csv>')
        sys.exit(1)
    main(sys.argv[1])
